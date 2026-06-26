"""App-prompt consistency eval: generate K structures per prompt, relax, then LM-judge property consistency. Shard via --row_start/--row_end; needs OPENAI_API_KEY."""
from __future__ import annotations


import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Pauling electronegativity.*")
warnings.filterwarnings("ignore", message=".*fractional coordinates.*")

import numpy as np
import pyarrow.parquet as pq
import torch
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from llm_judge import (  # noqa: E402
    DEFAULT_MODEL, batch_judge, build_app_consistency_messages,
    get_failure_counts, parse_score, reset_failure_counts,
)
from paths import DATA_ROOT  # noqa: E402


# Stratified mode samples max_rows/N_categories rows per category.
APP_CATEGORIES = [
    ("narrow-bandgap",  ["narrow-bandgap"]),
    ("wide-bandgap",    ["wide-bandgap", "wide bandgap"]),
    ("semiconductor",   ["semiconductor"]),
    ("thermal",         ["thermal"]),
    ("magnetic",        ["magnetic"]),
    ("catalyst",        ["catalyst"]),
    ("battery",         ["battery"]),
    ("photovoltaic",    ["photovoltaic"]),
    ("superconductor",  ["superconductor"]),
    ("ferroelectric",   ["ferroelectric"]),
    ("perovskite",      ["perovskite"]),
]


def _classify_app_prompt(prompt: str) -> str:
    p = (prompt or "").lower()
    for cat, kws in APP_CATEGORIES:
        if any(kw in p for kw in kws):
            return cat
    return "other"


def _selected_app_rows(parquet_path: Path, max_rows: int, seed: int,
                       stratify_per_category: bool = False) -> list[dict]:
    """Hash-deterministic eval-only subset; stratify_per_category balances tail categories like perovskite."""
    pf = pq.ParquetFile(parquet_path)
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=10000, columns=["row_id", "user_prompt"]):
        for r in batch.to_pylist():
            h = int(hashlib.md5(f"{r['row_id']}:{seed}".encode()).hexdigest(), 16)
            rows.append({**r, "_h": h})
    rows.sort(key=lambda r: r["_h"])
    if not stratify_per_category:
        return rows[:max_rows]
    buckets = defaultdict(list)
    for r in rows:
        cat = _classify_app_prompt(r["user_prompt"])
        buckets[cat].append({**r, "_cat": cat})
    n_cats = len(APP_CATEGORIES)
    per_cat = max(1, max_rows // n_cats)
    selected: list[dict] = []
    for cat, _ in APP_CATEGORIES:
        selected.extend(buckets.get(cat, [])[:per_cat])
    return selected[:max_rows]


def _formula_summary(struct: Structure) -> dict:
    elements_set = sorted({str(e) for e in struct.composition.elements})
    formula = struct.composition.reduced_formula
    try:
        sg = SpacegroupAnalyzer(struct, symprec=0.1).get_space_group_symbol()
    except Exception:
        sg = "?"
    n_atoms = int(struct.num_sites)
    vol = float(struct.volume)
    vpa = vol / max(n_atoms, 1)
    mass_amu = sum(atomic_masses[atomic_numbers[el]] for el in [str(s.specie) for s in struct])
    density = float(mass_amu * 1.66054 / max(vol, 1e-3))
    return {
        "formula": formula, "space_group": sg, "n_atoms": n_atoms,
        "elements": elements_set, "density": density, "volume_per_atom": vpa,
    }


def _formation_energy_per_atom_from_relaxed(atoms: Atoms) -> float:
    """Total energy/atom (eV) from MatterSim; elemental reference not subtracted, fine for relative-rank judging."""
    e = atoms.info.get("total_energy")
    if e is None:
        return float("nan")
    try:
        return float(e) / max(1, len(atoms))
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True)
    ap.add_argument("--atoms_mapper", required=True)
    ap.add_argument("--app_parquet", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_app.parquet")))
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_rows", type=int, default=50,
                    help="Held-out app prompts to evaluate. Hash-deterministic via --seed.")
    ap.add_argument("--K", type=int, default=20,
                    help="Generations per prompt.")
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify_per_category", action="store_true",
                    help="Sample max_rows/N rows per app category (top-10 categories) "
                         "instead of hash-random over the full parquet. Lifts perov + "
                         "tail categories to comparable n with bulk semicond/bandgap.")
    ap.add_argument("--judge_model", default=DEFAULT_MODEL)
    ap.add_argument("--judge_concurrency", type=int, default=16)
    ap.add_argument("--judge_only", action="store_true",
                    help="Skip generation + relax + judge_items construction; "
                         "read existing predictions.jsonl from --out_dir and re-run "
                         "ONLY the LLM judge phase. Useful when a prior run hit "
                         "rate-limit 429s and you don't want to redo relax.")
    ap.add_argument("--judge_max_per_prompt", type=int, default=0,
                    help="Cap number of judge calls per prompt (default 0 = no cap). "
                         "Set to e.g. 3 to dramatically reduce OpenAI API volume when "
                         "rate-limited. Per-prompt-mean is computed over the sampled "
                         "subset, so headlines remain comparable.")
    ap.add_argument("--mattersim_potential_path", type=str, default=None)
    ap.add_argument("--skip_relax", action="store_true",
                    help="Skip MatterSim relaxation (judge sees raw generation properties; "
                         "only useful for fast iteration on the judge prompt).")
    ap.add_argument("--row_start", type=int, default=0)
    ap.add_argument("--row_end", type=int, default=-1)
    ap.add_argument("--diffusion_seed", type=int, default=1337,
                    help="Seed for diffusion noise. Per-prompt offset added so "
                         "reordering doesn't change individual outputs.")
    ap.add_argument("--skip_generation_use_existing", action="store_true",
                    help="Skip generation; read existing structures from "
                         "out_dir/generations/<row_id>/generated_crystals_cif.zip. "
                         "Useful when a prior run crashed during relax/judge but "
                         "the on-disk CIFs are already there. Avoids reloading ALM + "
                         "MatterGen and re-running the diffusion sampler.")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_app] writing → {args.out_dir}", flush=True)
    t0 = time.time()

    # --judge_only: replay only the LLM judge phase against existing predictions.jsonl.
    if args.judge_only:
        preds_path = args.out_dir / "predictions.jsonl"
        if not preds_path.exists():
            raise SystemExit(f"--judge_only requires existing predictions.jsonl at {preds_path}")
        print(f"[eval_app] --judge_only: reading {preds_path}", flush=True)
        existing: list[dict] = []
        with open(preds_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                existing.append(json.loads(line))
        print(f"[eval_app] {len(existing)} existing predictions loaded", flush=True)

        per_prompt_count: dict[str, int] = defaultdict(int)
        judge_items: list[dict] = []
        keep_idx: list[int] = []
        n_dropped_cap = 0
        for i, ex in enumerate(existing):
            req = ("prompt", "formula", "density", "volume_per_atom",
                   "formation_energy_per_atom", "elements", "space_group", "n_atoms")
            if not all(k in ex for k in req):
                continue
            rid = ex["row_id"]
            if args.judge_max_per_prompt > 0 and per_prompt_count[rid] >= args.judge_max_per_prompt:
                n_dropped_cap += 1
                continue
            per_prompt_count[rid] += 1
            judge_items.append({k: ex[k] for k in (
                "row_id", "prompt", "formula", "space_group", "n_atoms",
                "elements", "density", "volume_per_atom", "formation_energy_per_atom",
            )})
            keep_idx.append(i)
        if args.judge_max_per_prompt > 0:
            print(f"[eval_app] judge_max_per_prompt={args.judge_max_per_prompt}: kept "
                  f"{len(judge_items)} / dropped {n_dropped_cap} calls", flush=True)

        reset_failure_counts()
        print(f"[eval_app] dispatching {len(judge_items)} judge calls "
              f"(model={args.judge_model}, concurrency={args.judge_concurrency}, retry+backoff on 429) ...",
              flush=True)
        verdicts = asyncio.run(batch_judge(
            items=judge_items,
            build_messages_fn=build_app_consistency_messages,
            model=args.judge_model,
            concurrency=args.judge_concurrency,
        ))
        fc = get_failure_counts()
        if fc:
            print(f"[eval_app] judge failures: {fc}", flush=True)

        per_prompt: dict[str, list[int]] = defaultdict(list)
        for back_i, verdict in zip(keep_idx, verdicts):
            score = parse_score(verdict, default=0)
            existing[back_i]["judge_score"] = score
            existing[back_i]["judge_verdict"] = verdict.get("verdict") if verdict else None
            existing[back_i]["judge_reason"] = verdict.get("reason") if verdict else None
            existing[back_i]["extracted_application"] = (
                verdict.get("extracted_application") if verdict else None
            )
            per_prompt[existing[back_i]["row_id"]].append(score)

        per_prompt_mean = {rid: float(np.mean(scores)) for rid, scores in per_prompt.items() if scores}
        overall_mean = float(np.mean(list(per_prompt_mean.values()))) if per_prompt_mean else 0.0
        score_2_rate = float(np.mean([
            1 if existing[back_i].get("judge_score") == 2 else 0 for back_i in keep_idx
        ]))

        metrics = {
            "n_judge_calls": len(judge_items),
            "n_judge_failures": int(sum(fc.values())) if fc else 0,
            "judge_model": args.judge_model,
            "judge_only_replay": True,
            "overall_consistency_mean_per_prompt": overall_mean,
            "fraction_score_2": score_2_rate,
            "per_prompt_mean": per_prompt_mean,
            "alm_checkpoint": str(args.alm_checkpoint),
            "wallclock_sec": time.time() - t0,
        }
        with open(args.out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(preds_path, "w") as f:
            for ex in existing:
                f.write(json.dumps(ex) + "\n")
        print(f"[eval_app] HEADLINE — overall_consistency_mean_per_prompt = {overall_mean:.3f} / 2.0", flush=True)
        print(f"[eval_app]            fraction_score_2 = {score_2_rate:.3f}", flush=True)
        print(f"[eval_app] DONE (judge-only replay) in {time.time()-t0:.0f}s", flush=True)
        return 0

    # ── 1. Pick rows ──
    rows = _selected_app_rows(args.app_parquet, args.max_rows, args.seed,
                              stratify_per_category=args.stratify_per_category)
    if args.row_end < 0:
        args.row_end = len(rows)
    rows = rows[args.row_start:args.row_end]
    print(f"[eval_app] {len(rows)} prompts (seed={args.seed}, range={args.row_start}-{args.row_end})", flush=True)

    # ── 2. Load model + generate (or skip generation, read existing CIFs) ──
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # alm/

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = [r["user_prompt"] for r in rows]
    prompt_ids = [r["row_id"] for r in rows]
    gen_root = args.out_dir / "generations"
    gen_root.mkdir(parents=True, exist_ok=True)

    if args.skip_generation_use_existing:
        from zipfile import ZipFile
        from pymatgen.io.cif import CifParser
        n_have = 0
        n_missing = 0
        kept_rows: list[dict] = []
        structures_per_prompt: list[list] = []
        for r in rows:
            zip_path = gen_root / r["row_id"] / "generated_crystals_cif.zip"
            if not zip_path.exists():
                n_missing += 1
                continue
            gens = []
            try:
                with ZipFile(zip_path) as zf:
                    for name in zf.namelist():
                        if not name.endswith(".cif"):
                            continue
                        cif = zf.read(name).decode("utf-8", errors="ignore")
                        try:
                            s = CifParser.from_str(cif).parse_structures(primitive=False)[0]
                            gens.append(s)
                        except Exception:
                            pass
            except Exception as exc:
                print(f"[eval_app] error reading {zip_path}: {exc}", flush=True)
                continue
            if not gens:
                n_missing += 1
                continue
            kept_rows.append(r)
            structures_per_prompt.append(gens)
            n_have += 1
        rows = kept_rows
        prompts = [r["user_prompt"] for r in rows]
        prompt_ids = [r["row_id"] for r in rows]
        print(f"[eval_app] reused on-disk generations: {n_have} prompts found, "
              f"{n_missing} missing/empty (skipped). "
              f"Total structures: {sum(len(g) for g in structures_per_prompt)}", flush=True)
        if n_have == 0:
            raise SystemExit("No on-disk generations found to reuse — disable --skip_generation_use_existing or check the out_dir path")
    else:
        from generate_stage3 import generate_for_prompts, load_alm_and_pl_module
        print(f"[eval_app] loading ALM + MatterGen on {device} ...", flush=True)
        alm, tokenizer, pl_module, K_tokens = load_alm_and_pl_module(
            alm_checkpoint=args.alm_checkpoint,
            atoms_mapper=args.atoms_mapper,
            mattergen_pretrained=args.mattergen_pretrained,
            device=device,
        )
        print(f"[eval_app] generating {len(prompts)} × K={args.K} structures ...", flush=True)
        structures_per_prompt = generate_for_prompts(
            prompts=prompts, alm=alm, tokenizer=tokenizer, pl_module=pl_module,
            out_root=gen_root, batch_size=args.K, num_batches=1,
            diffusion_guidance_factor=args.guidance_factor,
            prompt_ids=prompt_ids, save_meta=False,
            diffusion_seed=args.diffusion_seed,
        )
        print(f"[eval_app] generation done in {time.time()-t0:.0f}s", flush=True)

    # ── 3. Relax + characterize ──
    from structure_metrics import relax_structures_mattersim
    flat: list = []
    flat_back_idx: list[tuple[int, int]] = []  # (prompt_i, gen_j)
    n_filtered_z = 0
    for i, gens in enumerate(structures_per_prompt):
        for j, g in enumerate(gens):
            if isinstance(g, Structure):
                s = g
            else:
                try:
                    s = AseAtomsAdaptor.get_structure(g)
                except Exception:
                    continue
            # Drop Z outside [1,94] before batched relax: one bad Z device-asserts and corrupts the CUDA context.
            try:
                zs = [int(site.specie.Z) for site in s]
            except Exception:
                continue
            if not zs or any(z < 1 or z > 94 for z in zs):
                n_filtered_z += 1
                continue
            flat.append(s)
            flat_back_idx.append((i, j))
    if n_filtered_z:
        print(f"[eval_app] pre-filtered {n_filtered_z} structures with out-of-range Z (kept Z∈[1,94])", flush=True)

    if args.skip_relax:
        relaxed_atoms = [AseAtomsAdaptor.get_atoms(s) for s in flat]
        relaxed_atoms = [a for a in relaxed_atoms if a is not None]
    else:
        print(f"[eval_app] MatterSim relaxing {len(flat)} structures ...", flush=True)
        relaxed_atoms, _ = relax_structures_mattersim(
            flat, device=str(device),
            potential_path=args.mattersim_potential_path,
            fmax=0.05, max_n_steps=500,
        )
        print(f"[eval_app] relax done in {time.time()-t0:.0f}s total", flush=True)

    judge_items: list[dict] = []
    judge_back_idx: list[tuple[int, int]] = []  # (prompt_i, gen_j)
    skipped = 0
    for (pi, gj), atoms in zip(flat_back_idx, relaxed_atoms):
        try:
            struct = AseAtomsAdaptor.get_structure(atoms)
            summary = _formula_summary(struct)
            fe = _formation_energy_per_atom_from_relaxed(atoms)
            if not (fe == fe):  # NaN
                fe = 0.0
            judge_items.append({
                "row_id": rows[pi]["row_id"],
                "prompt": rows[pi]["user_prompt"],
                "formation_energy_per_atom": fe,
                **summary,
            })
            judge_back_idx.append((pi, gj))
        except Exception:
            skipped += 1
    print(f"[eval_app] characterized {len(judge_items)} structures, "
          f"{skipped} skipped (relaxation/parse failures)", flush=True)

    # ── 4. Batch LLM judge ──
    reset_failure_counts()
    print(f"[eval_app] dispatching {len(judge_items)} judge calls "
          f"(model={args.judge_model}, concurrency={args.judge_concurrency}) ...", flush=True)
    verdicts = asyncio.run(batch_judge(
        items=judge_items,
        build_messages_fn=build_app_consistency_messages,
        model=args.judge_model,
        concurrency=args.judge_concurrency,
    ))
    fc = get_failure_counts()
    if fc:
        print(f"[eval_app] judge failures: {fc}", flush=True)

    # ── 5. Aggregate ──
    per_prompt: dict[str, list[int]] = defaultdict(list)
    examples: list[dict] = []
    for item, verdict in zip(judge_items, verdicts):
        score = parse_score(verdict, default=0)
        per_prompt[item["row_id"]].append(score)
        examples.append({
            **item,
            "judge_verdict": verdict.get("verdict") if verdict else None,
            "judge_score": score,
            "judge_reason": verdict.get("reason") if verdict else None,
            "extracted_application": verdict.get("extracted_application") if verdict else None,
        })

    per_prompt_mean = {rid: float(np.mean(scores)) for rid, scores in per_prompt.items() if scores}
    overall_mean = float(np.mean(list(per_prompt_mean.values()))) if per_prompt_mean else 0.0
    overall_max_score_rate = float(np.mean([s == 2 for scores in per_prompt.values() for s in scores]))

    metrics = {
        "n_prompts": len(rows),
        "n_judge_calls": len(judge_items),
        "n_judge_failures": int(sum(fc.values())) if fc else 0,
        "judge_model": args.judge_model,
        "K": args.K,
        "guidance_factor": args.guidance_factor,
        "skip_relax": bool(args.skip_relax),
        "overall_consistency_mean_per_prompt": overall_mean,  # in [0,2]
        "fraction_score_2": overall_max_score_rate,  # fraction of gens scored "consistent"
        "per_prompt_mean": per_prompt_mean,
        "alm_checkpoint": str(args.alm_checkpoint),
        "wallclock_sec": time.time() - t0,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.out_dir / "predictions.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"[eval_app] wrote {args.out_dir}/metrics.json + predictions.jsonl", flush=True)
    print(f"[eval_app] HEADLINE — overall_consistency_mean_per_prompt = {overall_mean:.3f} / 2.0", flush=True)
    print(f"[eval_app]            fraction_score_2 = {overall_max_score_rate:.3f}", flush=True)
    print(f"[eval_app] DONE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
