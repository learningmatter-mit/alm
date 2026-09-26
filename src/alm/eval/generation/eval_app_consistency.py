"""App-prompt consistency eval: generate K structures per prompt, relax them, then LLM-judge property consistency. Shard via --row_start/--row_end; needs OPENAI_API_KEY.

Every requested generation stays in the per-prompt denominator: structures that fail conversion,
have Z outside 1-94, get a NaN energy after relaxation or fail characterization score 0.
"""
from __future__ import annotations


import argparse
import asyncio
import hashlib
import json
import os
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

from llm_judge import (  # noqa: E402
    DEFAULT_MODEL, batch_judge, build_app_consistency_messages,
    get_failure_counts, parse_score, reset_failure_counts,
)
from paths import ALM_BENCH  # noqa: E402


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


def _app_messages(item: dict) -> list[dict]:
    """Judge messages; a missing energy (None, e.g. under --skip_relax) is shown to the judge as unknown."""
    if item.get("formation_energy_per_atom") is not None:
        return build_app_consistency_messages(item)
    msgs = build_app_consistency_messages({**item, "formation_energy_per_atom": float("nan")})
    for m in msgs:
        m["content"] = m["content"].replace(
            "formation_energy_per_atom: nan eV/atom", "formation_energy_per_atom: unknown")
    return msgs


def _formation_energy_per_atom_from_relaxed(atoms: Atoms) -> float:
    """Total energy/atom (eV) from MatterSim; no elemental reference is subtracted (fine for relative judging)."""
    e = atoms.info.get("total_energy")
    if e is None:
        return float("nan")
    return float(e) / max(1, len(atoms))


def aggregate_scores(row_ids: list[str], scored: list[tuple[str, int]]) -> dict:
    """Per-prompt mean judge score over every prompt in row_ids (a prompt with no scores gets 0), plus the fraction of 2s over all scores."""
    per_prompt: dict[str, list[int]] = {rid: [] for rid in row_ids}
    for rid, sc in scored:
        per_prompt.setdefault(rid, []).append(sc)
    per_prompt_mean = {rid: float(np.mean(v)) if v else 0.0 for rid, v in per_prompt.items()}
    all_scores = [sc for v in per_prompt.values() for sc in v]
    return {
        "per_prompt_mean": per_prompt_mean,
        "overall_consistency_mean_per_prompt": (
            float(np.mean(list(per_prompt_mean.values()))) if per_prompt_mean else 0.0),
        "fraction_score_2": float(np.mean([sc == 2 for sc in all_scores])) if all_scores else 0.0,
    }


def _failed_item(row: dict, failure: str, summary: dict | None = None) -> dict:
    return {"row_id": row["row_id"], "prompt": row["user_prompt"], "failed": True,
            "failure": failure, **(summary or {})}


def flatten_for_relax(structures_per_prompt: list[list], rows: list[dict], K: int):
    """Collect relaxable structures (first K per prompt); unconvertible, out-of-range-Z and missing generations become failed items."""
    flat: list = []
    flat_back_idx: list[tuple[int, int]] = []  # (prompt_i, gen_j)
    failed_items: list[dict] = []
    for i, gens in enumerate(structures_per_prompt):
        for j, g in enumerate(gens[:K]):
            try:
                s = g if isinstance(g, Structure) else AseAtomsAdaptor.get_structure(g)
                zs = [int(site.specie.Z) for site in s]
            except Exception:
                failed_items.append(_failed_item(rows[i], "conversion_error"))
                continue
            # Z outside [1, 94] makes MatterSim device-assert, which corrupts the CUDA context for the batch.
            if not zs or any(z < 1 or z > 94 for z in zs):
                failed_items.append(_failed_item(rows[i], "z_out_of_range"))
                continue
            flat.append(s)
            flat_back_idx.append((i, j))
        for _ in range(K - min(len(gens), K)):
            failed_items.append(_failed_item(rows[i], "missing_generation"))
    return flat, flat_back_idx, failed_items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True)
    ap.add_argument("--atoms_mapper", required=True)
    ap.add_argument("--app_parquet", type=Path,
                    default=Path(os.path.join(ALM_BENCH, "alm_bench/eval/app.parquet")))
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_rows", type=int, default=50,
                    help="Number of held-out app prompts (hash-selected with --seed).")
    ap.add_argument("--K", type=int, default=20,
                    help="Generations per prompt.")
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stratify_per_category", action="store_true",
                    help="Sample an equal number of prompts from each application category.")
    ap.add_argument("--judge_model", default=DEFAULT_MODEL)
    ap.add_argument("--judge_concurrency", type=int, default=16)
    ap.add_argument("--judge_only", action="store_true",
                    help="Re-run only the LLM judge on an existing predictions.jsonl in --out_dir.")
    ap.add_argument("--judge_max_per_prompt", type=int, default=0,
                    help="Cap on judge calls per prompt in --judge_only mode (0 = no cap).")
    ap.add_argument("--mattersim_potential_path", type=str, default=None)
    ap.add_argument("--skip_relax", action="store_true",
                    help="Skip MatterSim relaxation; the judge sees unrelaxed properties.")
    ap.add_argument("--row_start", type=int, default=0)
    ap.add_argument("--row_end", type=int, default=-1)
    ap.add_argument("--diffusion_seed", type=int, default=1337,
                    help="Diffusion noise seed; a per-prompt offset keeps outputs independent of prompt order.")
    ap.add_argument("--skip_generation_use_existing", action="store_true",
                    help="Skip generation and read out_dir/generations/<row_id>/generated_crystals_cif.zip from a previous run.")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_app] writing to {args.out_dir}", flush=True)
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
        failed_row_ids: list[str] = []
        for i, ex in enumerate(existing):
            if ex.get("failed"):
                failed_row_ids.append(ex["row_id"])
                continue
            req = ("prompt", "formula", "density", "volume_per_atom",
                   "formation_energy_per_atom", "elements", "space_group", "n_atoms")
            if not all(k in ex for k in req):
                failed_row_ids.append(ex["row_id"])  # incomplete record scores 0
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
            build_messages_fn=_app_messages,
            model=args.judge_model,
            concurrency=args.judge_concurrency,
        ))
        fc = get_failure_counts()
        if fc:
            print(f"[eval_app] judge failures: {fc}", flush=True)

        scored: list[tuple[str, int]] = [(rid, 0) for rid in failed_row_ids]
        for back_i, verdict in zip(keep_idx, verdicts):
            score = parse_score(verdict, default=0)
            existing[back_i]["judge_score"] = score
            existing[back_i]["judge_verdict"] = verdict.get("verdict") if verdict else None
            existing[back_i]["judge_reason"] = verdict.get("reason") if verdict else None
            existing[back_i]["extracted_application"] = (
                verdict.get("extracted_application") if verdict else None
            )
            scored.append((existing[back_i]["row_id"], score))

        agg = aggregate_scores(list(dict.fromkeys(ex["row_id"] for ex in existing)), scored)
        overall_mean = agg["overall_consistency_mean_per_prompt"]
        score_2_rate = agg["fraction_score_2"]
        per_prompt_mean = agg["per_prompt_mean"]

        metrics = {
            "n_judge_calls": len(judge_items),
            "n_judge_failures": int(sum(fc.values())) if fc else 0,
            "n_failed_structures": len(failed_row_ids),
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
        print(f"[eval_app] overall_consistency_mean_per_prompt = {overall_mean:.3f} / 2.0", flush=True)
        print(f"[eval_app] fraction_score_2 = {score_2_rate:.3f}", flush=True)
        print(f"[eval_app] done (judge-only replay) in {time.time()-t0:.0f}s", flush=True)
        return 0

    # 1. Pick rows.
    rows = _selected_app_rows(args.app_parquet, args.max_rows, args.seed,
                              stratify_per_category=args.stratify_per_category)
    if args.row_end < 0:
        args.row_end = len(rows)
    rows = rows[args.row_start:args.row_end]
    print(f"[eval_app] {len(rows)} prompts (seed={args.seed}, range={args.row_start}-{args.row_end})", flush=True)

    # 2. Load the model and generate, or read CIFs from a previous run.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = [r["user_prompt"] for r in rows]
    prompt_ids = [r["row_id"] for r in rows]
    gen_root = args.out_dir / "generations"
    gen_root.mkdir(parents=True, exist_ok=True)

    if args.skip_generation_use_existing:
        from zipfile import ZipFile
        from pymatgen.io.cif import CifParser
        n_have = 0
        structures_per_prompt: list[list] = []
        for r in rows:
            zip_path = gen_root / r["row_id"] / "generated_crystals_cif.zip"
            gens = []
            if zip_path.exists():
                with ZipFile(zip_path) as zf:
                    for name in zf.namelist():
                        if not name.endswith(".cif"):
                            continue
                        cif = zf.read(name).decode("utf-8", errors="ignore")
                        try:
                            gens.append(CifParser.from_str(cif).parse_structures(primitive=False)[0])
                        except Exception:
                            pass  # the missing slot is scored 0 below
            n_have += bool(gens)
            structures_per_prompt.append(gens)
        # Prompts with a missing or empty zip stay in the denominator and score 0.
        print(f"[eval_app] reused on-disk generations for {n_have}/{len(rows)} prompts "
              f"({sum(len(g) for g in structures_per_prompt)} structures)", flush=True)
        if n_have == 0:
            raise SystemExit(f"no generations found under {gen_root}")
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

    # 3. Relax and characterize. Dropped structures become failed items that score 0.
    from structure_metrics import relax_structures_mattersim
    flat, flat_back_idx, failed_items = flatten_for_relax(structures_per_prompt, rows, args.K)
    if failed_items:
        print(f"[eval_app] {len(failed_items)} generations missing or not relaxable "
              f"(conversion error, Z outside 1-94); they score 0", flush=True)

    if args.skip_relax:
        relaxed_atoms = [AseAtomsAdaptor.get_atoms(s) for s in flat]
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
    # Structures with a NaN energy after relaxation or that fail to characterize are not
    # sent to the judge; they score 0 and stay in the per-prompt denominator.
    for (pi, gj), atoms in zip(flat_back_idx, relaxed_atoms):
        try:
            struct = AseAtomsAdaptor.get_structure(atoms)
            summary = _formula_summary(struct)
            fe = _formation_energy_per_atom_from_relaxed(atoms)
            if not (fe == fe) and args.skip_relax:
                fe = None  # no relaxation was run, so the energy is unknown
            elif not (fe == fe):  # NaN after relaxation
                failed_items.append(_failed_item(rows[pi], "nan_energy", summary))
                continue
            judge_items.append({
                "row_id": rows[pi]["row_id"],
                "prompt": rows[pi]["user_prompt"],
                "formation_energy_per_atom": fe,
                **summary,
            })
            judge_back_idx.append((pi, gj))
        except Exception:
            failed_items.append(_failed_item(rows[pi], "characterization_error"))
    print(f"[eval_app] characterized {len(judge_items)} structures; {len(failed_items)} failed "
          f"and score 0", flush=True)

    # 4. Batch LLM judge.
    reset_failure_counts()
    print(f"[eval_app] dispatching {len(judge_items)} judge calls "
          f"(model={args.judge_model}, concurrency={args.judge_concurrency}) ...", flush=True)
    verdicts = asyncio.run(batch_judge(
        items=judge_items,
        build_messages_fn=_app_messages,
        model=args.judge_model,
        concurrency=args.judge_concurrency,
    ))
    fc = get_failure_counts()
    if fc:
        print(f"[eval_app] judge failures: {fc}", flush=True)

    # 5. Aggregate over every selected prompt and every requested generation.
    scored: list[tuple[str, int]] = []
    examples: list[dict] = []
    for item in failed_items:
        scored.append((item["row_id"], 0))
        examples.append({**item, "judge_verdict": None, "judge_score": 0,
                         "judge_reason": None, "extracted_application": None})
    for item, verdict in zip(judge_items, verdicts):
        score = parse_score(verdict, default=0)
        scored.append((item["row_id"], score))
        examples.append({
            **item,
            "judge_verdict": verdict.get("verdict") if verdict else None,
            "judge_score": score,
            "judge_reason": verdict.get("reason") if verdict else None,
            "extracted_application": verdict.get("extracted_application") if verdict else None,
        })

    agg = aggregate_scores([r["row_id"] for r in rows], scored)
    per_prompt_mean = agg["per_prompt_mean"]
    overall_mean = agg["overall_consistency_mean_per_prompt"]
    overall_max_score_rate = agg["fraction_score_2"]

    metrics = {
        "n_prompts": len(rows),
        "n_judge_calls": len(judge_items),
        "n_judge_failures": int(sum(fc.values())) if fc else 0,
        "n_failed_structures": len(failed_items),
        "n_expected_structures": len(rows) * args.K,
        "failed_structure_rate": len(failed_items) / max(1, len(rows) * args.K),
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

    print(f"[eval_app] wrote {args.out_dir}/metrics.json and predictions.jsonl", flush=True)
    print(f"[eval_app] overall_consistency_mean_per_prompt = {overall_mean:.3f} / 2.0", flush=True)
    print(f"[eval_app] fraction_score_2 = {overall_max_score_rate:.3f}", flush=True)
    print(f"[eval_app] done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
