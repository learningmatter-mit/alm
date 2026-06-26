"""Atom+text direction-correctness eval: did each candidate move the requested property the requested way?

Usage:
  export OPENAI_API_KEY=sk-...
  python -m alm.eval.eval_atomtxt_direction \\
      --alm_checkpoint <ckpt_dir> \\
      --atoms_mapper   <ckpt_dir>/atoms_mapper.pt \\
      --atomtxt_parquet <data_root>/pairs_atomtxt.parquet \\
      --cached_embs_root <data_root>/cached_embs_narratives \\
      --max_rows 100 --K 20 \\
      --out_dir <results_dir>/atomtxt_direction
"""
from __future__ import annotations


import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
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

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from llm_judge import (  # noqa: E402
    DEFAULT_MODEL, batch_judge, build_atomtxt_direction_messages,
    get_failure_counts, parse_score, reset_failure_counts,
)
from paths import DATA_ROOT  # noqa: E402

# atomtxt-{parent}-{input_idx}-to-{target_idx}-{prop}-{direction}
ROW_ID_RE = re.compile(
    r"^atomtxt-(?P<parent>[^-]+)-(?P<input_idx>\d+)-to-(?P<target_idx>\d+)-"
    r"(?P<prop>[a-z_]+)-(?P<direction>[a-z]+)$"
)

# band_gap excluded: MatterSim doesn't predict it directly.
SCOREABLE_PROPS = {"density", "volume", "formation_energy"}

DIRECTION_SIGN = {
    "higher": +1, "lower": -1,
    "larger": +1, "smaller": -1,
}


def _parse_row_id(row_id: str) -> dict | None:
    m = ROW_ID_RE.match(row_id)
    if not m:
        return None
    return m.groupdict()


def _ase_to_struct(atoms_struct: dict) -> Structure | None:
    try:
        elements = [str(e).strip() for e in atoms_struct["elements"]]
        coords = np.asarray(atoms_struct["coords"], dtype=np.float64)
        lattice = np.asarray(atoms_struct["lattice_mat"], dtype=np.float64)
        cartesian = bool(atoms_struct.get("cartesian", True))
        return Structure(
            lattice=lattice, species=elements, coords=coords,
            coords_are_cartesian=cartesian,
        )
    except Exception:
        return None


def _selected_atomtxt_rows(parquet_path: Path, max_rows: int, seed: int) -> list[dict]:
    """Hash-deterministic eval subset; only rows with scoreable props."""
    pf = pq.ParquetFile(parquet_path)
    candidates: list[dict] = []
    cols = ["row_id", "parent", "source_idx", "user_prompt", "atoms_struct",
            "input_atoms_struct", "input_source_idx"]
    for batch in pf.iter_batches(batch_size=10000, columns=cols):
        for r in batch.to_pylist():
            tag = _parse_row_id(r["row_id"])
            if tag is None or tag["prop"] not in SCOREABLE_PROPS:
                continue
            h = int(hashlib.md5(f"{r['row_id']}:{seed}".encode()).hexdigest(), 16)
            r["_h"] = h
            r["_tag"] = tag
            candidates.append(r)
    candidates.sort(key=lambda r: r["_h"])
    return candidates[:max_rows]


def _measure_struct(atoms: Atoms) -> dict:
    """density (g/cm³), volume_per_atom (Å³), formation_energy (eV/atom from MatterSim total energy)."""
    n = max(1, len(atoms))
    vol = float(abs(np.linalg.det(np.asarray(atoms.cell))))
    mass_amu = sum(atomic_masses[atomic_numbers[a.symbol]] for a in atoms)
    density = mass_amu * 1.66054 / max(vol, 1e-3)
    e = atoms.info.get("total_energy")
    fe = float(e) / n if e is not None else float("nan")
    return {
        "density": float(density),
        "volume_per_atom": float(vol / n),
        "volume": float(vol),
        "formation_energy_per_atom": float(fe),
        "n_atoms": n,
        "formula": atoms.get_chemical_formula(),
    }


def _load_orbv3_cache(cached_embs_root: Path, parents: set[str]) -> dict:
    """Load per-parent OrbV3 caches for the input-side <atoms> splice."""
    out = {"memmap": {}, "idx": {}}
    for parent in parents:
        bin_p = cached_embs_root / parent / "embeddings" / "orb_v3_direct_20_omat_atom.flat.bin"
        idx_p = cached_embs_root / parent / "embeddings" / "orb_v3_direct_20_omat_atom.flat.idx.json"
        if not bin_p.exists() or not idx_p.exists():
            print(f"[eval_atomtxt] WARN no cache for {parent} at {bin_p}", flush=True)
            continue
        with open(idx_p) as f:
            out["idx"][parent] = json.load(f)
        out["memmap"][parent] = np.memmap(bin_p, dtype=np.float32, mode="r").reshape(-1, 256)
    return out


def _input_atom_embed(cache, parent: str, input_idx: int, device) -> torch.Tensor | None:
    """(N_atoms, 256) OrbV3 features for the input structure, or None on miss."""
    idx_map = cache["idx"].get(parent)
    mm = cache["memmap"].get(parent)
    if idx_map is None or mm is None:
        return None
    ent = idx_map.get(str(int(input_idx)))
    if ent is None:
        return None
    off, length = int(ent[0]), int(ent[1])
    arr = np.asarray(mm[off:off + length], dtype=np.float32).copy()
    return torch.from_numpy(arr).to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True)
    ap.add_argument("--atoms_mapper", required=True)
    ap.add_argument("--atomtxt_parquet", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_atomtxt.parquet")))
    ap.add_argument("--cached_embs_root", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "cached_embs_narratives")))
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="Eval-set hash seed; pick a value never used in training (default 42).")
    ap.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True,
                    help="Whether to run the LLM judge overlay (default on).")
    ap.add_argument("--judge_model", default=DEFAULT_MODEL)
    ap.add_argument("--judge_concurrency", type=int, default=16)
    ap.add_argument("--judge_max_per_prompt", type=int, default=0,
                    help="Cap number of judge calls per prompt (default 0 = no cap). "
                         "Set to e.g. 3 to reduce OpenAI API volume when rate-limited. "
                         "The deterministic direction-correctness rate is unaffected; "
                         "this only caps the judge overlay.")
    ap.add_argument("--mattersim_potential_path", type=str, default=None)
    ap.add_argument("--row_start", type=int, default=0)
    ap.add_argument("--row_end", type=int, default=-1)
    ap.add_argument("--diffusion_seed", type=int, default=1337,
                    help="Seed for diffusion noise. Per-prompt offset added so "
                         "reordering doesn't change individual outputs.")
    # ── FK steering: preserve input's element set + stoichiometry ──
    ap.add_argument("--fk_n_particles", type=int, default=0,
                    help="0 = no FK (default; vanilla diffusion). When > 0, "
                         "REPLACES --K: each prompt produces fk_n_particles "
                         "structures via FK steering. Element mask + "
                         "stoich-match reward are auto-derived from input_atoms_struct.")
    ap.add_argument("--fk_lambda", type=float, default=0.5)
    ap.add_argument("--fk_log_w_clip", type=float, default=50.0)
    ap.add_argument("--fk_potential", type=str, default="sum",
                    choices=["sum", "diff", "max"])
    ap.add_argument("--fk_resample_every", type=int, default=10)
    ap.add_argument("--fk_t_start_frac", type=float, default=0.5)
    ap.add_argument("--fk_rewards", type=str,
                    default="stoich_match:1.0;count_l1:1.0;ratio_kl:1.0",
                    help="FK reward composition. target_counts is auto-set to the "
                         "input's atomic-number multiset.")
    ap.add_argument("--fk_mask_input_elements", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="When FK is on, hard-mask atomic-number logits to "
                         "input's element set (the atomtxt pairs' invariant). "
                         "Default ON.")
    ap.add_argument("--fk_enforce_target_counts", action="store_true",
                    help="Post-hoc Hungarian Z-override at end of denoising — "
                         "force exact input stoichiometry on every particle. "
                         "Default OFF (atomtxt may want some compositional flex).")
    ap.add_argument("--fk_n_atoms_exact_sum_target", action="store_true",
                    help="Force N_p = sum(input_counts) exactly. Default OFF "
                         "(atomtxt prompts can ask for cells of different sizes).")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_atomtxt] writing → {args.out_dir}", flush=True)
    t0 = time.time()

    # ── 1. Pick rows + load OrbV3 caches ──
    rows = _selected_atomtxt_rows(args.atomtxt_parquet, args.max_rows, args.seed)
    if args.row_end < 0:
        args.row_end = len(rows)
    rows = rows[args.row_start:args.row_end]
    print(f"[eval_atomtxt] {len(rows)} rows (seed={args.seed}, range={args.row_start}-{args.row_end})", flush=True)
    by_prop = Counter(r["_tag"]["prop"] for r in rows)
    by_dir = Counter(r["_tag"]["direction"] for r in rows)
    print(f"[eval_atomtxt] by prop: {dict(by_prop)}", flush=True)
    print(f"[eval_atomtxt] by direction: {dict(by_dir)}", flush=True)

    parents = {r["parent"] for r in rows}
    print(f"[eval_atomtxt] loading OrbV3 caches for parents: {parents}", flush=True)
    cache = _load_orbv3_cache(args.cached_embs_root, parents)

    # ── 2. Load model ──
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                       # alm/
    from generate_stage3 import load_alm_and_pl_module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_atomtxt] loading ALM + MatterGen on {device} ...", flush=True)
    alm, tokenizer, pl_module, K_tokens = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        device=device,
    )

    # ── 3. Generate per-prompt with input atom features ──
    # Run prompts one at a time via the primitives so cached input features get
    # spliced in, rather than forking generate_stage3's zero-length input path.
    from generate_stage3 import (
        get_alm_embedding, build_sampler_and_loader, draw_samples_from_sampler,
        _ensure_fk_hook_installed, _install_fk_on_sampler,
    )

    # FK plumbing
    fk_active = args.fk_n_particles > 0
    if fk_active:
        sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
        from fk_rewards import parse_rewards as _fk_parse_rewards
        _ensure_fk_hook_installed(pl_module)
        print(f"[eval_atomtxt] FK steering ON: K={args.fk_n_particles} "
              f"λ={args.fk_lambda} pot={args.fk_potential} "
              f"clip={args.fk_log_w_clip} t_start={args.fk_t_start_frac} "
              f"resample_every={args.fk_resample_every}", flush=True)
        print(f"[eval_atomtxt] rewards: {args.fk_rewards}", flush=True)
        print(f"[eval_atomtxt] mask_input_elements={args.fk_mask_input_elements} "
              f"enforce_target_counts={args.fk_enforce_target_counts} "
              f"n_atoms_exact_sum_target={args.fk_n_atoms_exact_sum_target}", flush=True)

    import random as _random
    import numpy as _np
    from collections import Counter as _Counter
    from ase.data import atomic_numbers as _atomic_numbers
    structures_per_prompt: list[list] = []
    for i, r in enumerate(rows):
        # Per-prompt seed: reproducible and order-independent.
        _s = (int(args.diffusion_seed) + i) & 0x7FFFFFFF
        torch.manual_seed(_s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_s)
        _np.random.seed(_s)
        _random.seed(_s)
        atom_embed = _input_atom_embed(cache, r["parent"], r["input_source_idx"], device)
        if atom_embed is None:
            print(f"[eval_atomtxt] {r['row_id']}: no cached input features → skip", flush=True)
            structures_per_prompt.append([])
            continue
        input_elems = [str(e) for e in r["input_atoms_struct"]["elements"]]
        json_counts = dict(_Counter(input_elems))
        # wrap_user_template=False: atomtxt prompts already start with "<atoms>\nGenerate a ...".
        alm_emb = get_alm_embedding(
            alm, tokenizer, r["user_prompt"], device,
            atom_embed=atom_embed,
            wrap_user_template=False,
            json_counts=json_counts,
        )
        # Per-prompt FK config derived from input.
        target_counts = None
        allowed_z = None
        constrain_exact = 0
        if fk_active:
            try:
                input_zs = [_atomic_numbers[e] for e in input_elems]
            except KeyError:
                # Unknown element symbol -> fall back to vanilla for this row.
                input_zs = None
            if input_zs is not None:
                target_counts = dict(_Counter(input_zs))
                allowed_z = set(input_zs) if args.fk_mask_input_elements else None
                if args.fk_n_atoms_exact_sum_target:
                    constrain_exact = int(sum(target_counts.values()))
        # hasattr guard: older pl_module instances lack _element_mask_state.
        if hasattr(pl_module, "_element_mask_state"):
            if fk_active and allowed_z is not None:
                pl_module._element_mask_state.allowed_z = allowed_z
            else:
                pl_module._element_mask_state.allowed_z = None

        sampler_batch_size = args.fk_n_particles if fk_active else args.K
        sampler, condition_loader = build_sampler_and_loader(
            pl_module=pl_module, batch_size=sampler_batch_size, num_batches=1,
            num_atoms_distribution="ALEX_MP_20",
            alm_emb_vec=alm_emb,
            diffusion_guidance_factor=args.guidance_factor,
            constrain_n_atoms_exact=constrain_exact,
        )
        if fk_active and target_counts is not None:
            reward = _fk_parse_rewards(
                args.fk_rewards,
                allowed_elements=None,  # masking handled by _element_mask_state above
                target_counts=target_counts,
                physical_bounds=None,
                target_sg=None,
            )
            st = pl_module._fk_state
            st.enabled = True
            st.reward = reward
            st.target_counts = target_counts
            st.enforce_target_counts = bool(args.fk_enforce_target_counts)
            st.n_particles = args.fk_n_particles
            st.resample_every = args.fk_resample_every
            st.t_start_frac = args.fk_t_start_frac
            st.lambda_ = args.fk_lambda
            st.potential = args.fk_potential
            st.ess_threshold_frac = 0.5
            st.keep_top_k = -1
            st.log_w_clip = args.fk_log_w_clip
            st.stratify_resample_by_n_atoms = False
            _install_fk_on_sampler(sampler, st)
        elif fk_active:
            # FK requested but no target derivable -> disable for this row.
            pl_module._fk_state.enabled = False

        # output_path=None: relax + score in-memory below, no on-disk artifacts needed.
        gens = draw_samples_from_sampler(
            sampler, condition_loader,
            output_path=None,
            properties_to_condition_on=None,
            record_trajectories=False,
        )
        structures_per_prompt.append(gens)
        if (i + 1) % 10 == 0:
            print(f"[eval_atomtxt] generated {i+1}/{len(rows)} prompts in {time.time()-t0:.0f}s", flush=True)

    # ── 4. Relax inputs + outputs ──
    from structure_metrics import relax_structures_mattersim
    print(f"[eval_atomtxt] relaxing inputs ...", flush=True)
    input_structs: list[Structure | None] = []
    for r in rows:
        s = _ase_to_struct(r["input_atoms_struct"])
        input_structs.append(s)
    # Defensive Z-range filter: one bad input would crash the whole relax batch.
    valid_inputs: list[Structure] = []
    n_input_filtered = 0
    for s in input_structs:
        if s is None:
            continue
        try:
            zs = [int(site.specie.Z) for site in s]
        except Exception:
            n_input_filtered += 1
            continue
        if not zs or any(z < 1 or z > 94 for z in zs):
            n_input_filtered += 1
            continue
        valid_inputs.append(s)
    if n_input_filtered:
        print(f"[eval_atomtxt] pre-filtered {n_input_filtered} input structures with out-of-range Z", flush=True)
    relaxed_inputs, _ = relax_structures_mattersim(
        valid_inputs, device=str(device),
        potential_path=args.mattersim_potential_path,
        fmax=0.05, max_n_steps=500,
    )
    rel_input_iter = iter(relaxed_inputs)
    relaxed_inputs_aligned: list[Atoms | None] = [
        next(rel_input_iter, None) if s is not None else None for s in input_structs
    ]

    print(f"[eval_atomtxt] relaxing outputs ...", flush=True)
    flat: list = []
    flat_back_idx: list[tuple[int, int]] = []
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
            # Drop Z outside [1, 94] before relax: one bad Z corrupts the CUDA context for the batch.
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
        print(f"[eval_atomtxt] pre-filtered {n_filtered_z} structures with out-of-range Z", flush=True)
    relaxed_outputs, _ = relax_structures_mattersim(
        flat, device=str(device),
        potential_path=args.mattersim_potential_path,
        fmax=0.05, max_n_steps=500,
    )
    print(f"[eval_atomtxt] relax done in {time.time()-t0:.0f}s total", flush=True)

    # ── 5. Score deterministically + collect judge inputs ──
    judge_items: list[dict] = []
    judge_back_idx: list[int] = []   # indices into examples
    examples: list[dict] = []
    per_prompt_correct: dict[str, list[bool]] = defaultdict(list)
    n_skipped = 0

    out_iter = iter(relaxed_outputs)
    for (pi, gj), out_atoms in zip(flat_back_idx, relaxed_outputs):
        r = rows[pi]
        tag = r["_tag"]
        prop = tag["prop"]
        direction = tag["direction"]
        sign = DIRECTION_SIGN.get(direction, 0)
        in_atoms = relaxed_inputs_aligned[pi]
        if in_atoms is None or sign == 0:
            n_skipped += 1
            continue
        in_meas = _measure_struct(in_atoms)
        out_meas = _measure_struct(out_atoms)
        if prop == "formation_energy":
            in_v = in_meas["formation_energy_per_atom"]
            out_v = out_meas["formation_energy_per_atom"]
        elif prop == "density":
            in_v = in_meas["density"]
            out_v = out_meas["density"]
        elif prop == "volume":
            in_v = in_meas["volume_per_atom"]
            out_v = out_meas["volume_per_atom"]
        else:
            n_skipped += 1
            continue
        if not (in_v == in_v) or not (out_v == out_v):
            n_skipped += 1
            continue
        delta = out_v - in_v
        # Correct = sign match AND move exceeds 5% of the input value.
        rel_threshold = 0.05 * abs(in_v) if abs(in_v) > 1e-6 else 0.0
        is_correct = (sign * delta) > rel_threshold
        per_prompt_correct[r["row_id"]].append(bool(is_correct))

        ex = {
            "row_id": r["row_id"],
            "prompt": r["user_prompt"],
            "prop_target": prop,
            "direction_target": direction,
            "input_formula": in_meas["formula"],
            "input_density": in_meas["density"],
            "input_vpa": in_meas["volume_per_atom"],
            "input_fe": in_meas["formation_energy_per_atom"],
            "output_formula": out_meas["formula"],
            "output_density": out_meas["density"],
            "output_vpa": out_meas["volume_per_atom"],
            "output_fe": out_meas["formation_energy_per_atom"],
            "input_value": in_v,
            "output_value": out_v,
            "delta": delta,
            "direction_correct": bool(is_correct),
        }
        examples.append(ex)
        if args.judge:
            if args.judge_max_per_prompt > 0:
                _existing = sum(1 for ji in judge_items if ji.get("row_id") == ex["row_id"])
                if _existing >= args.judge_max_per_prompt:
                    continue
            judge_items.append(ex)
            judge_back_idx.append(len(examples) - 1)

    print(f"[eval_atomtxt] {len(examples)} (input,output) pairs scored deterministically; "
          f"{n_skipped} skipped (relaxation/parsing failures or zero-direction)", flush=True)

    # ── 6. Optional LLM judge overlay ──
    judge_score_by_idx: dict[int, int] = {}
    if args.judge and judge_items:
        reset_failure_counts()
        print(f"[eval_atomtxt] dispatching {len(judge_items)} judge calls ...", flush=True)
        verdicts = asyncio.run(batch_judge(
            items=judge_items,
            build_messages_fn=build_atomtxt_direction_messages,
            model=args.judge_model,
            concurrency=args.judge_concurrency,
        ))
        fc = get_failure_counts()
        if fc:
            print(f"[eval_atomtxt] judge failures: {fc}", flush=True)
        for back_i, verdict in zip(judge_back_idx, verdicts):
            score = parse_score(verdict, default=0)
            judge_score_by_idx[back_i] = score
            examples[back_i]["judge_score"] = score
            examples[back_i]["judge_verdict"] = verdict.get("verdict") if verdict else None
            examples[back_i]["judge_reason"] = verdict.get("reason") if verdict else None

    # ── 7. Aggregates ──
    per_prompt_rate = {
        rid: float(np.mean(corrects))
        for rid, corrects in per_prompt_correct.items() if corrects
    }
    direction_rate = float(np.mean(list(per_prompt_rate.values()))) if per_prompt_rate else 0.0
    direction_rate_overall = float(np.mean([
        e["direction_correct"] for e in examples
    ])) if examples else 0.0

    judge_mean = None
    if judge_score_by_idx:
        judge_mean = float(np.mean(list(judge_score_by_idx.values())))

    by_prop_rate: dict[str, float] = {}
    for prop in SCOREABLE_PROPS:
        prop_examples = [e for e in examples if e["prop_target"] == prop]
        if prop_examples:
            by_prop_rate[prop] = float(np.mean([e["direction_correct"] for e in prop_examples]))

    metrics = {
        "n_rows": len(rows),
        "n_pairs_scored": len(examples),
        "n_skipped": n_skipped,
        "K": args.K,
        "judge_model": args.judge_model if args.judge else None,
        "guidance_factor": args.guidance_factor,
        "direction_correctness_rate_per_prompt_mean": direction_rate,
        "direction_correctness_rate_overall": direction_rate_overall,
        "judge_consistency_score_mean": judge_mean,
        "by_prop_rate": by_prop_rate,
        "alm_checkpoint": str(args.alm_checkpoint),
        "wallclock_sec": time.time() - t0,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.out_dir / "predictions.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"[eval_atomtxt] HEADLINE — direction_correctness_rate = {direction_rate:.3f}", flush=True)
    if judge_mean is not None:
        print(f"[eval_atomtxt]            judge_consistency_score = {judge_mean:.3f} / 2.0", flush=True)
    print(f"[eval_atomtxt]            by-prop: {by_prop_rate}", flush=True)
    print(f"[eval_atomtxt] DONE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
