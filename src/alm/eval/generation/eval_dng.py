"""De-novo generation eval: generate N structures, score validity/metastability/SUN via MatterSim relax + MP2020 hull. Unconditional (g=0) or conditional (--prompts_from_parquet, g>0)."""
from __future__ import annotations


import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase.io import write as ase_write
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # alm/

from runs import run_dir, write_run  # noqa: E402
from structure_metrics import (  # noqa: E402
    cdvae_matcher,
    e_above_hull_per_atom,
    load_hull_reference,
    novel_mask_by_formula,
    relax_structures_mattersim,
    total_energy_per_atom,
    unique_indices,
    validity_full,
)
from paths import DATA_ROOT  # noqa: E402


def _atoms_struct_to_pymatgen(struct_dict: dict) -> Structure:
    """Convert a pairs.parquet `atoms_struct` row to a pymatgen Structure."""
    from pymatgen.core import Lattice
    lat = Lattice(np.asarray(struct_dict["lattice_mat"]))
    coords = np.asarray(struct_dict["coords"])
    elements = [s.strip() for s in struct_dict["elements"]]
    cartesian = bool(struct_dict.get("cartesian", False))
    return Structure(lat, elements, coords, coords_are_cartesian=cartesian)


def load_training_reference(parquet_path: Path, n_sample: int, seed: int = 42) -> list[Structure]:
    """Sample a reference set from pairs.parquet for novelty checks (proxy for the full Alex-MP-20 training set)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(parquet_path))
    total = pf.metadata.num_rows
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, total)
    target = sorted(rng.choice(total, size=n_sample, replace=False).tolist())
    target_set = set(int(i) for i in target)
    out = []
    cursor = 0
    for batch in pf.iter_batches(batch_size=8192, columns=["atoms_struct"]):
        b = batch.to_pydict()
        for i, struct in enumerate(b["atoms_struct"]):
            if cursor in target_set:
                if hasattr(struct, "as_py"):
                    struct = struct.as_py()
                try:
                    out.append(_atoms_struct_to_pymatgen(struct))
                except Exception:
                    pass
            cursor += 1
            if len(out) == len(target_set):
                break
        if len(out) == len(target_set):
            break
    return out


def _sample_prompts_from_parquet(parquet_path: Path, n_prompts: int, seed: int,
                                 parent_filter: str | None = None,
                                 slice_start: int = 0,
                                 slice_end: int | None = None):
    """Pick `n_prompts` prompts from pairs.parquet (seed+n_prompts fully determines the list, so worker slices stay consistent), return prompts[slice_start:slice_end]."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(parquet_path))
    total = pf.metadata.num_rows
    rng = np.random.default_rng(seed)
    schema = pf.schema_arrow
    cols = ["row_id", "parent", "narrative", "atoms_struct"]
    if "user_prompt" in schema.names:
        cols.append("user_prompt")

    if parent_filter is not None:
        # Scan parent column then sample, so a sparse parent survives rejection.
        matching_idx: list[int] = []
        cursor = 0
        for batch in pf.iter_batches(batch_size=16384, columns=["parent"]):
            b = batch.to_pydict()
            n = len(b["parent"])
            for i in range(n):
                if b["parent"][i] == parent_filter:
                    matching_idx.append(cursor)
                cursor += 1
        if len(matching_idx) < n_prompts:
            raise RuntimeError(
                f"only {len(matching_idx)} rows match parent={parent_filter!r} in "
                f"{parquet_path} (need {n_prompts}). Loosen --prompts_parent_filter "
                f"or lower --num_samples."
            )
        sample_idx = sorted(rng.choice(len(matching_idx), size=n_prompts,
                                       replace=False).tolist())
        target_set = set(int(matching_idx[i]) for i in sample_idx)
    else:
        # Oversample 4x to survive any per-row rejections.
        sample_idx = sorted(rng.choice(total, size=min(n_prompts * 4, total),
                                       replace=False).tolist())
        target_set = set(int(i) for i in sample_idx)

    prompts: list[str] = []
    prompt_ids: list[str] = []
    elements_per_prompt: list[list[str]] = []
    json_counts_per_prompt: list[dict[str, int]] = []
    cursor = 0
    for batch in pf.iter_batches(batch_size=8192, columns=cols):
        b = batch.to_pydict()
        n = len(b["row_id"])
        for i in range(n):
            if cursor in target_set:
                if parent_filter is not None and b["parent"][i] != parent_filter:
                    cursor += 1
                    continue
                p = b.get("user_prompt", [None]*n)[i] if "user_prompt" in b else None
                if not p:
                    p = b["narrative"][i]
                if p:
                    prompts.append(p)
                    prompt_ids.append(str(b["row_id"][i]))
                    # Ground-truth elements feed the chemical_system cond_field at inference.
                    struct = b["atoms_struct"][i]
                    if hasattr(struct, "as_py"):
                        struct = struct.as_py()
                    raw_elems = [str(e) for e in (struct.get("elements") or [])]
                    elems = sorted(set(raw_elems))
                    elements_per_prompt.append(elems)
                    json_counts_per_prompt.append(dict(Counter(raw_elems)))
                if len(prompts) >= n_prompts:
                    break
            cursor += 1
        if len(prompts) >= n_prompts:
            break
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"only sampled {len(prompts)}/{n_prompts} prompts from {parquet_path} "
            f"(parent_filter={parent_filter}); raise --prompts_seed oversampling or "
            f"loosen --prompts_parent_filter."
        )
    if slice_end is None:
        slice_end = len(prompts)
    # Fail loud: an out-of-range slice_start would silently yield 0 gens / NaN metrics.
    if slice_start >= len(prompts):
        raise RuntimeError(
            f"prompt slice_start={slice_start} >= n_prompts={len(prompts)} — this shard "
            f"would generate 0 structures. Check the launcher's slice/num_samples units "
            f"(prompt list length = num_samples/batch_size)."
        )
    return (prompts[slice_start:slice_end],
            prompt_ids[slice_start:slice_end],
            elements_per_prompt[slice_start:slice_end],
            json_counts_per_prompt[slice_start:slice_end])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True)
    ap.add_argument("--atoms_mapper", required=True)
    ap.add_argument("--num_samples", type=int, default=1024,
                    help="Total unconditional structures to generate.")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Per-call MatterGen batch size; total = ceil(num_samples/batch_size).")
    ap.add_argument("--diffusion_snr", type=float, default=None,
                    help="Sampling temperature analog (scales Langevin corrector "
                         "SNR vs defaults pos=0.4 / cell=0.2). <1.0 = warmer, "
                         ">1.0 = cooler, None = default.")
    ap.add_argument("--guidance_factor", type=float, default=0.0,
                    help="0 = strictly unconditional. Default 0.")
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--mattergen_model_path", default=None,
                    help="Local MatterGen backbone dir (e.g. csp_backbone / csp_backbone). "
                         "Overrides --mattergen_pretrained — REQUIRED for DNG on bridge models "
                         "trained against a local model_path, else the WRONG backbone is loaded "
                         "(mattergen_base) and the bridge's cond layers never match.")
    ap.add_argument("--num_atoms_distribution", default="ALEX_MP_20")
    ap.add_argument("--mattersim_potential_path", default=None,
                    help="Path to MatterSim checkpoint. If None, MatterSim's default is used "
                         "(downloaded on first call to Potential.from_checkpoint).")
    ap.add_argument("--metastable_threshold", type=float, default=0.1,
                    help="E_hull threshold (eV/atom) for 'metastable'. Default 0.1.")
    ap.add_argument("--reference_parquet",
                    default=os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs.parquet"),
                    help="Source for the novelty reference set.")
    ap.add_argument("--reference_size", type=int, default=10000,
                    help="Number of training-set rows to sample for novelty.")
    ap.add_argument("--prompts_from_parquet", type=Path, default=None,
                    help="If set, sample prompts from this parquet (uses the `user_prompt` column "
                         "when present, falling back to `narrative`). One distinct prompt per "
                         "generation chunk of size --batch_size; total generations = num_samples. "
                         "When unset, uses a placeholder prompt + ZerosEmbedding (true unconditional).")
    ap.add_argument("--prompts_seed", type=int, default=1337,
                    help="RNG seed for prompt sampling (distinct from training seeds).")
    ap.add_argument("--prompts_parent_filter", type=str, default=None,
                    help="If set, restrict to one pairs.parquet `parent` (e.g. 'dft_3d').")
    ap.add_argument("--prompt_slice_start", type=int, default=0,
                    help="Multi-GPU slicing: 0-indexed start within the (seed-determined) "
                         "prompt list. Each worker generates prompts[start:end] only.")
    ap.add_argument("--prompt_slice_end", type=int, default=None,
                    help="Multi-GPU slicing: exclusive end index. None = generate to "
                         "the end of the prompt list. With --prompt_slice_start this lets "
                         "N parallel workers split the same N=num_samples prompt set.")
    # FK stoichiometry steering: in-diffusion SMC, ESS-gated resampling toward the
    # prompt composition via a Hungarian reward on the Tweedie x-hat-0 Z-logits.
    ap.add_argument("--fk_n_particles", type=int, default=0,
                    help="0 = no FK (default). >0 = K particles/prompt with FK stoich steering.")
    ap.add_argument("--fk_rewards", type=str, default="stoich_match:1.0",
                    help="FK reward composition. Default = Hungarian stoichiometry match.")
    ap.add_argument("--fk_resample_every", type=int, default=10)
    ap.add_argument("--fk_t_start_frac", type=float, default=0.5)
    ap.add_argument("--fk_lambda", type=float, default=0.5)
    ap.add_argument("--fk_potential", type=str, default="sum", choices=["diff", "sum", "max"])
    ap.add_argument("--fk_ess_threshold_frac", type=float, default=0.5)
    ap.add_argument("--fk_log_w_clip", type=float, default=10.0)
    # Count-lock: N_p = multiple of sum(target_counts) lets the sum potential work on DNG.
    ap.add_argument("--fk_constrain_n_atoms_to_target_multiple", action="store_true",
                    help="DNG recipe: force N_p = integer multiple of sum(target_counts).")
    ap.add_argument("--fk_n_atoms_exact_sum_target", action="store_true",
                    help="Force N_p = sum(target_counts) exactly.")
    ap.add_argument("--fk_enforce_target_counts", action="store_true",
                    help="Post-hoc Hungarian Z-override to the exact target composition.")
    ap.add_argument("--skip_relax", action="store_true",
                    help="Skip MatterSim relaxation (then stability/SUN are NaN). Useful for "
                         "smoke-testing the generation+novelty path quickly.")
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--diffusion_seed", type=int, default=1337,
                    help="Seed for diffusion noise. Per-prompt offset is added so "
                         "reordering prompts doesn't change individual outputs. "
                         "Set to a different integer to draw a fresh sample set.")
    # LLM-temperature CoT sampling between "Structure: " and [atoms_0..7].
    ap.add_argument("--cot_tokens", type=int, default=0,
                    help="K' = number of LLM-sampled tokens spliced between the "
                         "assistant anchor 'Structure: ' and the K=8 [atoms_i] "
                         "tokens. 0 (default) = deterministic, current behavior.")
    ap.add_argument("--llm_temperature", type=float, default=1.0,
                    help="Sampling temperature for the CoT (used when --cot_tokens > 0).")
    ap.add_argument("--cot_top_p", type=float, default=0.9,
                    help="Nucleus-sampling cutoff for the CoT (used when --cot_tokens > 0).")
    ap.add_argument("--cot_seed", type=int, default=None,
                    help="PRNG seed for CoT sampling. Defaults to --diffusion_seed.")
    args = ap.parse_args()

    from generate_stage3 import generate_for_prompts, load_alm_and_pl_module  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alm, tokenizer, pl_module, K = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        model_path=args.mattergen_model_path,
        device=device,
    )

    if args.out_root is not None:
        os.environ["ALM_EVAL_RESULTS_ROOT"] = str(args.out_root)
    bench_name = f"stage3b_dng_g{int(args.guidance_factor*10):02d}"
    rd = run_dir(bench_name, args.alm_checkpoint, run_id=args.run_id)
    print(f"[dng] writing results to {rd}", flush=True)

    # ── 1. Generate ────────────────────────────────────────────────────────
    # FK steering forces batch dim = particle dim (K per prompt), so n_calls = num_samples/K.
    fk_active = args.fk_n_particles > 0
    gen_batch = args.fk_n_particles if fk_active else args.batch_size
    n_calls = (args.num_samples + gen_batch - 1) // gen_batch
    if args.prompts_from_parquet is not None:
        if args.guidance_factor == 0.0:
            print("[dng] WARNING: --prompts_from_parquet was given but --guidance_factor=0; "
                  "the prompt content will be ignored (ZerosEmbedding). Set "
                  "--guidance_factor > 0 to actually condition on the prompt.", flush=True)
        prompts, prompt_ids, elements_per_prompt, json_counts_per_prompt = _sample_prompts_from_parquet(
            Path(args.prompts_from_parquet),
            n_prompts=n_calls,
            seed=args.prompts_seed,
            parent_filter=args.prompts_parent_filter,
            slice_start=args.prompt_slice_start,
            slice_end=args.prompt_slice_end,
        )
        print(f"[dng] sampled {len(prompts)} conditioning prompts from "
              f"{args.prompts_from_parquet}", flush=True)
    else:
        # g=0 routes through ZerosEmbedding, so prompt content is ignored.
        prompts = ["__unconditional__"] * n_calls
        prompt_ids = [f"dng_chunk_{i:04d}" for i in range(n_calls)]
        elements_per_prompt = None
        json_counts_per_prompt = None
    # FK target composition per prompt (reduced Z-counts).
    fk_target_counts_per_prompt = None
    if fk_active:
        if json_counts_per_prompt is None:
            raise SystemExit("--fk_n_particles>0 requires --prompts_from_parquet "
                             "(FK stoich reward needs a per-prompt target composition)")
        def _reduced_z(counts):
            from ase.data import atomic_numbers as _z
            from functools import reduce as _rd
            from math import gcd
            clean = {str(s): int(n) for s, n in counts.items() if int(n) > 0}
            if not clean:
                return {}
            g = max(_rd(gcd, clean.values()), 1)
            return {int(_z[s]): int(n // g) for s, n in clean.items()}
        fk_target_counts_per_prompt = [_reduced_z(c) for c in json_counts_per_prompt]
        print(f"[dng-fk] FK stoich steering ON: K={args.fk_n_particles} rewards={args.fk_rewards} "
              f"t_start={args.fk_t_start_frac} lambda={args.fk_lambda} potential={args.fk_potential} "
              f"row0_target={fk_target_counts_per_prompt[0]}", flush=True)
    gen_root = rd / "generations"
    gen_root.mkdir(parents=True, exist_ok=True)
    structures_per_chunk = generate_for_prompts(
        prompts=prompts,
        alm=alm, tokenizer=tokenizer, pl_module=pl_module,
        out_root=gen_root,
        batch_size=gen_batch,
        num_batches=1,
        fk_n_particles=args.fk_n_particles,
        fk_rewards=args.fk_rewards,
        fk_target_counts_per_prompt=fk_target_counts_per_prompt,
        fk_resample_every=args.fk_resample_every,
        fk_t_start_frac=args.fk_t_start_frac,
        fk_lambda=args.fk_lambda,
        fk_potential=args.fk_potential,
        fk_ess_threshold_frac=args.fk_ess_threshold_frac,
        fk_log_w_clip=args.fk_log_w_clip,
        fk_constrain_n_atoms_to_target_multiple=args.fk_constrain_n_atoms_to_target_multiple,
        fk_n_atoms_exact_sum_target=args.fk_n_atoms_exact_sum_target,
        fk_enforce_target_counts=args.fk_enforce_target_counts,
        diffusion_guidance_factor=args.guidance_factor,
        diffusion_snr=args.diffusion_snr,
        num_atoms_distribution=args.num_atoms_distribution,
        prompt_ids=prompt_ids,
        save_meta=False,
        diffusion_seed=args.diffusion_seed,
        # Populates chemical_system cond_field if the ckpt has it; no-op otherwise.
        chemical_system_per_prompt=elements_per_prompt,
        json_counts_per_prompt=json_counts_per_prompt,
        cot_tokens=int(args.cot_tokens),
        llm_temperature=float(args.llm_temperature),
        cot_top_p=float(args.cot_top_p),
        cot_seed=(int(args.cot_seed) if args.cot_seed is not None
                  else int(args.diffusion_seed)),
    )
    raw = []
    for chunk in structures_per_chunk:
        for s in chunk:
            raw.append(s if isinstance(s, Structure) else AseAtomsAdaptor.get_structure(s))
    raw = raw[:args.num_samples]
    print(f"[dng] generated {len(raw)} structures", flush=True)

    ase_atoms_pre = [AseAtomsAdaptor.get_atoms(s) for s in raw]
    ase_write(rd / "pre_relax.extxyz", ase_atoms_pre, format="extxyz")

    # ── 2. Validity ────────────────────────────────────────────────────────
    valid_flags = [validity_full(s) for s in raw]
    # dtype=bool: empty-list np.array() defaults to float64, breaking the `&` below.
    valid_geom = np.array([v["geom"] for v in valid_flags], dtype=bool)
    valid_charge = np.array([v["charge"] for v in valid_flags], dtype=bool)
    print(f"[dng] validity_geom = {valid_geom.mean():.3f}, "
          f"validity_charge = {valid_charge.mean():.3f}", flush=True)

    # ── 3. Relax + score energy ────────────────────────────────────────────
    e_per_atom = np.full(len(raw), np.nan)
    e_hull = np.full(len(raw), np.nan)
    relaxed_atoms = None
    if not args.skip_relax:
        # Relax only geom-valid structures; drop Z>94 (MatterSim CUDA assert kills the shard).
        def _max_z(idx):
            try:
                return int(max(ase_atoms_pre[idx].numbers))
            except (ValueError, IndexError):
                return 0
        relax_idx = [i for i, v in enumerate(valid_geom) if v and _max_z(i) <= 94]
        n_dropped_z = sum(1 for i, v in enumerate(valid_geom) if v and _max_z(i) > 94)
        if n_dropped_z:
            print(f"[dng] dropped {n_dropped_z} structures with Z>94 (MatterSim unsupported)",
                  flush=True)
        print(f"[dng] relaxing {len(relax_idx)} of {len(raw)} (geom-valid + Z<=94)...",
              flush=True)
        atoms_to_relax = [ase_atoms_pre[i] for i in relax_idx]
        if atoms_to_relax:
            relaxed_atoms_subset, total_energies_subset = relax_structures_mattersim(
                atoms_to_relax,
                device=str(device),
                potential_path=args.mattersim_potential_path,
                fmax=0.05, max_n_steps=500,
                output_extxyz=rd / "post_relax.extxyz",
            )
            print(f"[dng] relaxed {len(relaxed_atoms_subset)} structures", flush=True)

            print(f"[dng] loading hull reference + scoring E_hull...", flush=True)
            hull_ref = load_hull_reference()
            for sub_i, idx in enumerate(relax_idx):
                ra = relaxed_atoms_subset[sub_i]
                e_per_atom[idx] = total_energy_per_atom(ra)
                relaxed_struct = AseAtomsAdaptor.get_structure(ra)
                e_hull[idx] = e_above_hull_per_atom(
                    relaxed_struct, ra.info.get("total_energy", float("nan")), hull_ref,
                )
            relaxed_atoms = relaxed_atoms_subset

    stable_mask = (e_hull <= 0.0) & np.isfinite(e_hull)
    metastable_mask = (e_hull <= args.metastable_threshold) & np.isfinite(e_hull)

    # ── 4. Uniqueness within batch ─────────────────────────────────────────
    print(f"[dng] computing uniqueness (within {len(raw)} samples)...", flush=True)
    matcher = cdvae_matcher()
    unique_idx = unique_indices(raw, matcher=matcher)
    unique_mask = np.zeros(len(raw), dtype=bool)
    unique_mask[unique_idx] = True
    print(f"[dng] uniqueness = {unique_mask.mean():.3f}", flush=True)

    # ── 5. Novelty against training reference ──────────────────────────────
    print(f"[dng] loading {args.reference_size} reference structures from {args.reference_parquet}...",
          flush=True)
    ref_structs = load_training_reference(Path(args.reference_parquet), args.reference_size)
    print(f"[dng] loaded {len(ref_structs)} references; computing novelty...", flush=True)
    novel_mask = novel_mask_by_formula(raw, ref_structs, matcher=matcher)
    print(f"[dng] novelty = {novel_mask.mean():.3f}", flush=True)

    # ── 6. S.U.N. ──────────────────────────────────────────────────────────
    sun_mask = stable_mask & unique_mask & novel_mask
    metastable_sun_mask = metastable_mask & unique_mask & novel_mask

    # ── 7. Write ───────────────────────────────────────────────────────────
    predictions = []
    for i, s in enumerate(raw):
        predictions.append({
            "idx": i,
            "formula": s.composition.reduced_formula,
            "n_atoms": len(s),
            "valid_geom": bool(valid_geom[i]),
            "valid_charge": bool(valid_charge[i]),
            "e_per_atom": float(e_per_atom[i]) if np.isfinite(e_per_atom[i]) else None,
            "e_hull": float(e_hull[i]) if np.isfinite(e_hull[i]) else None,
            "stable": bool(stable_mask[i]),
            "metastable": bool(metastable_mask[i]),
            "unique": bool(unique_mask[i]),
            "novel": bool(novel_mask[i]),
            "sun": bool(sun_mask[i]),
            "metastable_sun": bool(metastable_sun_mask[i]),
        })

    metrics = {
        "n_generated": len(raw),
        "guidance_factor": args.guidance_factor,
        "validity_geom_pct": float(valid_geom.mean()),
        "validity_charge_pct": float(valid_charge.mean()),
        "validity_full_pct": float((valid_geom & valid_charge).mean()),
        "stable_pct": float(stable_mask.mean()),
        "metastable_pct": float(metastable_mask.mean()),
        "unique_pct": float(unique_mask.mean()),
        "novel_pct": float(novel_mask.mean()),
        "sun_pct": float(sun_mask.mean()),  # strict S.U.N. (E_hull <= 0)
        "metastable_sun_pct": float(metastable_sun_mask.mean()),  # MatterGen MSUN convention (E_hull <= 0.1)
        "n_relaxation_inputs": int(valid_geom.sum()),
        "n_with_e_hull": int(np.isfinite(e_hull).sum()),
        "alm_checkpoint": str(args.alm_checkpoint),
        "atoms_mapper": str(args.atoms_mapper),
        "metastable_threshold": args.metastable_threshold,
    }
    write_run(rd, metrics, predictions)

    print()
    print(f"[dng] DONE — {len(raw)} samples")
    print(f"  validity_geom    = {metrics['validity_geom_pct']:.3f}")
    print(f"  validity_charge  = {metrics['validity_charge_pct']:.3f}")
    print(f"  metastable       = {metrics['metastable_pct']:.3f}")
    print(f"  stable           = {metrics['stable_pct']:.3f}")
    print(f"  unique           = {metrics['unique_pct']:.3f}")
    print(f"  novel            = {metrics['novel_pct']:.3f}")
    print(f"  S.U.N.           = {metrics['sun_pct']:.3f}")
    print(f"  M.S.U.N. (≤0.1)  = {metrics['metastable_sun_pct']:.3f}")
    print(f"  results in {rd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
