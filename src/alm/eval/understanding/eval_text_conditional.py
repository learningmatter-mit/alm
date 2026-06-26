"""Text-conditional eval: composition-match / density / energy MAE for text -> structure on a held-out pairs.parquet slice."""
from __future__ import annotations


import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # alm/

from paths import DATA_ROOT  # noqa: E402
from runs import run_dir, write_run  # noqa: E402
from structure_metrics import (  # noqa: E402
    composition_match_ratio,
    composition_set,
    density_g_per_cm3,
    relax_structures_mattersim,
    total_energy_per_atom,
    validity_full,
)


def _atoms_struct_to_pymatgen(struct_dict: dict) -> Structure:
    from pymatgen.core import Lattice
    lat = Lattice(np.asarray(struct_dict["lattice_mat"]))
    coords = np.asarray(struct_dict["coords"])
    elements = [s.strip() for s in struct_dict["elements"]]
    cartesian = bool(struct_dict.get("cartesian", False))
    return Structure(lat, elements, coords, coords_are_cartesian=cartesian)


def sample_eval_rows(parquet_path: Path, n_rows: int, seed: int = 1337,
                     parent_filter: str | None = None):
    """Deterministic random sample of pairs.parquet rows; pairs are not held out from training, so this is in-distribution eval."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(parquet_path))
    total = pf.metadata.num_rows
    rng = np.random.default_rng(seed)
    sample_idx = sorted(rng.choice(total, size=min(n_rows * 4, total),
                                   replace=False).tolist())
    target_set = set(int(i) for i in sample_idx)
    out = []
    cursor = 0
    cols = ["row_id", "parent", "narrative", "user_prompt", "atoms_struct"]
    for batch in pf.iter_batches(batch_size=8192, columns=cols):
        b = batch.to_pydict()
        for i in range(len(b["row_id"])):
            if cursor in target_set:
                if parent_filter is not None and b["parent"][i] != parent_filter:
                    cursor += 1
                    continue
                struct = b["atoms_struct"][i]
                if hasattr(struct, "as_py"):
                    struct = struct.as_py()
                try:
                    py_struct = _atoms_struct_to_pymatgen(struct)
                except Exception:
                    cursor += 1
                    continue
                # Pre-templated user_prompt matches what the model saw at train time.
                out.append({
                    "row_id": b["row_id"][i],
                    "parent": b["parent"][i],
                    "narrative": b["narrative"][i],
                    "user_prompt": b["user_prompt"][i],
                    "target": py_struct,
                })
                if len(out) >= n_rows:
                    break
            cursor += 1
        if len(out) >= n_rows:
            break
    return out[:n_rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True)
    ap.add_argument("--atoms_mapper", required=True)
    ap.add_argument("--pairs_parquet", default=os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs.parquet"))
    ap.add_argument("--n_test_rows", type=int, default=200)
    ap.add_argument("--samples_per_prompt", type=int, default=8)
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--num_atoms_distribution", default="ALEX_MP_20")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--parent_filter", default=None,
                    help="Restrict to one pairs.parquet parent (e.g., 'dft_3d').")
    ap.add_argument("--score_energy", action="store_true",
                    help="Relax both target + generations with MatterSim and report "
                         "per-atom-energy MAE. Adds ~30s/structure.")
    ap.add_argument("--mattersim_potential_path", default=None)
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--run_id", type=str, default=None)
    ap.add_argument("--bench_tag", type=str, default="text_cond",
                    help="Bench-dir tag so different buckets sharing this eval don't collide: "
                         "describe -> 'text_cond' (default), ood -> 'ood'. Output goes to "
                         "stage3b_{bench_tag}_g{NN}/{run_id}.")
    args = ap.parse_args()

    from generate_stage3 import generate_for_prompts, load_alm_and_pl_module  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alm, tokenizer, pl_module, K = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        device=device,
    )
    # load_alm_and_pl_module's .to() can miss buffers/property_embeddings (CPU/cuda mismatch); force all here.
    pl_module = pl_module.to(device)
    pl_module.diffusion_module = pl_module.diffusion_module.to(device)
    pl_module.diffusion_module.model = pl_module.diffusion_module.model.to(device)

    if args.out_root is not None:
        os.environ["ALM_EVAL_RESULTS_ROOT"] = str(args.out_root)
    bench_name = f"stage3b_{args.bench_tag}_g{int(args.guidance_factor*10):02d}"
    rd = run_dir(bench_name, args.alm_checkpoint, run_id=args.run_id)
    print(f"[text-cond] writing results to {rd}", flush=True)

    rows = sample_eval_rows(
        Path(args.pairs_parquet),
        n_rows=args.n_test_rows,
        seed=args.seed,
        parent_filter=args.parent_filter,
    )
    print(f"[text-cond] sampled {len(rows)} test rows", flush=True)
    if not rows:
        print("[text-cond] no rows — nothing to do", flush=True)
        return 0

    prompts = [r["user_prompt"] for r in rows]
    prompt_ids = [r["row_id"] for r in rows]
    json_counts_per_prompt = [
        {str(el): int(round(r["target"].composition[el]))
         for el in r["target"].composition.elements}
        for r in rows
    ]
    gen_root = rd / "generations"
    structures_per_prompt = generate_for_prompts(
        prompts=prompts,
        alm=alm, tokenizer=tokenizer, pl_module=pl_module,
        out_root=gen_root,
        batch_size=args.samples_per_prompt,
        num_batches=1,
        diffusion_guidance_factor=args.guidance_factor,
        num_atoms_distribution=args.num_atoms_distribution,
        prompt_ids=prompt_ids,
        json_counts_per_prompt=json_counts_per_prompt,
        save_meta=False,
    )

    predictions = []
    composition_match_ratios = []
    composition_match_pct_at_least_one = []
    density_errs = []
    energy_errs = []  # only populated when --score_energy

    target_e_per_atom: dict[str, float] = {}
    if args.score_energy:
        print("[text-cond] relaxing target structures (one-time)...", flush=True)
        target_atoms_list = [AseAtomsAdaptor.get_atoms(r["target"]) for r in rows]
        relaxed_targets, _ = relax_structures_mattersim(
            target_atoms_list, device=str(device),
            potential_path=args.mattersim_potential_path,
            fmax=0.05, max_n_steps=300,
        )
        for r, ra in zip(rows, relaxed_targets):
            target_e_per_atom[r["row_id"]] = total_energy_per_atom(ra)

    for r, gens in zip(rows, structures_per_prompt):
        target = r["target"]
        target_elems = composition_set(target)
        target_density = density_g_per_cm3(target)

        gen_structs = []
        for g in gens:
            gen_structs.append(g if isinstance(g, Structure) else AseAtomsAdaptor.get_structure(g))

        sample_records = []
        per_sample_cmr = []
        per_sample_dens_err = []
        per_sample_energy_err = []
        gen_e_per_atom_list = None
        if args.score_energy and gen_structs:
            atoms_to_relax = [AseAtomsAdaptor.get_atoms(s) for s in gen_structs]
            try:
                relaxed_gens, _ = relax_structures_mattersim(
                    atoms_to_relax, device=str(device),
                    potential_path=args.mattersim_potential_path,
                    fmax=0.05, max_n_steps=300,
                )
                gen_e_per_atom_list = [total_energy_per_atom(ra) for ra in relaxed_gens]
            except Exception as exc:
                print(f"[text-cond] relax failed for {r['row_id']}: {exc}")
                gen_e_per_atom_list = [float("nan")] * len(gen_structs)

        for j, s in enumerate(gen_structs):
            cmr = composition_match_ratio(s, target_elems)
            per_sample_cmr.append(cmr)
            d = density_g_per_cm3(s)
            d_err = abs(d - target_density) if (np.isfinite(d) and np.isfinite(target_density)) else float("nan")
            per_sample_dens_err.append(d_err)
            v = validity_full(s)

            energy_err = float("nan")
            if gen_e_per_atom_list is not None and r["row_id"] in target_e_per_atom:
                gen_e = gen_e_per_atom_list[j]
                tgt_e = target_e_per_atom[r["row_id"]]
                if np.isfinite(gen_e) and np.isfinite(tgt_e):
                    energy_err = abs(gen_e - tgt_e)
                    per_sample_energy_err.append(energy_err)

            sample_records.append({
                "sample_idx": j,
                "formula": s.composition.reduced_formula,
                "composition_match_ratio": cmr,
                "elements_present": sorted(composition_set(s) & target_elems),
                "elements_missing": sorted(target_elems - composition_set(s)),
                "elements_extra": sorted(composition_set(s) - target_elems),
                "density_err": d_err if np.isfinite(d_err) else None,
                "energy_per_atom_err": energy_err if np.isfinite(energy_err) else None,
                "valid_geom": bool(v["geom"]),
                "valid_charge": bool(v["charge"]),
            })

        if per_sample_cmr:
            mean_cmr = float(np.mean(per_sample_cmr))
            any_match = any(c >= 1.0 for c in per_sample_cmr)
            at_least_one = any(c > 0.0 for c in per_sample_cmr)
        else:
            mean_cmr = 0.0
            any_match = False
            at_least_one = False

        composition_match_ratios.append(mean_cmr)
        composition_match_pct_at_least_one.append(at_least_one)
        valid_dens_errs = [e for e in per_sample_dens_err if np.isfinite(e)]
        if valid_dens_errs:
            density_errs.append(float(np.mean(valid_dens_errs)))
        if per_sample_energy_err:
            energy_errs.append(float(np.mean(per_sample_energy_err)))

        predictions.append({
            "row_id": r["row_id"],
            "parent": r["parent"],
            "target_formula": target.composition.reduced_formula,
            "target_elements": sorted(target_elems),
            "target_density": target_density,
            "n_samples": len(gen_structs),
            "mean_composition_match_ratio": mean_cmr,
            "any_full_match": any_match,
            "any_partial_match": at_least_one,
            "samples": sample_records,
        })

    metrics = {
        "n_test_rows": len(rows),
        "samples_per_prompt": args.samples_per_prompt,
        "guidance_factor": args.guidance_factor,
        "mean_composition_match_ratio": float(np.mean(composition_match_ratios)) if composition_match_ratios else 0.0,
        "pct_with_any_partial_match": float(np.mean(composition_match_pct_at_least_one)) if composition_match_pct_at_least_one else 0.0,
        "density_mae": float(np.mean(density_errs)) if density_errs else None,
        "density_mae_n": len(density_errs),
        "energy_per_atom_mae": float(np.mean(energy_errs)) if energy_errs else None,
        "energy_per_atom_mae_n": len(energy_errs),
        "alm_checkpoint": str(args.alm_checkpoint),
        "atoms_mapper": str(args.atoms_mapper),
        "scored_energy": args.score_energy,
    }
    write_run(rd, metrics, predictions)

    print()
    print(f"[text-cond] DONE — {len(rows)} prompts × {args.samples_per_prompt} samples")
    print(f"  mean_composition_match_ratio = {metrics['mean_composition_match_ratio']:.3f}")
    print(f"  pct_with_any_partial_match   = {metrics['pct_with_any_partial_match']:.3f}")
    print(f"  density_mae (g/cm³)          = {metrics['density_mae']}")
    if args.score_energy:
        print(f"  energy_per_atom_mae (eV/at)  = {metrics['energy_per_atom_mae']}")
    print(f"  results in {rd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
