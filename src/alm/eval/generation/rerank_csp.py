"""Post-hoc re-rank CSP candidates by SMACT charge validity then ORB-v3 energy, rewriting metrics.json with new M@1/M@K'."""

import argparse
import json
import sys
import os
import time
import warnings
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

warnings.filterwarnings("ignore")

# Requires PYTHONPATH to include alm/.
_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from structure_metrics import (  # noqa: E402
    cdvae_matcher, match_one, validity_charge,
)
from paths import DATA_ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path,
                   help="directory containing the per-shard subdirs")
    p.add_argument("--src_prefix", default="", type=str,
                   help="prefix of shard subdirs. Default empty = match anything; "
                        "use 'shard_' for MG-CSP-fs layout, or the run's shard-dir prefix "
                        "for the ALM-CSP layout.")
    p.add_argument("--gen_dirname", default=None, type=str,
                   help="name of the per-shard subdir holding per-prompt outputs. "
                        "Default: auto-detect ('generations' for ALM-CSP, 'gens' for MG-CSP-fs).")
    p.add_argument("--bench", type=Path,
                   default=Path(os.path.join(DATA_ROOT, "eval_data/csp/mp_20/test.csv")))
    p.add_argument("--K_new", type=int, default=20,
                   help="top-K' to retain after re-ranking (also evaluated for K=1)")
    p.add_argument("--K_orig", type=int, default=256, help="original K")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--orb_model", type=str, default="orb_v3_direct_20_omat")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit_rows", type=int, default=None,
                   help="for smoke-testing")
    return p.parse_args()


class OrbEnergyScorer:
    """Batched per-atom energy prediction via an orb_models pretrained model."""

    def __init__(self, variant: str, device: str):
        from orb_models.forcefield import pretrained
        loader = getattr(pretrained, variant)
        self.model = loader(device=device, precision="float32-high")
        self.model.eval()
        self.system_config = self.model.system_config
        self.device = device

    @torch.no_grad()
    def energies_per_atom(self, atoms_list) -> list[float]:
        """Per-atom energy per input; inf on failure."""
        from orb_models.forcefield import atomic_system
        from orb_models.forcefield.base import batch_graphs

        graphs = []
        valid_indices = []
        n_atoms = []
        for i, a in enumerate(atoms_list):
            try:
                g = atomic_system.ase_atoms_to_atom_graphs(
                    a, system_config=self.system_config, device=self.device,
                )
                graphs.append(g)
                valid_indices.append(i)
                n_atoms.append(len(a))
            except Exception:
                pass

        result = [float("inf")] * len(atoms_list)
        if not graphs:
            return result
        try:
            batch = batch_graphs(graphs)
            out = self.model.predict(batch)
            energies = out["energy"].detach().cpu().tolist()
        except Exception:
            return result
        for i, e, n in zip(valid_indices, energies, n_atoms):
            try:
                result[i] = float(e) / max(1, n)
            except Exception:
                pass
        return result


def load_gt(bench_csv: Path) -> dict[str, Structure]:
    df = pd.read_csv(bench_csv)
    gt = {}
    id_col = "material_id" if "material_id" in df.columns else df.columns[0]
    cif_col = "cif" if "cif" in df.columns else None
    for _, row in df.iterrows():
        try:
            s = Structure.from_str(row[cif_col], fmt="cif")
            gt[row[id_col]] = s
        except Exception:
            pass
    return gt


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rerank] src      = {args.src}")
    print(f"[rerank] K_new    = {args.K_new} (from original K={args.K_orig})")
    print(f"[rerank] out_dir  = {args.out_dir}")

    print(f"[rerank] loading ORB ({args.orb_model}) ...")
    scorer = OrbEnergyScorer(args.orb_model, args.device)

    print(f"[rerank] loading ground truth from {args.bench} ...")
    gt = load_gt(args.bench)
    print(f"  loaded {len(gt)} ground-truth structures")

    matcher = cdvae_matcher()

    # Empty --src_prefix matches everything except the aggregator output.
    shards = sorted([p for p in args.src.iterdir()
                     if p.is_dir() and p.name.startswith(args.src_prefix)
                     and "aggregated" not in p.name])
    print(f"[rerank] {len(shards)} shards found (prefix='{args.src_prefix}')")
    if not shards:
        print(f"[rerank] no shards under {args.src} matching prefix '{args.src_prefix}'")
        return

    new_preds = []
    rerank_stats = {
        "n_rows": 0,
        "n_with_extxyz": 0,
        "n_skipped": 0,
        "rerank_time_s": 0.0,
    }

    for shard in shards:
        if args.gen_dirname is not None:
            gen_root = shard / args.gen_dirname
        else:
            # ALM-CSP wrote 'generations/', MG-CSP-fs writes 'gens/'.
            gen_root = shard / "generations"
            if not gen_root.exists():
                gen_root = shard / "gens"
        if not gen_root.exists():
            continue
        per_row = sorted(gen_root.iterdir())
        for row_dir in per_row:
            if args.limit_rows and rerank_stats["n_rows"] >= args.limit_rows:
                break
            row_id = row_dir.name
            extxyz = row_dir / "generated_crystals.extxyz"
            if not extxyz.exists():
                rerank_stats["n_skipped"] += 1
                continue
            rerank_stats["n_rows"] += 1
            rerank_stats["n_with_extxyz"] += 1

            try:
                atoms_list = ase.io.read(str(extxyz), ":", format="extxyz")
            except Exception as e:
                print(f"  [warn] {row_id}: extxyz read failed: {e}")
                rerank_stats["n_skipped"] += 1
                continue
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            t0 = time.time()

            structures: list[Structure] = []
            for a in atoms_list:
                try:
                    structures.append(AseAtomsAdaptor.get_structure(a))
                except Exception:
                    structures.append(None)

            smact_valid = [validity_charge(s) if s is not None else False
                           for s in structures]
            energies = scorer.energies_per_atom(atoms_list)
            rerank_stats["rerank_time_s"] += time.time() - t0

            # Rank smact-valid first, then ascending energy.
            n = len(structures)
            order = sorted(range(n),
                           key=lambda i: (not smact_valid[i], energies[i]))
            new_topK = order[: args.K_new]

            ref = gt.get(row_id)
            if ref is None:
                rerank_stats["n_skipped"] += 1
                continue

            top1_struct = structures[new_topK[0]] if structures[new_topK[0]] else None
            matched_n1, rmse_n1 = (False, None)
            if top1_struct is not None:
                matched_n1, rmse_n1 = match_one(top1_struct, ref, matcher)
            matched_nK = False
            rmse_nK = None
            match_idx = []
            for i in new_topK:
                s = structures[i]
                if s is None:
                    continue
                m, r = match_one(s, ref, matcher)
                if m:
                    matched_nK = True
                    match_idx.append(i)
                    if rmse_nK is None or r < rmse_nK:
                        rmse_nK = r

            new_preds.append({
                "row_id": row_id,
                "formula": atoms_list[0].get_chemical_formula() if atoms_list else None,
                "n_gen": n,
                "K_new": args.K_new,
                "n_smact_valid": int(sum(smact_valid)),
                "matched_n1": matched_n1, "rmse_n1": rmse_n1,
                "matched_nK_new": matched_nK, "rmse_nK_new": rmse_nK,
                "match_idx": match_idx,
            })

        if args.limit_rows and rerank_stats["n_rows"] >= args.limit_rows:
            break

    n_scored = len([p for p in new_preds if p.get("matched_n1") is not None])
    n_m1 = sum(1 for p in new_preds if p.get("matched_n1"))
    n_mK = sum(1 for p in new_preds if p.get("matched_nK_new"))
    rmses_n1 = [p["rmse_n1"] for p in new_preds if p["rmse_n1"] is not None]
    rmses_nK = [p["rmse_nK_new"] for p in new_preds if p["rmse_nK_new"] is not None]
    metrics = {
        "n_rows": rerank_stats["n_rows"],
        "n_with_extxyz": rerank_stats["n_with_extxyz"],
        "n_skipped": rerank_stats["n_skipped"],
        "K_orig": args.K_orig,
        "K_new": args.K_new,
        "match_rate@1": n_m1 / max(1, rerank_stats["n_rows"]),
        "match_rate@K_new": n_mK / max(1, rerank_stats["n_rows"]),
        "rmse@1_mean": float(np.mean(rmses_n1)) if rmses_n1 else None,
        "rmse@K_new_mean": float(np.mean(rmses_nK)) if rmses_nK else None,
        "rerank_time_s": rerank_stats["rerank_time_s"],
    }
    out_metrics = args.out_dir / "metrics.json"
    out_preds = args.out_dir / "predictions.jsonl"
    json.dump(metrics, open(out_metrics, "w"), indent=2)
    with open(out_preds, "w") as f:
        for p in new_preds:
            f.write(json.dumps(p) + "\n")

    print()
    print(f"[rerank] DONE — {rerank_stats['n_rows']} rows, "
          f"rerank wallclock {rerank_stats['rerank_time_s']:.1f}s")
    print(f"  M@1     (new) = {metrics['match_rate@1']:.3f}  ({n_m1}/{rerank_stats['n_rows']})")
    print(f"  M@{args.K_new:<3d}  (new) = {metrics['match_rate@K_new']:.3f}  ({n_mK}/{rerank_stats['n_rows']})")
    rmK = metrics.get("rmse@K_new_mean")
    if rmK is not None:
        print(f"  RMSE@K_new   = {rmK:.4f}")
    print(f"  → metrics:    {out_metrics}")
    print(f"  → predictions:{out_preds}")


if __name__ == "__main__":
    main()
