"""Post-hoc RMSE@1 and RMSE@K for sharded CSP eval outputs (--out_root = per-shard parent dir).

RMSE@1 is the matched RMSD of candidate 0; RMSE@K is the minimum matched RMSD over the K candidates
(CDVAE/CrystaLLM convention, same as eval_csp.py and structure_metrics.py). Both average over matched rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from statistics import mean

import warnings
warnings.filterwarnings("ignore")

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

TOL = dict(ltol=0.3, stol=0.5, angle_tol=10.0)


def load_gt_structures_csv(csv_path: Path):
    out = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mp_id, cif = row.get("material_id"), row.get("cif")
            if mp_id and cif:
                try:
                    out[mp_id] = Structure.from_str(cif, fmt="cif")
                except Exception:
                    pass
    return out


def load_gt_structures_parquet(parquet_path: Path, want_row_ids: set | None = None):
    """GT structures from atoms_struct, keyed by row_id; want_row_ids filters while streaming large parquets."""
    import pyarrow.parquet as pq
    import numpy as np
    pf = pq.ParquetFile(str(parquet_path))
    out = {}
    cols = ["row_id", "atoms_struct"]
    for batch in pf.iter_batches(batch_size=8192, columns=cols):
        b = batch.to_pydict()
        for rid, a in zip(b["row_id"], b["atoms_struct"]):
            if want_row_ids is not None and rid not in want_row_ids:
                continue
            a = a or {}
            elems = a.get("elements")
            lattice = a.get("lattice_mat")
            coords = a.get("coords")
            cartesian = a.get("cartesian", False)
            if not (rid and elems and lattice and coords):
                continue
            try:
                s = Structure(
                    lattice=np.asarray(lattice, dtype=float),
                    species=list(elems),
                    coords=np.asarray(coords, dtype=float),
                    coords_are_cartesian=bool(cartesian),
                )
                out[rid] = s
            except Exception:
                pass
        if want_row_ids is not None and len(out) >= len(want_row_ids):
            break
    return out


def load_gt_structures(source: Path, want_row_ids: set | None = None):
    if source.suffix == ".parquet":
        return load_gt_structures_parquet(source, want_row_ids=want_row_ids)
    return load_gt_structures_csv(source)


def load_generated_cifs(zip_path: Path) -> list[Structure | None]:
    """All candidates in the zip, in sorted-name order (candidate 0 first); None for unparseable CIFs."""
    out: list[Structure | None] = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            try:
                with z.open(name) as f:
                    out.append(Structure.from_str(f.read().decode("utf-8"), fmt="cif"))
            except Exception:
                out.append(None)
    return out


def find_gens_zip(out_root: Path, row_id: str) -> Path | None:
    """Zip may live in any shard_*/ dir."""
    for shard in out_root.glob("shard_*"):
        p = shard / "gens" / row_id / "generated_crystals_cif.zip"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True,
                    help="Per-shard parent dir; predictions.jsonl read from here.")
    ap.add_argument("--max_rows", type=int, default=-1,
                    help="Cap for fast sanity (-1 = all).")
    ap.add_argument("--gt_source", type=Path, required=True,
                    help="CSV (MP-20/MPTS-52 test split) or parquet with an atoms_struct column "
                         "holding the GT structures.")
    args = ap.parse_args()

    print(f"[rmse] gt_source: {args.gt_source}", flush=True)

    pred = args.out_root / "predictions.jsonl"
    if not pred.exists():
        raise SystemExit(f"missing {pred}")
    rows = [json.loads(l) for l in open(pred) if l.strip()]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    print(f"[rmse] {len(rows)} rows from {pred}", flush=True)

    # Only load GT for predicted row_ids (large parquets).
    want_ids = set(r["row_id"] for r in rows if r.get("row_id"))
    gt = load_gt_structures(args.gt_source, want_row_ids=want_ids)
    print(f"[rmse] {len(gt)} GT structures loaded from {args.gt_source.name} (of {len(want_ids)} requested)", flush=True)
    matcher = StructureMatcher(**TOL)

    rmse_n1 = []
    rmse_nK = []
    n_matched_n1_with_rmse = 0
    n_matched_nK_with_rmse = 0
    n_zip_missing = 0
    n_match_recompute_fail = 0

    for i, r in enumerate(rows):
        if (i % 100) == 0:
            print(f"  [{i:4d}/{len(rows)}] running RMSE...", flush=True)
        mp_id = r["row_id"]
        if mp_id not in gt:
            continue
        target = gt[mp_id]

        if r.get("first_match_idx", -1) < 0:
            continue
        zip_path = find_gens_zip(args.out_root, mp_id)
        if zip_path is None:
            n_zip_missing += 1
            continue
        cands = load_generated_cifs(zip_path)
        rmsds: list[float | None] = []
        for s in cands:
            rd = None
            if s is not None:
                try:
                    rd = matcher.get_rms_dist(target, s)
                except Exception:
                    rd = None
            rmsds.append(float(rd[0]) if rd is not None else None)
        matched = [v for v in rmsds if v is not None]
        if not matched:
            n_match_recompute_fail += 1
            continue
        rmse_nK.append(min(matched))
        n_matched_nK_with_rmse += 1
        if rmsds[0] is not None:
            rmse_n1.append(rmsds[0])
            n_matched_n1_with_rmse += 1

    n = len(rows)
    headline = {
        "n_rows": n,
        "RMSE@1_mean": mean(rmse_n1) if rmse_n1 else None,
        "RMSE@1_n": n_matched_n1_with_rmse,
        "RMSE@K_mean": mean(rmse_nK) if rmse_nK else None,
        "RMSE@K_n": n_matched_nK_with_rmse,
        "n_zip_missing": n_zip_missing,
        "n_match_recompute_fail": n_match_recompute_fail,
        "note": ("RMSE@1 = matched RMSD of candidate 0, over rows where it matches. "
                 "RMSE@K = min matched RMSD over the K candidates, over rows with any match."),
    }
    out_path = args.out_root / "rmse_posthoc.json"
    out_path.write_text(json.dumps(headline, indent=2))
    print(f"\n[rmse] wrote {out_path}")
    for k, v in headline.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
