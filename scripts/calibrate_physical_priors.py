"""Compute empirical 1st/99th-percentile physical-prior bounds on a CSP benchmark.

Usage:
  python scripts/calibrate_physical_priors.py \\
      --benchmark mp_20 \\
      --out_path  <results_dir>/mp20_prior_bounds.json
"""
from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alm" / "eval"))
from csp_recovery import read_test_rows  # noqa: E402


def _min_pair_distance(struct) -> float:
    """Smallest interatomic distance in the unit cell, considering PBC."""
    n = len(struct)
    if n < 2:
        return float("inf")
    dmat = struct.distance_matrix
    iu = np.triu_indices(n, k=1)
    return float(dmat[iu].min())


def _aspect_ratio(struct) -> float:
    """max(a,b,c) / min(a,b,c), a proxy for layered / rod-like cells."""
    abc = np.array(struct.lattice.abc)
    return float(abc.max() / abc.min())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="mp_20",
                    choices=["mp_20", "mpts_52", "perov_5", "carbon_24"])
    ap.add_argument("--out_path", type=Path, required=True)
    ap.add_argument("--lo_pct", type=float, default=1.0,
                    help="Lower percentile for bounds (default 1.0).")
    ap.add_argument("--hi_pct", type=float, default=99.0,
                    help="Upper percentile for bounds (default 99.0).")
    args = ap.parse_args()

    rows = list(read_test_rows(args.benchmark))
    print(f"[calibrate] loaded {len(rows)} {args.benchmark} test rows")

    densities = []
    vol_per_atom = []
    min_pair = []
    aspect = []
    angles_min = []
    angles_max = []
    per_row = []
    for r in rows:
        struct = r["ref_structure"]
        d = float(struct.density)
        vpa = float(struct.volume / len(struct))
        mp = _min_pair_distance(struct)
        ar = _aspect_ratio(struct)
        ang = list(struct.lattice.angles)
        densities.append(d)
        vol_per_atom.append(vpa)
        min_pair.append(mp)
        aspect.append(ar)
        angles_min.append(min(ang))
        angles_max.append(max(ang))
        per_row.append({
            "row_id": r["row_id"], "formula": r["formula"],
            "density": d, "vol_per_atom": vpa,
            "min_pair_distance": mp, "aspect_ratio": ar,
            "angle_min": min(ang), "angle_max": max(ang),
        })

    def _pct(xs, p):
        return float(np.percentile(np.asarray(xs), p))

    bounds = {
        "benchmark": args.benchmark,
        "n_rows": len(rows),
        "lo_pct": args.lo_pct,
        "hi_pct": args.hi_pct,
        "density_min": _pct(densities, args.lo_pct),
        "density_max": _pct(densities, args.hi_pct),
        "vol_per_atom_min": _pct(vol_per_atom, args.lo_pct),
        "vol_per_atom_max": _pct(vol_per_atom, args.hi_pct),
        "min_pair_distance_min": _pct(min_pair, args.lo_pct),
        # No upper bound; large min-pair distance is fine.
        "aspect_ratio_max": _pct(aspect, args.hi_pct),
        "angle_min_min": _pct(angles_min, args.lo_pct),
        "angle_max_max": _pct(angles_max, args.hi_pct),
    }

    # At-boundary = within ~5% of either end; surfaces structures the prior might suppress.
    def _near(val, lo, hi):
        if lo is None or hi is None:
            return False
        span = hi - lo if hi > lo else 1.0
        return val < lo + 0.05 * span or val > hi - 0.05 * span

    at_risk = []
    for row in per_row:
        flags = []
        if _near(row["density"], bounds["density_min"], bounds["density_max"]):
            flags.append("density")
        if _near(row["vol_per_atom"], bounds["vol_per_atom_min"], bounds["vol_per_atom_max"]):
            flags.append("vol_per_atom")
        if row["min_pair_distance"] < bounds["min_pair_distance_min"] * 1.05:
            flags.append("min_pair_distance")
        if row["aspect_ratio"] > bounds["aspect_ratio_max"] * 0.95:
            flags.append("aspect_ratio")
        if row["angle_min"] < bounds["angle_min_min"] * 1.05:
            flags.append("angle_min")
        if row["angle_max"] > bounds["angle_max_max"] * 0.95:
            flags.append("angle_max")
        if flags:
            at_risk.append({**row, "flags": flags})

    out = {"bounds": bounds, "at_risk": at_risk[:50]}

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[calibrate] wrote {args.out_path}")
    print(f"[calibrate] bounds:")
    for k, v in bounds.items():
        if isinstance(v, float):
            print(f"  {k:32s} = {v:.4f}")
        else:
            print(f"  {k:32s} = {v}")
    print(f"[calibrate] {len(at_risk)} of {len(rows)} rows ({100*len(at_risk)/len(rows):.1f}%) "
          f"sit at boundary on >=1 metric")
    if at_risk:
        print(f"[calibrate] first 10 at-risk rows:")
        for row in at_risk[:10]:
            print(f"  {row['row_id']:24s} {row['formula']:20s} flags={row['flags']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
