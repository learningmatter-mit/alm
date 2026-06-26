#!/usr/bin/env python
"""Precompute spglib space-group numbers for the MP-20 cache splits."""

import argparse
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def detect_frac(pos: np.ndarray) -> bool:
    lo, hi = float(np.nanmin(pos)), float(np.nanmax(pos))
    return lo >= -0.05 and hi <= 1.05


def run_split(split_dir: Path, symprec: float) -> None:
    an = np.load(split_dir / "atomic_numbers.npy")
    pos = np.load(split_dir / "pos.npy")
    cell = np.load(split_dir / "cell.npy")            # (S, 3, 3)
    num_atoms = np.load(split_dir / "num_atoms.npy")  # (S,)
    n_struct = len(num_atoms)
    frac = detect_frac(pos)
    print(f"  {split_dir.name}: {n_struct} structures, pos={'fractional' if frac else 'CARTESIAN'} "
          f"(range [{pos.min():.3f}, {pos.max():.3f}])", flush=True)

    sgs = np.ones(n_struct, dtype=np.int16)  # SG=1 (P1) is the on-failure default
    n_fail = 0
    offset = 0
    for i in range(n_struct):
        n = int(num_atoms[i])
        z_i = an[offset:offset + n]
        coords_i = pos[offset:offset + n]
        offset += n
        try:
            struct = Structure(
                lattice=Lattice(cell[i]),
                species=[int(z) for z in z_i],
                coords=coords_i,
                coords_are_cartesian=not frac,
            )
            sgs[i] = SpacegroupAnalyzer(struct, symprec=symprec).get_space_group_number()
        except Exception:
            n_fail += 1
    assert offset == len(an), f"atom-count mismatch: consumed {offset} of {len(an)}"

    out = split_dir / "spacegroup.npy"
    np.save(out, sgs)
    # mostly-P1 means the coords convention was likely wrong
    p1_frac = float((sgs == 1).mean())
    uniq = len(np.unique(sgs))
    print(f"    -> wrote {out.name}  | SG=1 frac={p1_frac:.3f}  distinct_SGs={uniq}  failures={n_fail}", flush=True)
    if p1_frac > 0.9:
        print("    !! WARNING: >90% P1 — likely wrong coords convention or degenerate structures", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=str(Path(__file__).resolve().parents[1]
                    / "external" / "mattergen" / "datasets" / "cache" / "mp_20"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--symprec", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.cache_root)
    print(f"[sg-precompute] cache_root={root} symprec={args.symprec}", flush=True)
    for s in args.splits:
        d = root / s
        if not (d / "atomic_numbers.npy").exists():
            print(f"  SKIP {s}: no atomic_numbers.npy", flush=True)
            continue
        run_split(d, args.symprec)
    print("[sg-precompute] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
