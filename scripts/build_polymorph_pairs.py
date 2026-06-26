#!/usr/bin/env python
"""Build the ALM Bench `polymorph` editing pairs: for each same-composition pair (A, B) with E_hull(B) < E_hull(A), emit the edit A -> B.

  python scripts/build_polymorph_pairs.py --out_dir <data_root>/stage3_outputs/stage3a
"""

import argparse
import os
import random
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from paths import DATA_ROOT

PROMPTS = [
    "Generate a polymorph of this material below the convex hull.",
    "Build a more stable polymorph — closer to the convex hull.",
    "Design a structure with the same composition that sits below the convex hull.",
]
ANCHOR = "The crystal structure: "
PARENT = "mp_3d_2020"
EHULL_COL = "energy above hull (eV/atom)"


def _atoms_struct(atoms: dict) -> dict:
    return {k: atoms[k] for k in ("cartesian", "coords", "elements", "lattice_mat")}


def _n_uniq(atoms: dict) -> int:
    """Number of symmetry-inequivalent sites; falls back to unique coords on spglib failure."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    s = Structure(atoms["lattice_mat"], atoms["elements"], atoms["coords"],
                  coords_are_cartesian=bool(atoms["cartesian"]))
    try:
        return len(SpacegroupAnalyzer(s, symprec=0.1).get_symmetrized_structure().equivalent_sites)
    except Exception:
        return len(set(map(tuple, atoms["coords"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials/mp_3d_2020_gpt_narratives.parquet"))
    ap.add_argument("--out_dir", default=os.path.join(DATA_ROOT, "stage3_outputs/stage3a"))
    ap.add_argument("--id_col", default="material_id", help="id column for row_id; falls back to row index when absent")
    ap.add_argument("--max_per_formula", type=int, default=0, help="cap A->B pairs per formula (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cols = ["atoms", "reduced_formula", EHULL_COL]
    has_id = bool(args.id_col) and args.id_col in pq.ParquetFile(args.source).schema_arrow.names
    rows = pq.read_table(args.source, columns=cols + ([args.id_col] if has_id else [])).to_pylist()
    def rid(i, r): return str(r[args.id_col]) if has_id and r.get(args.id_col) is not None else f"{PARENT}-{i}"

    by_formula = defaultdict(list)
    for i, r in enumerate(rows):
        if r["atoms"] and r["reduced_formula"] and r[EHULL_COL] is not None:
            by_formula[r["reduced_formula"]].append((i, r))

    out = []
    for group in by_formula.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda ir: ir[1][EHULL_COL])           # most stable first
        pairs = [(a, b) for x, a in enumerate(group) for b in group[:x]]  # A is higher-E_hull, B lower
        if args.max_per_formula:
            rng.shuffle(pairs); pairs = pairs[:args.max_per_formula]
        for (ai, A), (bi, B) in pairs:
            eA, eB = A[EHULL_COL], B[EHULL_COL]
            row_id = (f"polymorph_under_hull-mp-{rid(ai,A)}-to-{rid(bi,B)}"
                      f"-ehull_{eA:.4f}_to_{eB:.4f}-n_uniq_{_n_uniq(B['atoms'])}")
            out.append({
                "row_id": row_id, "parent": PARENT, "source_idx": rid(bi, B),
                "n_atoms": len(B["atoms"]["elements"]), "narrative": "",
                "user_prompt": "<atoms>\n" + rng.choice(PROMPTS), "assistant_anchor": ANCHOR,
                "atoms_struct": _atoms_struct(B["atoms"]),
                "input_atoms_struct": _atoms_struct(A["atoms"]), "input_source_idx": rid(ai, A),
            })

    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, "pairs_polymorph_under_hull.parquet")
    pq.write_table(pa.Table.from_pylist(out), dst)
    print(f"wrote {len(out)} polymorph pairs -> {dst}")


if __name__ == "__main__":
    main()
