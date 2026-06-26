#!/usr/bin/env python
"""Build ALM Bench `doping` editing pairs from isostructural single-swap substitutions.

  python scripts/build_doping_pairs.py --out_dir <data_root>/stage3_outputs/stage3a
"""

import argparse
import os
import random
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from paths import DATA_ROOT

PROMPTS = [
    "Dope this material by replacing all {X} atoms with {Y}.",
    "Generate a doped variant: substitute {X} sites with {Y} throughout this structure.",
]
ANCHOR = "The substituted structure: "
PARENT = "mp_3d_2020"
VOL_COL = "volume (Å³)"
SG_COL = "space group symbol"


def _atoms_struct(atoms: dict) -> dict:
    return {k: atoms[k] for k in ("cartesian", "coords", "elements", "lattice_mat")}


def _sg_and_nuniq(atoms: dict):
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    s = Structure(atoms["lattice_mat"], atoms["elements"], atoms["coords"],
                  coords_are_cartesian=bool(atoms["cartesian"]))
    sga = SpacegroupAnalyzer(s, symprec=0.1)
    return sga.get_space_group_number(), len(sga.get_symmetrized_structure().equivalent_sites)


def _single_swap(ca: Counter, cb: Counter):
    """Return (X, Y) if A and B differ by exactly one X->Y swap with all other counts equal, else None."""
    only_a, only_b = ca.keys() - cb.keys(), cb.keys() - ca.keys()
    if len(only_a) != 1 or len(only_b) != 1:
        return None
    X, Y = next(iter(only_a)), next(iter(only_b))
    if ca[X] != cb[Y]:                       # swapped sublattice must have equal occupancy
        return None
    if {e: ca[e] for e in ca.keys() & cb.keys()} != {e: cb[e] for e in ca.keys() & cb.keys()}:
        return None                          # every shared element must keep its count
    return X, Y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials/mp_3d_2020_gpt_narratives.parquet"))
    ap.add_argument("--out_dir", default=os.path.join(DATA_ROOT, "stage3_outputs/stage3a"))
    ap.add_argument("--id_col", default="material_id", help="id column for row_id; falls back to row index when absent")
    ap.add_argument("--max_per_group", type=int, default=0, help="cap pairs per isostructural group (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cols = ["atoms", SG_COL, VOL_COL]
    has_id = bool(args.id_col) and args.id_col in pq.ParquetFile(args.source).schema_arrow.names
    rows = pq.read_table(args.source, columns=cols + ([args.id_col] if has_id else [])).to_pylist()
    def rid(i, r): return str(r[args.id_col]) if has_id and r.get(args.id_col) is not None else f"{PARENT}-{i}"

    # cheap isostructural signature: space group symbol + stoichiometry shape
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        a = r["atoms"]
        if not a or not r[SG_COL] or r[VOL_COL] is None:
            continue
        counts = Counter(a["elements"])
        sig = (r[SG_COL], tuple(sorted(counts.values())), len(a["elements"]))
        groups[sig].append((i, r, counts))

    out = []
    for group in groups.values():
        if len(group) < 2:
            continue
        pairs = [(group[x], group[y]) for x in range(len(group)) for y in range(len(group)) if x != y]
        if args.max_per_group:
            rng.shuffle(pairs); pairs = pairs[:args.max_per_group]
        for (ai, A, ca), (bi, B, cb) in pairs:
            swap = _single_swap(ca, cb)
            if swap is None:
                continue
            X, Y = swap
            vA, vB = A[VOL_COL], B[VOL_COL]
            dV = (vB - vA) / vA * 100.0
            try:
                sg, nuniq = _sg_and_nuniq(B["atoms"])
            except Exception:
                continue
            row_id = (f"doping_strain-mp-{rid(ai,A)}-to-{rid(bi,B)}-{X}_to_{Y}"
                      f"-dV_{dV:+.2f}pct-sg_{sg}-n_uniq_{nuniq}")
            out.append({
                "row_id": row_id, "parent": PARENT, "source_idx": rid(bi, B),
                "n_atoms": len(B["atoms"]["elements"]), "narrative": "",
                "user_prompt": "<atoms>\n" + rng.choice(PROMPTS).format(X=X, Y=Y),
                "assistant_anchor": ANCHOR, "atoms_struct": _atoms_struct(B["atoms"]),
                "input_atoms_struct": _atoms_struct(A["atoms"]), "input_source_idx": rid(ai, A),
            })

    os.makedirs(args.out_dir, exist_ok=True)
    dst = os.path.join(args.out_dir, "pairs_doping_strain.parquet")
    pq.write_table(pa.Table.from_pylist(out), dst)
    print(f"wrote {len(out)} doping/substitution pairs -> {dst}")


if __name__ == "__main__":
    main()
