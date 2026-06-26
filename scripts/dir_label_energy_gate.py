#!/usr/bin/env python3
"""Check whether MatterSim energy ordering agrees with DFT for same-composition polymorph pairs.

Usage: python scripts/dir_label_energy_gate.py --parents mp_3d_2020 oqmd --n_pairs 150
"""

import argparse, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ALM_ROOT, "alm"))
sys.path.insert(0, os.path.join(_ALM_ROOT, "helper_scripts"))
from pymatgen.core import Composition
from eval_edit import _ase_atoms_from_struct
from eval.structure_metrics import relax_structures_mattersim, total_energy_per_atom
from paths import DATA_ROOT

NARR = Path(os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials"))
FE_COL = {
    "dft_3d": "formation energy per atom (eV/atom)",
    "mp_3d_2020": "formation energy per atom (eV/atom)",
    "aflow2": "formation energy per atom (eV/atom)",
    "oqmd": "_oqmd_delta_e",
}
MARGIN_REL = 0.05  # matches DELTA_THRESHOLDS[formation_energy]


def _reduced(elements):
    try:
        return Composition(Counter(str(e).strip() for e in elements)).reduced_formula
    except Exception:
        return None


DB_SOURCES = {  # ASE-db sources with a formation-energy field in row.data
    "cantor_hea": (os.path.join(DATA_ROOT, "LLM4Mat-Bench/cantor_hea/train.db"), "Ef_per_atom"),
}


def _collect_parquet(parent, max_scan):
    """parent -> dict[reduced_formula] -> list[(ase_atoms, E_dft)]."""
    p = NARR / f"{parent}_gpt_narratives.parquet"
    if not p.exists():
        print(f"[gate] {parent}: parquet missing → skip"); return None, 0
    fe_col = FE_COL[parent]
    pf = pq.ParquetFile(p)
    cols = [c for c in ["atoms", fe_col] if c in pf.schema_arrow.names]
    clusters = defaultdict(list); n = 0
    for b in pf.iter_batches(batch_size=10000, columns=cols):
        for r in b.to_pylist():
            n += 1
            a = r.get("atoms"); v = r.get(fe_col)
            if a is None or a.get("elements") is None or v is None:
                continue
            f = _reduced(a["elements"])
            if f is None:
                continue
            try:
                clusters[f].append((_ase_atoms_from_struct(a), float(v)))
            except (TypeError, ValueError):
                pass
        if n >= max_scan:
            break
    return clusters, n


def _collect_db(name, max_scan):
    """ASE-db source -> dict[reduced_formula] -> list[(ase_atoms, E_dft)]."""
    from ase.db import connect
    dbpath, ef_key = DB_SOURCES[name]
    d = connect(dbpath); clusters = defaultdict(list); n = 0
    for row in d.select():
        n += 1
        data = row.data or {}
        ef = data.get(ef_key)
        if ef is None:
            continue
        try:
            atoms = row.toatoms()
            f = data.get("reduced_formula") or _reduced(atoms.get_chemical_symbols())
            clusters[f].append((atoms, float(ef)))
        except Exception:
            pass
        if n >= max_scan:
            break
    return clusters, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", nargs="+", default=[])
    ap.add_argument("--db_sources", nargs="+", default=[], help="ASE-db sources (e.g. cantor_hea)")
    ap.add_argument("--n_pairs", type=int, default=150)
    ap.add_argument("--max_scan", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    device = "cuda"

    sources = [("parquet", p) for p in args.parents] + [("db", s) for s in args.db_sources]
    for kind, parent in sources:
        clusters, n = (_collect_parquet(parent, args.max_scan) if kind == "parquet"
                       else _collect_db(parent, args.max_scan))
        if clusters is None:
            continue
        multi = {f: m for f, m in clusters.items() if len(m) >= 2}
        print(f"[gate] {parent}: scanned {n}, {len(multi)} multi-polymorph compositions", flush=True)

        pairs = []  # (atoms_A, atoms_B, E_dft_A, E_dft_B)
        forms = list(multi.keys()); rng.shuffle(forms)
        for f in forms:
            members = multi[f]
            rng.shuffle(members)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    (aA, eA), (aB, eB) = members[i], members[j]
                    if abs(eA) < 1e-6:
                        ok = abs(eB - eA) > MARGIN_REL
                    else:
                        ok = abs(eB - eA) / abs(eA) > MARGIN_REL
                    if ok:
                        pairs.append((aA, aB, eA, eB))
                        break
                if len(pairs) >= args.n_pairs:
                    break
            if len(pairs) >= args.n_pairs:
                break
        print(f"[gate] {parent}: {len(pairs)} sampled pairs w/ meaningful DFT ΔE", flush=True)
        if not pairs:
            continue

        # members already hold ASE Atoms; don't re-run _ase_atoms_from_struct (wants a struct dict)
        flat, meta = [], []
        for k, (aA, aB, eA, eB) in enumerate(pairs):
            flat.append(aA); meta.append((k, "A"))
            flat.append(aB); meta.append((k, "B"))
        print(f"[gate] {parent}: relaxing {len(flat)} structures ...", flush=True)
        relaxed, _ = relax_structures_mattersim(flat, device=device, fmax=0.05, max_n_steps=300)
        e_ms = {}
        for (k, ab), atoms in zip(meta, relaxed):
            e_ms[(k, ab)] = total_energy_per_atom(atoms)

        agree = disagree = nan = 0
        for k, (aA, aB, eA, eB) in enumerate(pairs):
            msA, msB = e_ms.get((k, "A"), float("nan")), e_ms.get((k, "B"), float("nan"))
            if not (msA == msA and msB == msB):
                nan += 1; continue
            dft_sign = np.sign(eB - eA); ms_sign = np.sign(msB - msA)
            if dft_sign == ms_sign:
                agree += 1
            else:
                disagree += 1
        tot = agree + disagree
        rate = agree / tot if tot else float("nan")
        print(f"\n[gate] === {parent} ORDERING AGREEMENT ===")
        print(f"  scored={tot}  unscorable(NaN relax)={nan}")
        print(f"  DFT↔MatterSim same-sign ΔE = {agree}/{tot} = {rate:.3f}")
        print(f"  VERDICT: {'USE DFT labels (high agreement)' if rate >= 0.8 else 'RELABEL with MatterSim (low agreement)'}\n", flush=True)


if __name__ == "__main__":
    main()
