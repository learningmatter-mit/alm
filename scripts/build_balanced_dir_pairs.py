#!/usr/bin/env python3
"""Build balanced directional polymorph pairs (phases: scan, energy, pair)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ALM_ROOT, "alm"))
sys.path.insert(0, os.path.join(_ALM_ROOT, "helper_scripts"))

from paths import DATA_ROOT

NARR = Path(os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials"))
ATOMS_STRUCT_TYPE = pa.struct([
    ("elements", pa.list_(pa.string())),
    ("coords", pa.list_(pa.list_(pa.float64()))),
    ("lattice_mat", pa.list_(pa.list_(pa.float64()))),
    ("cartesian", pa.bool_()),
])
OUT_SCHEMA = pa.schema([
    ("row_id", pa.string()),
    ("parent", pa.string()),
    ("source_idx", pa.int64()),
    ("n_atoms", pa.int32()),
    ("narrative", pa.string()),
    ("user_prompt", pa.string()),
    ("assistant_anchor", pa.string()),
    ("atoms_struct", ATOMS_STRUCT_TYPE),
    ("input_atoms_struct", ATOMS_STRUCT_TYPE),
    ("input_source_idx", pa.int64()),
])

# Direction prompt templates; one picked per pair by hash.
TEMPLATES = {
    "lower": [
        "Generate a more thermodynamically stable version of this material.",
        "Design a structure with the same elements but lower formation energy.",
        "Build a more stable polymorph of this material.",
        "Make a version of this with stronger atomic binding (lower formation energy per atom).",
    ],
    "higher": [
        "Generate a metastable variant of this material with higher formation energy.",
        "Design a less thermodynamically favored polymorph of this material.",
        "Build a higher-energy polymorph of this material.",
        "Make a less stable variant of this material with the same composition.",
    ],
}
ASSISTANT_ANCHOR = "Here is the structure: "


def _reduced_formula(elements) -> str | None:
    from pymatgen.core import Composition
    try:
        return Composition(Counter(str(e).strip() for e in elements)).reduced_formula
    except Exception:
        return None


def _atoms_struct_arrays(a: dict):
    """atoms_struct dict -> (numbers, frac_pos, cell), or None if malformed."""
    from ase.data import atomic_numbers as Z_OF
    try:
        elements = [str(e).strip() for e in (a.get("elements") or [])]
        if len(elements) < 2:
            return None
        numbers = np.array([Z_OF[e] for e in elements], dtype=np.int64)
        cell = np.asarray(a.get("lattice_mat"), dtype=np.float64)
        coords = np.asarray(a.get("coords"), dtype=np.float64)
        if cell.shape != (3, 3) or coords.shape != (len(elements), 3):
            return None
        if a.get("cartesian"):
            inv = np.linalg.inv(cell)
            frac = coords @ inv
        else:
            frac = coords
        frac = frac - np.floor(frac)  # wrap to [0,1)
        return numbers, frac.astype(np.float64), cell
    except Exception:
        return None


def _fingerprint(numbers, frac, cell) -> str:
    order = np.lexsort((frac[:, 2], frac[:, 1], frac[:, 0], numbers))
    z = numbers[order].astype(np.int32).tobytes()
    p = np.round(frac[order], 3).astype(np.float32).tobytes()
    c = np.round(cell, 3).astype(np.float32).tobytes()
    return hashlib.md5(z + p + c).hexdigest()


def phase_scan(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    pool_rows = {c: [] for c in OUT_SCHEMA.names}  # placeholder; build small dict below
    pool = {"parent": [], "source_idx": [], "reduced_formula": [],
            "n_atoms": [], "atoms_struct": []}
    clusters_out = {}  # parent -> {formula: [source_idx, ...]}

    for parent in args.parents:
        p = NARR / f"{parent}_gpt_narratives.parquet"
        if not p.exists():
            print(f"[scan] {parent}: parquet missing at {p} → skip", flush=True)
            continue
        pf = pq.ParquetFile(p)
        # formula -> {md5: (source_idx, n_atoms, atoms_struct)}, dedup polymorphs
        byform: dict[str, dict] = defaultdict(dict)
        n = 0
        for b in pf.iter_batches(batch_size=10000, columns=["atoms"]):
            for r in b.to_pylist():
                idx = n; n += 1
                a = r.get("atoms")
                if a is None or a.get("elements") is None:
                    continue
                form = _reduced_formula(a["elements"])
                if form is None:
                    continue
                arrs = _atoms_struct_arrays(a)
                if arrs is None:
                    continue
                md5 = _fingerprint(*arrs)
                if md5 in byform[form]:
                    continue
                byform[form][md5] = (idx, len(a["elements"]), a)
            if args.max_scan and n >= args.max_scan:
                break
        multi = {f: list(m.values()) for f, m in byform.items() if len(m) >= 2}
        cl = {}
        n_pool0 = len(pool["source_idx"])
        for form, members in multi.items():
            if len(members) > args.max_members_per_cluster:
                members = rng.sample(members, args.max_members_per_cluster)
            ids = []
            for (idx, na, a) in members:
                pool["parent"].append(parent)
                pool["source_idx"].append(int(idx))
                pool["reduced_formula"].append(form)
                pool["n_atoms"].append(int(na))
                pool["atoms_struct"].append(a)
                ids.append(int(idx))
            cl[form] = ids
        clusters_out[parent] = cl
        print(f"[scan] {parent}: scanned {n:,}; {len(multi):,} multi-polymorph "
              f"compositions; +{len(pool['source_idx'])-n_pool0:,} pool structures",
              flush=True)

    pool_schema = pa.schema([
        ("parent", pa.string()), ("source_idx", pa.int64()),
        ("reduced_formula", pa.string()), ("n_atoms", pa.int32()),
        ("atoms_struct", ATOMS_STRUCT_TYPE),
    ])
    pq.write_table(pa.Table.from_pydict(pool, schema=pool_schema),
                   out_dir / "pool.parquet", compression="snappy")
    (out_dir / "clusters.json").write_text(json.dumps(clusters_out))
    print(f"[scan] wrote {len(pool['source_idx']):,} pool structures → "
          f"{out_dir/'pool.parquet'}", flush=True)
    print(f"[scan] wrote clusters → {out_dir/'clusters.json'}", flush=True)


def phase_energy(args):
    out_dir = Path(args.out_dir)
    edir = out_dir / "energy"; edir.mkdir(parents=True, exist_ok=True)
    from eval_edit import _ase_atoms_from_struct
    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield.potential import Potential
    import torch

    pf = pq.ParquetFile(out_dir / "pool.parquet")
    rows = []
    for b in pf.iter_batches(batch_size=10000):
        rows.extend(b.to_pylist())
    shard_rows = [r for i, r in enumerate(rows) if (i % args.nshards) == args.shard]
    print(f"[energy] shard {args.shard}/{args.nshards}: {len(shard_rows):,}/{len(rows):,} structures",
          flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    potential = Potential.from_checkpoint(device=device, load_training_state=False)

    keys, atoms_list = [], []
    for r in shard_rows:
        try:
            at = _ase_atoms_from_struct(r["atoms_struct"])
        except Exception:
            continue
        keys.append((r["parent"], int(r["source_idx"])))
        atoms_list.append(at)

    out = {"parent": [], "source_idx": [], "e_per_atom": []}
    B = args.energy_block
    for s in range(0, len(atoms_list), B):
        chunk = atoms_list[s:s + B]
        kchunk = keys[s:s + B]
        try:
            dl = build_dataloader(chunk, only_inference=True, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0)
            energies, _, _ = potential.predict_properties(dl, include_forces=False,
                                                          include_stresses=False)
        except Exception as e:
            print(f"[energy]   block {s} failed ({e}); marking NaN", flush=True)
            energies = [float("nan")] * len(chunk)
        for (parent, sidx), at, E in zip(kchunk, chunk, energies):
            out["parent"].append(parent)
            out["source_idx"].append(sidx)
            out["e_per_atom"].append(float(E) / max(1, len(at)))
        if (s // B) % 20 == 0:
            print(f"[energy]   {s+len(chunk):,}/{len(atoms_list):,}", flush=True)

    eschema = pa.schema([("parent", pa.string()), ("source_idx", pa.int64()),
                         ("e_per_atom", pa.float64())])
    pq.write_table(pa.Table.from_pydict(out, schema=eschema),
                   edir / f"energy_shard{args.shard}.parquet", compression="snappy")
    print(f"[energy] wrote {len(out['source_idx']):,} energies → "
          f"{edir/f'energy_shard{args.shard}.parquet'}", flush=True)


def phase_pair(args):
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    pf = pq.ParquetFile(out_dir / "pool.parquet")
    atoms_by_key = {}
    natoms_by_key = {}
    for b in pf.iter_batches(batch_size=10000):
        for r in b.to_pylist():
            k = (r["parent"], int(r["source_idx"]))
            atoms_by_key[k] = r["atoms_struct"]
            natoms_by_key[k] = int(r["n_atoms"])

    e_by_key = {}
    for ep in sorted((out_dir / "energy").glob("energy_shard*.parquet")):
        epf = pq.ParquetFile(ep)
        for b in epf.iter_batches(batch_size=10000):
            for r in b.to_pylist():
                e = r["e_per_atom"]
                if e == e:  # not NaN
                    e_by_key[(r["parent"], int(r["source_idx"]))] = float(e)
    print(f"[pair] loaded energies for {len(e_by_key):,}/{len(atoms_by_key):,} pool structures",
          flush=True)

    clusters = json.loads((out_dir / "clusters.json").read_text())

    out = {c: [] for c in OUT_SCHEMA.names}
    n_higher = n_lower = n_src_both = n_src_total = 0
    for parent, byform in clusters.items():
        for form, ids in byform.items():
            mem = [(int(i), e_by_key[(parent, int(i))]) for i in ids
                   if (parent, int(i)) in e_by_key]
            if len(mem) < 2:
                continue
            for src_idx, e_src in mem:
                margin = max(args.margin_abs, args.margin_frac * abs(e_src))
                higher = [(j, ej) for j, ej in mem if ej > e_src + margin]
                lower = [(j, ej) for j, ej in mem if ej < e_src - margin]
                n_src_total += 1
                if higher and lower:
                    n_src_both += 1
                emitted = []
                # Pick the target with the largest energy gap (clearest signal).
                if higher:
                    emitted.append(("higher", max(higher, key=lambda t: t[1] - e_src)[0]))
                if lower:
                    emitted.append(("lower", min(lower, key=lambda t: t[1] - e_src)[0]))
                for direction, tgt_idx in emitted:
                    sk = (parent, src_idx); tk = (parent, tgt_idx)
                    src_a = atoms_by_key[sk]; tgt_a = atoms_by_key[tk]
                    tmpls = TEMPLATES[direction]
                    h = hash((parent, src_idx, tgt_idx, direction))
                    out["row_id"].append(
                        f"atomtxtbal-{parent}-{src_idx}-to-{tgt_idx}-formation_energy-{direction}")
                    out["parent"].append(parent)
                    out["source_idx"].append(tgt_idx)
                    out["n_atoms"].append(natoms_by_key[tk])
                    out["narrative"].append("")
                    out["user_prompt"].append("<atoms>\n" + tmpls[h % len(tmpls)])
                    out["assistant_anchor"].append(ASSISTANT_ANCHOR)
                    out["atoms_struct"].append(tgt_a)
                    out["input_atoms_struct"].append(src_a)
                    out["input_source_idx"].append(src_idx)
                    if direction == "higher":
                        n_higher += 1
                    else:
                        n_lower += 1

    n_total = len(out["row_id"])
    if args.max_total_pairs and n_total > args.max_total_pairs:
        keep = rng.sample(range(n_total), args.max_total_pairs)
        keep_set = set(keep)
        for c in out:
            out[c] = [v for i, v in enumerate(out[c]) if i in keep_set]
        n_total = len(out["row_id"])
        n_higher = sum(1 for rid in out["row_id"] if rid.endswith("higher"))
        n_lower = n_total - n_higher

    pq.write_table(pa.Table.from_pydict(out, schema=OUT_SCHEMA),
                   out_dir / "pairs_atomtxt_balanced.parquet", compression="snappy")
    print(f"[pair] sources w/ both dirs available: {n_src_both:,}/{n_src_total:,} "
          f"({100*n_src_both/max(1,n_src_total):.2f}%)", flush=True)
    print(f"[pair] wrote {n_total:,} rows  (higher={n_higher:,}  lower={n_lower:,}  "
          f"= {100*n_higher/max(1,n_total):.1f}% / {100*n_lower/max(1,n_total):.1f}%) → "
          f"{out_dir/'pairs_atomtxt_balanced.parquet'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--parents", nargs="+",
                   default=["dft_3d", "mp_3d_2020", "aflow2", "oqmd"])
    s.add_argument("--out_dir", required=True)
    s.add_argument("--max_members_per_cluster", type=int, default=12)
    s.add_argument("--max_scan", type=int, default=0, help="0 = all rows")
    s.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("energy")
    e.add_argument("--out_dir", required=True)
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--nshards", type=int, default=1)
    e.add_argument("--batch_size", type=int, default=64)
    e.add_argument("--energy_block", type=int, default=4096,
                   help="Structures per dataloader build (memory cap).")

    p = sub.add_parser("pair")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--margin_abs", type=float, default=0.02,
                   help="Min |ΔE/atom| in eV/atom (20 meV/atom = standard polymorph threshold).")
    p.add_argument("--margin_frac", type=float, default=0.0,
                   help="Optional relative margin on |E/atom| (added via max()).")
    p.add_argument("--max_total_pairs", type=int, default=0, help="0 = no cap")
    p.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    {"scan": phase_scan, "energy": phase_energy, "pair": phase_pair}[args.phase](args)


if __name__ == "__main__":
    main()
