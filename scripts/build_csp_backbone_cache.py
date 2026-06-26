#!/usr/bin/env python
"""Build the csp_backbone MatterGen cache (de-leaked train + MP-20 val/test)."""

import argparse
import hashlib
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect

from paths import DATA_ROOT

_REPO = Path(__file__).resolve().parents[1]
LLM4MAT_ROOT = Path(os.environ.get("ALM_DATA_ROOT", _REPO / "data")) / "LLM4Mat-Bench"
MP_20_CACHE = _REPO / "external" / "mattergen" / "datasets" / "cache" / "mp_20"

HULL_DATASETS = {
    "mp":         "energy_above_hull",
    "jarvis_dft": "ehull",
    "cantor_hea": "e_above_hull",
}
NOHULL_DATASETS = ["gnome", "hmof", "jarvis_qetb", "omdb", "oqmd", "qmof", "snumat"]

# Excluded from train so it stays disjoint from the CSP eval splits.
CSP_EVAL_CSVS = [
    os.path.join(DATA_ROOT, "eval_data/csp/mp_20/test.csv"),
    os.path.join(DATA_ROOT, "eval_data/csp/mp_20/val.csv"),
    os.path.join(DATA_ROOT, "eval_data/csp/mpts_52/test.csv"),
    os.path.join(DATA_ROOT, "eval_data/csp/mpts_52/val.csv"),
]


def fingerprint(numbers, frac_pos, cell) -> str:
    """Cheap structure fingerprint for cross-subdataset dedup."""
    # Canonical atom order: sort by Z then fractional position.
    order = np.lexsort((frac_pos[:, 2], frac_pos[:, 1], frac_pos[:, 0], numbers))
    z = numbers[order].astype(np.int32).tobytes()
    p = np.round(frac_pos[order], 3).astype(np.float32).tobytes()
    c = np.round(cell, 3).astype(np.float32).tobytes()
    return hashlib.md5(z + p + c).hexdigest()


def load_hull_lookup(ds: str) -> dict:
    """db_row_id (int) -> E_hull, resolved via train.id_index.json."""
    import json
    col = HULL_DATASETS[ds]
    csv_path = LLM4MAT_ROOT / ds / "train.csv"
    idx_path = LLM4MAT_ROOT / ds / "train.id_index.json"
    if not csv_path.exists() or not idx_path.exists():
        return {}
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    id_col = next((c for c in ("material_id", "jarvis_id", "hea_id", "jid", "id", "structure_id")
                   if c in header), None)
    if id_col is None:
        print(f"  [{ds}] WARN: no id column in train.csv ({header[:5]}...) — skipping hull lookup")
        return {}
    df = pd.read_csv(csv_path, usecols=[id_col, col])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    csv_hull_by_did = dict(zip(df[id_col].astype(str), df[col].astype(float)))
    forward = json.loads(idx_path.read_text())  # {dataset_id: db_row_id}
    return {int(db_id): csv_hull_by_did[ds_id]
            for ds_id, db_id in forward.items() if ds_id in csv_hull_by_did}


def load_exclusion(csv_paths: list[str]):
    """CSP-eval material_ids (primary key) + structure fingerprints (catches non-mp ids) to de-leak."""
    from pymatgen.core import Structure
    mpids, fps = set(), set()
    for p in csv_paths:
        p = Path(p)
        if not p.exists():
            print(f"  [exclude] WARN: {p} missing — skipping", flush=True)
            continue
        df = pd.read_csv(p)
        if "material_id" in df.columns:
            mpids.update(df["material_id"].astype(str).tolist())
        n_fp = 0
        if "cif" in df.columns:
            for cif in df["cif"].dropna().tolist():
                try:
                    s = Structure.from_str(cif, fmt="cif")
                except Exception:
                    continue
                numbers = np.array(s.atomic_numbers, dtype=np.int64)
                frac = (np.array(s.frac_coords, dtype=np.float64) % 1.0)
                cell = np.array(s.lattice.matrix, dtype=np.float64)
                fps.add(fingerprint(numbers, frac, cell))
                n_fp += 1
        print(f"  [exclude] {p.name}: {len(df)} rows, +{n_fp} fingerprints", flush=True)
    print(f"[exclude] total: {len(mpids)} material_ids, {len(fps)} fingerprints", flush=True)
    return mpids, fps


def iter_filtered(ds: str, split: str, max_atoms: int, hull_thresh: float,
                  hull_lookup: dict | None, exclude_db_ids: set | None = None):
    """Yield (sid, numbers, frac_pos, cell) passing atom-cap + (if available) E_hull filter."""
    db_path = LLM4MAT_ROOT / ds / f"{split}.db"
    if not db_path.exists():
        return
    has_hull = hull_lookup is not None and len(hull_lookup) > 0
    with connect(db_path) as db:
        for row in db.select(f"natoms<={max_atoms}"):
            if exclude_db_ids and row.id in exclude_db_ids:
                continue
            sid = f"{ds}_db{row.id}"
            if has_hull:
                h = hull_lookup.get(int(row.id))
                if h is None or h > hull_thresh:
                    continue
            atoms = row.toatoms()
            yield sid, atoms.numbers.astype(np.int64), \
                  atoms.get_scaled_positions(wrap=True).astype(np.float64), \
                  atoms.cell.array.astype(np.float64)


def reservoir_sample(stream, k: int, rng: random.Random):
    """Algorithm R reservoir sampling: keep k items from a stream of unknown length."""
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randrange(i + 1)
            if j < k:
                reservoir[j] = item
    return reservoir


def collect_records(args, rng, exclude_mpids=None):
    """Returns list of (sid, numbers, frac_pos, cell)."""
    import json
    all_records = []

    # E_hull subdatasets: take ALL metastable.
    for ds in HULL_DATASETS:
        hull = load_hull_lookup(ds)
        print(f"[{ds}] hull lookup: {len(hull)} entries (col={HULL_DATASETS[ds]})", flush=True)
        # CSP-eval mp-ids live only in the mp subdataset; map to db rows and skip.
        excl_db = None
        if exclude_mpids and ds == "mp":
            idx_path = LLM4MAT_ROOT / ds / "train.id_index.json"
            if idx_path.exists():
                forward = json.loads(idx_path.read_text())  # {material_id: db_row_id}
                excl_db = {int(forward[m]) for m in exclude_mpids if m in forward}
                print(f"[{ds}] de-leak: excluding {len(excl_db)} db rows "
                      f"(of {len(exclude_mpids)} CSP-eval material_ids)", flush=True)
        recs = list(iter_filtered(ds, "train", args.max_atoms, args.hull_thresh, hull, excl_db))
        print(f"[{ds}] kept {len(recs)} metastable (atom<={args.max_atoms}, E_hull<={args.hull_thresh})",
              flush=True)
        all_records += recs

    # No-E_hull subdatasets: reservoir-sample N each.
    for ds in NOHULL_DATASETS:
        recs = reservoir_sample(
            iter_filtered(ds, "train", args.max_atoms, args.hull_thresh, None),
            args.sample_per_nohull, rng,
        )
        print(f"[{ds}] reservoir-sampled {len(recs)} (atom<={args.max_atoms}, no hull filter)",
              flush=True)
        all_records += recs

    return all_records


def dedup(records, exclude_fps=None):
    """Fingerprint dedup; also drops any structure matching a CSP-eval fingerprint."""
    seen, out = set(), []
    exclude_fps = exclude_fps or set()
    n_excl = 0
    for sid, numbers, frac_pos, cell in records:
        fp = fingerprint(numbers, frac_pos, cell)
        if fp in exclude_fps:
            n_excl += 1
            continue
        if fp not in seen:
            seen.add(fp)
            out.append((sid, numbers, frac_pos, cell))
    if exclude_fps:
        print(f"[dedup]   fingerprint-excluded {n_excl} CSP-eval structures (de-leak supplement)",
              flush=True)
    return out


def write_cache(records, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        print(f"  [{out_dir}] EMPTY — skipping write", flush=True)
        return
    an_flat = np.concatenate([r[1] for r in records]).astype(np.int64)
    pos_flat = np.concatenate([r[2] for r in records]).astype(np.float64)
    cell_arr = np.stack([r[3] for r in records]).astype(np.float64)
    natoms = np.array([len(r[1]) for r in records], dtype=np.int64)
    # Fixed-width unicode, not dtype=object: MatterGen loads with allow_pickle=False.
    sid_list = [r[0] for r in records]
    max_len = max(len(s) for s in sid_list)
    sids = np.array(sid_list, dtype=f"U{max_len}")
    np.save(out_dir / "atomic_numbers.npy", an_flat)
    np.save(out_dir / "pos.npy", pos_flat)
    np.save(out_dir / "cell.npy", cell_arr)
    np.save(out_dir / "num_atoms.npy", natoms)
    np.save(out_dir / "structure_id.npy", sids)
    print(f"  [{out_dir}] wrote {len(records)} structures, {int(natoms.sum())} atoms total",
          flush=True)


def copy_mp20_split(out_root: Path, split: str):
    """Copy mp_20's val/test split into csp_backbone so eval uses the MP-20 benchmark."""
    src = MP_20_CACHE / ("val" if split == "val" else "test")
    dst = out_root / split
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("atomic_numbers.npy", "pos.npy", "cell.npy",
                 "num_atoms.npy", "structure_id.npy"):
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    print(f"  [{dst}] copied MP-20 {split} cache (shared eval split)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path,
                    default=MP_20_CACHE.parent / "csp_backbone")
    ap.add_argument("--max_atoms", type=int, default=40)
    ap.add_argument("--hull_thresh", type=float, default=0.1)
    ap.add_argument("--sample_per_nohull", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke: only mp, sample_per_nohull=200 (skips no-hull subdatasets except 1).")
    ap.add_argument("--no_exclude_csp_eval", action="store_true",
                    help="(debug) DON'T de-leak — keep MP-20/MPTS-52 test+val in train.")
    ap.add_argument("--csp_eval_csvs", type=str, default=",".join(CSP_EVAL_CSVS),
                    help="Comma list of CSP-eval csvs whose material_ids/structures are excluded from train.")
    args = ap.parse_args()

    if args.smoke:
        global HULL_DATASETS, NOHULL_DATASETS
        HULL_DATASETS = {"mp": HULL_DATASETS["mp"]}
        NOHULL_DATASETS = ["oqmd"]
        args.sample_per_nohull = 200
        args.out_root = args.out_root.parent / "csp_backbone_smoke"
        print("[smoke] only mp + 200-sample from oqmd", flush=True)

    rng = random.Random(args.seed)
    print(f"[builder] out={args.out_root} max_atoms={args.max_atoms} "
          f"hull<={args.hull_thresh} sample/nohull={args.sample_per_nohull}", flush=True)

    exclude_mpids, exclude_fps = set(), set()
    if not args.no_exclude_csp_eval:
        exclude_mpids, exclude_fps = load_exclusion(args.csp_eval_csvs.split(","))

    records = collect_records(args, rng, exclude_mpids)
    print(f"[collect] total before dedup: {len(records)}", flush=True)
    records = dedup(records, exclude_fps)
    print(f"[dedup]   total after dedup: {len(records)}", flush=True)

    rng.shuffle(records)
    write_cache(records, args.out_root / "train")

    if not args.smoke:
        copy_mp20_split(args.out_root, "val")
        copy_mp20_split(args.out_root, "test")
    print("[builder] DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
