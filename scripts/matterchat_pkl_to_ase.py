"""Convert MatterChat Zenodo pickles to ASE dbs with metadata in row.data."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from ase.db import connect
from tqdm import tqdm


_DROP = {"structure", "structure_mace"}   # don't duplicate geometry into metadata


def _to_atoms(rec):
    """Use pre-cached structure_mace Atoms if present, else pymatgen Structure."""
    if "structure_mace" in rec and rec["structure_mace"] is not None:
        return rec["structure_mace"]
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_atoms(rec["structure"])


def _coerce(v):
    """Cast numpy scalars/arrays to JSON-able Python types for ASE db data."""
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)):    return bool(v)
    if isinstance(v, np.ndarray):     return v.tolist()
    return v


def convert(pkl_path: Path, db_path: Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if db_path.exists():
        db_path.unlink()
    db = connect(db_path)
    id_index = {}
    for mid, rec in tqdm(data.items(), desc=db_path.name, unit="row"):
        atoms = _to_atoms(rec)
        meta = {k: _coerce(v) for k, v in rec.items() if k not in _DROP}
        meta["material_id"] = mid
        kvp = {
            "material_id": mid,
            "chemical_formula": rec.get("chemical_formula", ""),
        }
        # Promote labels into kvp for db queries; skip None (ASE rejects nullable kvp).
        for k in ("is_metal", "is_magnetic", "direct_bandgap", "stable",
                  "bandgap", "formation_energy", "energy_above_hull"):
            v = rec.get(k)
            if v is not None:
                kvp[k] = _coerce(v)
        row_id = db.write(atoms, key_value_pairs=kvp, data=meta)
        id_index[mid] = row_id

    idx_path = db_path.with_suffix(".id_index.json")
    with open(idx_path, "w") as f:
        json.dump({k: int(v) for k, v in id_index.items()}, f)
    print(f"wrote {len(data):,} rows → {db_path}")
    print(f"wrote material_id → row_id index → {idx_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_dir", required=True)
    p.add_argument("--out_dir", default=None, help="default: --pkl_dir")
    args = p.parse_args()

    pkl_dir = Path(args.pkl_dir)
    out_dir = Path(args.out_dir) if args.out_dir else pkl_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pkls = sorted(pkl_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"no *.pkl in {pkl_dir}")
    for pkl_path in pkls:
        convert(pkl_path, out_dir / (pkl_path.stem + ".db"))
