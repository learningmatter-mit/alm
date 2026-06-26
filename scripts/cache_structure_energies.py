"""Cache MatterSim energy + MP-2020 hull stability per parquet shard; --merge concatenates shards and writes a stable-row-index file."""
from __future__ import annotations


import argparse
import json
import math
import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# alm/ and external/mattergen on the path so structure_metrics + eval imports resolve.
_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ALM_ROOT, "alm"))
sys.path.insert(0, _ALM_ROOT)
sys.path.insert(0, os.path.join(_ALM_ROOT, "external", "mattergen"))

from pymatgen.core import Structure  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402
from paths import DATA_ROOT  # noqa: E402

from eval.structure_metrics import (  # noqa: E402
    relax_structures_mattersim,
    e_above_hull_per_atom,
    load_hull_reference,
)
from eval.eval_dng import _atoms_struct_to_pymatgen  # noqa: E402


# Narrative buckets state DFT properties in prose only; permissive regexes pull them as a reference column.
import re  # noqa: E402

_NUM = r"(-?\d+\.?\d*)"
_PROP_PATTERNS = {
    "dft_formation_energy": re.compile(rf"formation energy per atom (?:is |of )?{_NUM}", re.I),
    "dft_e_above_hull": re.compile(rf"energy above (?:the )?hull(?: per atom)? (?:is |of )?{_NUM}", re.I),
    "dft_total_energy_per_atom": re.compile(rf"total energy per atom (?:is |of )?{_NUM}", re.I),
    "dft_band_gap": re.compile(rf"band gap (?:is |of )?{_NUM}", re.I),
    "dft_density": re.compile(rf"density (?:is |of )?(?:a high density of )?{_NUM}", re.I),
}


def _extract_dft_props(narrative: str | None) -> dict:
    out = {k: None for k in _PROP_PATTERNS}
    if not narrative:
        return out
    for k, pat in _PROP_PATTERNS.items():
        m = pat.search(narrative)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out


# Cheap pre-relax filter: tags n_atoms<=2 / tiny / collided cells unstable so we skip a wasted relax step.
def _is_degenerate_struct(s: Structure) -> bool:
    try:
        if len(s) <= 2:
            return True
        d = s.distance_matrix
        n = len(s)
        if n > 1:
            iu = np.triu_indices(n, k=1)
            if iu[0].size and float(d[iu].min()) < 0.5:
                return True
        vol = float(s.volume)
        if not math.isfinite(vol) or vol <= 1e-3 or (vol / max(1, n)) > 1.0e4:
            return True
    except Exception:
        return True
    return False


DEDUP_ADAPTERS = {
    # narrative buckets share structures → dedup on (parent, source_idx); editing buckets are distinct → row_id.
    "narrative": lambda row: f"{row.get('parent')}::{row.get('source_idx')}",
    "row_id": lambda row: str(row.get("row_id")),
}


def _shard_rows(pf: pq.ParquetFile, shard_idx: int, num_shards: int, columns: list[str]):
    """Yield (global_index, row_dict) for rows where global_index % num_shards == shard_idx."""
    cursor = 0
    for batch in pf.iter_batches(batch_size=4096, columns=columns):
        rows = batch.to_pylist()
        for r in rows:
            gi = cursor
            cursor += 1
            if gi % num_shards == shard_idx:
                yield gi, r


def run_shard(args):
    pf = pq.ParquetFile(str(args.parquet))
    schema_cols = {f.name for f in pf.schema_arrow}
    wanted = ["row_id", "parent", "source_idx", "n_atoms", "atoms_struct"]
    if "narrative" in schema_cols:
        wanted.append("narrative")
    if "input_source_idx" in schema_cols:
        wanted.append("input_source_idx")
    columns = [c for c in wanted if c in schema_cols]

    dedup_fn = DEDUP_ADAPTERS[args.dedup_key]

    print(f"[cache-energy] {args.parquet.name} shard {args.shard_idx}/{args.num_shards} "
          f"total_rows={pf.metadata.num_rows} dedup_key={args.dedup_key}", flush=True)

    # Within-shard dedup on the adapter key; cross-shard dedup happens at merge.
    rows_meta: list[dict] = []
    structs: list[Structure] = []
    degenerate_flags: list[bool] = []
    seen_keys: set[str] = set()

    for gi, r in _shard_rows(pf, args.shard_idx, args.num_shards, columns):
        key = dedup_fn(r)
        if args.dedup_key == "narrative":
            if key in seen_keys:
                continue
            seen_keys.add(key)
        a = r["atoms_struct"]
        if hasattr(a, "as_py"):
            a = a.as_py()
        try:
            s = _atoms_struct_to_pymatgen(a)
        except Exception:
            s = None
        meta = {
            "parquet": args.parquet.name,
            "row_id": str(r.get("row_id")),
            "parent": r.get("parent"),
            "source_idx": r.get("source_idx"),
            "input_source_idx": r.get("input_source_idx"),
            "dedup_key": key,
            "global_index": gi,
        }
        meta.update(_extract_dft_props(r.get("narrative")))
        if s is None:
            meta.update(dict(
                n_atoms=int(r.get("n_atoms") or 0), composition=None,
                mattersim_total_energy=None, energy_per_atom=None,
                e_above_hull=None, is_stable=False, relax_status="decode_failed",
            ))
            rows_meta.append(meta)
            structs.append(None)
            degenerate_flags.append(True)
            continue
        meta["n_atoms"] = len(s)
        meta["composition"] = s.composition.reduced_formula
        rows_meta.append(meta)
        structs.append(s)
        degenerate_flags.append(_is_degenerate_struct(s))

    n = len(structs)
    n_degen = sum(degenerate_flags)
    print(f"[cache-energy] shard {args.shard_idx}: {n} unique structures "
          f"({n_degen} degenerate/decode-fail → scored unstable, no relax)", flush=True)

    # Relax only the non-degenerate ones.
    relax_idx = [i for i in range(n) if not degenerate_flags[i] and structs[i] is not None]
    energies = np.full(n, float("nan"), dtype=float)
    relaxed_atoms = [None] * n
    if relax_idx:
        reference = (
            load_hull_reference(args.hull_dir) if args.hull_dir is not None
            else load_hull_reference()
        )
        sub_structs = [structs[i] for i in relax_idx]
        try:
            r_atoms, r_energies = relax_structures_mattersim(
                sub_structs, device=args.mattersim_device,
            )
        except Exception as e:
            print(f"[cache-energy] relax batch err: {e}", flush=True)
            r_atoms = [None] * len(sub_structs)
            r_energies = np.full(len(sub_structs), float("nan"))
        for j, i in enumerate(relax_idx):
            relaxed_atoms[i] = r_atoms[j]
            energies[i] = float(r_energies[j])
    else:
        reference = None

    n_eh = 0
    for i, meta in enumerate(rows_meta):
        if "mattersim_total_energy" in meta:  # already finalized (decode_failed)
            continue
        e_total = energies[i]
        ase_atoms = relaxed_atoms[i]
        if degenerate_flags[i] or structs[i] is None:
            meta.update(dict(
                mattersim_total_energy=None, energy_per_atom=None,
                e_above_hull=None, is_stable=False, relax_status="degenerate",
            ))
            continue
        if ase_atoms is None or not math.isfinite(e_total):
            meta.update(dict(
                mattersim_total_energy=None, energy_per_atom=None,
                e_above_hull=None, is_stable=False, relax_status="relax_failed",
            ))
            continue
        try:
            s_relaxed = (
                ase_atoms if isinstance(ase_atoms, Structure)
                else AseAtomsAdaptor.get_structure(ase_atoms)
            )
            n_at = max(1, len(s_relaxed))
            eh = e_above_hull_per_atom(
                structure=s_relaxed, total_energy_eV=float(e_total),
                hull_reference=reference,
            )
            if math.isnan(eh):
                meta.update(dict(
                    mattersim_total_energy=float(e_total),
                    energy_per_atom=float(e_total) / n_at,
                    e_above_hull=None, is_stable=False, relax_status="hull_missing_chemsys",
                ))
            else:
                n_eh += 1
                meta.update(dict(
                    mattersim_total_energy=float(e_total),
                    energy_per_atom=float(e_total) / n_at,
                    e_above_hull=float(eh),
                    is_stable=bool(eh <= args.stable_threshold),
                    relax_status="ok",
                ))
        except Exception as ex:
            meta.update(dict(
                mattersim_total_energy=float(e_total) if math.isfinite(e_total) else None,
                energy_per_atom=(float(e_total) / max(1, len(structs[i]))) if math.isfinite(e_total) else None,
                e_above_hull=None, is_stable=False,
                relax_status=f"hull_err:{type(ex).__name__}",
            ))

    valid_eh = [m["e_above_hull"] for m in rows_meta if m.get("e_above_hull") is not None]
    print(f"[cache-energy] shard {args.shard_idx}: {n_eh}/{n} E_h computed; "
          f"{sum(1 for m in rows_meta if m.get('is_stable'))} stable "
          f"(@<= {args.stable_threshold}); "
          + (f"mean E_h={np.mean(valid_eh):.4f}" if valid_eh else "no E_h"), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.parquet.stem}.shard_{args.shard_idx}_of_{args.num_shards}.parquet"
    _write_table(rows_meta, out_path)
    print(f"[cache-energy] wrote {out_path}", flush=True)


_OUT_FIELDS = [
    ("parquet", pa.string()), ("row_id", pa.string()), ("parent", pa.string()),
    ("source_idx", pa.int64()), ("input_source_idx", pa.int64()),
    ("dedup_key", pa.string()), ("global_index", pa.int64()),
    ("n_atoms", pa.int64()), ("composition", pa.string()),
    ("mattersim_total_energy", pa.float64()), ("energy_per_atom", pa.float64()),
    ("e_above_hull", pa.float64()), ("is_stable", pa.bool_()),
    ("relax_status", pa.string()),
    ("dft_formation_energy", pa.float64()), ("dft_e_above_hull", pa.float64()),
    ("dft_total_energy_per_atom", pa.float64()), ("dft_band_gap", pa.float64()),
    ("dft_density", pa.float64()),
]


def _coerce(v, ptype):
    if v is None:
        return None
    if ptype == pa.int64():
        try:
            return int(v)
        except (ValueError, TypeError):
            return None
    if ptype == pa.float64():
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (ValueError, TypeError):
            return None
    if ptype == pa.bool_():
        return bool(v)
    return None if v is None else str(v)


def _write_table(rows_meta: list[dict], out_path: Path):
    cols = {name: [] for name, _ in _OUT_FIELDS}
    for m in rows_meta:
        for name, ptype in _OUT_FIELDS:
            cols[name].append(_coerce(m.get(name), ptype))
    arrays = [pa.array(cols[name], type=ptype) for name, ptype in _OUT_FIELDS]
    table = pa.table(arrays, names=[name for name, _ in _OUT_FIELDS])
    pq.write_table(table, out_path)


def run_merge(args):
    """Concatenate a parquet's shards into one energy table plus a stable-row-index file."""
    shard_files = sorted(args.out_dir.glob(f"{args.parquet.stem}.shard_*_of_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(
            f"no shard files for {args.parquet.stem} under {args.out_dir}"
        )
    print(f"[merge] {args.parquet.stem}: {len(shard_files)} shards", flush=True)
    tables = [pq.read_table(f) for f in shard_files]
    table = pa.concat_tables(tables)

    df = table.to_pandas()
    if args.dedup_key == "narrative":
        before = len(df)
        df = df.drop_duplicates(subset=["dedup_key"], keep="first").reset_index(drop=True)
        print(f"[merge] narrative dedup {before} → {len(df)} unique structures", flush=True)

    merged_path = args.out_dir / f"{args.parquet.stem}.energies.parquet"
    df.to_parquet(merged_path, index=False)
    print(f"[merge] wrote {merged_path} ({len(df)} rows)", flush=True)

    # Stable-row-index file: row_ids to KEEP for stability-filtered retraining.
    stable = df[df["is_stable"] == True]  # noqa: E712
    keep_ids = stable["row_id"].tolist()
    n_eh = int(df["e_above_hull"].notna().sum())
    idx_path = args.out_dir / f"{args.parquet.stem}.stable_row_ids.json"
    idx_path.write_text(json.dumps({
        "parquet": args.parquet.name,
        "stable_threshold": args.stable_threshold,
        "n_total": int(len(df)),
        "n_eh_computed": n_eh,
        "n_stable": int(len(keep_ids)),
        "stable_frac_of_eh": (len(keep_ids) / n_eh) if n_eh else None,
        "mean_e_above_hull": (float(df["e_above_hull"].mean(skipna=True))
                              if n_eh else None),
        "median_e_above_hull": (float(df["e_above_hull"].median(skipna=True))
                                if n_eh else None),
        "row_ids": keep_ids,
    }, indent=2))
    print(f"[merge] wrote {idx_path}: {len(keep_ids)} stable / {n_eh} E_h-computed / "
          f"{len(df)} total (thr={args.stable_threshold})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/energies")))
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--dedup_key", choices=list(DEDUP_ADAPTERS), default="narrative",
                    help="narrative=dedup on (parent,source_idx); row_id=keep all (editing buckets)")
    ap.add_argument("--stable_threshold", type=float, default=0.1,
                    help="e_above_hull (eV/atom) cutoff for the stable-row-index file")
    ap.add_argument("--hull_dir", type=Path, default=None,
                    help="MP-2020 hull dir; default = MatterGen-bundled via load_hull_reference()")
    ap.add_argument("--mattersim_device", default="cuda")
    ap.add_argument("--merge", action="store_true",
                    help="merge mode: concatenate this parquet's shards + write stable-row index")
    args = ap.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
