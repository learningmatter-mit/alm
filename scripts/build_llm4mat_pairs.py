"""Build pairs.parquet-style datasets from LLM4Mat-Bench subsets."""

import argparse
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from paths import DATA_ROOT

warnings.filterwarnings("ignore")

SYSTEM_PROMPT = "You are an expert materials scientist."
ASSISTANT_ANCHOR = "Structure: "


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path,
                   default=Path(os.path.join(DATA_ROOT, "LLM4Mat-Bench")))
    p.add_argument("--out_dir", type=Path,
                   default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a")))
    p.add_argument("--max_atoms", type=int, default=20,
                   help="Filter to structures with ≤max_atoms (matches Alex-MP-20 / MatterGen scale)")
    p.add_argument("--splits", type=str, default="train,validation",
                   help="Comma-separated list of splits to include in the parquet")
    p.add_argument("--subsets", type=str, default=None,
                   help="Comma-separated subset names (default: all)")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def parse_cif_to_atoms_struct(cif_str: str) -> dict | None:
    """Parse a CIF string into our atoms_struct dict (cif is safer than the mp subset's repr-dict `structure`)."""
    if not isinstance(cif_str, str) or not cif_str.strip():
        return None
    try:
        from pymatgen.core import Structure
        s = Structure.from_str(cif_str, fmt="cif")
    except Exception:
        return None
    return {
        "elements": [str(site.specie) for site in s],
        "coords": [[float(c) for c in site.coords] for site in s],
        "lattice_mat": s.lattice.matrix.tolist(),
        "cartesian": True,
    }


def _process_one(args_tuple):
    """Worker: returns the pair dict, or None on failure."""
    parent, source_idx, mat_id, description, cif_str, max_atoms = args_tuple
    atoms = parse_cif_to_atoms_struct(cif_str)
    if atoms is None:
        return None
    n_atoms = len(atoms["elements"])
    if n_atoms > max_atoms or n_atoms == 0:
        return None
    if not isinstance(description, str) or len(description) < 50:
        return None
    user_prompt = f"Generate a crystal structure described as: {description}"
    return {
        "row_id": f"{parent}-{mat_id}",
        "parent": parent,
        "source_idx": int(source_idx),
        "n_atoms": int(n_atoms),
        "narrative": description,
        "user_prompt": user_prompt,
        "assistant_anchor": ASSISTANT_ANCHOR,
        "atoms_struct": atoms,
    }


def build_subset_parquet(subset: str, data_root: Path, out_dir: Path,
                         splits: list[str], max_atoms: int, workers: int) -> Path | None:
    sub_dir = data_root / subset
    if not sub_dir.is_dir():
        print(f"[{subset}] missing dir {sub_dir}, skipping")
        return None

    rows = []
    for split in splits:
        csv_path = sub_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[{subset}] missing {csv_path}, skipping split")
            continue
        df = pl.read_csv(csv_path, infer_schema_length=10000)
        if "description" not in df.columns or "cif_structure" not in df.columns:
            print(f"[{subset}/{split}] missing description or cif_structure column, skipping")
            continue
        # First column is the id (material_id / jarvis_id / hmof_id / ...).
        id_col = df.columns[0]
        descs = df["description"].to_list()
        structs = df["cif_structure"].to_list()
        ids = df[id_col].to_list()
        n_rows = len(df)
        print(f"[{subset}/{split}] {n_rows} rows")
        for i, (mat_id, desc, struct_str) in enumerate(zip(ids, descs, structs)):
            rows.append((subset, i, mat_id, desc, struct_str, max_atoms))

    if not rows:
        return None
    print(f"[{subset}] parsing {len(rows)} candidate rows with {workers} workers ...")
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(_process_one, rows, chunksize=64)
    kept = [r for r in results if r is not None]
    print(f"[{subset}] kept {len(kept)}/{len(rows)} after filters (max_atoms={max_atoms}) "
          f"in {time.time() - t0:.1f}s")

    if not kept:
        return None
    out_path = out_dir / f"pairs_robocrys_{subset}.parquet"
    table = pa.Table.from_pylist(kept)
    pq.write_table(table, out_path, compression="snappy")
    print(f"[{subset}] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(kept)} rows)")
    return out_path


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_subsets = [d.name for d in args.data_root.iterdir() if d.is_dir()]
    if args.subsets:
        wanted = [s.strip() for s in args.subsets.split(",")]
        all_subsets = [s for s in all_subsets if s in wanted]
    splits = [s.strip() for s in args.splits.split(",")]
    print(f"[main] subsets={all_subsets}, splits={splits}, max_atoms={args.max_atoms}")

    out_files = []
    for subset in sorted(all_subsets):
        f = build_subset_parquet(subset, args.data_root, args.out_dir,
                                 splits, args.max_atoms, args.workers)
        if f:
            out_files.append(f)

    if out_files:
        tables = [pq.read_table(f) for f in out_files]
        combined = pa.concat_tables(tables)
        combined_path = args.out_dir / "pairs_robocrys.parquet"
        pq.write_table(combined, combined_path, compression="snappy")
        print(f"\n[main] combined: {combined_path} "
              f"({combined.num_rows:,} rows, {combined_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print("[main] no subsets produced output")


if __name__ == "__main__":
    main()
