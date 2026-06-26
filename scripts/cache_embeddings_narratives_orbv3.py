"""Pre-compute OrbV3 atom-wise embeddings for a GPT-Narratives parquet, keyed by row index."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from ase import Atoms
from ase.db import connect
from tqdm import tqdm

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs


def _atoms_struct_to_ase(a):
    cell = a["lattice_mat"]
    coords = a["coords"]
    symbols = [s.strip() for s in a["elements"]]
    kw = {"symbols": symbols, "cell": cell, "pbc": True}
    kw["positions" if a["cartesian"] else "scaled_positions"] = coords
    return Atoms(**kw)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = f"{args.model_name}_atom"
    bin_path = out_dir / f"{basename}.flat.bin"
    idx_path = out_dir / f"{basename}.flat.idx.json"

    db_path = Path(args.db_path) if args.db_path else out_dir.parent / "atoms.db"
    db_idx_path = db_path.with_name(db_path.stem + ".id_index.json")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()   # ASE db appends otherwise

    pf = pq.ParquetFile(args.parquet_path)
    total = pf.metadata.num_rows
    print(f"Streaming {total:,} rows from {args.parquet_path}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orbff = getattr(pretrained, args.model_name)(device=device, precision="float32-high")

    db = connect(db_path)
    id_index = {}
    index = {}
    byte_offset = 0
    row_cursor = 0
    with open(bin_path, "wb") as fout, tqdm(total=total, desc="cache") as pbar:
        for record_batch in pf.iter_batches(batch_size=args.batch_size, columns=["atoms"]):
            atoms_structs = record_batch.column(0).to_pylist()
            ase_batch = [_atoms_struct_to_ase(a) for a in atoms_structs]
            graphs = [
                atomic_system.ase_atoms_to_atom_graphs(a, orbff.system_config, device=device)
                for a in ase_batch
            ]
            graph = batch_graphs(graphs)
            node_feats = orbff.model(graph)["node_features"].detach().cpu().numpy().astype(np.float32)
            n_per = graph.n_node.cpu().tolist()
            ptr = 0
            for i, (atoms, n_atoms) in enumerate(zip(ase_batch, n_per)):
                row_id = str(row_cursor + i)
                node_feats[ptr:ptr + n_atoms].tofile(fout)
                index[row_id] = [byte_offset, n_atoms]
                # data['smiles']=row_id matches AtomisticLanguageDataset's lookup convention
                id_index[row_id] = db.write(atoms, data={"smiles": row_id})
                byte_offset += n_atoms
                ptr += n_atoms
            row_cursor += len(atoms_structs)
            pbar.update(len(atoms_structs))
            
            if row_cursor % 10000 == 0:
                with open(idx_path, "w") as f:
                    json.dump(index, f)
                with open(db_idx_path, "w") as f:
                    json.dump(id_index, f)
                size_gb = bin_path.stat().st_size / 1e9
                print(f"Wrote {byte_offset:,} atoms → {bin_path} ({size_gb:.2f} GB)")
                print(f"Wrote cache index for {len(index):,} rows → {idx_path}")
                print(f"Wrote ASE db ({db_path.stat().st_size / 1e9:.2f} GB) → {db_path}")
                print(f"Wrote db id_index → {db_idx_path}")

    with open(idx_path, "w") as f:
        json.dump(index, f)
    with open(db_idx_path, "w") as f:
        json.dump(id_index, f)
    size_gb = bin_path.stat().st_size / 1e9
    print(f"Wrote {byte_offset:,} atoms → {bin_path} ({size_gb:.2f} GB)")
    print(f"Wrote cache index for {len(index):,} rows → {idx_path}")
    print(f"Wrote ASE db ({db_path.stat().st_size / 1e9:.2f} GB) → {db_path}")
    print(f"Wrote db id_index → {db_idx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory for {model}_atom.flat.bin + .flat.idx.json")
    parser.add_argument("--db_path", type=str, default=None,
                        help="ASE db output path (default: {out_dir.parent}/atoms.db). "
                             "Companion id_index.json written alongside.")
    parser.add_argument("--model_name", type=str, default="orb_v3_direct_20_omat")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
