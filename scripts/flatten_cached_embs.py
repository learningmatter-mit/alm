"""Re-pack dict-of-numpy .pt caches into mmap-friendly .flat.bin + .flat.idx.json so DDP ranks share one page-cache copy."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from paths import DATA_ROOT


DEFAULT_EMBED_DIM = 256


def flatten_one(src_pt: Path, dst_bin: Path, dst_idx: Path, embed_dim: int = DEFAULT_EMBED_DIM):
    cache = torch.load(src_pt, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise RuntimeError(f"{src_pt}: expected dict, got {type(cache)}")

    # Preallocate the flat file: sum atom counts first.
    total_atoms = 0
    for arr in cache.values():
        if arr.ndim != 2 or arr.shape[1] != embed_dim:
            raise RuntimeError(f"{src_pt}: unexpected shape {arr.shape} (expected dim {embed_dim})")
        total_atoms += arr.shape[0]

    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    flat = np.memmap(dst_bin, dtype=np.float32, mode="w+", shape=(total_atoms, embed_dim))

    index: dict[str, list[int]] = {}
    offset = 0
    for sample_id, arr in tqdm(cache.items(), desc=src_pt.name, leave=False):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        n = arr.shape[0]
        flat[offset : offset + n] = arr
        index[str(sample_id)] = [offset, n]
        offset += n
    assert offset == total_atoms

    flat.flush()
    del flat
    with open(dst_idx, "w") as f:
        json.dump(index, f)
    print(f"wrote {dst_bin} ({total_atoms:,} atoms) and {dst_idx} ({len(index):,} ids)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=str, default=os.path.join(DATA_ROOT, "cached_embs"),
                        help="parent dir containing {dataset}/embeddings/*.pt")
    parser.add_argument("--model_name", type=str, default="orb_v3_direct_20_omat")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=DEFAULT_EMBED_DIM,
                        help="Per-atom feature dim. OrbV3=256, UMA=128, PET-MAD variable.")
    args = parser.parse_args()

    parent = Path(args.parent)
    for ds_dir in sorted(parent.iterdir()):
        if not ds_dir.is_dir():
            continue
        for split in args.splits:
            src = ds_dir / "embeddings" / f"{args.model_name}_{split}_atom.pt"
            if not src.exists():
                print(f"skip (no cache): {src}")
                continue
            dst_bin = src.with_suffix(".flat.bin")
            dst_idx = src.with_suffix(".flat.idx.json")
            if dst_bin.exists() and dst_idx.exists() and not args.overwrite:
                print(f"skip (exists): {dst_bin}")
                continue
            flatten_one(src, dst_bin, dst_idx, embed_dim=args.embed_dim)


if __name__ == "__main__":
    main()
