"""Unified atomistic-encoder (orb/uma/pet) caching runner writing the .flat.bin + .flat.idx.json layout Stage 2 consumes."""

import argparse
import gc
import json
import os
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ─── Source iterators ─────────────────────────────────────────────────────────

def iter_ase_db(db_path: Path, batch_size: int, id_key: str):
    """Yield (ids, atoms_list) batches from an ASE SQLite DB."""
    from ase.db import connect
    db = connect(str(db_path))
    rows = db.select()
    while True:
        batch = list(islice(rows, batch_size))
        if not batch:
            return
        ids = [r.data[id_key] for r in batch]
        atoms = [r.toatoms() for r in batch]
        yield ids, atoms


def iter_narratives_parquet(parquet_path: Path, batch_size: int):
    """Yield (ids, atoms_list) batches; ids are parquet row index as str (Stage 2 GPTNarrativeDataset key)."""
    import pyarrow.parquet as pq
    from ase import Atoms

    def _to_atoms(a):
        cell = a["lattice_mat"]
        coords = a["coords"]
        symbols = [s.strip() for s in a["elements"]]
        kw = {"symbols": symbols, "cell": cell, "pbc": True}
        kw["positions" if a["cartesian"] else "scaled_positions"] = coords
        return Atoms(**kw)

    pf = pq.ParquetFile(str(parquet_path))
    global_idx = 0
    for rec_batch in pf.iter_batches(batch_size=batch_size, columns=["atoms"]):
        batch_atoms = rec_batch.column("atoms").to_pylist()
        ids = [str(global_idx + i) for i in range(len(batch_atoms))]
        atoms = [_to_atoms(a) for a in batch_atoms]
        global_idx += len(batch_atoms)
        yield ids, atoms


# ─── Encoder adapters ─────────────────────────────────────────────────────────

class OrbV3Adapter:
    """OrbV3, 256-d per-atom node features."""
    feature_dim = 256

    def __init__(self, variant: str, device: str):
        self.variant = variant
        self.device = device
        self._model = None

    def load(self):
        from orb_models.forcefield import pretrained
        loader = getattr(pretrained, self.variant)
        self._model = loader(device=self.device, precision="float32-high")
        self._model.model.eval()

    @torch.no_grad()
    def encode_batch(self, atoms_list):
        """Returns (n_atoms_per_struct: list[int], per_atom_feats: CPU tensor)."""
        from orb_models.forcefield import atomic_system
        from orb_models.forcefield.base import batch_graphs
        graphs = [
            atomic_system.ase_atoms_to_atom_graphs(a, self._model.system_config, device=self.device)
            for a in atoms_list
        ]
        graph = batch_graphs(graphs)
        out = self._model.model(graph)
        feats = out["node_features"].detach().cpu()
        n_atoms = graph.n_node.detach().cpu().tolist()
        return n_atoms, feats


class UMAAdapter:
    """UMA via fairchem.core.pretrained_mlip."""
    feature_dim = 128  # updated from first encode_batch if different

    def __init__(self, variant: str, device: str, cache_dir: str | None = None,
                 task: str = "omat", inference_settings: str | None = None):
        self.variant = variant
        self.device = device
        self.cache_dir = cache_dir
        self.task = task
        self.inference_settings = inference_settings
        self._predictor = None
        self._dim_locked = False

    def load(self):
        from fairchem.core import pretrained_mlip
        kwargs = {"device": self.device}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
            kwargs["local_files_only"] = True
        if self.inference_settings:
            kwargs["inference_settings"] = self.inference_settings
        self._predictor = pretrained_mlip.get_predict_unit(self.variant, **kwargs)

    @torch.no_grad()
    def encode_batch(self, atoms_list):
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
        atomic_data_list = [AtomicData.from_ase(a, task_name=self.task) for a in atoms_list]
        batched = atomicdata_list_to_batch(atomic_data_list)
        out = self._predictor.model.module.backbone(batched)
        # UMA/eSEN node_embedding is [N, L, H]; take L=0 to get [N, H]
        embs = out["node_embedding"].narrow(1, 0, 1).squeeze(1)
        if not self._dim_locked:
            self.feature_dim = int(embs.shape[-1])
            self._dim_locked = True
        feats = embs.detach().cpu()
        n_atoms = [len(a) for a in atoms_list]
        return n_atoms, feats


class PETAdapter:
    """PET-MAD via upet.calculator.UPETCalculator."""
    feature_dim = None  # set on first encode_batch; varies across xs/s/m sizes

    def __init__(self, variant: str, device: str, version: str = "latest",
                 checkpoint_path: str | None = None):
        self.variant = variant
        self.device = device
        self.version = version
        self.checkpoint_path = checkpoint_path
        self._calc = None
        self._supported_types = None
        self._dim_locked = False

    def load(self):
        from upet.calculator import UPETCalculator
        self._calc = UPETCalculator(
            model=None if self.checkpoint_path else self.variant,
            version=self.version,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
        )
        self._supported_types = set(int(z) for z in
                                    self._calc.calculator._model.capabilities().atomic_types)

    @torch.no_grad()
    def encode_batch(self, atoms_list):
        from metatomic.torch import ModelOutput
        from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
        # Drop unsupported-Z structures; n_atoms=0 marks them skipped in idx-json.
        keep_mask = [
            all(int(z) in self._supported_types for z in a.numbers)
            for a in atoms_list
        ]
        filtered_atoms = [a for a, k in zip(atoms_list, keep_mask) if k]
        if not filtered_atoms:
            return [0] * len(atoms_list), torch.empty(0, self.feature_dim or 0)
        requested = {"features": ModelOutput(per_atom=True)}
        try:
            outputs = self._calc.calculator.run_model(filtered_atoms, requested)
        except NeighborOverflowError:
            # Per-structure fallback so one offender doesn't kill the batch.
            per_struct = []
            for a in filtered_atoms:
                try:
                    single = self._calc.calculator.run_model([a], requested)
                    per_struct.append(single["features"].block().values)
                except NeighborOverflowError:
                    per_struct.append(torch.empty(0, self.feature_dim or 0))
            if not any(t.numel() for t in per_struct):
                return [0] * len(atoms_list), torch.empty(0, self.feature_dim or 0)
            kept_feats = torch.cat(per_struct, dim=0)
        else:
            kept_feats = outputs["features"].block().values

        if not self._dim_locked:
            self.feature_dim = int(kept_feats.shape[-1])
            self._dim_locked = True
        kept_feats = kept_feats.detach().cpu()

        n_atoms_full = [
            len(a) if k else 0
            for a, k in zip(atoms_list, keep_mask)
        ]
        return n_atoms_full, kept_feats


ADAPTERS = {"orb": OrbV3Adapter, "uma": UMAAdapter, "pet": PETAdapter}


# ─── .flat.bin writer ─────────────────────────────────────────────────────────

def peek_partial_done_ids(out_bin: Path, out_idx: Path) -> set[str]:
    """Return IDs already encoded in a prior interrupted run's <out_idx>.partial."""
    partial_idx = out_idx.with_suffix(out_idx.suffix + ".partial")
    if not partial_idx.exists():
        return set()
    try:
        with open(partial_idx) as f:
            meta = json.load(f)
        return set(meta.get("_index", {}).keys())
    except Exception as e:
        print(f"[flatbin] WARN: failed to read partial index {partial_idx}: {e} — starting fresh")
        return set()


class FlatBinWriter:
    """Streaming durable writer: appends feats to .partial, checkpoints the index, atomically promotes on finalize; auto-resumes from .partial after SLURM requeues."""

    CKPT_EVERY = 50  # batches between durable index flushes

    def __init__(self, out_bin: Path, out_idx: Path, feature_dim: int):
        self.out_bin = out_bin
        self.out_idx = out_idx
        self.feature_dim = feature_dim
        self._partial_bin = out_bin.with_suffix(out_bin.suffix + ".partial")
        self._partial_idx = out_idx.with_suffix(out_idx.suffix + ".partial")
        self._index: dict[str, list[int]] = {}
        self._offset = 0
        self._bin_fp = None
        self._batches_since_ckpt = 0
        out_bin.parent.mkdir(parents=True, exist_ok=True)
        if self._partial_bin.exists() and self._partial_idx.exists():
            self._resume()
        else:
            # Wipe a half-written .partial (e.g. index without bin) before fresh start.
            for stale in (self._partial_bin, self._partial_idx):
                if stale.exists():
                    print(f"[flatbin] removing stale {stale}")
                    stale.unlink()
            self._bin_fp = open(self._partial_bin, "wb")

    def _resume(self):
        with open(self._partial_idx) as f:
            meta = json.load(f)
        saved_fd = meta.get("_feature_dim")
        if saved_fd != self.feature_dim:
            raise RuntimeError(
                f"[flatbin] resume aborted: saved feature_dim={saved_fd} != current "
                f"{self.feature_dim} for {self._partial_bin}. Delete the .partial pair "
                f"to start fresh."
            )
        self._offset = int(meta["_offset"])
        self._index = dict(meta["_index"])
        # Defensive: a crash can flush bin bytes without persisting the index.
        expected_bytes = self._offset * self.feature_dim * 4  # float32
        on_disk_bytes = self._partial_bin.stat().st_size
        if on_disk_bytes != expected_bytes:
            print(f"[flatbin] resume size sync: bin={on_disk_bytes}b, expected={expected_bytes}b "
                  f"— truncating to {expected_bytes}b")
            with open(self._partial_bin, "r+b") as fp:
                fp.truncate(expected_bytes)
        self._bin_fp = open(self._partial_bin, "ab")
        print(f"[flatbin] resumed from {self._partial_bin}: {len(self._index):,} ids "
              f"already encoded, offset={self._offset:,}")

    def add(self, ids: list[str], n_atoms_per: list[int], feats: torch.Tensor):
        """`feats` concatenates all rows' atoms (sum of n_atoms_per); n_atoms=0 rows get no idx entry."""
        feats_np = np.asarray(feats.numpy(), dtype=np.float32, order="C")
        total = sum(n_atoms_per)
        if feats_np.shape[0] != total:
            raise RuntimeError(
                f"FlatBinWriter.add: feats has {feats_np.shape[0]} atoms but "
                f"n_atoms_per sums to {total}"
            )
        if feats_np.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"FlatBinWriter.add: expected feature_dim={self.feature_dim} "
                f"but got {feats_np.shape[1]}"
            )
        if total > 0:
            self._bin_fp.write(feats_np.tobytes())
        cursor = 0
        for sid, n in zip(ids, n_atoms_per):
            if n == 0:
                continue
            self._index[str(sid)] = [self._offset + cursor, n]
            cursor += n
        self._offset += total
        self._batches_since_ckpt += 1
        if self._batches_since_ckpt >= self.CKPT_EVERY:
            self.checkpoint()

    def checkpoint(self):
        """fsync the .partial bin and atomically rewrite the index sidecar."""
        if self._bin_fp is not None:
            self._bin_fp.flush()
            os.fsync(self._bin_fp.fileno())
        meta = {
            "_offset": self._offset,
            "_feature_dim": self.feature_dim,
            "_index": self._index,
        }
        tmp = self._partial_idx.with_suffix(self._partial_idx.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, self._partial_idx)
        self._batches_since_ckpt = 0

    def finalize(self):
        if self._bin_fp is None or self._offset == 0:
            print(f"[flatbin] no atoms collected — skipping write of {self.out_bin}")
            for stale in (self._partial_bin, self._partial_idx):
                if stale.exists():
                    stale.unlink()
            return
        self._bin_fp.flush()
        os.fsync(self._bin_fp.fileno())
        self._bin_fp.close()
        self._bin_fp = None
        # Final index drops the resume-meta wrapper; consumers expect flat {id: [off, n]}.
        tmp = self._partial_idx.with_suffix(self._partial_idx.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._index, f)
        os.replace(tmp, self._partial_idx)
        os.replace(self._partial_bin, self.out_bin)
        os.replace(self._partial_idx, self.out_idx)
        print(f"[flatbin] wrote {self.out_bin} ({self._offset:,} atoms × {self.feature_dim} dim) "
              f"+ {self.out_idx} ({len(self._index):,} ids)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", required=True, choices=list(ADAPTERS.keys()))
    p.add_argument("--variant", required=True,
                   help="Encoder-specific model name (orb_v3_direct_20_omat, "
                        "uma-omat-v1p1-s, pet-mad-s, etc.). Used both to select "
                        "the model and as the cache-filename tag.")
    p.add_argument("--source", required=True, choices=["ase_db", "narratives_parquet"])
    p.add_argument("--data_path", required=True, help="ASE DB path or parquet path")
    p.add_argument("--out_dir", required=True, help="Output directory for .flat.bin / .flat.idx.json")
    p.add_argument("--split", default=None,
                   help="Split tag for ASE DB sources (train/validation/test). "
                        "Omit for narratives parquet (uses '' to match Stage 2's "
                        "GPTNarrativeDataset filename pattern).")
    p.add_argument("--id_key", default="smiles",
                   help="row.data field used as the per-sample id (ASE DB source only). "
                        "'smiles' for legacy LLM4Mat caches; 'material_id' for MatterChat.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache_dir", default=None,
                   help="UMA / fairchem model cache dir (--encoder uma).")
    p.add_argument("--task", default="omat", help="UMA task name (omat / omol / ...).")
    p.add_argument("--inference_settings", default=None,
                   help="UMA only: pass 'turbo' to enable fairchem's compiled-inference "
                        "fast path (faster forwards, slightly longer load). Default None.")
    p.add_argument("--version", default="latest", help="PET-MAD --version.")
    p.add_argument("--checkpoint_path", default=None,
                   help="PET-MAD checkpoint override (ignores --variant/--version).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing .flat.bin; default is idempotent skip.")
    p.add_argument("--limit_batches", type=int, default=None,
                   help="Smoke-test cap on number of batches to encode.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    # Filename tag must match what Stage 2's datasets expect.
    if args.source == "ase_db":
        if args.split is None:
            raise ValueError("--split required for --source ase_db")
        tag = f"{args.variant}_{args.split}_atom"
    else:
        tag = f"{args.variant}_atom"
    out_bin = out_dir / f"{tag}.flat.bin"
    out_idx = out_dir / f"{tag}.flat.idx.json"

    if out_bin.exists() and out_idx.exists() and not args.force:
        print(f"[cache] {out_bin} already exists; skip (--force to overwrite)")
        return

    # Skip already-encoded ids so a SLURM requeue doesn't redo hours of work.
    done_ids = peek_partial_done_ids(out_bin, out_idx)
    if done_ids:
        print(f"[cache] resuming: {len(done_ids):,} ids already encoded in .partial — will skip")

    print(f"[cache] encoder={args.encoder} variant={args.variant} source={args.source}")
    print(f"[cache]   data: {args.data_path}")
    print(f"[cache]   out:  {out_bin}")

    AdapterCls = ADAPTERS[args.encoder]
    if args.encoder == "orb":
        adapter = AdapterCls(variant=args.variant, device=args.device)
    elif args.encoder == "uma":
        adapter = AdapterCls(variant=args.variant, device=args.device,
                             cache_dir=args.cache_dir, task=args.task,
                             inference_settings=args.inference_settings)
    else:  # pet
        adapter = AdapterCls(variant=args.variant, device=args.device,
                             version=args.version, checkpoint_path=args.checkpoint_path)
    print(f"[cache] loading {args.encoder} model ...")
    adapter.load()
    print(f"[cache] model loaded; declared feature_dim={adapter.feature_dim}")

    if args.source == "ase_db":
        iterator = iter_ase_db(Path(args.data_path), args.batch_size, args.id_key)
    else:
        iterator = iter_narratives_parquet(Path(args.data_path), args.batch_size)

    writer = None  # built after first batch so UMA/PET can lock feature_dim
    n_batches = 0
    total_rows = 0
    skipped_rows = 0
    resumed_rows = 0
    for ids, atoms in tqdm(iterator, desc=f"{args.encoder}/{args.variant}"):
        # Drop rows already in the .partial index, filtering ids+atoms together.
        if done_ids:
            keep = [i for i, sid in enumerate(ids) if str(sid) not in done_ids]
            if len(keep) < len(ids):
                resumed_rows += len(ids) - len(keep)
                ids = [ids[i] for i in keep]
                atoms = [atoms[i] for i in keep]
        if not atoms:
            continue
        try:
            n_atoms_per, feats = adapter.encode_batch(atoms)
        except Exception as e:
            print(f"[cache] batch {n_batches}: encode failed: {e}")
            skipped_rows += len(atoms)
            n_batches += 1
            continue
        if writer is None:
            fd = adapter.feature_dim
            if fd is None:
                raise RuntimeError(f"[cache] adapter {args.encoder} did not lock feature_dim "
                                   f"on first batch — empty input?")
            writer = FlatBinWriter(out_bin, out_idx, feature_dim=fd)
        writer.add(ids, n_atoms_per, feats)
        total_rows += len(ids)
        skipped_rows += sum(1 for n in n_atoms_per if n == 0)
        n_batches += 1
        if n_batches % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        if args.limit_batches is not None and n_batches >= args.limit_batches:
            print(f"[cache] hit --limit_batches={args.limit_batches}, stopping early")
            break

    if writer is None:
        if done_ids:
            # Every remaining row was already in .partial: just finalize it.
            fd = adapter.feature_dim
            if fd is not None:
                writer = FlatBinWriter(out_bin, out_idx, feature_dim=fd)
                writer.finalize()
                print(f"[cache] DONE (resume-only) — {resumed_rows:,} ids carried over from .partial")
                return
        print(f"[cache] no rows produced features; nothing written.")
        return
    writer.finalize()
    print(f"[cache] DONE — encoded {total_rows - skipped_rows}/{total_rows} new rows "
          f"({skipped_rows} skipped, {resumed_rows:,} carried from .partial), "
          f"feature_dim={writer.feature_dim}")


if __name__ == "__main__":
    main()
