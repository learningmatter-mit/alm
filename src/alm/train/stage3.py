"""Stage 3a/3b joint bridge training (GILL/DreamLLM pattern); see argparse --help for usage."""

# Map legacy checkpoint bridge_kind values to current names.
_LEGACY_BRIDGE = {"qformer": "producer-consumer", "qformer_pool": "producer-consumer-pool", "ipadapter": "consumer-only"}
def _norm_bridge(bk):
    return _LEGACY_BRIDGE.get(bk, bk)


import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from omegaconf import OmegaConf

from mattergen.scripts.finetune import init_adapter_lightningmodule_from_pretrained
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate as mg_collate
from mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string

# PYTHONPATH must include the repo's alm/ directory.
from aux_heads import build_aux_head
from loader import load_alm
from paths import DATA_ROOT

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert materials scientist. When asked to generate a crystal "
    "structure, provide a detailed description and conclude with the structure "
    "generation tokens."
)

# user_prompt and assistant_anchor are pre-baked into pairs.parquet at build time.


def _atoms_struct_to_tensors(struct: dict):
    """atoms_struct dict -> (frac_pos, cell, atomic_numbers)."""
    from ase import Atoms
    from ase.data import atomic_numbers as ase_Z

    cell = np.asarray(struct["lattice_mat"], dtype=np.float32)
    coords = np.asarray(struct["coords"], dtype=np.float64)
    symbols = [s.strip() for s in struct["elements"]]

    if struct.get("cartesian", False):
        atoms = Atoms(symbols=symbols, positions=coords, cell=cell, pbc=True)
        frac = atoms.get_scaled_positions(wrap=True).astype(np.float32)
    else:
        frac = (np.mod(coords, 1.0)).astype(np.float32)

    Z = np.array([ase_Z[s] for s in symbols], dtype=np.int64)
    return frac, cell, Z


def _direction_tag(row_id) -> str:
    """row_id '...-{property}-{higher|lower}' -> ' Target: lower formation energy.' ('' if non-directional)."""
    s = str(row_id)
    if s.endswith("higher"):
        d = "higher"
    elif s.endswith("lower"):
        d = "lower"
    else:
        return ""
    parts = s.rsplit("-", 2)
    prop = parts[1].replace("_", " ") if len(parts) == 3 else "target property"
    return f" Target: {d} {prop}."


class Stage3aDataset(Dataset):
    """Reads pairs.parquet; emits tokenized prompt + structure + optional aux_target."""

    def __init__(self, parquet_path: str, tokenizer, max_num_tokens: int = 2048,
                 aux_target_kind: str | None = None,
                 num_output_atom_tokens: int = 8,
                 cached_embs_root: str | None = None,
                 description_in_assistant_turn: bool = False,
                 direction_in_assistant_turn: bool = False,
                 atomistic_feature_dim: int = 256,
                 atomistic_model_name: str = "orb_v3_direct_20_omat",
                 lm_loss_json: bool = False,
                 atoms_before_json: bool = False,
                 use_task_direction: bool = False):
        # Per-encoder dim (orb 256, uma 128, pet-mad-xs 640, pet-mad-s 1280); sizes the text-only sentinel + cache reshape.
        self.atomistic_feature_dim = atomistic_feature_dim
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        self.rows = table.to_pydict()
        self.n = len(self.rows["row_id"])
        # Per-row +-1 direction from row_id suffix (-higher=+1, -lower=-1, else NaN=unconditional).
        self.use_task_direction = use_task_direction
        self.task_direction_labels = None
        if use_task_direction:
            rids = self.rows["row_id"]
            labels = np.full(self.n, float("nan"), dtype=np.float32)
            for i, rid in enumerate(rids):
                r = str(rid)
                if r.endswith("higher"):
                    labels[i] = 1.0
                elif r.endswith("lower"):
                    labels[i] = -1.0
            self.task_direction_labels = labels
            if is_main_process():
                n_pos = int((labels == 1.0).sum()); n_neg = int((labels == -1.0).sum())
                print(f"[stage3a] task_direction: +1={n_pos} -1={n_neg} "
                      f"nan={self.n - n_pos - n_neg} of {self.n} rows ({parquet_path})")
        # Same-input opposite-direction partner map for the directional contrastive loss.
        self.dir_partner_idx = None
        if use_task_direction and "input_source_idx" in self.rows:
            by_src: dict = {}
            isrc = self.rows["input_source_idx"]
            lab = self.task_direction_labels
            for i in range(self.n):
                d = float(lab[i])
                if d == 1.0 or d == -1.0:
                    key = str(isrc[i])
                    by_src.setdefault(key, {})[int(d)] = i
            # Keep only sources with both directions; partner[i] = opposite-direction row index.
            partner = np.full(self.n, -1, dtype=np.int64)
            n_paired_src = 0
            for key, dd in by_src.items():
                if 1 in dd and -1 in dd:
                    n_paired_src += 1
                    partner[dd[1]] = dd[-1]
                    partner[dd[-1]] = dd[1]
            self.dir_partner_idx = partner
            if is_main_process():
                n_with_partner = int((partner >= 0).sum())
                print(f"[stage3a] dir-contrastive partners: {n_paired_src} input_source_idx "
                      f"with both directions → {n_with_partner} rows have a partner "
                      f"({parquet_path})")
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        self.num_output_atom_tokens = num_output_atom_tokens
        self.output_atom_token_ids = tokenizer.convert_tokens_to_ids(
            [f"[atoms_{i}]" for i in range(num_output_atom_tokens)]
        )
        self.output_atom_str = "".join(f"[atoms_{i}]" for i in range(num_output_atom_tokens))
        # Prepend a JSON composition to the assistant turn and supervise only those tokens with LM-CE.
        self.lm_loss_json = bool(lm_loss_json)
        # Emit [atoms_i] before the JSON so they're read pre-composition-commit (causal).
        self.atoms_before_json = bool(atoms_before_json)
        self._output_atom_id_set = set(self.output_atom_token_ids)
        self.description_in_assistant_turn = bool(description_in_assistant_turn)
        if self.description_in_assistant_turn and "narrative" not in self.rows:
            raise ValueError(
                f"--description_in_assistant_turn requires a `narrative` column in "
                f"{parquet_path}; columns are {list(self.rows.keys())}"
            )
        # Echo the directional instruction onto the assistant turn just before [atoms_0] (pure text, not a cond_field).
        self.direction_in_assistant_turn = bool(direction_in_assistant_turn)

        # Input-side <atoms> splice; only fires for atomtxt buckets with a cache root.
        self.has_input_atoms = "input_source_idx" in self.rows
        self.cached_embs = None
        self.cached_embs_idx = None
        if self.has_input_atoms and cached_embs_root is not None:
            cer = Path(cached_embs_root)
            self.cached_embs = {}      # parent -> np.memmap
            self.cached_embs_idx = {}  # parent -> {row_idx_str: (offset, length)}
            seen_parents = sorted(set(self.rows["parent"]))
            for parent in seen_parents:
                bin_p = cer / parent / "embeddings" / f"{atomistic_model_name}_atom.flat.bin"
                idx_p = cer / parent / "embeddings" / f"{atomistic_model_name}_atom.flat.idx.json"
                if not bin_p.exists() or not idx_p.exists():
                    if is_main_process():
                        print(f"[stage3a] WARN: no cached {atomistic_model_name} features for {parent} "
                              f"at {bin_p} — atomtxt rows from this parent will be dropped.")
                    continue
                with open(idx_p) as f:
                    self.cached_embs_idx[parent] = json.load(f)
                self.cached_embs[parent] = np.memmap(
                    bin_p, dtype=np.float32, mode="r"
                ).reshape(-1, atomistic_feature_dim)
            if is_main_process():
                print(f"[stage3a] cached {atomistic_model_name} input features loaded for parents: "
                      f"{list(self.cached_embs.keys())}")

        self.aux_target_kind = aux_target_kind
        self.aux_targets = None        # (n, target_dim) numpy array

        if aux_target_kind == "composition":
            from composition_utils import composition_multihot, symbol_to_z
            sym2z = symbol_to_z()
            self.aux_targets = np.zeros((self.n, 100), dtype=np.float32)
            atoms_structs = self.rows["atoms_struct"]
            for i in range(self.n):
                struct = atoms_structs[i]
                if hasattr(struct, "as_py"):
                    struct = struct.as_py()
                self.aux_targets[i] = composition_multihot(struct["elements"], sym2z)
            if is_main_process():
                pos_per_row = self.aux_targets.sum(axis=1).mean()
                print(f"[stage3a] aux_target=composition: precomputed (n={self.n}, dim=100), "
                      f"avg ~{pos_per_row:.2f} elements/structure")

        elif aux_target_kind == "composition_count":
            from composition_utils import (
                composition_count_vec, MAX_COUNT, N_ELEMENTS, symbol_to_z,
            )
            sym2z = symbol_to_z()
            # (n, N_ELEMENTS) integer counts clamped to MAX_COUNT, float32 for torch.stack.
            self.aux_targets = np.zeros((self.n, N_ELEMENTS), dtype=np.float32)
            n_clamped = 0
            atoms_structs = self.rows["atoms_struct"]
            for i in range(self.n):
                struct = atoms_structs[i]
                if hasattr(struct, "as_py"):
                    struct = struct.as_py()
                v = composition_count_vec(struct["elements"], sym2z, dtype=np.int64)
                if v.max() == MAX_COUNT:
                    # Re-count without clamping to detect actual saturation.
                    raw_max = max(
                        (struct["elements"].count(sym) for sym in set(struct["elements"])),
                        default=0,
                    )
                    if raw_max > MAX_COUNT:
                        n_clamped += 1
                self.aux_targets[i] = v.astype(np.float32, copy=False)
            if is_main_process():
                pos_per_row = (self.aux_targets > 0).sum(axis=1).mean()
                avg_count_when_pos = (
                    self.aux_targets[self.aux_targets > 0].mean()
                    if (self.aux_targets > 0).any() else 0.0
                )
                print(f"[stage3a] aux_target=composition_count: precomputed "
                      f"(n={self.n}, dim={N_ELEMENTS}, MAX_COUNT={MAX_COUNT}), "
                      f"avg ~{pos_per_row:.2f} elements/structure, "
                      f"avg count-when-present {avg_count_when_pos:.2f}, "
                      f"{n_clamped} rows had ≥1 element clamped to MAX_COUNT")

        elif aux_target_kind not in (None, "none", ""):
            raise ValueError(f"unknown aux_target_kind: {aux_target_kind!r}; "
                             f"expected composition, composition_count, or none")

    def __len__(self):
        return self.n

    def _tokenize_prompt(self, user_prompt, assistant_anchor, atoms_struct):
        """-> (input_ids, attention_mask, labels), or None if truncation ate any K trailing [atoms_i]."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
            truncation=True, max_length=self.max_num_tokens - 16,
        )
        # Commit composition as a JSON prefix in the assistant turn; [atoms_i] stay at the end.
        _json_prefix = None
        if self.lm_loss_json:
            import json as _json
            from collections import Counter as _Counter
            _elems = atoms_struct.get("elements") or []
            _cnt = {str(k): int(v) for k, v in _Counter(_elems).items()}
            _json_prefix = _json.dumps({"counts": _cnt}, separators=(",", ":"))
            if not self.atoms_before_json:
                assistant_anchor = _json_prefix + "\n" + assistant_anchor
        if self.lm_loss_json and self.atoms_before_json:
            # [atoms_i] precede the JSON, so their hidden states are pre-commitment (causal).
            assistant_content = assistant_anchor + self.output_atom_str + "\n" + _json_prefix
        else:
            assistant_content = assistant_anchor + self.output_atom_str
        full_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": assistant_content}],
            tokenize=True, add_generation_prompt=False,
            enable_thinking=False, truncation=True, max_length=self.max_num_tokens,
        )
        # Verify all K [atoms_i] IDs survived truncation (mid-sequence under atoms_before_json, else tail).
        if self.atoms_before_json:
            present = set(full_ids) & self._output_atom_id_set
        else:
            trailing = full_ids[-(len(self.output_atom_token_ids) + 2):]  # +2 for EOS/IM_END
            present = set(trailing) & set(self.output_atom_token_ids)
        if len(present) < len(self.output_atom_token_ids):
            return None

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones(len(full_ids), dtype=torch.long)
        labels = torch.tensor(
            [-100] * len(prompt_ids) + full_ids[len(prompt_ids):], dtype=torch.long
        )
        # Mask [atoms_i] positions out of the LM-CE labels (they are the bridge's, not a text target).
        if self.lm_loss_json:
            for _j, _tid in enumerate(full_ids):
                if _tid in self._output_atom_id_set:
                    labels[_j] = -100
        return input_ids, attention_mask, labels

    def __getitem__(self, idx):
        user_prompt = self.rows["user_prompt"][idx]
        assistant_anchor = self.rows["assistant_anchor"][idx]
        atoms_struct = self.rows["atoms_struct"][idx]

        # Move the narrative into the assistant turn (before the anchor) so the bridge reads it.
        if self.description_in_assistant_turn:
            narrative = self.rows["narrative"][idx]
            user_prompt = "Generate a crystal structure."
            assistant_anchor = (narrative or "").strip() + "\n" + assistant_anchor
        if self.direction_in_assistant_turn:
            assistant_anchor = assistant_anchor + _direction_tag(self.rows["row_id"][idx])
        if hasattr(atoms_struct, "as_py"):
            atoms_struct = atoms_struct.as_py()

        # Reject geometries that NaN the diffusion score: n_atoms<=1, VPA>100 A^3/atom
        # (outside MatterGen's 15-50 schedule), non-finite/wrong-shape lattice, det<=0.
        try:
            import numpy as _np
            _elems = atoms_struct.get("elements") or []
            _n = len(_elems)
            if _n <= 1:
                return None
            _lat = _np.asarray(atoms_struct.get("lattice_mat") or [], dtype=_np.float64)
            if _lat.shape != (3, 3) or not _np.isfinite(_lat).all():
                return None
            _det = float(_np.linalg.det(_lat))
            if abs(_det) < 1e-3 or _det <= 0:
                return None
            _vol = abs(_det)
            if _vol / _n > 100.0:
                return None
            _coords = _np.asarray(atoms_struct.get("coords") or [], dtype=_np.float64)
            if _coords.size and not _np.isfinite(_coords).all():
                return None
        except Exception:
            return None

        _tok = self._tokenize_prompt(user_prompt, assistant_anchor, atoms_struct)
        if _tok is None:
            return None
        input_ids, attention_mask, labels = _tok

        frac, cell, Z = _atoms_struct_to_tensors(atoms_struct)

        # Input-side <atoms> splice: zero-length no-op for non-atomtxt rows; cached features for atomtxt.
        atom_embed = torch.zeros(0, self.atomistic_feature_dim, dtype=torch.float32)
        if self.has_input_atoms and self.cached_embs is not None:
            parent = self.rows["parent"][idx]
            input_source_idx = self.rows["input_source_idx"][idx]
            cache = self.cached_embs.get(parent)
            cache_idx = self.cached_embs_idx.get(parent) if cache is not None else None
            if cache is not None and cache_idx is not None:
                key = str(int(input_source_idx))
                ent = cache_idx.get(key)
                if ent is not None:
                    off, length = int(ent[0]), int(ent[1])
                    atom_embed = torch.from_numpy(
                        np.asarray(cache[off:off + length], dtype=np.float32).copy()
                    )
                else:
                    return None  # no cached features for this row; skip to avoid splice mismatch

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "frac": torch.from_numpy(frac),      # (N_atoms, 3)
            "cell": torch.from_numpy(cell),       # (3, 3)
            "Z": torch.from_numpy(Z),             # (N_atoms,)
            "n_atoms": torch.tensor(len(Z), dtype=torch.long),
            "atom_embed": atom_embed,             # (n_input_atoms, 256) or (0, 256)
        }
        if self.task_direction_labels is not None:
            sample["task_direction"] = float(self.task_direction_labels[idx])
        # Attach the opposite-direction sibling's tokenized prompt (reuses this row's atom_embed).
        if self.dir_partner_idx is not None:
            pj = int(self.dir_partner_idx[idx])
            if pj >= 0:
                p_user = self.rows["user_prompt"][pj]
                p_anchor = self.rows["assistant_anchor"][pj]
                if self.description_in_assistant_turn:
                    p_narr = self.rows["narrative"][pj]
                    p_user = "Generate a crystal structure."
                    p_anchor = (p_narr or "").strip() + "\n" + p_anchor
                if self.direction_in_assistant_turn:
                    p_anchor = p_anchor + _direction_tag(self.rows["row_id"][pj])
                p_struct = self.rows["atoms_struct"][pj]
                if hasattr(p_struct, "as_py"):
                    p_struct = p_struct.as_py()
                p_tok = self._tokenize_prompt(p_user, p_anchor, p_struct)
                if p_tok is not None:
                    p_ids, p_attn, _ = p_tok
                    sample["partner_input_ids"] = p_ids
                    sample["partner_attention_mask"] = p_attn
        if self.aux_target_kind in ("composition", "composition_count"):
            sample["aux_target"] = torch.from_numpy(self.aux_targets[idx])
        return sample


def stage3a_collate(samples):
    """Collate Stage3aDataset samples, filtering None entries."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None

    input_ids = [s["input_ids"] for s in samples]
    attention_mask = [s["attention_mask"] for s in samples]
    labels = [s["labels"] for s in samples]

    max_len = max(x.shape[0] for x in input_ids)
    input_ids_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, max_len - x.shape[0]), value=0) for x in input_ids]
    )
    attn_mask_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, max_len - x.shape[0]), value=0) for x in attention_mask]
    )
    labels_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, max_len - x.shape[0]), value=-100) for x in labels]
    )

    # Keep structures as a list; mg_collate batches them via pyg.
    fracs = [s["frac"] for s in samples]
    cells = [s["cell"] for s in samples]
    Zs = [s["Z"] for s in samples]
    n_atoms = [s["n_atoms"] for s in samples]

    out = {
        "input_ids": input_ids_padded,         # (B, max_len)
        "attention_mask": attn_mask_padded,    # (B, max_len)
        "labels": labels_padded,               # (B, max_len)
        "fracs": fracs,
        "cells": cells,
        "Zs": Zs,
        "n_atoms": n_atoms,
        "atom_embeds": [s["atom_embed"] for s in samples],  # list of (N_in, 256)
    }
    if "task_direction" in samples[0]:
        out["task_direction"] = torch.tensor(
            [float(s["task_direction"]) for s in samples], dtype=torch.float32
        )
    # Partners exist on only some rows; store as per-sample lists (not a padded stack,
    # which would create [atoms_i]-free placeholder rows and trip the "exactly K" assert).
    if any("partner_input_ids" in s for s in samples):
        partner_ids_list = []
        partner_attn_list = []
        has_partner = []
        for s in samples:
            if "partner_input_ids" in s:
                partner_ids_list.append(s["partner_input_ids"])
                partner_attn_list.append(s["partner_attention_mask"])
                has_partner.append(True)
            else:
                partner_ids_list.append(None)
                partner_attn_list.append(None)
                has_partner.append(False)
        out["partner_input_ids"] = partner_ids_list
        out["partner_attention_mask"] = partner_attn_list
        out["has_partner"] = torch.tensor(has_partner, dtype=torch.bool)
    if "aux_target" in samples[0]:
        t0 = samples[0]["aux_target"]
        if isinstance(t0, dict):
            out["aux_target"] = {
                k: torch.stack([s["aux_target"][k] for s in samples]) for k in t0
            }
        else:
            out["aux_target"] = torch.stack([s["aux_target"] for s in samples])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MatterGen loading
# ─────────────────────────────────────────────────────────────────────────────

def _make_adapter_cfg(pretrained_name: str, full_finetuning: bool,
                      hidden_dim: int = 4096, K: int = 8, mid_dim: int = 2048,
                      bridge_kind: str = "pool", cond_adapt_n_heads: int = 4,
                      use_task_direction_cond: bool = False,
                      use_alm_embedding_cond: bool = True,
                      bridge_gate_init: float = 1.0,
                      model_path: str | None = None,
                      qformer_num_queries: int = 16,
                      qformer_depth: int = 2,
                      qformer_heads: int = 8,
                      qformer_context_tokens: int = 128,
                      qformer_input_atoms: int = 0,
                      bridge_out_norm: bool = False,
                      bridge_learnable_null: bool = False,
                      bridge_noise_gate: bool = False,
                      bridge_tenc_fuse: bool = False):
    """Build the OmegaConf DictConfig for init_adapter_lightningmodule_from_pretrained."""
    if bridge_kind not in ("pool", "producer-consumer", "producer-consumer-pool", "consumer-only"):
        raise ValueError(f"Unknown bridge_kind={bridge_kind!r}; expected pool|qformer|qformer_pool|ipadapter")

    # Skip alm_embedding for the discrete-cond-only / no-LLM-bridge control.
    property_embedding_cfg = {}
    if use_alm_embedding_cond:
        if bridge_kind == "producer-consumer":
            # Learnable null opts out of the zeros-rewrite so CFG steers (cond - learned_null).
            _qf_uncond_target = (
                "mattergen.property_embeddings.LearnedEmbeddingSequence"
                if bridge_learnable_null
                else "mattergen.property_embeddings.EmbeddingSequence"
            )
            property_embedding_cfg["alm_embedding"] = {
                "_target_": "mattergen.property_embeddings.PropertyEmbedding",
                "name": "alm_embedding",
                "unconditional_embedding_module": {
                    "_target_": _qf_uncond_target,
                    "hidden_dim": 512,
                    "K": qformer_num_queries,
                },
                "conditional_embedding_module": {
                    "_target_": "bridge.AtomsMapperProducerConsumer",
                    "hidden_dim": hidden_dim,
                    "out_dim": 512,
                    "num_queries": qformer_num_queries,
                    "depth": qformer_depth,
                    "n_heads": qformer_heads,
                    "source_len": qformer_context_tokens + qformer_input_atoms + K,
                    "n_context": qformer_context_tokens + qformer_input_atoms,
                    "out_norm": bridge_out_norm,   # bound cond-vector magnitude
                },
                "scaler": {"_target_": "torch.nn.Identity"},
            }
        elif bridge_kind == "producer-consumer-pool":
            # Same Q-Former producer, mean-pooled to (B, 512) for the additive concat-MLP path.
            property_embedding_cfg["alm_embedding"] = {
                "_target_": "mattergen.property_embeddings.PropertyEmbedding",
                "name": "alm_embedding",
                "unconditional_embedding_module": {
                    "_target_": "mattergen.property_embeddings.EmbeddingVector",
                    "hidden_dim": 512,
                },
                "conditional_embedding_module": {
                    "_target_": "bridge.AtomsMapperProducerConsumer",
                    "hidden_dim": hidden_dim,
                    "out_dim": 512,
                    "num_queries": qformer_num_queries,
                    "depth": qformer_depth,
                    "n_heads": qformer_heads,
                    "source_len": qformer_context_tokens + qformer_input_atoms + K,
                    "n_context": qformer_context_tokens + qformer_input_atoms,
                    "pool": "mean",
                },
                "scaler": {"_target_": "torch.nn.Identity"},
            }
        elif bridge_kind == "consumer-only":
            # Eval-only: (B, K, 512) per-position MLP into the same IP-Adapter cross-attention.
            property_embedding_cfg["alm_embedding"] = {
                "_target_": "mattergen.property_embeddings.PropertyEmbedding",
                "name": "alm_embedding",
                "unconditional_embedding_module": {
                    "_target_": "mattergen.property_embeddings.EmbeddingSequence",
                    "hidden_dim": 512,
                    "K": K,
                },
                "conditional_embedding_module": {
                    "_target_": "bridge.AtomsMapperConsumerOnly",
                    "hidden_dim": hidden_dim,
                    "mid_dim": mid_dim,
                    "out_dim": 512,
                    "K": K,
                },
                "scaler": {"_target_": "torch.nn.Identity"},
            }
        else:
            property_embedding_cfg["alm_embedding"] = {
                "_target_": "mattergen.property_embeddings.PropertyEmbedding",
                "name": "alm_embedding",
                "unconditional_embedding_module": {
                    "_target_": "mattergen.property_embeddings.EmbeddingVector",
                    "hidden_dim": 512,
                },
                "conditional_embedding_module": {
                    "_target_": "bridge.AtomsMapper",
                    "hidden_dim": hidden_dim,
                    "mid_dim": mid_dim,
                    "out_dim": 512,
                    "K": K,
                },
                "scaler": {"_target_": "torch.nn.Identity"},
            }

    # Optional scalar +-1 task_direction cond_field (off by default; sinusoidal NoiseLevelEncoding, NaN=unconditional).
    if use_task_direction_cond:
        property_embedding_cfg["task_direction"] = {
            "_target_": "mattergen.property_embeddings.PropertyEmbedding",
            "name": "task_direction",
            "unconditional_embedding_module": {
                "_target_": "mattergen.property_embeddings.EmbeddingVector",
                "hidden_dim": 512,
            },
            "conditional_embedding_module": {
                "_target_": "mattergen.diffusion.model_utils.NoiseLevelEncoding",
                "d_model": 512,
            },
            "scaler": {"_target_": "torch.nn.Identity"},
        }

    adapter_cfg = {
        "_target_": "mattergen.adapter.GemNetTAdapter",
        "property_embeddings_adapt": property_embedding_cfg,
    }
    # gemnet kwargs flowing into GemNetTCtrl via Hydra.
    if bridge_kind == "producer-consumer":
        adapter_cfg["gemnet"] = {
            "cond_adapt_use_ipa": ["alm_embedding"],
            "cond_adapt_n_heads": cond_adapt_n_heads,
            "cond_adapt_depth": 1,
            "bridge_gate_init": bridge_gate_init,
            "bridge_noise_gate": bridge_noise_gate,   # linear 1-t gate
            "bridge_tenc_fuse": bridge_tenc_fuse,     # learned t_enc token fusion
        }
    elif bridge_kind == "consumer-only":
        # Eval-only; depth fixed at 1 (zero-init V only holds for a single MHA).
        adapter_cfg["gemnet"] = {
            "cond_adapt_use_ipa": ["alm_embedding"],
            "cond_adapt_n_heads": cond_adapt_n_heads,
            "cond_adapt_depth": 1,
            "bridge_gate_init": bridge_gate_init,
        }
    elif bridge_kind in ("pool", "producer-consumer-pool"):
        # Single (B, 512) vector into the additive concat-MLP path; only the gate init matters.
        adapter_cfg["gemnet"] = {
            "bridge_gate_init": bridge_gate_init,
        }

    return OmegaConf.create({
        # model_path set -> init_adapter loads a LOCAL ckpt dir; else HF hub.
        "pretrained_name": None if model_path is not None else pretrained_name,
        "model_path": model_path,
        "load_epoch": "last",
        "full_finetuning": full_finetuning,
        "adapter": adapter_cfg,
    })


def _make_lightning_module_cfg(lr: float):
    return OmegaConf.create({
        "_target_": "mattergen.diffusion.lightning_module.DiffusionLightningModule",
        "optimizer_partial": {
            "_target_": "torch.optim.AdamW",
            "_partial_": True,
            "lr": lr,
            "weight_decay": 0.0,
            "amsgrad": True,
        },
        "scheduler_partials": [],
    })


def load_mattergen_adapter(pretrained_name: str, lr: float,
                           hidden_dim: int = 4096, K: int = 8, mid_dim: int = 2048,
                           bridge_kind: str = "pool", cond_adapt_n_heads: int = 4,
                           use_task_direction_cond: bool = False,
                           use_alm_embedding_cond: bool = True,
                           full_finetuning: bool = False,
                           bridge_gate_init: float = 1.0,
                           model_path: str | None = None,
                           qformer_num_queries: int = 16,
                           qformer_depth: int = 2,
                           qformer_heads: int = 8,
                           qformer_context_tokens: int = 128,
                           qformer_input_atoms: int = 0,
                           bridge_out_norm: bool = False,
                           bridge_learnable_null: bool = False,
                           bridge_noise_gate: bool = False,
                           bridge_tenc_fuse: bool = False):
    """Load MatterGen with the bridge adapter; backbone frozen unless full_finetuning, model_path picks a local CSP backbone."""
    adapter_cfg = _make_adapter_cfg(
        pretrained_name, full_finetuning=full_finetuning,
        hidden_dim=hidden_dim, K=K, mid_dim=mid_dim,
        bridge_kind=bridge_kind, cond_adapt_n_heads=cond_adapt_n_heads,
        use_task_direction_cond=use_task_direction_cond,
        use_alm_embedding_cond=use_alm_embedding_cond,
        bridge_gate_init=bridge_gate_init,
        model_path=model_path,
        qformer_num_queries=qformer_num_queries,
        qformer_depth=qformer_depth,
        qformer_heads=qformer_heads,
        qformer_context_tokens=qformer_context_tokens,
        qformer_input_atoms=qformer_input_atoms,
        bridge_out_norm=bridge_out_norm,
        bridge_learnable_null=bridge_learnable_null,
        bridge_noise_gate=bridge_noise_gate,
        bridge_tenc_fuse=bridge_tenc_fuse,
    )
    lm_cfg = _make_lightning_module_cfg(lr)
    pl_module, _ = init_adapter_lightningmodule_from_pretrained(adapter_cfg, lm_cfg)
    return pl_module


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(step: int, alm, diffusion_module, optimizer, out_dir: Path,
                    include_optimizer: bool = True, aux_head=None, stage_3b: bool = False,
                    bridge_kind: str = "pool", cond_adapt_n_heads: int = 4,
                    cond_adapt_depth: int = 1, llm_full_ft: bool = False,
                    full_finetuning: bool = False,
                    num_output_atom_tokens: int = 8, use_last_prompt_token: bool = False,
                    bridge_source: str = "atoms_tokens",
                    init_atoms_tokens_from_eos: bool = False,
                    qformer_num_queries: int = 16, qformer_depth: int = 2,
                    qformer_heads: int = 8, qformer_context_tokens: int = 128,
                    qformer_input_atoms: int = 0):
    """Save AtomsMapper + cond_adapt/mixin + optional aux head/optimizer (and LoRA adapter when stage_3b)."""
    save_dir = out_dir / f"step={step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    dm = diffusion_module.module if hasattr(diffusion_module, "module") else diffusion_module
    # No-LLM-bridge control has no AtomsMapper; save an empty state dict for it.
    if "alm_embedding" in dm.model.property_embeddings_adapt:
        atoms_mapper = (
            dm.model
            .property_embeddings_adapt["alm_embedding"]
            .conditional_embedding_module
        )
        atoms_mapper_state_dict = atoms_mapper.state_dict()
    else:
        atoms_mapper_state_dict = {}
    trainable_state = {
        k: v for k, v in dm.model.state_dict().items()
        if "property_embeddings_adapt" in k
        or "cond_adapt_layers" in k
        or "cond_mixin_layers" in k
        or "tenc_fuse" in k          # learned timestep-fusion MLP/LN
        or "tenc_encoding" in k
    }
    # Under full-FT the filter above misses the co-adapted backbone, so persist the complete state.
    mattergen_full_state = (
        {k: v.detach().cpu() for k, v in dm.model.state_dict().items()}
        if full_finetuning else None
    )
    payload = {
        "step": step,
        "atoms_mapper_state_dict": atoms_mapper_state_dict,
        "trainable_state_dict": trainable_state,
        # Full backbone weights only under full_finetuning; load sites overlay strict=False.
        "mattergen_full_state_dict": mattergen_full_state,
        "full_finetuning": bool(full_finetuning),
        # Bridge metadata so load sites pick the right class; absence means 'pool'.
        "bridge_kind": bridge_kind,
        "cond_adapt_n_heads": cond_adapt_n_heads,
        "cond_adapt_depth": cond_adapt_depth,
        # ALM-side bridge metadata so inference rebuilds the ALM identically (defaults if absent).
        "num_output_atom_tokens": num_output_atom_tokens,
        "use_last_prompt_token": use_last_prompt_token,
        "bridge_source": bridge_source,
        "init_atoms_tokens_from_eos": init_atoms_tokens_from_eos,
        # Q-Former hparams so eval rebuilds the producer; ignored for other bridge_kinds.
        "qformer_num_queries": qformer_num_queries,
        "qformer_depth": qformer_depth,
        "qformer_heads": qformer_heads,
        "qformer_context_tokens": qformer_context_tokens,
        "qformer_input_atoms": qformer_input_atoms,
    }
    if aux_head is not None:
        payload["aux_head_kind"] = aux_head.target_kind
        payload["aux_head_state_dict"] = aux_head.state_dict()
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    # Stage 3b: write the irreplaceable LoRA/projector first, then the recomputable atoms_mapper.pt,
    # so a mid-save disk-full loses the recoverable file, not the LoRA.
    if stage_3b:
        lora_dir = save_dir / "lora_adapter"
        alm_module = alm.module if hasattr(alm, "module") else alm
        alm_module.llm.save_pretrained(str(lora_dir), save_embedding_layers=False)
        # save_embedding_layers=False drops the K=8 [atoms_i] rows; persist them explicitly.
        _state_blob = {"projector_state_dict": alm_module.projector.state_dict()}
        try:
            _emb_w = alm_module.llm.get_input_embeddings().weight.data
            _ids = list(alm_module.output_atom_token_ids)
            _state_blob["atoms_i_embed_rows"] = _emb_w[_ids].detach().cpu().clone()
            _state_blob["output_atom_token_ids"] = _ids
        except Exception as _e:
            print(f"[save] WARN could not persist atoms_i rows: {_e}", flush=True)
        torch.save(_state_blob, save_dir / "projector_and_state.pt")
    elif llm_full_ft:
        # FSDP gathers the per-layer FlatParameters onto rank-0 host RAM (offload_to_cpu).
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )
        alm_module = alm.module if hasattr(alm, "module") else alm
        # On a single GPU the LLM is unwrapped; FSDP.state_dict_type would raise, so fall back.
        _llm_is_fsdp = any(isinstance(m, FSDP) for m in alm_module.llm.modules())
        if _llm_is_fsdp:
            with FSDP.state_dict_type(
                alm_module.llm,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                llm_state = alm_module.llm.state_dict()
        else:
            llm_state = alm_module.llm.state_dict()
        if is_main_process():
            llm_ft_dir = save_dir / "llm_full_ft"
            llm_ft_dir.mkdir(parents=True, exist_ok=True)
            torch.save(llm_state, llm_ft_dir / "qwen3_state_dict.pt")
            torch.save(
                {"projector_state_dict": alm_module.projector.state_dict()},
                save_dir / "projector_and_state.pt",
            )

    # Guard against multi-rank races (full-FT enters this on all ranks for the gather).
    if is_main_process():
        torch.save(payload, save_dir / "atoms_mapper.pt")


def resume_checkpoint(ckpt_path: str, diffusion_module, optimizer, device, aux_head=None):
    """Restore diffusion-side state (AtomsMapper + cond_adapt + optional aux head + optimizer). LoRA resumes via load_alm."""
    ckpt = torch.load(ckpt_path, map_location=device)
    dm = diffusion_module.module if hasattr(diffusion_module, "module") else diffusion_module

    atoms_mapper = (
        dm.model
        .property_embeddings_adapt["alm_embedding"]
        .conditional_embedding_module
    )
    atoms_mapper.load_state_dict(ckpt["atoms_mapper_state_dict"])

    if ckpt.get("mattergen_full_state_dict") is not None:
        cur_sd = dm.model.state_dict()
        cur_sd.update(ckpt["mattergen_full_state_dict"])
        miss, unexp = dm.model.load_state_dict(cur_sd, strict=False)
        print(f"[stage3a] resumed FULL MatterGen backbone (full-FT): "
              f"{len(ckpt['mattergen_full_state_dict'])} tensors "
              f"(missing={len(miss)}, unexpected={len(unexp)})")
    elif "trainable_state_dict" in ckpt:
        cur_sd = dm.model.state_dict()
        cur_sd.update(ckpt["trainable_state_dict"])
        dm.model.load_state_dict(cur_sd, strict=True)

    if aux_head is not None:
        saved_kind = ckpt.get("aux_head_kind")
        if saved_kind == aux_head.target_kind and "aux_head_state_dict" in ckpt:
            aux_head.load_state_dict(ckpt["aux_head_state_dict"])
            print(f"[stage3a] resumed aux_head ({aux_head.target_kind}) state")
        else:
            print(f"[stage3a] aux_head kind mismatch (ckpt={saved_kind!r}, "
                  f"current={aux_head.target_kind!r}) — aux_head starts fresh")

    # Optimizer state is saved only periodically; weights-only resumes start fresh.
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[stage3a] resumed optimizer state from step {ckpt.get('step', 0)}")
    else:
        print(f"[stage3a] no optimizer state in ckpt — starting with fresh optimizer "
              f"(step {ckpt.get('step', 0)} weights restored)")
    return ckpt.get("step", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Build ChemGraph batch from raw structure tensors
# ─────────────────────────────────────────────────────────────────────────────

def build_chemgraph_batch(fracs, cells, Zs, n_atoms, device,
                          task_direction=None):
    """Batched ChemGraph from per-structure tensors; optional (B,) +-1 task_direction (NaN -> unconditional)."""
    graphs = []
    for i, (frac, cell, Z, na) in enumerate(zip(fracs, cells, Zs, n_atoms)):
        kwargs = dict(
            pos=frac.to(device, dtype=torch.float32),
            cell=cell.to(device, dtype=torch.float32).unsqueeze(0),  # (1, 3, 3)
            atomic_numbers=Z.to(device, dtype=torch.long),
            num_atoms=na.to(device),
        )
        if task_direction is not None:
            td = float(task_direction[i].item() if hasattr(task_direction[i], "item")
                       else task_direction[i])
            kwargs["task_direction"] = torch.tensor([td], dtype=torch.float32, device=device)
        g = ChemGraph(**kwargs)
        g = symmetrize_lattice(g)
        g = set_chemical_system_string(g)
        graphs.append(g)
    return mg_collate(graphs)


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _group_grad_norm(params):
    """L2 norm of grads across a parameter group (skips grad-less params)."""
    sq = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            sq += g.float().pow(2).sum().item()
    return sq ** 0.5


def _group_weight_norm(params):
    sq = 0.0
    for p in params:
        sq += p.detach().float().pow(2).sum().item()
    return sq ** 0.5


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alm_checkpoint", required=True,
                   help="Stage 2 checkpoint dir (lora_adapter/ + projector_and_state.pt)")
    p.add_argument("--pairs_parquet", default=None,
                   help="Output of build_stage3a_pairs.py. Single-bucket mode. "
                        "Mutually exclusive with --pairs_parquets.")
    p.add_argument("--pairs_parquets", default=None,
                   help="Comma-separated list of parquet paths for multi-bucket "
                        "sampling. Each bucket draws from one parquet; sampler "
                        "picks bucket per step via --pairs_weights, then a row "
                        "within. Mutually exclusive with --pairs_parquet.")
    p.add_argument("--pairs_weights", default=None,
                   help="Comma-separated bucket weights aligned 1:1 with "
                        "--pairs_parquets. Need not sum to 1 (normalized internally). "
                        "Required when --pairs_parquets is set.")
    p.add_argument("--cached_embs_root", type=str, default=None,
                   help="Root dir of per-parent OrbV3 caches (e.g. "
                        "<data_root>/cached_embs_narratives). Used by "
                        "the atomtxt bucket to load cached features for input_atoms_struct. "
                        "Non-atomtxt buckets ignore this; their atom_embed is zero-length.")
    p.add_argument("--mattergen_pretrained", default="mattergen_base",
                   help="MatterGen pretrained name (HF hub) or path")
    p.add_argument("--mattergen_model_path", type=str, default=None,
                   help="Local CSP-mode backbone dir. When set, "
                        "the adapter loads this LOCAL ckpt instead of --mattergen_pretrained, "
                        "wraps its plain GemNetT into GemNetTCtrl, and attaches the "
                        "alm_embedding cond field. CSP corruption (atoms observed, pos+cell "
                        "denoised) + CSP loss are inherited from the backbone's saved config. "
                        "This is the planner architecture: JSON-composition observed "
                        "atoms + atoms_i->CFG task conditioning.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="LR for AtomsMapper + cond_adapt/mixin layers + aux head.")
    p.add_argument("--lora_lr", type=float, default=0.0,
                   help="LR for Stage 2 LoRA params on Qwen3. Default 0 (Stage 3a, frozen LLM). "
                        "Set >0 to enter Stage 3b (unfreeze LoRA on Qwen3). Recommended: 1e-5 "
                        "to 5e-5 — much lower than Stage 2's 2e-4 since the diffusion+aux "
                        "gradient is noisier, and LoRA at high lr destroys Qwen3's "
                        "hidden-state diversity. With aux composition loss "
                        "as anchor, low-lr LoRA can refine [atoms_i] hidden states without "
                        "drifting into the degenerate equilibrium.")
    p.add_argument("--contrastive_lambda", type=float, default=0.0,
                   help="Weight on the off-diag-cosine contrastive loss that decorrelates "
                        "AtomsMapper outputs across the batch. Default 0 (disabled). Note: "
                        "without aux loss, contrastive alone leads to composition-irrelevant "
                        "decorrelation. Pair it with composition aux at a small weight "
                        "(0.02) as a rank-collapse safety net.")
    p.add_argument("--aux_target_kind", type=str, default="composition",
                   choices=["composition", "composition_count", "none"],
                   help="Auxiliary supervision target on AtomsMapper output. 'composition' "
                        "predicts a multi-hot Z=1..100 of the target structure (BCE). "
                        "'composition_count' adds a per-element CE on the exact integer count "
                        "(clamped to MAX_COUNT=20) on top of the same presence BCE — drop-in "
                        "replacement for 'composition' that pressures stoichiometry.")
    p.add_argument("--aux_lambda", type=float, default=1.0,
                   help="Weight on the auxiliary loss in total = L_diff + λ_aux * L_aux + "
                        "λ_contrastive * L_contrastive.")
    p.add_argument("--count_lambda", type=float, default=1.0,
                   help="Weight on the count-CE branch *inside* the composition_count head, "
                        "relative to the presence-BCE branch (= L_aux). 1.0 starts even.")
    p.add_argument("--use_last_prompt_token", action="store_true",
                   help="Bypass the K random-init [atoms_i] embeddings and pool from "
                        "the position immediately preceding the first [atoms_i] token. "
                        "Returns K copies of that hidden state so AtomsMapper API stays "
                        "the same. Tests whether the random-init [atoms_i] tokens are "
                        "the bottleneck for archetype-knowledge preservation.")
    p.add_argument("--aux_warmup_steps", type=int, default=0,
                   help="Pre-train AtomsMapper on aux loss only for N steps (zero-out L_diff). "
                        "Useful if mapper output starts uninformative; lets the aux head "
                        "shape AtomsMapper before the noisy diffusion gradient comes online.")
    p.add_argument("--diffusion_loss_weight", type=float, default=1.0,
                   help="Multiplier on the diffusion loss L_diff in the combined loss. "
                        "Default 1.0 (no change).")
    p.add_argument("--bridge_gate_init", type=float, default=1.0,
                   help="Initial value of GemNetTCtrl.bridge_gate (per-int_block scalar "
                        "multiplying the alm_embedding cond_mixin residual before the "
                        "additive sum). 1.0 (default) = plain additive behavior.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--save_optimizer_every", type=int, default=5,
                   help="Include optimizer state every Nth periodic save (else weights-only). "
                        "Final save always includes optimizer.")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--num_output_atom_tokens", type=int, default=8,
                   help="K — number of [atoms_{i}] output-side tokens emitted at the end "
                        "of the assistant turn. Default 8. Reduce to 4 for the "
                        "K=4 ablation; increase to 16 to give Qwen3 more capacity for "
                        "spreading structural info across positions before AtomsMapper "
                        "pooling. Each change requires a fresh Stage 3b run from the Stage 2 "
                        "ckpt — load_alm migrates the saved [atoms_i] embedding rows to the "
                        "new K (truncating prefix on shrink, freshly init'ing extras on grow).")
    p.add_argument("--atoms_mapper_mid_dim", type=int, default=2048,
                   help="Hidden width of AtomsMapper's two-Linear-with-GELU MLP. Default "
                        "2048. Bump to 4096 to give the K*4096 -> 512 "
                        "projection more capacity for capturing geometric / structural detail "
                        "before pooling into MatterGen's 512-d conditioning vector. Doubling "
                        "mid_dim from 2048 to 4096 roughly doubles AtomsMapper params "
                        "(~70M -> ~140M); the LLM-side cost is unchanged.")
    p.add_argument("--p_unconditional", type=float, default=0.2,
                   help="CFG dropout probability (fraction of samples that use ZerosEmbedding)")
    p.add_argument("--bridge_kind", choices=("pool", "producer-consumer", "producer-consumer-pool", "consumer-only", "auto"), default="auto",
                   help="LLM→MatterGen bridge architecture. 'producer-consumer' uses "
                        "AtomsMapperProducerConsumer (M learned queries cross-attending the N context + K "
                        "atoms hidden states → (B, M, 512)) consumed by GemNetTCtrl's post-block "
                        "IP-Adapter cross-attention (zero-init V). 'producer-consumer-pool' mean-pools the M "
                        "query tokens to a single 512-d vector for the additive concat-MLP path. "
                        "'pool' uses AtomsMapper (per-position projection + terminal "
                        "mean-pool) → single 512-d vector → concat-MLP cond_adapt. "
                        "'consumer-only' is the IP-Adapter producer: K per-token "
                        "MLP outputs consumed by GemNet IP-Adapter cross-attention. "
                        "'auto' (default): sniff bridge_kind from --resume_atoms_mapper if set, "
                        "else default to 'pool' for backward compatibility.")
    p.add_argument("--cond_adapt_n_heads", type=int, default=4,
                   help="Number of attention heads for the qformer IP-Adapter cross-attention "
                        "cond_adapt. Must divide MatterGen's emb_size_atom=512. Ignored when "
                        "bridge_kind=pool.")
    # ── CFG-interface controls (compose freely) ───────────────────────────────
    p.add_argument("--bridge_out_norm", action="store_true",
                   help="Magnitude control: LayerNorm the QFormer cond-vector output to ~unit "
                        "RMS (FT-head convention) so the CFG delta has bounded scale. "
                        "Currently wired for --bridge_kind qformer.")
    p.add_argument("--bridge_learnable_null", action="store_true",
                   help="Learned null: use a LEARNABLE unconditional embedding "
                        "(LearnedEmbeddingSequence, opts out of the adapter zeros-rewrite) so CFG "
                        "steers (cond − learned_null) not (cond − 0). For --bridge_kind qformer.")
    p.add_argument("--bridge_noise_gate", action="store_true",
                   help="Hand-set linear 1−t gate; prefer --bridge_tenc_fuse. "
                        "Scales the bridge contribution by w(t)=(1−t).")
    p.add_argument("--bridge_tenc_fuse", action="store_true",
                   help="Learned timestep fusion: make the QFormer/IP-Adapter "
                        "cond TOKENS noise-aware — cond_t=LN(cond+MLP([cond, t_enc])), t_enc=native "
                        "NoiseLevelEncoding, MLP final zero-init (no-op at step 0). The model LEARNS "
                        "the noise-dependence instead of the hand-set 1−t gate. For --bridge_kind qformer.")
    # ── Q-Former bridge (--bridge_kind qformer) ──────────────────────────────
    p.add_argument("--qformer_num_queries", type=int, default=16,
                   help="M: number of learned query tokens for the Q-Former bridge. "
                        "AtomsMapperProducerConsumer emits (B, M, 512); GemNet cond_adapt_use_ipa "
                        "cross-attends over these M conditioning tokens. Only used when "
                        "--bridge_kind qformer.")
    p.add_argument("--qformer_depth", type=int, default=2,
                   help="Number of {cross-attn, self-attn, MLP} Q-Former blocks. "
                        "Only used when --bridge_kind qformer.")
    p.add_argument("--qformer_heads", type=int, default=8,
                   help="Number of attention heads inside each Q-Former block (must divide "
                        "out_dim=512). Only used when --bridge_kind qformer.")
    p.add_argument("--qformer_context_tokens", type=int, default=128,
                   help="N: number of LLM context hidden states (the N states immediately "
                        "before the first [atoms_0] token) prepended to the K [atoms_i] states "
                        "to form the Q-Former source sequence S = N + K. Plumbed to the ALM as "
                        "qformer_n_context. Only used when --bridge_kind qformer.")
    p.add_argument("--qformer_input_atoms", type=int, default=0,
                   help="L_in: when >0 (and --bridge_kind qformer), prepend up to L_in "
                        "input-side <atoms> hidden states (the P_in(H) projected encoder "
                        "features) to the Q-Former source, GUARANTEEING the input structure "
                        "is in the producer context regardless of prompt length. Source "
                        "becomes [input_atoms(L_in) ++ context(N) ++ atoms_i(K)]; the Q-Former "
                        "is built with source_len = L_in+N+K, n_context = L_in+N.")
    p.add_argument("--qformer_dir_aux_lambda", type=float, default=0.0,
                   help="Weight for the direction-aux CE loss on the Q-Former output. When >0 "
                        "(and --bridge_kind qformer), AtomsMapperProducerConsumer.direction_logits(am_out) "
                        "predicts the ±1 task_direction sign; CE is computed on finite-direction "
                        "rows only. Does NOT attach task_direction as a ChemGraph cond_field.")
    p.add_argument("--qformer_dir_aux_pool", choices=("mean", "query0"), default="mean",
                   help="How AtomsMapperProducerConsumer.direction_logits pools the M query tokens before "
                        "the 2-way direction head: 'mean' over queries or the first query 'query0'.")
    p.add_argument("--qformer_dir_margin_lambda", type=float, default=0.0,
                   help="Weight for the directional COSINE-MARGIN loss (bridge-internal). Unlike "
                        "the CE dir-aux (satisfied by tiny linear separability → cond stays at "
                        "high cross-direction cosine), this forces pooled am_out to carry an "
                        "ANGULAR directional component along ±dir_proto: relu(margin − sign(td)·"
                        "cos(pooled, dir_proto)) on finite-direction rows. Drives the cross-direction "
                        "cosine DOWN while leaving the orthogonal complement free for composition. "
                        "No external cond_field, no extra forward, no pairs.")
    p.add_argument("--qformer_dir_margin", type=float, default=0.3,
                   help="Target angular margin for --qformer_dir_margin_lambda: each higher row is "
                        "pushed to cos(pooled, dir_proto) ≥ margin, each lower row ≤ −margin. Larger "
                        "= stronger directional separation (lower cross-direction cosine) at more "
                        "cost to composition headroom. 0.3 ≈ leave ~95%% of the norm for composition.")
    p.add_argument("--qformer_contrastive_lambda", type=float, default=0.0,
                   help="Reserved Q-Former-specific contrastive weight (default 0.0; the shared "
                        "--contrastive_lambda path already decorrelates mapper outputs across the "
                        "batch for any bridge_kind).")
    p.add_argument("--qformer_dir_contrastive_lambda", type=float, default=0.0,
                   help="Weight for the same-input directional contrastive loss on the RAW LLM "
                        "hidden states (extract_atoms_hidden_states output, NOT am_out). For each "
                        "directional row with an opposite-direction sibling sharing the same "
                        "input_source_idx, the sibling's prompt is forwarded through the SAME bridge "
                        "extractor and the loss is relu(cos(flatten(raw_i), flatten(partner_raw_i)) - "
                        "target), averaged over rows that have a partner. Forces Qwen3 to encode "
                        "higher/lower direction at the bridge-extracted positions (where the "
                        "cross-direction cosine otherwise stays high). No learnable prototype, no "
                        "MatterGen cond_field — bridge-internal.")
    p.add_argument("--qformer_dir_contrastive_target", type=float, default=0.5,
                   help="Max-allowed cosine for --qformer_dir_contrastive_lambda: the loss only "
                        "penalizes a same-input higher/lower pair whose RAW-hidden-state cosine "
                        "EXCEEDS this target (relu(cos - target)). Lower = stronger separation. "
                        "0.5 leaves the pair free to share up to half its direction while still "
                        "pulling the otherwise-high cosine down.")
    p.add_argument("--qformer_dir_contrastive_on", choices=["raw", "am_out"], default="raw",
                   help="WHICH layer the directional contrastive acts on. 'raw': the "
                        "flattened extract_atoms_hidden_states output (context_plus_atoms, S*4096) — the "
                        "128 context tokens trivially differ by prompt word, so the loss is absorbed there "
                        "without pressuring the MatterGen-facing signal (cond-token cosine unchanged). "
                        "'am_out': the POST-MAPPER cond tokens (M*512) — the EXACT layer MatterGen "
                        "consumes; runs the QFormer on the partner's raw states and minimizes "
                        "relu(cos(flatten(am_out_i), flatten(am_out_partner_i)) - target). Excludes the "
                        "trivially-differing context tokens → forces the QFormer OUTPUT to encode direction.")
    p.add_argument("--init_atoms_tokens_from_eos", action="store_true",
                   help="CoT-style initialization: copy the LLM's <|im_end|> "
                        "embedding row into the K=8 [atoms_i] rows at construction. "
                        "The K positions enter the LLM forward with a semantically "
                        "meaningful 'assistant-turn-closes-here' embedding instead "
                        "of random init. Causal attention + RoPE diverge them across "
                        "positions during forward.")
    p.add_argument("--bridge_source", choices=("atoms_tokens", "last_k_prompt", "context_plus_atoms"),
                   default="atoms_tokens",
                   help="Which hidden states to feed the bridge. 'atoms_tokens' (default): "
                        "the K=8 random-init [atoms_i] output token positions in the assistant "
                        "turn. 'last_k_prompt': the K hidden states immediately preceding the "
                        "[atoms_0] anchor --- captures the LLM's processing of the actual prompt "
                        "content. Pair with the existing pairs.parquet for the last-user-prompt-tokens "
                        "variant, or with a description-in-assistant-turn data variant. "
                        "'context_plus_atoms': "
                        "Q-Former source — N=qformer_context_tokens context states immediately "
                        "before [atoms_0] followed by the K [atoms_i] states, returning (B, N+K, "
                        "hidden_dim). Set automatically when --bridge_kind qformer.")
    p.add_argument("--description_in_assistant_turn", action="store_true",
                   help="At training time, move the description (the `narrative` column of "
                        "pairs.parquet) from the user turn into the assistant turn, before the "
                        "[atoms_0..K] anchor. Combined with --bridge_source last_k_prompt, the "
                        "last K hidden states the bridge reads are description-content tokens "
                        "that the LLM has explicitly emitted. Requires the `narrative` column "
                        "in the parquet.")
    p.add_argument("--direction_in_assistant_turn", action="store_true",
                   help="Echo the directional instruction (parsed from the row_id suffix, e.g. "
                        "'...-formation_energy-lower' -> ' Target: lower formation energy.') onto "
                        "the END of the assistant anchor, immediately before [atoms_0]. Puts the "
                        "directional word in the last context tokens the QFormer's queries attend "
                        "most (otherwise the word is buried in the user turn). Pure TEXT processed "
                        "by the LLM — NOT a hand-set vector/cond_field.")
    p.add_argument("--lora_attn_lr", type=float, default=None,
                   help="Override --lora_lr for LoRA on q_proj/k_proj/v_proj/o_proj. "
                        "If unset (default), all LoRA params share --lora_lr. When set, "
                        "MLP LoRA (--lora_mlp_lr, defaults to --lora_lr) trains at a "
                        "separate rate. Multimodal-LLM pattern: hit attention "
                        "modules harder than MLP.")
    p.add_argument("--lora_mlp_lr", type=float, default=None,
                   help="Override --lora_lr for LoRA on gate_proj/up_proj/down_proj. "
                        "Defaults to --lora_lr when unset. See --lora_attn_lr.")
    p.add_argument("--p_stage2_text", type=float, default=0.0,
                   help="Fraction of training steps that additionally include a text-only "
                        "LM-CE micro-batch from Stage 2 data (MaScQA), interleaved with the "
                        "diffusion step. Default 0.0 (off). Set 0.1 to fire one text batch "
                        "every ~10 optimizer steps. Addresses the Stage-3a Q&A response-format "
                        "regression.")
    p.add_argument("--stage2_text_lambda", type=float, default=1.0,
                   help="Weight on the Stage-2-text LM CE loss in the combined gradient.")
    p.add_argument("--lm_loss_json_lambda", type=float, default=0.0,
                   help="Weight on the JSON-composition LM-CE loss. When >0, the dataset "
                        "prepends `{\"counts\": {El: cell_count}}` to the assistant turn and "
                        "supervises those JSON tokens (masking [atoms_i]); the main step adds "
                        "lambda * LM-CE via a second ALM forward. Teaches the model to emit "
                        "clean JSON jointly with the bridge — avoids the diffusion-LoRA "
                        "repetition-collapse in the self-planner. Try 0.5.")
    p.add_argument("--atoms_before_json", action="store_true",
                   help="Variant A: emit [atoms_i] BEFORE the {\"counts\":...} JSON in the "
                        "assistant turn (requires --lm_loss_json_lambda>0). The bridge then "
                        "reads the atoms hidden states pre-composition-commit (causal: they "
                        "cannot see the JSON to their right), surfacing the direction the "
                        "JSON-first layout compresses away. Inference must match "
                        "(generate_stage3.get_alm_embedding atoms_before_json=True).")
    p.add_argument("--stage2_text_batch_size", type=int, default=2,
                   help="Per-rank batch size for the interleaved text micro-batch. Keep "
                        "small — text samples skip the atomistic splice but still cost "
                        "one full Qwen3 forward.")
    p.add_argument("--cond_adapt_depth", type=int, default=1,
                   help="No-op (the qformer bridge uses a single IP-Adapter "
                        "cross-attention layer). Retained for checkpoint-metadata "
                        "compatibility; leave at the default 1.")
    p.add_argument("--full_finetuning", action="store_true",
                   help="Unfreeze the entire MatterGen base (GemNetT backbone "
                        "+ property embeddings). Default: backbone frozen, only "
                        "AtomsMapper+cond_adapt/mixin trainable. Setting this matches "
                        "MatterGen's own conditional-checkpoint recipe (full FT at "
                        "lr=5e-6). Significantly larger memory footprint; expect "
                        "to need batch_size=2 or grad_accum at batch_size=4.")
    p.add_argument("--llm_full_finetuning", action="store_true",
                   help="Unfreeze the ENTIRE Qwen3-8B base LLM (all decoder block "
                        "weights, embeddings, lm_head). At Stage 3, Stage 2 LoRA "
                        "is merged into the base by load_alm, so unfreezing here "
                        "trains every Qwen3 weight. Mutually exclusive with "
                        "--lora_lr > 0 (LoRA + full-FT stacking would double-count "
                        "deltas). Requires bitsandbytes PagedAdamW8bit; an fp32 "
                        "AdamW state for 8B params alone is 64 GB/rank which "
                        "won't fit on H200. Pair with very small --llm_lr (5e-7 "
                        "default) to avoid catastrophic forgetting of Stage 2 "
                        "instruction-following.")
    p.add_argument("--llm_lr", type=float, default=5e-7,
                   help="LR for the full-Qwen3-8B optimizer group when "
                        "--llm_full_finetuning is set. Default 5e-7 (much smaller "
                        "than --lora_lr's 1e-5 default, because full-FT touches "
                        "every weight not just the LoRA delta).")
    p.add_argument("--unfreeze_atoms_i_embeds", action="store_true",
                   help="Stage-3b targeted-embedding-unfreeze: train ONLY the K=8 "
                        "[atoms_i] input-embedding rows (32K params) on top of LoRA, "
                        "via a gradient mask that zeros all other embed rows. Unlike "
                        "--llm_full_finetuning (8B full-FT → FSDP), the trainable set "
                        "stays tiny + replicable, so the run is plain DDP (one model "
                        "per GPU). Requires Stage 3b (--lora_lr > 0). atoms_i rows ride "
                        "the lora optimizer group (weight_decay=0).")
    p.add_argument("--no_alm_embedding_cond", action="store_true",
                   help="CONTROL: drop the alm_embedding cond_field entirely (no-LLM-bridge "
                        "baseline). Skips the ALM forward at training (much faster). When "
                        "set, --aux_target_kind must be `none` (no AtomsMapper → no aux head).")
    p.add_argument("--handset_direction_token", action="store_true",
                   help="Directional editing: with --num_output_atom_tokens 9, "
                        "OVERWRITE the 9th [atoms_8] token's hidden block with a "
                        "deterministic direction code (first-half one-hot=higher, "
                        "second-half=lower, all-zero=unconditional), scale-matched to the "
                        "8 real structure tokens' RMS. The 8 real tokens carry structure "
                        "(grad flows to LoRA); the 9th is hand-set, not learned. Reuses "
                        "the task_direction ±1 parse (no separate cond_field added).")
    p.add_argument("--use_task_direction_cond", action="store_true",
                   help="Directional editing: add `task_direction` as a scalar "
                        "±1 cond_field (sinusoidal NoiseLevelEncoding, like dft_band_gap) "
                        "ALONGSIDE the alm_embedding bridge. +1=higher, -1=lower, parsed "
                        "from each row_id suffix; rows without a -higher/-lower suffix get "
                        "NaN (unconditional). The bridge carries the input structure; this "
                        "clean channel carries the direction bit. Forced by balanced data.")
    p.add_argument("--fresh_lora_rank", type=int, default=None,
                   help="If set (Stage 3b only): merge the Stage 2 LoRA into base, then "
                        "attach a NEW small LoRA at this rank for Stage 3 training. Lets you "
                        "decouple Stage 3 capacity from the Stage 2 LoRA rank — Stage 2 "
                        "capability is baked into the merged base, Stage 3 adapts on top with "
                        "a fresh randomly-initialized A/B. Pair with a higher --lora_lr "
                        "(~1e-4..2e-4) since the LoRA starts at zero, not warm.")
    p.add_argument("--fresh_lora_alpha", type=int, default=None,
                   help="α for the fresh Stage 3 LoRA (default: 2 × fresh_lora_rank, matching "
                        "Stage 2 scaling=2.0). Drop to fresh_lora_rank for scaling=1.0.")
    p.add_argument("--lora_resume_dir", default=None,
                   help="Resume a fresh-LoRA run. Path to a step=N dir from a prior fresh-LoRA "
                        "training (contains lora_adapter/ with the trained r=fresh_lora_rank "
                        "weights). With --fresh_lora_rank set, --alm_checkpoint must point at "
                        "the ORIGINAL Stage 2 ckpt (so the merged base is recovered exactly), "
                        "and these saved fresh-LoRA weights overwrite the random init. Pair "
                        "with --resume_atoms_mapper pointing at the same step=N's "
                        "atoms_mapper.pt to restore optimizer + AtomsMapper + cond_adapt/mixin.")
    p.add_argument("--resume_atoms_mapper", default=None,
                   help="Path to atoms_mapper.pt checkpoint to resume from")
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--wandb_project", default="alm-stage3a")
    args = p.parse_args()
    # --no_alm_embedding_cond has no AtomsMapper, so the aux head must be off.
    if args.no_alm_embedding_cond and args.aux_target_kind != "none":
        raise ValueError(
            f"--no_alm_embedding_cond requires --aux_target_kind=none (no "
            f"AtomsMapper → no aux head). Got --aux_target_kind={args.aux_target_kind!r}. "
            f"Either pass --aux_target_kind none, or drop --no_alm_embedding_cond."
        )
    # --use_last_prompt_token and a non-default --bridge_source are conflicting bridge mechanisms.
    if args.use_last_prompt_token and args.bridge_source != "atoms_tokens":
        raise ValueError(
            f"--use_last_prompt_token and --bridge_source={args.bridge_source!r} "
            f"are mutually exclusive bridge mechanisms. Use --bridge_source last_k_prompt "
            f"alone (supersedes --use_last_prompt_token) OR --use_last_prompt_token "
            f"alone (with default bridge_source=atoms_tokens). Not both."
        )
    return args


def _resolve_bridge_kind(args) -> str:
    """Resolve bridge_kind: 'auto' sniffs the resume ckpt (default 'pool'); explicit values assert-match a resume ckpt."""
    ckpt_bridge = None
    if args.resume_atoms_mapper:
        try:
            ckpt_meta = torch.load(
                args.resume_atoms_mapper, map_location="cpu", weights_only=False
            )
        except TypeError:
            ckpt_meta = torch.load(args.resume_atoms_mapper, map_location="cpu")
        ckpt_bridge = _norm_bridge(ckpt_meta.get("bridge_kind", "pool"))

    if args.bridge_kind == "auto":
        return ckpt_bridge if ckpt_bridge is not None else "pool"

    if ckpt_bridge is not None and ckpt_bridge != args.bridge_kind:
        raise ValueError(
            f"bridge_kind mismatch: ckpt={ckpt_bridge!r}, cli={args.bridge_kind!r}. "
            "Either re-run with --bridge_kind auto, or start a fresh run for the "
            "requested bridge."
        )
    return args.bridge_kind


def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    out_dir = Path(args.out_dir)

    # Stage 3a (lora_lr=0): merge_lora, all frozen, ALM forward in no_grad.
    # Stage 3b (lora_lr>0): PEFT live, only lora_A/B trainable; aux loss anchors [atoms_i] against collapse.
    stage_3b = args.lora_lr > 0
    fresh_lora = stage_3b and args.fresh_lora_rank is not None
    llm_full_ft = bool(getattr(args, "llm_full_finetuning", False))
    if llm_full_ft and stage_3b:
        raise ValueError("--llm_full_finetuning is mutually exclusive with "
                         "--lora_lr > 0. Set lora_lr=0 to merge Stage 2 LoRA "
                         "and unfreeze all Qwen3 weights instead.")
    if fresh_lora and args.fresh_lora_rank <= 0:
        raise ValueError("--fresh_lora_rank must be a positive int")
    if args.fresh_lora_rank is not None and not stage_3b and not llm_full_ft:
        raise ValueError("--fresh_lora_rank requires Stage 3b (--lora_lr > 0) "
                         "or --llm_full_finetuning (in which case fresh_lora_rank "
                         "is ignored — full Qwen3 weights are trained, no LoRA).")
    if is_main_process():
        if fresh_lora:
            a = args.fresh_lora_alpha or 2 * args.fresh_lora_rank
            mode = (f"Stage 3b — Stage 2 LoRA MERGED, fresh LoRA "
                    f"(r={args.fresh_lora_rank}, α={a}) attached for Stage 3")
        elif stage_3b:
            mode = "Stage 3b — LoRA UNFROZEN"
        else:
            mode = "Stage 3a — LoRA frozen+merged"
        print(f"[stage3a] Loading ALM from {args.alm_checkpoint} ({mode}) ...")
    # Resolve bridge_kind early; the Q-Former bridge forces the context_plus_atoms source.
    bridge_kind = _resolve_bridge_kind(args)
    _alm_bridge_source = args.bridge_source
    if bridge_kind in ("producer-consumer", "producer-consumer-pool"):
        _alm_bridge_source = "context_plus_atoms"
    alm, tokenizer = load_alm(
        checkpoint=args.alm_checkpoint,
        merge_lora=(not stage_3b) or fresh_lora,
        is_trainable=stage_3b and not fresh_lora,
        use_cached_embeddings=True,
        device=device,
        num_output_atom_tokens=args.num_output_atom_tokens,
        use_last_prompt_token=args.use_last_prompt_token,
        bridge_source=_alm_bridge_source,
        qformer_n_context=args.qformer_context_tokens,
        qformer_input_atoms=args.qformer_input_atoms,
        init_atoms_tokens_from_eos=args.init_atoms_tokens_from_eos,
    )
    # Auto-detect per-encoder feature dim from the projector to size the sentinel + pick the cache file.
    # uma_s vs uma_m is ambiguous at 128-d; uma-s-1p1 is a benign default (shared idx layout).
    _ATOMISTIC_NAME_BY_DIM = {
        256: "orb_v3_direct_20_omat",
        128: "uma-s-1p1",
        640: "pet-mad-xs",
        1280: "pet-mad-s",
    }
    atomistic_feature_dim = int(alm.projector[0].in_features)
    atomistic_model_name = _ATOMISTIC_NAME_BY_DIM.get(
        atomistic_feature_dim, "orb_v3_direct_20_omat"
    )
    if is_main_process():
        print(f"[stage3a] atomistic_feature_dim={atomistic_feature_dim} "
              f"(auto-detected) → atomistic_model_name={atomistic_model_name}")
    if fresh_lora:
        # Stage 2 LoRA is merged into base; attach a fresh small LoRA for Stage 3.
        from peft import LoraConfig, get_peft_model
        from loader import LORA_TARGET_MODULES
        from safetensors.torch import load_file as _load_safetensors
        fresh_alpha = args.fresh_lora_alpha or 2 * args.fresh_lora_rank
        fresh_cfg = LoraConfig(
            r=args.fresh_lora_rank, lora_alpha=fresh_alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
        )
        alm.llm = get_peft_model(alm.llm, fresh_cfg)
        if args.lora_resume_dir is not None:
            # Overwrite the random init with saved A/B from --lora_resume_dir/lora_adapter/.
            resume_lora_path = (Path(args.lora_resume_dir)
                                / "lora_adapter" / "adapter_model.safetensors")
            if not resume_lora_path.exists():
                raise FileNotFoundError(
                    f"--lora_resume_dir set but {resume_lora_path} not found")
            sd = _load_safetensors(str(resume_lora_path))
            sd = {k.replace(".lora_A.weight", ".lora_A.default.weight")
                   .replace(".lora_B.weight", ".lora_B.default.weight"): v
                  for k, v in sd.items()}
            missing, unexpected = alm.llm.load_state_dict(sd, strict=False)
            if is_main_process():
                # unexpected (saved keys we couldn't place) should be 0.
                print(f"[stage3a] resumed fresh-LoRA weights from {args.lora_resume_dir} "
                      f"(loaded {len(sd)} tensors, {len(unexpected)} unexpected)")
    if stage_3b:
        for name, p in alm.named_parameters():
            p.requires_grad_("lora_A" in name or "lora_B" in name)
        # Train only the K [atoms_i] embed rows via a backward hook zeroing all other rows.
        if getattr(args, "unfreeze_atoms_i_embeds", False):
            emb = alm.llm.get_input_embeddings()
            emb.weight.requires_grad_(True)
            _ids = list(alm.output_atom_token_ids)
            _row_mask = torch.zeros(emb.weight.shape[0], 1, device=emb.weight.device,
                                    dtype=emb.weight.dtype)
            _row_mask[_ids] = 1.0
            emb.weight.register_hook(lambda g, _m=_row_mask: g * _m)
            if is_main_process():
                print(f"[stage3a] targeted-unfreeze: embed_tokens trainable, grad masked "
                      f"to K={len(_ids)} [atoms_i] rows (token_ids={_ids}); DDP one-per-GPU "
                      f"(no FSDP). atoms_i rows train at lora_lr={args.lora_lr:.0e}.")
        alm.llm.gradient_checkpointing_enable()
        alm.llm.enable_input_require_grads()
    elif llm_full_ft:
        # Full-FT: Stage 2 LoRA already merged; unfreeze every Qwen3 weight.
        for name, p in alm.named_parameters():
            p.requires_grad_(True)
        alm.llm.gradient_checkpointing_enable()
        alm.llm.enable_input_require_grads()

        # FSDP-wrap alm.llm before optimizer creation (it changes param identities) so AdamW state shards.
        if world_size > 1:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                ShardingStrategy,
            )
            # Per-decoder-layer FSDP wrap; top-level alm.llm stays unwrapped so
            # get_input_embeddings() sees a 2-D weight. embed_tokens/lm_head stay replicated.
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            num_layers = len(alm.llm.model.layers)
            if is_main_process():
                print(f"[stage3a] FSDP per-decoder-layer wrap "
                      f"({num_layers} layers, FULL_SHARD, bf16 mixed precision, "
                      f"world_size={world_size}); embed_tokens + lm_head stay replicated")
            for i in range(num_layers):
                alm.llm.model.layers[i] = FSDP(
                    alm.llm.model.layers[i],
                    mixed_precision=mp_policy,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=torch.cuda.current_device(),
                    use_orig_params=True,
                )
    else:
        for p in alm.parameters():
            p.requires_grad_(False)
    alm.eval()
    K = len(alm.output_atom_token_ids)

    # ── bridge_kind already resolved above (before load_alm) ─────────────────
    if is_main_process():
        print(f"[stage3a] bridge_kind={bridge_kind!r} "
              f"(cli={args.bridge_kind!r}, "
              f"cond_adapt_n_heads={args.cond_adapt_n_heads}, "
              f"alm_bridge_source={_alm_bridge_source!r})")

    # ── Load MatterGen adapter ───────────────────────────────────────────────
    if is_main_process():
        _backbone_src = (f"LOCAL model_path={args.mattergen_model_path} (CSP-mode, atoms observed)"
                         if args.mattergen_model_path
                         else f"HF {args.mattergen_pretrained} (DNG-mode)")
        print(f"[stage3a] Loading MatterGen adapter from {_backbone_src} ...")
    diffusion_pl = load_mattergen_adapter(
        pretrained_name=args.mattergen_pretrained,
        lr=args.lr,
        hidden_dim=alm.llm_hidden_dim,
        K=K,
        mid_dim=args.atoms_mapper_mid_dim,
        bridge_kind=bridge_kind,
        cond_adapt_n_heads=args.cond_adapt_n_heads,
        use_task_direction_cond=args.use_task_direction_cond,
        use_alm_embedding_cond=not args.no_alm_embedding_cond,
        full_finetuning=args.full_finetuning,
        bridge_gate_init=args.bridge_gate_init,
        model_path=args.mattergen_model_path,
        qformer_num_queries=args.qformer_num_queries,
        qformer_depth=args.qformer_depth,
        qformer_heads=args.qformer_heads,
        qformer_context_tokens=args.qformer_context_tokens,
        qformer_input_atoms=args.qformer_input_atoms,
        bridge_out_norm=args.bridge_out_norm,
        bridge_learnable_null=args.bridge_learnable_null,
        bridge_noise_gate=args.bridge_noise_gate,
        bridge_tenc_fuse=args.bridge_tenc_fuse,
    )
    diffusion_pl = diffusion_pl.to(device)
    diffusion_module = diffusion_pl.diffusion_module

    # Override the pre_corruption_fn's p_unconditional if needed
    if hasattr(diffusion_module.pre_corruption_fn, "p_unconditional"):
        diffusion_module.pre_corruption_fn.p_unconditional = args.p_unconditional
    # iid per-field dropout: else a NaN task_direction AND-gates the mask and starves alm_embedding.
    if args.use_task_direction_cond and hasattr(diffusion_module.pre_corruption_fn, "dropout_fields_iid"):
        diffusion_module.pre_corruption_fn.dropout_fields_iid = True
        if is_main_process():
            print("[stage3a] dropout_fields_iid=True (decouple task_direction CFG from alm_embedding)")

    # ── Param groups + optimizer ─────────────────────────────────────────────
    mapper_params = [p for p in diffusion_module.parameters() if p.requires_grad]
    # No-LLM-bridge control: no AtomsMapper module exists.
    if args.no_alm_embedding_cond:
        atoms_mapper_module = None
        atoms_mapper_params = []
    else:
        atoms_mapper_module = (
            diffusion_module.model
            .property_embeddings_adapt["alm_embedding"]
            .conditional_embedding_module
        )
        atoms_mapper_params = list(atoms_mapper_module.parameters())
    # cond_adapt/mixin layers exist per cond_field; pull from all (they train jointly).
    cond_adapt_params = []
    cond_mixin_params = []
    for _cf in diffusion_module.model.gemnet.cond_adapt_layers.keys():
        cond_adapt_params += list(
            diffusion_module.model.gemnet.cond_adapt_layers[_cf].parameters()
        )
        cond_mixin_params += list(
            diffusion_module.model.gemnet.cond_mixin_layers[_cf].parameters()
        )

    aux_head = build_aux_head(args.aux_target_kind, in_dim=512,
                              count_lambda=args.count_lambda)
    aux_head_params: list = []
    if aux_head is not None:
        aux_head = aux_head.to(device)
        aux_head_params = list(aux_head.parameters())
        mapper_params = mapper_params + aux_head_params

    # Stage 3b: LoRA gets its own optimizer group; optionally split attn vs MLP for asymmetric lr.
    lora_params = []
    lora_attn_params = []
    lora_mlp_params = []
    asym_lora = stage_3b and (
        args.lora_attn_lr is not None or args.lora_mlp_lr is not None
    )
    if stage_3b:
        _ATTN_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")
        _MLP_KEYS = ("gate_proj", "up_proj", "down_proj")
        for n, p in alm.named_parameters():
            if not p.requires_grad:
                continue
            lora_params.append(p)
            if asym_lora:
                if any(k in n for k in _ATTN_KEYS):
                    lora_attn_params.append(p)
                elif any(k in n for k in _MLP_KEYS):
                    lora_mlp_params.append(p)
                else:
                    lora_attn_params.append(p)  # unmatched LoRA params go with attn

    # Full-FT: every Qwen3 weight, in its own optimizer group at --llm_lr.
    llm_full_params = []
    if llm_full_ft:
        for n, p in alm.named_parameters():
            if p.requires_grad:
                llm_full_params.append(p)

    if is_main_process():
        n_mapper = sum(p.numel() for p in mapper_params)
        n_diff_total = sum(p.numel() for p in diffusion_module.parameters())
        n_am = sum(p.numel() for p in atoms_mapper_params)
        n_ca = sum(p.numel() for p in cond_adapt_params)
        n_cm = sum(p.numel() for p in cond_mixin_params)
        n_aux = sum(p.numel() for p in aux_head_params)
        n_lora = sum(p.numel() for p in lora_params)
        n_llm_full = sum(p.numel() for p in llm_full_params)
        if stage_3b:
            print(f"[stage3a] LoRA: TRAINABLE (Stage 3b, lora_lr={args.lora_lr:.0e})")
            print(f"[stage3a] Trainable LoRA:         {n_lora/1e6:6.1f}M")
        elif llm_full_ft:
            print(f"[stage3a] LLM: FULL-FT (Stage 4, llm_lr={args.llm_lr:.0e})")
            print(f"[stage3a] Trainable LLM (full):  {n_llm_full/1e9:6.2f}B")
        else:
            print(f"[stage3a] LoRA: FROZEN (Stage 3a, merged into base)")
        print(f"[stage3a] Trainable AtomsMapper:  {n_am/1e6:6.1f}M")
        print(f"[stage3a] Trainable cond_adapt:   {n_ca/1e6:6.1f}M")
        print(f"[stage3a] Trainable cond_mixin:   {n_cm/1e6:6.1f}M (zero-init)")
        if aux_head is not None:
            print(f"[stage3a] Trainable aux_head:     {n_aux/1e6:6.3f}M ({aux_head.target_kind}, "
                  f"target_dim={aux_head.target_dim})")
        else:
            print(f"[stage3a] aux_head:               disabled (--aux_target_kind=none)")
        print(f"[stage3a] Trainable total:        {(n_mapper + n_lora + n_llm_full)/1e6:6.1f}M")

    if stage_3b:
        if asym_lora:
            attn_lr = args.lora_attn_lr if args.lora_attn_lr is not None else args.lora_lr
            mlp_lr  = args.lora_mlp_lr  if args.lora_mlp_lr  is not None else args.lora_lr
            if is_main_process():
                print(f"[stage3a] LoRA lr asymmetric: attn={attn_lr:.1e} ({len(lora_attn_params)} t), "
                      f"mlp={mlp_lr:.1e} ({len(lora_mlp_params)} t)")
            groups = [
                {"params": mapper_params,    "lr": args.lr, "name": "mapper"},
                {"params": lora_attn_params, "lr": attn_lr, "name": "lora_attn"},
                {"params": lora_mlp_params,  "lr": mlp_lr,  "name": "lora_mlp"},
            ]
            groups = [g for g in groups if g["params"]]
            optimizer = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.95))
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": mapper_params, "lr": args.lr,      "name": "mapper"},
                    {"params": lora_params,   "lr": args.lora_lr, "name": "lora"},
                ],
                weight_decay=0.0, betas=(0.9, 0.95),
            )
    elif llm_full_ft:
        # FSDP shards the AdamW state across ranks; the wrap happened earlier.
        n_llm = sum(p.numel() for p in llm_full_params)
        if is_main_process():
            print(f"[stage3a] LLM full-FT: {n_llm/1e9:.2f}B params at llm_lr={args.llm_lr:.0e} "
                  f"(FSDP FULL_SHARD; per-rank AdamW state ~{n_llm * 8 / 1e9 / max(world_size,1):.1f} GB)")
        optimizer = torch.optim.AdamW(
            [
                {"params": mapper_params,    "lr": args.lr,       "name": "mapper"},
                {"params": llm_full_params,  "lr": args.llm_lr,   "name": "llm_full"},
            ],
            weight_decay=0.0, betas=(0.9, 0.95),
        )
    else:
        optimizer = torch.optim.AdamW(
            mapper_params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
        )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume_atoms_mapper is not None:
        start_step = resume_checkpoint(args.resume_atoms_mapper, diffusion_module,
                                       optimizer, device, aux_head=aux_head)
        if is_main_process():
            print(f"[stage3a] Resumed from step {start_step}")

    # ── DDP wrap ─────────────────────────────────────────────────────────────
    if world_size > 1:
        diffusion_module = DDP(
            diffusion_module,
            device_ids=[local_rank],
            find_unused_parameters=False,  # all mapper params see grad every step
        )
        if stage_3b:
            # Wrap the OUTER ALM (Stage 2 pattern); find_unused_parameters=True for
            # grad-ckpt + LoRA paths that may go unused in a given microbatch.
            alm = DDP(alm, device_ids=[local_rank], find_unused_parameters=True)
        # llm_full_ft: no outer DDP wrap; alm.llm is FSDP-wrapped, the rest is frozen.

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    # (a) --pairs_parquet -> DistributedSampler; (b) --pairs_parquets -> ConcatDataset + BucketedDistributedSampler.
    if args.pairs_parquets and args.pairs_parquet:
        raise ValueError("--pairs_parquet and --pairs_parquets are mutually exclusive")
    if not args.pairs_parquets and not args.pairs_parquet:
        raise ValueError("must specify either --pairs_parquet or --pairs_parquets")

    if args.pairs_parquets:
        from torch.utils.data import ConcatDataset
        from samplers import BucketedDistributedSampler
        bucket_paths = [p.strip() for p in args.pairs_parquets.split(",") if p.strip()]
        if not args.pairs_weights:
            raise ValueError("--pairs_weights required with --pairs_parquets")
        weights_raw = [float(w) for w in args.pairs_weights.split(",") if w.strip()]
        if len(weights_raw) != len(bucket_paths):
            raise ValueError(
                f"--pairs_weights ({len(weights_raw)}) must align 1:1 with "
                f"--pairs_parquets ({len(bucket_paths)})"
            )
        if any(w < 0 for w in weights_raw):
            raise ValueError("--pairs_weights must be non-negative")
        # Normalize for readability; the sampler does its own multinomial.
        wsum = sum(weights_raw) or 1.0
        weights = [w / wsum for w in weights_raw]
        buckets = []
        for path in bucket_paths:
            buckets.append(Stage3aDataset(
                path, tokenizer, max_num_tokens=args.max_num_tokens,
                aux_target_kind=args.aux_target_kind,
                num_output_atom_tokens=args.num_output_atom_tokens,
                cached_embs_root=args.cached_embs_root,
                description_in_assistant_turn=args.description_in_assistant_turn,
                direction_in_assistant_turn=args.direction_in_assistant_turn,
                atomistic_feature_dim=atomistic_feature_dim,
                atomistic_model_name=atomistic_model_name,
                lm_loss_json=(args.lm_loss_json_lambda > 0),
                atoms_before_json=args.atoms_before_json,
                use_task_direction=(args.use_task_direction_cond or args.handset_direction_token
                                    or args.qformer_dir_aux_lambda > 0
                                    or args.qformer_dir_contrastive_lambda > 0),
            ))
        dataset = ConcatDataset(buckets)
        bucket_lengths = [len(b) for b in buckets]
        bucket_offsets, off = [], 0
        for n in bucket_lengths:
            bucket_offsets.append(off); off += n
        # total_steps x grad_accum_steps x world_size (one sample-index per microbatch).
        total_microbatches = (
            args.total_steps
            * max(1, getattr(args, "grad_accum_steps", 1))
            * max(1, world_size)
        )
        if is_main_process():
            print(f"[stage3a] multi-bucket sampling — {len(buckets)} buckets:")
            print(f"[stage3a]   {'bucket':40s}  {'rows':>10s}  {'weight':>7s}  "
                  f"{'visits':>10s}  {'visits/row':>11s}")
            for path, n_rows, w in zip(bucket_paths, bucket_lengths, weights):
                visits = int(total_microbatches * w)
                vpr = visits / max(n_rows, 1)
                short = path.split("/")[-1]
                print(f"[stage3a]   {short:40s}  {n_rows:>10,}  {w:>7.3f}  "
                      f"{visits:>10,}  {vpr:>11.3f}")
            print(f"[stage3a] total_microbatches={total_microbatches:,}")
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = BucketedDistributedSampler(
            bucket_lengths=bucket_lengths,
            bucket_offsets=bucket_offsets,
            weights=weights,
            num_microbatches=total_microbatches,
            num_replicas=max(1, world_size),
            rank=rank,
            seed=42,
        )
    else:
        dataset = Stage3aDataset(
            args.pairs_parquet, tokenizer, max_num_tokens=args.max_num_tokens,
            aux_target_kind=args.aux_target_kind,
            num_output_atom_tokens=args.num_output_atom_tokens,
            cached_embs_root=args.cached_embs_root,
            description_in_assistant_turn=args.description_in_assistant_turn,
            direction_in_assistant_turn=args.direction_in_assistant_turn,
            atomistic_feature_dim=atomistic_feature_dim,
            atomistic_model_name=atomistic_model_name,
            lm_loss_json=(args.lm_loss_json_lambda > 0),
            atoms_before_json=args.atoms_before_json,
            use_task_direction=(args.use_task_direction_cond or args.handset_direction_token
                                or args.qformer_dir_aux_lambda > 0
                                or args.qformer_dir_contrastive_lambda > 0),
        )
        sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=stage3a_collate,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = _WANDB and is_main_process() and not args.disable_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = start_step
    diffusion_module.train()

    def _unwrap(m):
        return m.module if hasattr(m, "module") else m

    def _inf_loader():
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                if batch is not None:
                    yield batch
            epoch += 1

    # ── Optional Stage-2 text data interleave ────────────────────────────────
    # When --p_stage2_text > 0, add a MaScQA LM-CE forward every Nth step to refresh Q&A format.
    text_loader_iter = None
    if args.p_stage2_text > 0.0:
        from utils import MaScQADataset, custom_collate_fn
        text_ds = MaScQADataset(
            tokenizer=tokenizer,
            questions_json=os.path.join(DATA_ROOT, "MaScQA/mascqa-eval.json"),
            scoresheet_xlsx=os.path.join(DATA_ROOT, "MaScQA/scoresheets/all_questions.xlsx"),
            max_num_tokens=args.max_num_tokens,
            split="train",
        )
        text_sampler = DistributedSampler(text_ds, shuffle=True) if world_size > 1 else None
        text_loader = DataLoader(
            text_ds,
            batch_size=args.stage2_text_batch_size,
            sampler=text_sampler,
            num_workers=1,
            collate_fn=custom_collate_fn,
            drop_last=True,
        )
        def _inf_text_loader():
            epoch = 0
            while True:
                if text_sampler is not None:
                    text_sampler.set_epoch(epoch)
                for batch in text_loader:
                    if batch is not None:
                        yield batch
                epoch += 1
        text_loader_iter = _inf_text_loader()
        if is_main_process():
            print(f"[stage3a] Stage-2 text mixing ON: p={args.p_stage2_text}, "
                  f"lambda={args.stage2_text_lambda}, source=MaScQA (n={len(text_ds)})")

    out_dir.mkdir(parents=True, exist_ok=True)

    for batch in _inf_loader():
        if global_step >= args.total_steps:
            break

        input_ids = batch["input_ids"].to(device)      # (B, max_len)
        attn_mask = batch["attention_mask"].to(device)  # (B, max_len)
        fracs = batch["fracs"]
        cells = batch["cells"]
        Zs = batch["Zs"]
        n_atoms_list = batch["n_atoms"]
        B = input_ids.shape[0]

        # Stage 3a: frozen ALM in no_grad + detach. Stage 3b: gradient flows back into LoRA.
        atom_embeds = [a.to(device) for a in batch["atom_embeds"]]
        input_ids_list = [input_ids[b] for b in range(B)]
        attn_mask_list = [attn_mask[b] for b in range(B)]

        # No-LLM-bridge control: skip the ALM forward and the alm_embedding attach.
        skip_alm = args.no_alm_embedding_cond
        lm_json_loss_raw = None
        if not skip_alm:
            if stage_3b or llm_full_ft:
                # alm is DDP-wrapped; one forward also returns LM-CE when JSON-loss is on
                # (a separate 2nd forward breaks multi-node DDP -> SIGABRT at step 1).
                if args.lm_loss_json_lambda > 0.0:
                    _lab = batch["labels"].to(device)
                    _lab_list = [_lab[b] for b in range(B)]
                    hidden_states, lm_json_loss_raw = alm(
                        input_ids_list, attn_mask_list, labels=_lab_list,
                        atom_embeds=atom_embeds, output_atoms_hidden_states=True,
                    )
                else:
                    hidden_states = alm(
                        input_ids_list, attn_mask_list, labels=None,
                        atom_embeds=atom_embeds, output_atoms_hidden_states=True,
                    )  # (B, K, hidden_dim) — autograd live, DDP grad-sync hooks armed
                alm_emb = hidden_states.flatten(1).float()
            else:
                # Stage 3a: alm unwrapped + frozen.
                with torch.no_grad():
                    hidden_states = alm.extract_atoms_hidden_states(
                        input_ids_list, attn_mask_list, atom_embeds=atom_embeds
                    )
                alm_emb = hidden_states.flatten(1).float().detach()

        # Hand-set the 9th [atoms_8] block via the shared direction_code helper (matches inference).
        if getattr(args, "handset_direction_token", False) and not skip_alm:
            td = batch.get("task_direction")
            if td is not None:
                from direction_code import apply_handset_direction
                alm_emb = apply_handset_direction(alm_emb, K, td)

        # ── Build ChemGraph and attach alm_embedding ─────────────────────────
        # Attach task_direction as a cond_field only under --use_task_direction_cond
        # (else it would AND-gate the NaN mask and starve alm_embedding).
        chemgraph = build_chemgraph_batch(
            fracs, cells, Zs, n_atoms_list, device,
            task_direction=(batch.get("task_direction") if args.use_task_direction_cond else None),
        )
        if not skip_alm:
            chemgraph["alm_embedding"] = alm_emb

        # ── Diffusion loss ───────────────────────────────────────────────────
        dm_inner = _unwrap(diffusion_module)
        loss_diff, metrics = dm_inner.calc_loss(chemgraph)

        # Extra AtomsMapper forward giving a clean handle for the aux/contrastive heads
        # (calc_loss runs it again internally; shared params, so grads accumulate correctly).
        _need_am_out = (args.contrastive_lambda > 0 or aux_head is not None
                        or (bridge_kind in ("producer-consumer", "producer-consumer-pool") and args.qformer_dir_aux_lambda > 0)
                        or (args.qformer_dir_contrastive_lambda > 0
                            and args.qformer_dir_contrastive_on == "am_out"))
        if _need_am_out and atoms_mapper_module is not None:
            am_out = atoms_mapper_module(chemgraph["alm_embedding"])  # (B, out_dim) or (B, M/K, out_dim)
        else:
            am_out = None

        # qformer bridge: am_out is (B, M, out_dim) and contrastive/aux pool over M.
        am_out_is_seq = am_out is not None and am_out.dim() == 3

        # ── Contrastive regularization (decorrelate mapper outputs across batch) ──
        if args.contrastive_lambda > 0 and B > 1 and am_out is not None:
            am_for_contrastive = am_out.mean(dim=1) if am_out_is_seq else am_out
            am_norm = torch.nn.functional.normalize(am_for_contrastive, dim=-1)
            sim = am_norm @ am_norm.T
            off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
            off_diag = sim[off_diag_mask]
            loss_contrastive = off_diag.pow(2).mean()
            metrics["contrastive_offdiag_mean"] = off_diag.mean().detach()
            metrics["contrastive_offdiag_max"] = off_diag.max().detach()
            metrics["contrastive_loss"] = loss_contrastive.detach()
        else:
            loss_contrastive = torch.tensor(0.0, device=device)

        # ── Auxiliary supervised loss on AtomsMapper output ──────────────────
        # Aux head predicts a composition fingerprint from am_out, pinning its output to structure.
        if aux_head is not None:
            aux_raw = batch["aux_target"]
            if isinstance(aux_raw, dict):
                aux_target = {k: v.to(device) for k, v in aux_raw.items()}
            else:
                aux_target = aux_raw.to(device)

            if am_out_is_seq:
                # (B, K, out_dim) -> (B*K, out_dim); broadcast target over K.
                B_, K_, _ = am_out.shape
                am_in = am_out.reshape(B_ * K_, -1)
                def _expand(t):
                    return (t.unsqueeze(1)
                              .expand(-1, K_, *([-1] * (t.dim() - 1)))
                              .reshape(B_ * K_, *t.shape[1:]))
                if isinstance(aux_target, dict):
                    aux_target_expanded = {k: _expand(v) for k, v in aux_target.items()}
                else:
                    aux_target_expanded = _expand(aux_target)
                aux_pred = aux_head(am_in)
                loss_aux = aux_head.loss(aux_pred, aux_target_expanded)
                metrics["aux/loss"] = loss_aux.detach()
                for k, v in aux_head.metrics(aux_pred, aux_target_expanded).items():
                    metrics[f"aux/{k}"] = v.detach() if torch.is_tensor(v) else v
            else:
                aux_pred = aux_head(am_out)
                loss_aux = aux_head.loss(aux_pred, aux_target)
                metrics["aux/loss"] = loss_aux.detach()
                for k, v in aux_head.metrics(aux_pred, aux_target).items():
                    metrics[f"aux/{k}"] = v.detach() if torch.is_tensor(v) else v
        else:
            loss_aux = torch.tensor(0.0, device=device)

        # ── Q-Former direction-aux loss ──────────────────────────────────────
        # CE predicting the +-1 task_direction sign from am_out (finite rows only); supervision, not a cond_field.
        loss_qformer_dir = torch.tensor(0.0, device=device)
        if (bridge_kind in ("producer-consumer", "producer-consumer-pool") and args.qformer_dir_aux_lambda > 0
                and am_out is not None
                and atoms_mapper_module is not None
                and hasattr(atoms_mapper_module, "direction_logits")):
            td_raw = batch.get("task_direction")
            if td_raw is not None:
                td = td_raw.to(device).float()                     # (B,) ±1 / NaN
                finite = torch.isfinite(td)
                n_finite = int(finite.sum().item())
                metrics["qformer_dir_aux_n_finite"] = torch.tensor(float(n_finite))
                if n_finite > 0:
                    dir_logits = atoms_mapper_module.direction_logits(
                        am_out, pool=args.qformer_dir_aux_pool)    # (B, 2)
                    dir_target = (td > 0).long()                   # >0 → 1, <0 → 0
                    dl_f = dir_logits[finite]
                    dt_f = dir_target[finite]
                    loss_qformer_dir = torch.nn.functional.cross_entropy(dl_f, dt_f)
                    dir_acc = (dl_f.argmax(dim=-1) == dt_f).float().mean()
                    metrics["qformer_dir_aux_loss"] = loss_qformer_dir.detach()
                    metrics["qformer_dir_acc"] = dir_acc.detach()
                    metrics["qformer_cond_norm"] = am_out.norm(dim=-1).mean().detach()
                    metrics["qformer_cond_std"] = am_out.std().detach()

        # ── Directional COSINE-MARGIN loss (bridge-internal cosine attack) ────
        # Forces pooled am_out angularly along +-dir_proto so cond_higher/lower become non-collinear.
        loss_dir_margin = torch.tensor(0.0, device=device)
        if (bridge_kind in ("producer-consumer", "producer-consumer-pool") and args.qformer_dir_margin_lambda > 0
                and am_out is not None and atoms_mapper_module is not None
                and hasattr(atoms_mapper_module, "direction_cosine")):
            td_raw = batch.get("task_direction")
            if td_raw is not None:
                td = td_raw.to(device).float()
                finite = torch.isfinite(td)
                if int(finite.sum().item()) > 0:
                    dcos = atoms_mapper_module.direction_cosine(
                        am_out, pool=args.qformer_dir_aux_pool)          # (B,) in [-1,1]
                    sgn = torch.sign(td[finite])                         # ±1
                    dcos_f = dcos[finite]
                    loss_dir_margin = torch.relu(
                        args.qformer_dir_margin - sgn * dcos_f).mean()
                    metrics["qformer_dir_margin_loss"] = loss_dir_margin.detach()
                    # Per-direction mean cosine vs prototype (separation diagnostic).
                    if (sgn > 0).any():
                        metrics["qformer_dir_cos_pos"] = dcos_f[sgn > 0].mean().detach()
                    if (sgn < 0).any():
                        metrics["qformer_dir_cos_neg"] = dcos_f[sgn < 0].mean().detach()

        # ── Same-input directional contrastive loss on the RAW LLM hidden states ──
        # Forward the opposite-direction sibling and push the two hidden-state tensors
        # apart: relu(cos - target) over has_partner rows. No prototype, no cond_field.
        loss_dir_contrastive = torch.tensor(0.0, device=device)
        if (args.qformer_dir_contrastive_lambda > 0 and not skip_alm
                and (stage_3b or llm_full_ft)
                and batch.get("has_partner") is not None
                and bool(batch["has_partner"].any())):
            # Forward only has_partner rows, bypassing the DDP wrapper (via alm.module)
            # so the reducer isn't double-armed; shared params still accumulate grad.
            hp_mask = batch["has_partner"]                              # (B,) bool, cpu
            idx_hp = [b for b in range(B) if bool(hp_mask[b])]
            if len(idx_hp) > 0:
                _alm = alm.module if hasattr(alm, "module") else alm
                p_ids_list = [batch["partner_input_ids"][b].to(device) for b in idx_hp]
                p_attn_list = [batch["partner_attention_mask"][b].to(device) for b in idx_hp]
                # Partner reuses the main row's atom_embed (same input structure).
                p_atom_embeds = [atom_embeds[b] for b in idx_hp]
                partner_raw = _alm.extract_atoms_hidden_states(
                    p_ids_list, p_attn_list, atom_embeds=p_atom_embeds,
                )  # (n_hp, S, H)
                idx_hp_t = torch.tensor(idx_hp, device=hidden_states.device,
                                        dtype=torch.long)
                if (args.qformer_dir_contrastive_on == "am_out"
                        and atoms_mapper_module is not None and am_out is not None):
                    # Compare POST-MAPPER cond tokens (the MatterGen-facing layer).
                    partner_am = atoms_mapper_module(partner_raw.flatten(1).float())
                    main_v = am_out[idx_hp_t].flatten(1).float()         # (n_hp, M*out)
                    partner_v = partner_am.flatten(1).float()            # (n_hp, M*out)
                else:
                    main_v = hidden_states[idx_hp_t].flatten(1).float()  # (n_hp, S*H)
                    partner_v = partner_raw.flatten(1).float()           # (n_hp, S*H)
                cos_hp = torch.nn.functional.cosine_similarity(
                    main_v, partner_v, dim=-1)                           # (n_hp,)
                loss_dir_contrastive = torch.relu(
                    cos_hp - args.qformer_dir_contrastive_target).mean()
                metrics["qformer_dir_contrastive_loss"] = loss_dir_contrastive.detach()
                metrics["qformer_dir_contrastive_cos"] = cos_hp.mean().detach()
                metrics["qformer_dir_contrastive_n"] = torch.tensor(float(len(idx_hp)))

        # ── Combined loss with optional warmup ────────────────────────────────
        # During aux warmup, zero L_diff so AtomsMapper learns from aux before diffusion comes online.
        diff_factor = 1.0
        if args.aux_warmup_steps > 0 and global_step < args.aux_warmup_steps and aux_head is not None:
            diff_factor = 0.0

        loss = (diff_factor * args.diffusion_loss_weight * loss_diff
                + args.contrastive_lambda * loss_contrastive
                + args.aux_lambda * loss_aux
                + args.qformer_dir_aux_lambda * loss_qformer_dir
                + args.qformer_dir_margin_lambda * loss_dir_margin
                + args.qformer_dir_contrastive_lambda * loss_dir_contrastive)

        # JSON-composition LM loss was computed in the bridge forward above; just weight + add it.
        loss_lm_json = torch.tensor(0.0, device=device)
        if lm_json_loss_raw is not None:
            loss_lm_json = args.lm_loss_json_lambda * lm_json_loss_raw
            metrics["lm_json_loss"] = lm_json_loss_raw.detach()
            loss = loss + loss_lm_json

        loss_text = torch.tensor(0.0, device=device)
        if text_loader_iter is not None:
            text_every = max(1, int(round(1.0 / max(args.p_stage2_text, 1e-9))))
            if global_step % text_every == 0:
                tb = next(text_loader_iter)
                t_ids = [t.to(device) for t in tb["input_ids"]]
                t_lab = [t.to(device) for t in tb["labels"]]
                t_attn = [t.to(device) for t in tb["attention_mask"]]
                t_atom_embeds = [t.to(device) for t in tb["atom_embeds"]]
                # Loss-branch forward for LM CE, via the underlying module to keep the graph attached.
                _alm = alm.module if hasattr(alm, "module") else alm
                t_out = _alm(t_ids, t_attn, labels=t_lab, atom_embeds=t_atom_embeds)
                loss_text = args.stage2_text_lambda * t_out.loss
                metrics["text_lm_loss"] = t_out.loss.detach()
                loss = loss + loss_text

        optimizer.zero_grad()
        loss.backward()

        # Per-group pre-clip gradient norms at log cadence.
        will_log = (global_step + 1) % args.log_every == 0 and is_main_process()
        if will_log:
            grad_norms = {
                "atoms_mapper": _group_grad_norm(atoms_mapper_params),
                "cond_adapt":   _group_grad_norm(cond_adapt_params),
                "cond_mixin":   _group_grad_norm(cond_mixin_params),
            }
            if aux_head_params:
                grad_norms["aux_head"] = _group_grad_norm(aux_head_params)
            if stage_3b and lora_params:
                grad_norms["lora"] = _group_grad_norm(lora_params)

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mapper_params, args.grad_clip)
            if stage_3b and lora_params:
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        optimizer.step()
        global_step += 1

        # ── Logging ──────────────────────────────────────────────────────────
        if global_step % args.log_every == 0 and is_main_process():
            log = {"loss": loss.item(), "step": global_step}
            log.update({k: v.item() for k, v in metrics.items()})
            log.update({f"grad_norm/{k}": v for k, v in grad_norms.items()})
            log.update({
                "weight_norm/atoms_mapper": _group_weight_norm(atoms_mapper_params),
                "weight_norm/cond_adapt":   _group_weight_norm(cond_adapt_params),
                "weight_norm/cond_mixin":   _group_weight_norm(cond_mixin_params),
            })
            if aux_head_params:
                log["weight_norm/aux_head"] = _group_weight_norm(aux_head_params)
            if stage_3b and lora_params:
                log["weight_norm/lora"] = _group_weight_norm(lora_params)
            # Per-int_block bridge_gate magnitude (is the bridge growing or near zero?).
            _dm_log = _unwrap(diffusion_module)
            _bg = getattr(_dm_log.model.gemnet, "bridge_gate", None)
            if _bg is not None:
                _bg_vals = _bg.detach().tolist()
                log["bridge_gate/mean_abs"] = float(_bg.detach().abs().mean().item())
                log["bridge_gate/values"] = _bg_vals
                log["bridge_gate_str"] = "[" + ",".join(f"{v:+.3f}" for v in _bg_vals) + "]"
            cd_mean = metrics.get("contrastive_offdiag_mean")
            cd_max = metrics.get("contrastive_offdiag_max")
            cd_str = (f"  cd_mean={cd_mean.item():+.3f} cd_max={cd_max.item():+.3f}"
                      if cd_mean is not None else "")
            aux_loss = metrics.get("aux/loss")
            aux_str = ""
            if aux_loss is not None:
                if "aux/recall" in metrics:
                    aux_str = (f"  aux={aux_loss.item():.3f}"
                               f" P={metrics['aux/precision'].item():.3f}"
                               f" R={metrics['aux/recall'].item():.3f}")
                elif "aux/cosine_sim" in metrics:
                    aux_str = (f"  aux={aux_loss.item():.3f}"
                               f" cos={metrics['aux/cosine_sim'].item():.3f}")
                else:
                    aux_str = f"  aux={aux_loss.item():.3f}"
            lora_str = ""
            if stage_3b and "lora" in grad_norms:
                lora_str = (f" lora={grad_norms['lora']:.2e}"
                            f"  |w_lora|={log['weight_norm/lora']:.2e}")
            print(f"[stage3a] step={global_step}/{args.total_steps}  loss={loss.item():.4f}  "
                  f"|g|: am={grad_norms['atoms_mapper']:.2e} "
                  f"ca={grad_norms['cond_adapt']:.2e} cm={grad_norms['cond_mixin']:.2e}"
                  f"{lora_str}  "
                  f"|w_cm|={log['weight_norm/cond_mixin']:.2e}{cd_str}{aux_str}"
                  + (f" lm_json={metrics['lm_json_loss']:.3f}" if 'lm_json_loss' in metrics else "")
                  + (f" dirmrg={metrics['qformer_dir_margin_loss']:.3f}"
                     f"(cos+{metrics.get('qformer_dir_cos_pos', float('nan')):+.2f}/"
                     f"{metrics.get('qformer_dir_cos_neg', float('nan')):+.2f})"
                     if 'qformer_dir_margin_loss' in metrics else "")
                  + (f" dircon={metrics['qformer_dir_contrastive_loss']:.3f}"
                     f"(cos={metrics['qformer_dir_contrastive_cos']:.3f}"
                     f",n={int(metrics.get('qformer_dir_contrastive_n', 0))})"
                     if 'qformer_dir_contrastive_loss' in metrics else ""))
            if use_wandb:
                wandb.log(log, step=global_step)

        # ── Checkpoint ───────────────────────────────────────────────────────
        # Periodic saves are weights-only; every save_optimizer_every-th also writes optimizer.
        # Under FSDP the state_dict gather is collective, so all ranks must enter together.
        if global_step % args.save_every == 0 and (is_main_process() or llm_full_ft):
            include_opt = (global_step % (args.save_every * args.save_optimizer_every) == 0)
            save_checkpoint(global_step, alm, diffusion_module, optimizer, out_dir,
                            include_optimizer=include_opt, aux_head=aux_head,
                            stage_3b=stage_3b,
                            bridge_kind=bridge_kind,
                            cond_adapt_n_heads=args.cond_adapt_n_heads,
                            cond_adapt_depth=args.cond_adapt_depth,
                            llm_full_ft=llm_full_ft,
                            full_finetuning=args.full_finetuning,
                            num_output_atom_tokens=args.num_output_atom_tokens,
                            use_last_prompt_token=args.use_last_prompt_token,
                            bridge_source=_alm_bridge_source,
                            init_atoms_tokens_from_eos=args.init_atoms_tokens_from_eos,
                            qformer_num_queries=args.qformer_num_queries,
                            qformer_depth=args.qformer_depth,
                            qformer_heads=args.qformer_heads,
                            qformer_context_tokens=args.qformer_context_tokens,
                            qformer_input_atoms=args.qformer_input_atoms)
            if is_main_process():
                print(f"[stage3a] Saved checkpoint at step {global_step}"
                      f" {'(with optimizer)' if include_opt else '(weights only)'}")
            # Barrier: under FSDP only rank 0 writes, so sync before the next collective op.
            if llm_full_ft and torch.distributed.is_initialized():
                torch.distributed.barrier()

    # Final save (with optimizer), skipped if the periodic save already covered this step.
    periodic_covers_final = (
        global_step % args.save_every == 0
        and global_step % (args.save_every * args.save_optimizer_every) == 0
    )
    if is_main_process() and periodic_covers_final:
        print(f"[stage3a] Final step={global_step} already saved by the "
              f"periodic-with-optimizer save; skipping redundant final save.")
    elif not periodic_covers_final and (is_main_process() or llm_full_ft):
        save_checkpoint(global_step, alm, diffusion_module, optimizer, out_dir,
                        include_optimizer=True, aux_head=aux_head, stage_3b=stage_3b,
                        bridge_kind=bridge_kind,
                        cond_adapt_n_heads=args.cond_adapt_n_heads,
                        cond_adapt_depth=args.cond_adapt_depth,
                        llm_full_ft=llm_full_ft,
                        full_finetuning=args.full_finetuning,
                        num_output_atom_tokens=args.num_output_atom_tokens,
                        use_last_prompt_token=args.use_last_prompt_token,
                        bridge_source=_alm_bridge_source,
                        init_atoms_tokens_from_eos=args.init_atoms_tokens_from_eos,
                        qformer_num_queries=args.qformer_num_queries,
                        qformer_depth=args.qformer_depth,
                        qformer_heads=args.qformer_heads,
                        qformer_context_tokens=args.qformer_context_tokens,
                        qformer_input_atoms=args.qformer_input_atoms)
    if is_main_process():
        print(f"[stage3a] Training complete. Final step: {global_step}")
        if use_wandb:
            wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
