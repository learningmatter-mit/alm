"""Load an ALM checkpoint for evaluation."""

import json
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import AtomisticLanguageModel  # alm/alm.py module, not package


LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def _apply_lora_adapter(llm, adapter_dir: Path, lora_rank=None, lora_alpha=None,
                        merge: bool = True):
    """Attach a PEFT LoRA adapter onto `llm` and optionally merge it; replayable for a second adapter atop an already-merged base."""
    with open(adapter_dir / "adapter_config.json") as f:
        saved_cfg = json.load(f)
    r = lora_rank if lora_rank is not None else saved_cfg["r"]
    a = lora_alpha if lora_alpha is not None else saved_cfg["lora_alpha"]
    lora_cfg = LoraConfig(
        r=r, lora_alpha=a, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=saved_cfg.get("target_modules", LORA_TARGET_MODULES),
    )
    llm = get_peft_model(llm, lora_cfg)
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    sd = {k.replace(".lora_A.weight", ".lora_A.default.weight")
           .replace(".lora_B.weight", ".lora_B.default.weight"): v
          for k, v in sd.items()}
    cur_sd = llm.state_dict()
    for k in list(sd.keys()):
        if k in cur_sd and sd[k].shape != cur_sd[k].shape:
            old, cur = sd[k], cur_sd[k]
            if old.ndim != cur.ndim:
                continue
            if all(o <= c for o, c in zip(old.shape, cur.shape)):
                new = cur.clone()
                new[tuple(slice(0, s) for s in old.shape)] = old.to(new.dtype)
                sd[k] = new
                print(f"  resized (grow) {k}: {tuple(old.shape)} → {tuple(new.shape)}")
            elif all(o >= c for o, c in zip(old.shape, cur.shape)):
                sd[k] = old[tuple(slice(0, s) for s in cur.shape)].to(cur.dtype)
                print(f"  resized (truncate) {k}: {tuple(old.shape)} → {tuple(cur.shape)}")
    llm.load_state_dict(sd, strict=False)
    if merge:
        llm = llm.merge_and_unload()
    return llm


def _infer_from_projector(ckpt_dir: Path, base_model_default: str) -> tuple[str, int]:
    """Read (base_model, atomistic_feature_dim) from the Stage-2 projector weight shape; falls back to (default, 256)."""
    try:
        state = torch.load(ckpt_dir / "projector_and_state.pt", map_location="cpu",
                           weights_only=False)
        w = state["projector_state_dict"]["0.weight"]
        out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
    except Exception:
        return base_model_default, 256
    base_model = base_model_default
    # out_dim is the Qwen3 hidden_size; map to the model id.
    for hidden_size, model_id in (
        (1024, "Qwen/Qwen3-0.6B"),
        (2048, "Qwen/Qwen3-1.7B"),
        (2560, "Qwen/Qwen3-4B"),
        (4096, "Qwen/Qwen3-8B"),
        (5120, "Qwen/Qwen3-14B"),
        (6144, "Qwen/Qwen3-32B"),
    ):
        if out_dim == hidden_size:
            base_model = model_id
            break
    return base_model, in_dim


def load_alm(checkpoint=None, stage1_projector=None,
             base_model="Qwen/Qwen3-8B",
             lora_rank=None, lora_alpha=None,
             merge_lora=True, max_atoms=2048 - 256, device=None,
             use_cached_embeddings=True, is_trainable=False,
             num_output_atom_tokens: int = 8,
             attn_implementation: str = "flash_attention_2",
             use_last_prompt_token: bool = False,
             bridge_source: str = 'atoms_tokens',
             qformer_n_context: int = 128,
             qformer_input_atoms: int = 0,
             init_atoms_tokens_from_eos: bool = False,
             atom_bidirectional_attention: bool = False,
             stage2_base=None):
    """Returns (model, tokenizer); pass exactly one of `checkpoint` (Stage 2 dir) or `stage1_projector` (.pt)."""
    if (checkpoint is None) == (stage1_projector is None):
        raise ValueError("pass exactly one of --checkpoint or --stage1_projector")
    if is_trainable and merge_lora:
        raise ValueError("is_trainable=True requires merge_lora=False")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect LLM size and feature dim from the Stage-2 projector shape.
    atomistic_feature_dim = 256
    if checkpoint is not None:
        base_model, atomistic_feature_dim = _infer_from_projector(Path(checkpoint), base_model)

    model = AtomisticLanguageModel(
        llm_name=base_model, atomistic_model_name="orb_v3_direct_20_omat",
        atomistic_feature_dim=atomistic_feature_dim,
        device=device, use_cached_embeddings=use_cached_embeddings, max_atoms=max_atoms,
        num_output_atom_tokens=num_output_atom_tokens,
        attn_implementation=attn_implementation,
        use_last_prompt_token=use_last_prompt_token,
        bridge_source=bridge_source,
        qformer_n_context=qformer_n_context,
        qformer_input_atoms=qformer_input_atoms,
        init_atoms_tokens_from_eos=init_atoms_tokens_from_eos,
        atom_bidirectional_attention=atom_bidirectional_attention,
    )

    if stage1_projector is not None:
        ckpt = torch.load(stage1_projector, map_location=device)
        proj_state = ckpt.get("projector_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.projector.load_state_dict(proj_state)
    elif (Path(checkpoint) / "llm_full_ft" / "qwen3_state_dict.pt").exists():
        # Full Qwen3-8B fine-tune (Stage 4): saved state_dict is merged weights, no PEFT wrap.
        llm_full_path = Path(checkpoint) / "llm_full_ft" / "qwen3_state_dict.pt"
        full_state = torch.load(str(llm_full_path), map_location=device)
        missing, unexpected = model.llm.load_state_dict(full_state, strict=False)
        print(f"[load_alm] full-FT ckpt: loaded {len(full_state)} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")
        state = torch.load(Path(checkpoint) / "projector_and_state.pt", map_location=device)
        model.projector.load_state_dict(state["projector_state_dict"])
    elif (stage2_base or os.environ.get("ALM_STAGE2_BASE")) and \
            (Path(checkpoint) / "lora_adapter").exists():
        # Fresh-r8 bridge two-stage load: the r8 adapter only makes sense atop the Stage-2-merged base.
        if is_trainable:
            raise ValueError("stage2_base two-stage load is eval-only (it merges both "
                             "adapters); drop stage2_base for training resume.")
        s2 = Path(stage2_base or os.environ["ALM_STAGE2_BASE"])
        print(f"[load_alm] fresh-r8 two-stage load: Stage-2 base {s2} → "
              f"bridge adapter {Path(checkpoint)/'lora_adapter'}", flush=True)
        model.llm = _apply_lora_adapter(model.llm, s2 / "lora_adapter", merge=True)
        model.llm = _apply_lora_adapter(model.llm, Path(checkpoint) / "lora_adapter",
                                        lora_rank=lora_rank, lora_alpha=lora_alpha,
                                        merge=True)
        state = torch.load(Path(checkpoint) / "projector_and_state.pt", map_location=device)
        model.projector.load_state_dict(state["projector_state_dict"])
    else:
        adapter_dir = Path(checkpoint) / "lora_adapter"
        with open(adapter_dir / "adapter_config.json") as f:
            saved_cfg = json.load(f)
        r = lora_rank if lora_rank is not None else saved_cfg["r"]
        a = lora_alpha if lora_alpha is not None else saved_cfg["lora_alpha"]
        lora_cfg = LoraConfig(
            r=r, lora_alpha=a, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
        )
        # Manual load (not PEFT from_pretrained) to side-step its strict shape-check; preserves requires_grad.
        model.llm = get_peft_model(model.llm, lora_cfg)
        sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
        # PEFT strips the adapter name during save_pretrained; re-insert "default".
        sd = {k.replace(".lora_A.weight", ".lora_A.default.weight")
               .replace(".lora_B.weight", ".lora_B.default.weight"): v
              for k, v in sd.items()}
        # Vocab-resize migration: copy the matching prefix when num_output_atom_tokens differs.
        cur_sd = model.llm.state_dict()
        for k in list(sd.keys()):
            if k in cur_sd and sd[k].shape != cur_sd[k].shape:
                old, cur = sd[k], cur_sd[k]
                if old.ndim != cur.ndim:
                    continue
                if all(o <= c for o, c in zip(old.shape, cur.shape)):
                    new = cur.clone()
                    new[tuple(slice(0, s) for s in old.shape)] = old.to(new.dtype)
                    sd[k] = new
                    print(f"  resized (grow) {k}: {tuple(old.shape)} → {tuple(new.shape)}")
                elif all(o >= c for o, c in zip(old.shape, cur.shape)):
                    sd[k] = old[tuple(slice(0, s) for s in cur.shape)].to(cur.dtype)
                    print(f"  resized (truncate) {k}: {tuple(old.shape)} → {tuple(cur.shape)}")
        model.llm.load_state_dict(sd, strict=False)
        state = torch.load(Path(checkpoint) / "projector_and_state.pt", map_location=device)
        model.projector.load_state_dict(state["projector_state_dict"])
        if merge_lora:
            model.llm = model.llm.merge_and_unload()

    model = model.to(device)
    if not is_trainable:
        model = model.eval()
    return model, model.tokenizer


def load_base_only(base_model="Qwen/Qwen3-8B", device=None):
    """Vanilla base LLM wrapped as AtomisticLanguageModel for harness compatibility (projector unused at eval)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AtomisticLanguageModel(
        llm_name=base_model, atomistic_model_name="orb_v3_direct_20_omat",
        device=device, use_cached_embeddings=True, max_atoms=2048 - 256,
    )
    return model.to(device).eval(), model.tokenizer
