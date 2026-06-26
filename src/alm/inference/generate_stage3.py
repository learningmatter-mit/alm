"""End-to-end inference: text prompt -> ALM -> bridge -> diffusion sampler -> CIF."""

# Map legacy checkpoint bridge_kind values to current names.
_LEGACY_BRIDGE = {"qformer": "producer-consumer", "qformer_pool": "producer-consumer-pool", "ipadapter": "consumer-only"}
def _norm_bridge(bk):
    return _LEGACY_BRIDGE.get(bk, bk)


import argparse
import os
import random
import sys
from pathlib import Path
from typing import Mapping

import hydra
import hydra.core.global_hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
from loader import load_alm  # noqa: E402
from stage3 import _make_adapter_cfg, _make_lightning_module_cfg  # noqa: E402

from mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH  # noqa: E402
from mattergen.generator import draw_samples_from_sampler  # noqa: E402
from mattergen.scripts.finetune import init_adapter_lightningmodule_from_pretrained  # noqa: E402


# Must byte-match the trainer's SYSTEM_PROMPT: the bridge's [atoms_i] hidden states depend on the full upstream text.
SYSTEM_PROMPT = (
    "You are an expert materials scientist. When asked to generate a crystal "
    "structure, provide a detailed description and conclude with the structure "
    "generation tokens."
)
USER_TEMPLATE = "Generate a crystal structure described as: {prompt}"
# Must match the trainer's ASSISTANT_ANCHOR exactly.
ASSISTANT_ANCHOR = "Structure: "


def build_pl_module(atoms_mapper_path: Path, mattergen_pretrained: str,
                    hidden_dim: int, K: int, mid_dim: int, device,
                    model_path: str | None = None):
    """Build pl_module via the training path, then overlay Stage 3a state; bridge_kind is sniffed from the ckpt.

    model_path: optional LOCAL MatterGen training dir; loads the backbone from there (CSP-mode) instead of the HF name.
    """
    # CPU load avoids pulling multi-GB optimizer_state_dict into GPU before sampling.
    try:
        ckpt = torch.load(atoms_mapper_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(atoms_mapper_path, map_location="cpu")
    bridge_kind = _norm_bridge(ckpt.get("bridge_kind", "pool"))
    cond_adapt_n_heads = ckpt.get("cond_adapt_n_heads", 4)
    cond_adapt_depth = ckpt.get("cond_adapt_depth", 1)
    # Q-Former hparams; rebuild AtomsMapperProducerConsumer so source_len/n_context match training.
    qformer_num_queries = ckpt.get("qformer_num_queries", 16)
    qformer_depth = ckpt.get("qformer_depth", 2)
    qformer_heads = ckpt.get("qformer_heads", 8)
    qformer_context_tokens = ckpt.get("qformer_context_tokens", 128)
    qformer_input_atoms = ckpt.get("qformer_input_atoms", 0)
    # Bridge must rebuild with the same CFG-interface flags it trained with; auto-detect from saved tensors.
    _am_keys = list(ckpt.get("atoms_mapper_state_dict", {}).keys())
    _ts_keys_all = list(ckpt.get("trainable_state_dict", {}).keys())
    bridge_out_norm = ckpt.get("bridge_out_norm",
                               any(k.startswith("out_norm.") for k in _am_keys))
    bridge_learnable_null = ckpt.get("bridge_learnable_null",
        any("alm_embedding.unconditional_embedding_module.embedding" in k for k in _ts_keys_all))
    bridge_noise_gate = ckpt.get("bridge_noise_gate", False)  # no params -> metadata-only
    bridge_tenc_fuse = ckpt.get("bridge_tenc_fuse",
        any("tenc_fuse" in k or "tenc_encoding" in k for k in _ts_keys_all))

    # Auto-detect trained discrete cond_fields by scanning trainable_state_dict (CLI flags aren't persisted).
    ts_keys = list(ckpt.get("trainable_state_dict", {}).keys())
    use_chemical_system_cond = any(
        k.startswith("gemnet.cond_adapt_layers.chemical_system.")
        or "property_embeddings_adapt.chemical_system" in k
        for k in ts_keys
    )
    use_space_group_cond = any(
        k.startswith("gemnet.cond_adapt_layers.space_group.")
        or "property_embeddings_adapt.space_group" in k
        for k in ts_keys
    )
    # If alm_embedding cond_field unused, the no-LLM-bridge control branch in _make_adapter_cfg fires.
    use_alm_embedding_cond = any(
        k.startswith("gemnet.cond_adapt_layers.alm_embedding.")
        or "property_embeddings_adapt.alm_embedding" in k
        for k in ts_keys
    )
    # Auto-detect task_direction cond_field so its ckpt weights load (else load_state_dict rejects them).
    use_task_direction_cond = any(
        k.startswith("gemnet.cond_adapt_layers.task_direction.")
        or "property_embeddings_adapt.task_direction" in k
        for k in ts_keys
    )
    # Pre-cond-field-era ckpts have no trainable_state_dict.
    if not ts_keys:
        use_alm_embedding_cond = True

    adapter_cfg = _make_adapter_cfg(
        mattergen_pretrained, full_finetuning=False,
        hidden_dim=hidden_dim, K=K, mid_dim=mid_dim,
        bridge_kind=bridge_kind, cond_adapt_n_heads=cond_adapt_n_heads,
        use_task_direction_cond=use_task_direction_cond,
        use_alm_embedding_cond=use_alm_embedding_cond,
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
    print(f"[gen] bridge fix flags: out_norm={bridge_out_norm} "
          f"learnable_null={bridge_learnable_null} noise_gate={bridge_noise_gate} "
          f"tenc_fuse={bridge_tenc_fuse}", flush=True)
    if model_path is not None:
        print(f"[gen] backbone from LOCAL model_path={model_path} "
              f"(CSP-mode if trained so; atoms observed)")
    lm_cfg = _make_lightning_module_cfg(lr=1e-4)
    pl_module, _ = init_adapter_lightningmodule_from_pretrained(adapter_cfg, lm_cfg)
    pl_module = pl_module.to(device).eval()
    print(f"[gen] bridge_kind={bridge_kind!r}, "
          f"alm_emb={use_alm_embedding_cond}, "
          f"chem_sys={use_chemical_system_cond}, "
          f"space_grp={use_space_group_cond} "
          f"(cond_adapt_n_heads={cond_adapt_n_heads})")
    if bridge_kind in ("producer-consumer", "producer-consumer-pool"):
        print(f"[gen] qformer: num_queries={qformer_num_queries} depth={qformer_depth} "
              f"heads={qformer_heads} n_context={qformer_context_tokens + qformer_input_atoms} "
              f"source_len={qformer_context_tokens + qformer_input_atoms + K} "
              f"input_atoms={qformer_input_atoms} pool={'mean' if bridge_kind=='producer-consumer-pool' else 'none'}")

    # Overlay Stage 3a's AtomsMapper + cond_adapt/mixin for all active cond_fields.
    diffusion_model = pl_module.diffusion_module.model
    if use_alm_embedding_cond and "atoms_mapper_state_dict" in ckpt:
        atoms_mapper = (
            diffusion_model.property_embeddings_adapt["alm_embedding"]
            .conditional_embedding_module
        )
        # strict=False so pre-dir_proto qformer ckpts load (dir_proto is train-only); surface other mismatches.
        _ld = atoms_mapper.load_state_dict(ckpt["atoms_mapper_state_dict"], strict=False)
        _miss = [k for k in _ld.missing_keys if "dir_proto" not in k]
        if _miss or _ld.unexpected_keys:
            print(f"[load] AtomsMapper non-dir_proto mismatch: missing={_miss} "
                  f"unexpected={_ld.unexpected_keys}", flush=True)

    if ckpt.get("mattergen_full_state_dict") is not None:
        # Full MatterGen-FT ckpt: overlay trained backbone + adapter, else eval runs on vanilla pretrained.
        cur = diffusion_model.state_dict()
        cur.update(ckpt["mattergen_full_state_dict"])
        _ld = diffusion_model.load_state_dict(cur, strict=False)
        print(f"[gen] overlaid FULL MatterGen backbone (full-FT): "
              f"{len(ckpt['mattergen_full_state_dict'])} tensors "
              f"(missing={len(_ld.missing_keys)}, unexpected={len(_ld.unexpected_keys)})")
    elif "trainable_state_dict" in ckpt:
        cur = diffusion_model.state_dict()
        cur.update(ckpt["trainable_state_dict"])
        diffusion_model.load_state_dict(cur, strict=True)
        print(f"[gen] overlaid {len(ckpt['trainable_state_dict'])} trainable tensors "
              f"(cond_adapt/mixin)")

    return pl_module


def get_alm_embedding(
    alm, tokenizer, prompt: str, device,
    atom_embed: torch.Tensor | None = None,
    wrap_user_template: bool = True,
    cot_tokens: int = 0,
    llm_temperature: float = 1.0,
    cot_top_p: float = 0.9,
    cot_seed: int | None = None,
    json_counts: dict | None = None,
    task_direction: float | None = None,
    atoms_before_json: bool = False,
) -> torch.Tensor:
    """Run ALM forward on the prompt, return the flattened (K*4096,) [atoms_i] hidden states."""
    output_atoms_str = "".join(alm.output_atom_tokens)
    user_text = USER_TEMPLATE.format(prompt=prompt) if wrap_user_template else prompt
    # lm_loss_json training prepended a committed-composition JSON prefix before the anchor; byte-match it here.
    anchor = ASSISTANT_ANCHOR
    _prefix = None
    if json_counts is not None:
        import json as _json
        _prefix = _json.dumps({"counts": {str(k): int(v) for k, v in json_counts.items()}},
                              separators=(",", ":"))
        if not atoms_before_json:
            anchor = _prefix + "\n" + ASSISTANT_ANCHOR
    # atoms_before_json: atoms precede the JSON; must match the trainer's layout.
    if atoms_before_json and _prefix is not None:
        assistant_content = anchor + output_atoms_str + "\n" + _prefix
    else:
        assistant_content = anchor + output_atoms_str
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_content},
    ]
    full_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        enable_thinking=False, truncation=True, max_length=2048,
    )
    ids = torch.tensor(full_ids, dtype=torch.long, device=device)

    # Optional CoT: splice cot_tokens sampled tokens between the anchor and the [atoms_i] block.
    if cot_tokens > 0:
        K = len(alm.output_atom_token_ids)
        atoms_id_set = set(alm.output_atom_token_ids)
        atoms_mask = torch.tensor([int(t.item()) in atoms_id_set for t in ids],
                                  dtype=torch.bool)
        if int(atoms_mask.sum().item()) < K:
            raise RuntimeError(
                f"pre-CoT prompt missing [atoms_i] tokens "
                f"({int(atoms_mask.sum().item())}/{K})"
            )
        first_atom_pos = int(atoms_mask.nonzero(as_tuple=True)[0][0].item())
        prefix_ids = ids[:first_atom_pos]                 # ... "Structure: "
        atoms_ids = ids[first_atom_pos:first_atom_pos + K]  # [atoms_0..7]
        suffix_ids = ids[first_atom_pos + K:]              # <|im_end|> etc.

        # Suppress <|im_end|> mid-CoT so the turn can't end before the atoms tokens.
        bad_words_ids = None
        try:
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end_id, int) and im_end_id >= 0:
                bad_words_ids = [[int(im_end_id)]]
        except Exception:
            bad_words_ids = None

        gen_kwargs = dict(
            max_new_tokens=int(cot_tokens),
            min_new_tokens=int(cot_tokens),  # force exactly cot_tokens tokens
            do_sample=True,
            temperature=float(llm_temperature),
            top_p=float(cot_top_p),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )
        if bad_words_ids is not None:
            gen_kwargs["bad_words_ids"] = bad_words_ids
        gen_input = prefix_ids.unsqueeze(0).to(device)
        gen_attn = torch.ones_like(gen_input)
        # HF generate() takes no `generator` kwarg; seed the global PRNG that do_sample reads.
        if cot_seed is not None:
            torch.manual_seed(int(cot_seed) & 0x7FFFFFFF)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(cot_seed) & 0x7FFFFFFF)
        with torch.no_grad():
            out = alm.llm.generate(
                input_ids=gen_input, attention_mask=gen_attn, **gen_kwargs,
            )
        sampled = out[0, gen_input.shape[1]:].to(device=device, dtype=ids.dtype)
        # Trim at first <|im_end|> in case bad_words_ids didn't catch it (best-effort processor).
        if bad_words_ids is not None:
            im_end_id = int(bad_words_ids[0][0])
            hit = (sampled == im_end_id).nonzero(as_tuple=True)[0]
            if hit.numel() > 0:
                sampled = sampled[: int(hit[0].item())]
        ids = torch.cat([prefix_ids.to(device), sampled, atoms_ids.to(device),
                         suffix_ids.to(device)], dim=0)

    attn = torch.ones_like(ids)

    K = len(alm.output_atom_token_ids)
    atoms_id_set = set(alm.output_atom_token_ids)
    n_present = sum(int(t.item()) in atoms_id_set for t in ids)
    if n_present != K:
        raise RuntimeError(f"prompt was truncated: only {n_present}/{K} [atoms_i] tokens present")

    if atom_embed is None or atom_embed.shape[0] == 0:
        # Auto-detect from the projector; hardcoding 256 crashes non-orb ALMs (uma=128, pet-mad=640/1280).
        feat_dim = int(alm.projector[0].in_features)
        embeds = [torch.zeros(0, feat_dim, dtype=torch.float32, device=device)]
    else:
        embeds = [atom_embed.to(device).float()]
    with torch.no_grad():
        hidden = alm.extract_atoms_hidden_states(
            [ids], [attn], atom_embeds=embeds,
        )  # (1, K, 4096)
    v = hidden.flatten().float()  # (K*4096,)
    if task_direction is not None:
        # Reproduce the trained hand-set direction override (shared helper, no train/eval drift).
        from direction_code import apply_handset_direction
        v = apply_handset_direction(v, hidden.shape[1], task_direction).flatten()
    return v


def build_sampler_and_loader(pl_module, batch_size: int, num_batches: int,
                             num_atoms_distribution: str, alm_emb_vec: torch.Tensor,
                             diffusion_guidance_factor: float,
                             constrain_n_atoms_to_multiple_of: int = 0,
                             constrain_n_atoms_exact: int = 0,
                             min_n_atoms: int = 3,
                             diffusion_snr: float | None = None,
                             diffusion_steps: int | None = None,
                             chemical_system: str | None = None,
                             space_group: int | None = None,
                             composition_count=None):
    """Mirror CrystalGenerator.generate's hydra-config pipeline with a runtime alm_embedding vector.

    min_n_atoms (default 3) drops degenerate cells: n_atoms<=2 yield zero-edge graphs that crash
    GemNet with a cuda/cpu device mismatch; ALEX_MP_20 puts ~0.3% mass there.
    constrain_n_atoms_to_multiple_of restricts the distribution to multiples of sum(target_counts).
    """
    # Drop degenerate small cells before any constraint; the exact-N branch overrides this entirely.
    if min_n_atoms > 1 and constrain_n_atoms_exact <= 0:
        from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
        base = NUM_ATOMS_DISTRIBUTIONS[num_atoms_distribution]
        kept = {n: p for n, p in base.items() if n >= min_n_atoms and p > 0.0}
        if not kept:
            raise ValueError(
                f"min_n_atoms={min_n_atoms} drained {num_atoms_distribution} of all support."
            )
        if len(kept) != sum(1 for p in base.values() if p > 0):
            total = sum(kept.values())
            renorm = {n: p / total for n, p in kept.items()}
            filtered_name = f"{num_atoms_distribution}__min{min_n_atoms}"
            NUM_ATOMS_DISTRIBUTIONS[filtered_name] = renorm
            dropped = {n: p for n, p in base.items() if n < min_n_atoms and p > 0.0}
            print(f"[n_atoms] floor={min_n_atoms}; dropped {dropped} from "
                  f"{num_atoms_distribution} → {filtered_name}", flush=True)
            num_atoms_distribution = filtered_name

    if constrain_n_atoms_exact > 0:
        # Strict N_p = sum(target_counts); single-atom-count distribution (no supercells).
        from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
        n_exact = int(constrain_n_atoms_exact)
        new_name = f"{num_atoms_distribution}__exact_{n_exact}"
        NUM_ATOMS_DISTRIBUTIONS[new_name] = {n_exact: 1.0}
        print(f"[fk] N_p constraint: STRICT N_p = {n_exact} (no multiples)", flush=True)
        num_atoms_distribution = new_name
    elif constrain_n_atoms_to_multiple_of > 0:
        from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
        base = NUM_ATOMS_DISTRIBUTIONS[num_atoms_distribution]
        m = int(constrain_n_atoms_to_multiple_of)
        kept = {n: p for n, p in base.items() if n % m == 0 and p > 0.0}
        if not kept:
            raise ValueError(
                f"No N_p values in {num_atoms_distribution} are multiples of {m}; "
                f"distribution would be empty. Either lower target_counts or pick "
                f"a num_atoms_distribution with appropriate support."
            )
        total = sum(kept.values())
        renorm = {n: p / total for n, p in kept.items()}
        new_name = f"{num_atoms_distribution}__mult_{m}"
        NUM_ATOMS_DISTRIBUTIONS[new_name] = renorm
        kept_str = ", ".join(f"{n}:{round(p, 3)}" for n, p in sorted(renorm.items()))
        print(f"[fk] N_p constraint: filtered {num_atoms_distribution} → multiples of {m}, "
              f"support = {{{kept_str}}}", flush=True)
        num_atoms_distribution = new_name

    overrides = [
        f"+condition_loader_partial.num_atoms_distribution={num_atoms_distribution}",
        f"+condition_loader_partial.batch_size={batch_size}",
        f"+condition_loader_partial.num_samples={num_batches * batch_size}",
        f"sampler_partial.guidance_scale={diffusion_guidance_factor}",
    ]
    # Temperature analog: scale Langevin-corrector SNR off defaults (pos=0.4, cell=0.2). Higher = cooler.
    if diffusion_snr is not None:
        snr_pos = 0.4 * diffusion_snr
        snr_cell = 0.2 * diffusion_snr
        overrides.append(f"sampler_partial.corrector_partials.pos.snr={snr_pos}")
        overrides.append(f"sampler_partial.corrector_partials.cell.snr={snr_cell}")
    if diffusion_steps is not None:
        # Override predictor step count (default 1000); 'N' is the hydra key.
        overrides.append(f"sampler_partial.N={int(diffusion_steps)}")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(os.path.abspath(str(DEFAULT_SAMPLING_CONFIG_PATH)),
                                     version_base="1.1"):
        sampling_cfg = hydra.compose(config_name="default", overrides=overrides)

    # Stamp alm_embedding as (1, K*4096) so pyg cat -> (B, K*4096); a flat (K*4096,) would cat wrong.
    # Discrete cond fields must be stamped here too; properties_to_condition_on only feeds the trained-fields assert.
    condition_loader_partial = hydra.utils.instantiate(sampling_cfg.condition_loader_partial)
    props_to_stamp: dict = {}
    if alm_emb_vec is not None:
        props_to_stamp["alm_embedding"] = alm_emb_vec.detach().cpu().unsqueeze(0)
    if chemical_system is not None:
        # Dashed sorted string format ("Fe-O-Si").
        props_to_stamp["chemical_system"] = chemical_system
    if space_group is not None:
        # sg in [1,230]; encode missing as NaN so MG's dropout selector routes to unconditional.
        sg_int = int(space_group)
        if 1 <= sg_int <= 230:
            props_to_stamp["space_group"] = torch.tensor([float(sg_int)], dtype=torch.float32)
        else:
            props_to_stamp["space_group"] = torch.tensor([float("nan")], dtype=torch.float32)
    if composition_count is not None:
        # Per-element count vector (1, 101); MUST be a tensor (SetProperty only tensor-izes scalars).
        cc = composition_count
        if not torch.is_tensor(cc):
            cc = torch.as_tensor(cc, dtype=torch.float32)
        cc = cc.to(torch.float32).reshape(1, -1)
        props_to_stamp["composition_count"] = cc
    condition_loader = condition_loader_partial(properties=props_to_stamp)

    sampler_partial = hydra.utils.instantiate(sampling_cfg.sampler_partial)
    sampler = sampler_partial(pl_module=pl_module)
    return sampler, condition_loader


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed PRNGs so the CLI can retry stochastic sampler crashes with bumped seeds.
    if getattr(args, "diffusion_seed", None) is not None:
        s = int(args.diffusion_seed) & 0x7FFFFFFF
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        np.random.seed(s)
        random.seed(s)
        print(f"[gen] seeded torch/cuda/numpy/random with diffusion_seed={s}",
              flush=True)

    # ── 1. Load ALM (LoRA + projector merged) ────────────
    # Pre-read atoms_mapper.pt for ALM bridge metadata so non-default K / last_k_prompt / EOS-init load right.
    try:
        _meta_ckpt = torch.load(args.atoms_mapper, map_location="cpu", weights_only=False)
    except TypeError:
        _meta_ckpt = torch.load(args.atoms_mapper, map_location="cpu")
    _num_output_atom_tokens = _meta_ckpt.get("num_output_atom_tokens", 8)
    _use_last_prompt_token = _meta_ckpt.get("use_last_prompt_token", False)
    _bridge_source = _meta_ckpt.get("bridge_source", "atoms_tokens")
    _init_atoms_tokens_from_eos = _meta_ckpt.get("init_atoms_tokens_from_eos", False)
    # Q-Former: force context_plus_atoms extraction so N context + K [atoms_i] match training.
    _bridge_kind = _norm_bridge(_meta_ckpt.get("bridge_kind", "pool"))
    _qformer_n_context = _meta_ckpt.get("qformer_context_tokens", 128)
    _qformer_input_atoms = _meta_ckpt.get("qformer_input_atoms", 0)
    if _bridge_kind in ("producer-consumer", "producer-consumer-pool"):
        _bridge_source = "context_plus_atoms"
    del _meta_ckpt
    print(f"[gen] loading ALM from {args.alm_checkpoint} "
          f"K={_num_output_atom_tokens} bridge_source={_bridge_source!r} "
          f"last_k_prompt={_use_last_prompt_token} eos_init={_init_atoms_tokens_from_eos}"
          + (f" qformer_n_context={_qformer_n_context}" if _bridge_kind == "producer-consumer" else ""),
          flush=True)
    alm, tokenizer = load_alm(
        checkpoint=args.alm_checkpoint,
        merge_lora=True,
        use_cached_embeddings=True,
        device=device,
        num_output_atom_tokens=_num_output_atom_tokens,
        use_last_prompt_token=_use_last_prompt_token,
        bridge_source=_bridge_source,
        qformer_n_context=_qformer_n_context,
        qformer_input_atoms=_qformer_input_atoms,
        init_atoms_tokens_from_eos=_init_atoms_tokens_from_eos,
    )
    alm.eval()
    K = len(alm.output_atom_token_ids)
    print(f"[gen] K={K}, hidden_dim={alm.llm_hidden_dim}")

    # ── 2. Build MatterGen pl_module + overlay trained AtomsMapper ─────────
    print(f"[gen] building MatterGen adapter ({args.mattergen_pretrained}) ...", flush=True)
    print(f"[gen] overlaying AtomsMapper + cond_adapt/mixin from {args.atoms_mapper}")
    # Auto-detect mid_dim from the ckpt's AtomsMapper shape (irrelevant for no-LLM-bridge ckpts).
    _ckpt_for_mid = torch.load(args.atoms_mapper, map_location="cpu")
    _am_sd_for_mid = _ckpt_for_mid.get("atoms_mapper_state_dict") or {}
    if "proj.0.weight" in _am_sd_for_mid:
        _mid_dim = int(_am_sd_for_mid["proj.0.weight"].shape[0])
        print(f"[gen] detected AtomsMapper mid_dim={_mid_dim} from ckpt")
    else:
        _mid_dim = 2048
        print(f"[gen] no AtomsMapper in ckpt (no-LLM-bridge); placeholder mid_dim={_mid_dim}")
    pl_module = build_pl_module(
        Path(args.atoms_mapper), args.mattergen_pretrained,
        hidden_dim=alm.llm_hidden_dim, K=K, mid_dim=_mid_dim, device=device,
    )

    # ── 3. ALM forward → conditioning vector ───────────────────────────────
    # Skip the LLM forward for no-LLM-bridge control ckpts (no alm_embedding path).
    _has_alm_cond = "alm_embedding" in pl_module.diffusion_module.model.cond_fields_model_was_trained_on
    if _has_alm_cond:
        print(f"[gen] computing alm_embedding for prompt: {args.prompt!r}", flush=True)
        _cot_seed = (
            int(args.cot_seed) if getattr(args, "cot_seed", None) is not None
            else (int(args.diffusion_seed) if args.diffusion_seed is not None else 1337)
        )
        _json_counts = None
        if args.fk_target_counts or args.fk_target_counts_from_prompt_json:
            try:
                from ase.data import chemical_symbols as _chemical_symbols
                _z_counts = _parse_target_counts(
                    args.fk_target_counts,
                    from_prompt_json=args.fk_target_counts_from_prompt_json,
                )
                if _z_counts:
                    _json_counts = {
                        _chemical_symbols[int(z)]: int(n)
                        for z, n in _z_counts.items()
                    }
            except Exception as exc:  # noqa: BLE001
                print(f"[gen] WARNING: could not build json_counts prefix from "
                      f"target counts ({type(exc).__name__}: {exc}); "
                      f"continuing without JSON-counts prefix", flush=True)
        _alm_mapper = (
            pl_module.diffusion_module.model.property_embeddings_adapt["alm_embedding"]
            .conditional_embedding_module
        )
        if (
            type(_alm_mapper).__name__ == "AtomsMapperProducerConsumer"
            and _json_counts is None
            and float(args.diffusion_guidance_factor) != 0.0
        ):
            print(
                "[gen] WARNING: QFormer alm_embedding guidance is active but no "
                "committed JSON counts prefix was supplied. lm_loss_json/QFormer "
                "training conditioned the producer on that prefix, so this is an "
                "open-ended context-shift run unless that is intentional.",
                flush=True,
            )
        alm_emb = get_alm_embedding(
            alm, tokenizer, args.prompt, device,
            cot_tokens=int(getattr(args, "cot_tokens", 0)),
            llm_temperature=float(getattr(args, "llm_temperature", 1.0)),
            cot_top_p=float(getattr(args, "cot_top_p", 0.9)),
            cot_seed=_cot_seed if int(getattr(args, "cot_tokens", 0)) > 0 else None,
            json_counts=_json_counts,
        )  # (in_dim,)
        print(f"[gen] alm_embedding shape: {tuple(alm_emb.shape)}, "
              f"mean={alm_emb.mean().item():+.3f}, std={alm_emb.std().item():.3f}, "
              f"L2={alm_emb.float().norm(p=2).item():.4f}")
    else:
        print(f"[gen] ckpt was not trained with alm_embedding (no-LLM-bridge "
              f"control); skipping ALM forward, sampling unconditionally on LLM side.",
              flush=True)
        alm_emb = None

    del alm
    torch.cuda.empty_cache()

    # ── 4. Build sampler + condition_loader ────────────────────────────────
    # Under FK the sampler's batch dim IS the particle dim.
    fk_active = getattr(args, "fk_n_particles", 0) > 0
    sampler_batch_size = args.fk_n_particles if fk_active else args.batch_size
    sampler_num_batches = 1 if fk_active else args.num_batches

    # Parse target_counts early so it threads into build_sampler_and_loader, the FK reward, and init-types.
    init_types_active = bool(getattr(args, "init_types_at_target", False))
    lock_types_active = bool(getattr(args, "lock_types_via_mask", False))
    bias_types_active = bool(getattr(args, "bias_types_via_score", False))
    if lock_types_active and bias_types_active:
        raise ValueError(
            "--lock_types_via_mask and --bias_types_via_score are mutually exclusive. "
            "--bias_types_via_score is the soft version of --lock_types_via_mask; pick one."
        )
    if (lock_types_active or bias_types_active) and init_types_active:
        raise ValueError(
            "--init_types_at_target is superseded by --lock_types_via_mask "
            "and --bias_types_via_score. Don't combine."
        )
    if lock_types_active and bool(getattr(args, "fk_enforce_target_counts", False)):
        # Post-hoc Z-override is redundant when types are clamped throughout.
        print("[gen] WARNING: --lock_types_via_mask superseding "
              "--fk_enforce_target_counts (post-hoc Hungarian is redundant under hard "
              "lock). Disabling enforce_target_counts for this run.", flush=True)
        args.fk_enforce_target_counts = False
    need_target_counts = (
        fk_active or init_types_active or lock_types_active or bias_types_active
    )
    fk_target_counts = None
    if need_target_counts:
        fk_target_counts = _parse_target_counts(
            args.fk_target_counts,
            from_prompt_json=args.fk_target_counts_from_prompt_json,
        )
    constrain_mult = 0
    constrain_exact = 0
    if need_target_counts and (args.fk_constrain_n_atoms_to_target_multiple
                               or args.fk_n_atoms_exact_sum_target):
        if not fk_target_counts:
            raise ValueError(
                "N_p constraint flags require --fk_target_counts "
                "or --fk_target_counts_from_prompt_json"
            )
        if args.fk_n_atoms_exact_sum_target:
            # Exact wins over multiple.
            constrain_exact = int(sum(fk_target_counts.values()))
        elif args.fk_constrain_n_atoms_to_target_multiple:
            constrain_mult = int(sum(fk_target_counts.values()))
    if init_types_active and not args.fk_n_atoms_exact_sum_target:
        raise ValueError(
            "--init_types_at_target requires --fk_n_atoms_exact_sum_target so each "
            "particle has exactly sum(target_counts) atoms to populate."
        )
    if init_types_active and not fk_target_counts:
        raise ValueError(
            "--init_types_at_target requires --fk_target_counts (or "
            "--fk_target_counts_from_prompt_json)."
        )
    if lock_types_active and not args.fk_n_atoms_exact_sum_target:
        raise ValueError(
            "--lock_types_via_mask requires --fk_n_atoms_exact_sum_target so each "
            "particle has exactly sum(target_counts) atoms to populate."
        )
    if lock_types_active and not fk_target_counts:
        raise ValueError(
            "--lock_types_via_mask requires --fk_target_counts or "
            "--fk_target_counts_from_prompt_json."
        )
    if bias_types_active and not fk_target_counts:
        raise ValueError(
            "--bias_types_via_score requires --fk_target_counts or "
            "--fk_target_counts_from_prompt_json."
        )

    print(f"[gen] building sampler (batch_size={sampler_batch_size}, "
          f"num_batches={sampler_num_batches}, "
          f"guidance={args.diffusion_guidance_factor}, "
          f"num_atoms_distribution={args.num_atoms_distribution}, "
          f"fk={'on' if fk_active else 'off'}"
          + (f", N_p={constrain_exact}=sum(target)" if constrain_exact else "")
          + (f", N_p%{constrain_mult}=0" if constrain_mult else "")
          + (f", enforce_target_counts={args.fk_enforce_target_counts}" if fk_active else "")
          + ") ...", flush=True)
    _diffusion_steps_env = os.environ.get("DIFFUSION_STEPS", "")
    _diffusion_steps = int(_diffusion_steps_env) if _diffusion_steps_env else None
    sampler, condition_loader = build_sampler_and_loader(
        pl_module=pl_module,
        batch_size=sampler_batch_size,
        num_batches=sampler_num_batches,
        num_atoms_distribution=args.num_atoms_distribution,
        alm_emb_vec=alm_emb,
        diffusion_guidance_factor=args.diffusion_guidance_factor,
        diffusion_snr=args.diffusion_snr,
        diffusion_steps=_diffusion_steps,
        constrain_n_atoms_to_multiple_of=constrain_mult,
        constrain_n_atoms_exact=constrain_exact,
    )

    # ── 4b. Optional element masking ──────────────────────────────────────
    if args.allowed_elements:
        elements = [s.strip() for s in args.allowed_elements.split(",") if s.strip()]
        _ensure_element_mask_installed(pl_module)
        pl_module._element_mask_state.allowed_z = _z_set_from_elements(elements)
        print(f"[gen] element mask ON: allowed Z = {sorted(pl_module._element_mask_state.allowed_z)} "
              f"(elements: {elements})", flush=True)

    # ── 4c. Optional FK steering ──────────────────────────────────────────
    if fk_active:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
        from fk_rewards import parse_rewards as _fk_parse_rewards  # noqa: E402

        allowed_elements = (
            [s.strip() for s in args.allowed_elements.split(",") if s.strip()]
            if args.allowed_elements else None
        )
        reward = _fk_parse_rewards(
            args.fk_rewards,
            allowed_elements=allowed_elements,
            target_counts=fk_target_counts,
        )
        _ensure_fk_hook_installed(pl_module)
        st = pl_module._fk_state
        st.enabled = True
        st.reward = reward
        st.target_counts = fk_target_counts
        st.enforce_target_counts = bool(args.fk_enforce_target_counts)
        st.n_particles = args.fk_n_particles
        st.resample_every = args.fk_resample_every
        st.t_start_frac = args.fk_t_start_frac
        st.lambda_ = args.fk_lambda
        st.potential = args.fk_potential
        st.ess_threshold_frac = args.fk_ess_threshold_frac
        st.keep_top_k = args.fk_keep_top_k
        st.log_w_clip = args.fk_log_w_clip
        st.stratify_resample_by_n_atoms = args.fk_stratify_resample_by_n_atoms
        _install_fk_on_sampler(sampler, st)
        print(f"[fk] hook installed: N={st.n_particles}, k={st.resample_every}, "
              f"t_start_frac={st.t_start_frac}, λ={st.lambda_}, "
              f"potential={st.potential}, ess_thr={st.ess_threshold_frac}, "
              f"clip=±{st.log_w_clip}, rewards={args.fk_rewards}, "
              f"stratify_n_atoms={st.stratify_resample_by_n_atoms}", flush=True)

    # ── 4d. Optional init-types-at-target ─────────────────────────────────
    if init_types_active:
        _ensure_init_types_state(pl_module)
        it_state = pl_module._init_types_state
        it_state.enabled = True
        it_state.target_counts = fk_target_counts
        # Fixed seed for the single-prompt path; vary across runs for different permutations.
        it_state.seed = 1337
        _install_init_types_on_sampler(sampler, it_state)
        print(f"[init_types] enabled: target_counts={fk_target_counts}, "
              f"seed={it_state.seed}", flush=True)

    # ── 4e. Optional lock-types-via-mask ──────────────────────────────────
    if lock_types_active:
        _ensure_lock_types_state(pl_module)
        lk_state = pl_module._lock_types_state
        lk_state.enabled = True
        lk_state.target_counts = fk_target_counts
        lk_state.seed = 1337
        _install_lock_types_on_sampler(sampler, lk_state)
        print(f"[lock_types] enabled: target_counts={fk_target_counts}, "
              f"seed={lk_state.seed}", flush=True)

    # ── 4f. Optional bias-types-via-score ─────────────────────────────────
    if bias_types_active:
        _ensure_type_biasing_installed(
            pl_module, fk_target_counts,
            t_start=args.type_bias_t_start, t_end=args.type_bias_t_end,
        )
        pl_module._type_bias_state.enabled = True
        print(f"[type_bias] enabled: target_counts={fk_target_counts}, "
              f"t_start={args.type_bias_t_start}, t_end={args.type_bias_t_end}",
              flush=True)

    # ── 5. Sample ──────────────────────────────────────────────────────────
    print(f"[gen] sampling {args.num_batches * args.batch_size} structures → {out_dir}",
          flush=True)
    # cfg satisfies draw_samples_from_sampler's not-None assert; properties_to_condition_on only feeds
    # the trained-fields assert (actual conditioning is stamped into condition_loader).
    _props_for_assert = {}
    if alm_emb is not None:
        _props_for_assert["alm_embedding"] = alm_emb.detach().cpu()
    structures = draw_samples_from_sampler(
        sampler=sampler,
        condition_loader=condition_loader,
        properties_to_condition_on=_props_for_assert,
        output_path=out_dir,
        cfg=OmegaConf.create({}),
        record_trajectories=args.record_trajectories,
    )

    # Save prompt + alm_embedding alongside the CIFs for reproducibility.
    _cli_meta = {
        "prompt": args.prompt,
        "alm_checkpoint": str(args.alm_checkpoint),
        "atoms_mapper": str(args.atoms_mapper),
        "mattergen_pretrained": args.mattergen_pretrained,
        "diffusion_guidance_factor": args.diffusion_guidance_factor,
        "fk_target_counts": fk_target_counts if fk_active else None,
    }
    if alm_emb is not None:
        _cli_meta["alm_embedding"] = alm_emb.detach().cpu()
    torch.save(_cli_meta, out_dir / "stage3a_inference_meta.pt")

    if fk_active and pl_module._fk_state.reward_trajectory:
        st = pl_module._fk_state
        traj = {k: torch.stack(v) for k, v in st.reward_trajectory.items()}  # (n_steps, N)
        torch.save({
            "np_per_particle_final": st.np_per_particle,
            "trajectory": traj,
            "resample_log": st.resample_log,
            "n_fk_steps": st.n_fk_steps,
            "n_resamples": st.n_resamples,
            "n_clip_hits": st.n_clip_hits,
            "fk_t_start_frac": st.t_start_frac,
            "fk_lambda": st.lambda_,
            "fk_resample_every": st.resample_every,
            "fk_target_counts": fk_target_counts,
            "fk_rewards_spec": args.fk_rewards,
            "stratify_resample_by_n_atoms": st.stratify_resample_by_n_atoms,
        }, out_dir / "fk_trajectory.pt")
        print(f"[fk] wrote {out_dir / 'fk_trajectory.pt'} "
              f"(components: {sorted(traj.keys())}, "
              f"shape per component: {traj[next(iter(traj))].shape.numel()} elems)",
              flush=True)

    print(f"\n[gen] DONE — {len(structures)} structures generated. Outputs in {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Element-mask plumbing (CSP-style hard composition constraint at sample time)
# ─────────────────────────────────────────────────────────────────────────────

class _ElementMaskState:
    """Per-prompt element mask, read by the masked score_fn each denoising step."""
    def __init__(self):
        self.allowed_z: set[int] | None = None  # allowed atomic numbers (Z=1..100)


def _ensure_element_mask_installed(pl_module):
    """Idempotently wrap score_fn to mask atomic_numbers logits to allowed_z (no-op when None).

    Also masks the MASK absorbing-state token (index 100) since x_0 must be a real Z=1..100.
    CFG-safe: the same mask is added to conditional and unconditional logits and commutes with the combine.
    """
    if getattr(pl_module, "_element_mask_installed", False):
        return
    import torch as _torch
    state = _ElementMaskState()
    pl_module._element_mask_state = state
    diffusion_module = pl_module.diffusion_module
    orig_score_fn = diffusion_module.score_fn

    def masked_score_fn(x, t):
        output = orig_score_fn(x, t)
        if state.allowed_z is None:
            return output
        try:
            logits = output["atomic_numbers"]
        except (KeyError, TypeError):
            return output
        if not _torch.is_tensor(logits) or logits.ndim != 2:
            return output
        n_classes = logits.shape[-1]
        # index 0 <-> Z=1 ... index 99 <-> Z=100; index 100 (MASK token) is masked too.
        mask = _torch.full((n_classes,), -1.0e9, device=logits.device, dtype=logits.dtype)
        for z in state.allowed_z:
            i = int(z) - 1
            if 0 <= i < min(n_classes, 100):
                mask[i] = 0.0
        return output.replace(atomic_numbers=logits + mask)

    diffusion_module.score_fn = masked_score_fn
    pl_module._element_mask_installed = True


def _z_set_from_elements(elements):
    """Convert an iterable of element symbols (['Cu', 'Ni']) to {Z, ...} via ASE."""
    from ase.data import atomic_numbers as _ase_z
    out = set()
    for sym in elements:
        s = sym.strip()
        if s in _ase_z:
            out.add(int(_ase_z[s]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Peaked atomic_numbers init at sampler t=T
# ─────────────────────────────────────────────────────────────────────────────
# Override atomic_numbers right after _sample_prior with a permuted target multiset so denoising
# co-evolves positions with real types (vs the absorbing-state MASK prior). May be off-distribution.

class _InitTypesState:
    def __init__(self):
        self.enabled: bool = False
        self.target_counts = None        # dict[int Z -> int count]
        self.seed: int | None = None


def _ensure_init_types_state(pl_module):
    if not getattr(pl_module, "_init_types_state", None):
        pl_module._init_types_state = _InitTypesState()


def _override_atomic_numbers_to_target(batch, target_counts: dict, seed=None):
    """Replace batch.atomic_numbers with a per-particle permuted target multiset; needs N_p == sum(target_counts)."""
    device = batch["atomic_numbers"].device
    multiset = []
    for z, n in target_counts.items():
        multiset.extend([int(z)] * int(n))
    multiset_t = torch.tensor(multiset, dtype=batch["atomic_numbers"].dtype, device=device)
    expected_per = len(multiset)

    # ChemGraph's `batch` field maps each atomic_numbers/pos row to its particle slot.
    if hasattr(batch, "batch") and batch.batch is not None:
        batch_idx = batch.batch
    else:
        batch_idx = batch.get_batch_idx("atomic_numbers")
    n_particles = int(batch_idx.max().item()) + 1

    new_an = batch["atomic_numbers"].clone()
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(int(seed))
    for p in range(n_particles):
        atom_idxs = torch.where(batch_idx == p)[0]
        if int(atom_idxs.numel()) != expected_per:
            raise ValueError(
                f"[init_types] particle {p} has {int(atom_idxs.numel())} atoms but "
                f"target multiset has {expected_per}. Pass --fk_n_atoms_exact_sum_target "
                f"so the num-atoms distribution is locked to sum(target_counts)."
            )
        perm = torch.randperm(expected_per, generator=rng).to(device)
        new_an[atom_idxs] = multiset_t[perm]

    return batch.replace(atomic_numbers=new_an)


def _install_init_types_on_sampler(sampler, init_state: "_InitTypesState"):
    """Patch sampler._sample_maybe_record to override atomic_numbers after _sample_prior; idempotent."""
    if getattr(sampler, "_init_types_installed", False):
        return
    if not init_state.enabled:
        return

    import importlib
    pc_module = importlib.import_module("mattergen.diffusion.sampling.pc_sampler")
    _sample_prior_fn = pc_module._sample_prior

    @torch.no_grad()
    def patched_sample_maybe_record(conditioning_data, mask=None, record=False):
        if isinstance(sampler._diffusion_module, torch.nn.Module):
            sampler._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(sampler._device)
        mask = {k: v.to(sampler._device) for k, v in mask.items()}
        batch = _sample_prior_fn(sampler._multi_corruption, conditioning_data, mask=mask)
        if init_state.target_counts is not None:
            batch = _override_atomic_numbers_to_target(
                batch, init_state.target_counts, seed=init_state.seed,
            )
            print(f"[init_types] overrode atomic_numbers at t=T to permuted target "
                  f"multiset ({init_state.target_counts})", flush=True)
        return sampler._denoise(batch=batch, mask=mask, record=record)

    sampler._sample_maybe_record = patched_sample_maybe_record
    sampler._init_types_installed = True


# ─────────────────────────────────────────────────────────────────────────────
# --lock_types_via_mask (inpainting-style hard composition lock)
# ─────────────────────────────────────────────────────────────────────────────
# MatterGen's mask inpainting uses lerp (float-only), so instead re-clamp atomic_numbers to a
# cached per-particle permuted multiset after every predictor/corrector step. Same outcome, no dtype issues.

class _LockTypesState:
    def __init__(self):
        self.enabled: bool = False
        self.target_counts = None        # dict[int Z -> int count]
        self.seed: int | None = None
        self.cached_z_per_atom = None    # (n_atoms_total,) on device


def _ensure_lock_types_state(pl_module):
    if not getattr(pl_module, "_lock_types_state", None):
        pl_module._lock_types_state = _LockTypesState()


def _install_lock_types_on_sampler(sampler, lock_state: "_LockTypesState"):
    """Patch sampler to override atomic_numbers after _sample_prior and re-clamp them each denoise step.

    Not composed with the FK hook (which re-orders atoms on resample); --lock_types_via_mask excludes --fk_n_particles.
    """
    if getattr(sampler, "_lock_types_installed", False):
        return
    if not lock_state.enabled:
        return

    import importlib
    pc_module = importlib.import_module("mattergen.diffusion.sampling.pc_sampler")
    _sample_prior_fn = pc_module._sample_prior
    _mask_replace = pc_module._mask_replace
    apply_fn = pc_module.apply
    from tqdm import tqdm as _tqdm

    state = lock_state

    @torch.no_grad()
    def patched_sample_maybe_record(conditioning_data, mask=None, record=False):
        if isinstance(sampler._diffusion_module, torch.nn.Module):
            sampler._diffusion_module.eval()
        mask = mask or {}
        conditioning_data = conditioning_data.to(sampler._device)
        mask = {k: v.to(sampler._device) for k, v in mask.items()}
        batch = _sample_prior_fn(sampler._multi_corruption, conditioning_data, mask=mask)
        # Build and cache the per-particle permuted multiset once.
        batch = _override_atomic_numbers_to_target(
            batch, state.target_counts, seed=state.seed,
        )
        state.cached_z_per_atom = batch["atomic_numbers"].clone()
        first_8 = state.cached_z_per_atom[:8].cpu().tolist()
        print(f"[lock_types] cached per-atom Z assignment (first 8): {first_8}, "
              f"target_counts={state.target_counts}", flush=True)
        return _patched_denoise(batch=batch, mask=mask, record=record)

    @torch.no_grad()
    def _patched_denoise(batch, mask, record=False):
        recorded_samples = [] if record else None
        for k in sampler._predictors:
            mask.setdefault(k, None)
        for k in sampler._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()
        timesteps = torch.linspace(sampler._max_t, sampler._eps_t, sampler.N,
                                   device=sampler._device)
        dt = -torch.tensor((sampler._max_t - sampler._eps_t) / (sampler.N - 1)).to(sampler._device)

        for i in _tqdm(range(sampler.N), miniters=50, mininterval=5):
            t = torch.full((batch.get_batch_size(),), timesteps[i],
                           device=sampler._device)
            if sampler._correctors:
                for _ in range(sampler._n_steps_corrector):
                    score = sampler._score_fn(batch, t)
                    fns = {k: corrector.step_given_score
                           for k, corrector in sampler._correctors.items()}
                    samples_means = apply_fn(
                        fns=fns, broadcast={"t": t, "dt": dt}, x=batch, score=score,
                        batch_idx=sampler._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means, batch=batch,
                        mean_batch=mean_batch, mask=mask,
                    )
                    batch["atomic_numbers"] = state.cached_z_per_atom.clone()
                    mean_batch["atomic_numbers"] = state.cached_z_per_atom.clone()

            score = sampler._score_fn(batch, t)
            predictor_fns = {k: predictor.update_given_score
                             for k, predictor in sampler._predictors.items()}
            samples_means = apply_fn(
                fns=predictor_fns, x=batch, score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=sampler._multi_corruption._get_batch_indices(batch),
            )
            if record:
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means, batch=batch,
                mean_batch=mean_batch, mask=mask,
            )
            batch["atomic_numbers"] = state.cached_z_per_atom.clone()
            mean_batch["atomic_numbers"] = state.cached_z_per_atom.clone()

            if i == 0:
                vals = batch["atomic_numbers"][:8].cpu().tolist()
                print(f"[lock_types] post-step-0 atomic_numbers[:8] = {vals}",
                      flush=True)

        return batch, mean_batch, recorded_samples

    sampler._sample_maybe_record = patched_sample_maybe_record
    sampler._lock_types_installed = True


# ─────────────────────────────────────────────────────────────────────────────
# --bias_types_via_score (gradual score-fn type biasing)
# ─────────────────────────────────────────────────────────────────────────────
# Add log q(z|site,t) to atomic_numbers logits each step; q goes from uniform-over-target (high t)
# to Hungarian-peaked (low t) via alpha(t) = clip((t_start - t) / (t_start - t_end)). MASK column unbiased.

class _TypeBiasState:
    def __init__(self):
        self.enabled: bool = False
        self.target_zs = None              # sorted distinct target Zs
        self.target_multiset = None        # 1D long tensor, len = sum(target_counts)
        self.t_start: float = 0.5
        self.t_end: float = 0.05
        self.n_steps_called: int = 0


def _ensure_type_biasing_installed(pl_module, target_counts: dict,
                                   t_start: float = 0.5, t_end: float = 0.05):
    """Idempotently wrap score_fn to bias atomic_numbers logits toward the target composition on a t-schedule."""
    if getattr(pl_module, "_type_biasing_installed", False):
        # Update existing state (target_counts may differ per prompt).
        st = pl_module._type_bias_state
        st.target_zs = sorted(int(z) for z in target_counts.keys())
        ms = []
        for z, n in target_counts.items():
            ms.extend([int(z)] * int(n))
        st.target_multiset = torch.tensor(ms, dtype=torch.long)
        st.t_start = float(t_start)
        st.t_end = float(t_end)
        st.n_steps_called = 0
        return

    from scipy.optimize import linear_sum_assignment as _linsum
    state = _TypeBiasState()
    state.target_zs = sorted(int(z) for z in target_counts.keys())
    ms = []
    for z, n in target_counts.items():
        ms.extend([int(z)] * int(n))
    state.target_multiset = torch.tensor(ms, dtype=torch.long)
    state.t_start = float(t_start)
    state.t_end = float(t_end)
    pl_module._type_bias_state = state

    diffusion_module = pl_module.diffusion_module
    orig_score_fn = diffusion_module.score_fn

    def biased_score_fn(batch, t):
        output = orig_score_fn(batch, t)
        if not state.enabled:
            return output
        try:
            logits = output["atomic_numbers"]
        except (KeyError, TypeError):
            return output
        if not torch.is_tensor(logits) or logits.ndim != 2:
            return output

        device = logits.device
        n_atoms_total, n_classes = logits.shape
        n_eff = min(n_classes, 100)  # only bias real-element columns

        t_val = float(t.mean().item()) if torch.is_tensor(t) else float(t)
        denom = max(state.t_start - state.t_end, 1e-9)
        alpha = max(0.0, min(1.0, (state.t_start - t_val) / denom))

        # Uniform-over-target component.
        target_z_idx = torch.tensor([z - 1 for z in state.target_zs],
                                    device=device, dtype=torch.long)
        q_uniform = torch.zeros(n_eff, device=device, dtype=logits.dtype)
        q_uniform[target_z_idx] = 1.0 / float(len(state.target_zs))

        # Peaked component via per-particle Hungarian (when alpha > 0).
        q_peaked = torch.zeros(n_atoms_total, n_eff, device=device, dtype=logits.dtype)
        if alpha > 0:
            batch_idx = batch.get_batch_idx("atomic_numbers")
            if batch_idx is None:
                batch_idx = batch.get_batch_idx("pos")
            n_particles = int(batch_idx.max().item()) + 1
            ms_cpu = state.target_multiset
            target_cols = [int(z) - 1 for z in ms_cpu.tolist()]
            site_probs_full = torch.softmax(logits[:, :n_eff], dim=-1).detach()
            for p in range(n_particles):
                mask_p = (batch_idx == p)
                idxs = mask_p.nonzero(as_tuple=False).squeeze(-1)
                n_p = int(idxs.numel())
                if n_p == 0:
                    continue
                if n_p != ms_cpu.numel():
                    # N_p mismatch: skip Hungarian, leave q_peaked = 0 for this particle.
                    continue
                site_probs = site_probs_full[idxs][:, target_cols]  # (N_p, N_p)
                cost = -(site_probs + 1e-4).log().cpu().numpy()
                rows, cols = _linsum(cost)
                for atom_local_i, slot_idx in zip(rows, cols):
                    target_z = int(ms_cpu[int(slot_idx)].item())
                    q_peaked[idxs[int(atom_local_i)], target_z - 1] = 1.0

        q = (1.0 - alpha) * q_uniform.unsqueeze(0) + alpha * q_peaked
        log_q = (q + 1e-8).log().to(logits.dtype)

        # Bias the first n_eff columns; leave the MASK column untouched.
        new_logits = logits.clone()
        new_logits[:, :n_eff] = logits[:, :n_eff] + log_q
        state.n_steps_called += 1
        return output.replace(atomic_numbers=new_logits)

    diffusion_module.score_fn = biased_score_fn
    pl_module._type_biasing_installed = True


# ─────────────────────────────────────────────────────────────────────────────
# Feynman-Kac steering — N-particle SMC wrapper around the diffusion sampler
# ─────────────────────────────────────────────────────────────────────────────

class _FKState:
    """FK config + per-prompt cumulative state, read by the sampler hook each denoising step."""
    def __init__(self):
        self.enabled: bool = False
        self.reward = None                  # fk_rewards.Reward (callable)
        self.n_particles: int = 1
        self.resample_every: int = 5
        self.t_start_frac: float = 0.5
        self.lambda_: float = 1.0
        self.potential: str = "diff"        # "diff" | "sum" | "max"
        self.ess_threshold_frac: float = 0.5
        self.keep_top_k: int = -1           # -1 = keep all particles
        self.log_w_clip: float = 10.0
        self.stratify_resample_by_n_atoms: bool = False
        # Post-hoc Hungarian Z-override at end of denoising; forces exact target counts.
        self.enforce_target_counts: bool = False
        self.target_counts = None  # dict[int Z -> int count]
        # Per-prompt cumulative state (reset by _fk_reset_per_prompt):
        self.log_w = None                   # (N,) cumulative log-weights
        self.prev_r = None                  # (N,) reward at previous step
        self.just_resampled: bool = False
        self.n_resamples: int = 0
        self.n_clip_hits: int = 0
        self.n_fk_steps: int = 0
        # Trajectory diagnostics, serialized to fk_trajectory.pt at end of generation.
        self.np_per_particle = None                        # (N,) N_p per slot, re-indexed on resample
        self.reward_trajectory: dict[str, list] = {}       # name -> list of (N,) tensors
        self.resample_log: list = []                       # [(step_idx, idx_after_resample), ...]


def _fk_reset_per_prompt(state: "_FKState", n_particles: int, device):
    """Reset cumulative log_w / prev_r / counters for a fresh prompt."""
    state.n_particles = n_particles
    state.log_w = torch.zeros(n_particles, device=device)
    state.prev_r = None
    state.just_resampled = False
    state.n_resamples = 0
    state.n_clip_hits = 0
    state.n_fk_steps = 0
    state.np_per_particle = None
    state.reward_trajectory = {}
    state.resample_log = []


def _fk_check_batch_invariants(b, tag, n_particles):
    """Diagnostic: assert/log MatterGen batch invariants around an FK resample.
    Enabled via FK_DEBUG_INVARIANTS=1. Reports the FIRST broken invariant."""
    import torch as _t
    msgs = []
    def chk(name, cond, detail=""):
        msgs.append((name, bool(cond), detail))
    try:
        chk("get_batch_size==N", b.get_batch_size() == n_particles,
            f"{b.get_batch_size()} vs {n_particles}")
        chk("batch.min==0", int(b.batch.min()) == 0, f"min={int(b.batch.min())}")
        chk("batch.max==N-1", int(b.batch.max()) == n_particles - 1,
            f"max={int(b.batch.max())}")
        if "num_atoms" in b.keys():
            bc = _t.bincount(b.batch, minlength=n_particles)
            na = b["num_atoms"].flatten()
            chk("bincount(batch)==num_atoms", bool(_t.equal(bc.cpu(), na.cpu())),
                f"bincount={bc.tolist()} num_atoms={na.tolist()}")
        z = b["atomic_numbers"]
        chk("atomic_numbers in [1,100]", bool((z >= 1).all() and (z <= 100).all()),
            f"min={int(z.min())} max={int(z.max())}")
        chk("cell.shape[0]==N", b["cell"].shape[0] == n_particles,
            f"cell={tuple(b['cell'].shape)}")
        if "alm_embedding" in b.keys():
            ae = b["alm_embedding"]
            chk("alm_embedding.shape[0]==N", ae.shape[0] == n_particles,
                f"shape={tuple(ae.shape)}")
        if hasattr(b, "ptr"):
            chk("ptr len==N+1", b.ptr.numel() == n_particles + 1, f"ptr={b.ptr.tolist()}")
    except Exception as e:  # noqa: BLE001
        print(f"[fk-inv {tag}] EXC during invariant check: {type(e).__name__}: {e}", flush=True)
        return
    broken = [m for m in msgs if not m[1]]
    status = "ALL_OK" if not broken else f"BROKEN(first={broken[0][0]})"
    print(f"[fk-inv {tag}] {status}", flush=True)
    for name, ok, detail in msgs:
        if not ok:
            print(f"    [FAIL] {name}: {detail}", flush=True)


def _install_sega_on_sampler(sampler, alm_emb_opp: "torch.Tensor", g: float):
    """Prompt-difference (SEGA) guidance: s = s_null + g*(s_asked - s_opp), the pure directional residual.

    alm_emb_opp is this row's opposite-direction bridge vector (1, D) or (D,); call restore() after the draw.
    Costs ~3x score evals/step.
    """
    orig_score_fn = sampler._score_fn
    remove_fn = sampler._remove_conditioning_fn
    keep_fn = sampler._keep_conditioning_fn
    corrupted = sampler._multi_corruption.corrupted_fields
    raw = sampler.diffusion_module.score_fn          # un-guided score model
    opp_vec = alm_emb_opp.detach()  # opposite-direction bridge vector
    if opp_vec.dim() == 1:
        opp_vec = opp_vec.unsqueeze(0)

    def sega_score_fn(x, t):
        s_null = raw(remove_fn(x), t)                # bridge OFF
        x_ask = keep_fn(x)                           # bridge ON, alm_embedding = asked
        s_asked = raw(x_ask, t)
        cur = x_ask["alm_embedding"]
        opp = opp_vec.to(cur.device, cur.dtype)
        if opp.shape[0] != cur.shape[0]:
            opp = opp.expand(cur.shape[0], *opp.shape[1:])
        s_opp = raw(x_ask.replace(alm_embedding=opp), t)   # bridge ON, alm_embedding = opposite
        return s_null.replace(**{
            k: s_null[k] + g * (s_asked[k] - s_opp[k]) for k in corrupted
        })

    sampler._score_fn = sega_score_fn

    def restore():
        sampler._score_fn = orig_score_fn
    return restore


def _ensure_fk_hook_installed(pl_module):
    """Idempotently attach an _FKState to pl_module; the sampler-loop wrap is per-sampler (see _install_fk_on_sampler)."""
    if not getattr(pl_module, "_fk_state", None):
        pl_module._fk_state = _FKState()


def _install_fk_on_sampler(sampler, fk_state: "_FKState"):
    """Patch sampler._denoise to inject FK steering: reward eval, log_w accumulation, ESS-gated resampling.

    Per-sampler (built fresh per-prompt). Composes with the element-mask score_fn wrapper, which it doesn't touch.
    """
    if getattr(sampler, "_fk_installed", False):
        return
    if not fk_state.enabled:
        return

    import importlib
    pc_module = importlib.import_module(
        "mattergen.diffusion.sampling.pc_sampler"
    )
    _mask_replace = pc_module._mask_replace
    apply_fn = pc_module.apply
    from tqdm import tqdm as _tqdm
    from mattergen.diffusion.data.batched_data import collate_fn as _collate_fn

    orig_denoise = sampler._denoise
    state = fk_state

    def _x_hat_0_view(score, mean_batch, batch_idx_t):
        """Build the Tweedie-clean dict that rewards consume from score (logits) + mean_batch (x_hat_0 pos/cell)."""
        # CSP-mode atoms are observed: expose true per-atom Z so rewards score real chemistry, not noisy logit argmax.
        # DNG-mode has no observed Z -> key absent -> rewards fall back to logits.
        obs_z = mean_batch["atomic_numbers"] if "atomic_numbers" in mean_batch.keys() else None
        return {
            "atomic_numbers_logits": score["atomic_numbers"],
            "atomic_numbers": obs_z,           # observed Z (CSP) or None (DNG)
            "pos": mean_batch["pos"] if "pos" in mean_batch.keys() else None,
            "cell": mean_batch["cell"] if "cell" in mean_batch.keys() else None,
            "batch_idx": batch_idx_t,
        }

    import torch_geometric.data as _pyg_data
    import copy as _copy
    from mattergen.common.data.collate import collate as _mg_collate

    def _resample_one(batch, idx):
        """Re-index a PyG-Batched ChemGraph by particle indices `idx` via deepcopy + MatterGen collate."""
        # Deep-copy each chosen graph (idx duplicates alias the same object) before collating.
        data_list = batch.to_data_list()
        new_list = [_copy.deepcopy(data_list[j.item()]) for j in idx]
        new_batch = _mg_collate(new_list)
        if os.environ.get("FK_DEBUG_INVARIANTS"):
            _fk_check_batch_invariants(batch, "BEFORE_resample", int(idx.numel()))
            _fk_check_batch_invariants(new_batch, "AFTER_resample", int(idx.numel()))
        return new_batch

    def _resample_batch(batch, mean_batch, idx):
        """Multinomial-resample BOTH batch and mean_batch by particle index `idx`."""
        return _resample_one(batch, idx), _resample_one(mean_batch, idx)

    @torch.no_grad()
    def fk_denoise(batch, mask, record=False):
        # Mirror PredictorCorrector._denoise's setup.
        recorded_samples = [] if record else None
        for k in sampler._predictors:
            mask.setdefault(k, None)
        for k in sampler._correctors:
            mask.setdefault(k, None)
        mean_batch = batch.clone()

        n_particles = batch.get_batch_size()
        _fk_reset_per_prompt(state, n_particles, sampler._device)

        timesteps = torch.linspace(sampler._max_t, sampler._eps_t, sampler.N,
                                   device=sampler._device)
        dt = -torch.tensor(
            (sampler._max_t - sampler._eps_t) / (sampler.N - 1),
        ).to(sampler._device)
        # FK steers only after t_val < t_threshold (deferred start).
        t_threshold = sampler._max_t * (1.0 - state.t_start_frac)

        for i in _tqdm(range(sampler.N), miniters=50, mininterval=5):
            t = torch.full((batch.get_batch_size(),), timesteps[i],
                           device=sampler._device)

            if sampler._correctors:
                for _ in range(sampler._n_steps_corrector):
                    score = sampler._score_fn(batch, t)
                    fns = {
                        k: corrector.step_given_score
                        for k, corrector in sampler._correctors.items()
                    }
                    samples_means = apply_fn(
                        fns=fns,
                        broadcast={"t": t, "dt": dt},
                        x=batch,
                        score=score,
                        batch_idx=sampler._multi_corruption._get_batch_indices(batch),
                    )
                    if record:
                        recorded_samples.append(batch.clone().to("cpu"))
                    batch, mean_batch = _mask_replace(
                        samples_means=samples_means,
                        batch=batch,
                        mean_batch=mean_batch,
                        mask=mask,
                    )

            # ── Predictor update ──────────────────────────────────────────
            score = sampler._score_fn(batch, t)
            predictor_fns = {
                k: predictor.update_given_score
                for k, predictor in sampler._predictors.items()
            }
            samples_means = apply_fn(
                fns=predictor_fns,
                x=batch,
                score=score,
                broadcast=dict(t=t, batch=batch, dt=dt),
                batch_idx=sampler._multi_corruption._get_batch_indices(batch),
            )
            if record:
                recorded_samples.append(batch.clone().to("cpu"))
            batch, mean_batch = _mask_replace(
                samples_means=samples_means,
                batch=batch,
                mean_batch=mean_batch,
                mask=mask,
            )

            # ── FK injection (only after t_start_frac) ────────────────────
            t_val = float(timesteps[i].item())
            if t_val < t_threshold:
                state.n_fk_steps += 1
                batch_idx_t = sampler._multi_corruption._get_batch_indices(batch)
                # Use the 'pos' batch-idx (variable-len channel); cell is dense.
                pos_batch_idx = batch_idx_t.get("pos", None)
                if pos_batch_idx is None:
                    pos_batch_idx = next(iter(
                        v for v in batch_idx_t.values() if v is not None
                    ), None)
                if pos_batch_idx is not None:
                    x_hat_0 = _x_hat_0_view(score, mean_batch, pos_batch_idx)
                    cur_r = state.reward(x_hat_0, t_val, i)
                    # Drain GPU-reward work in-band so a reward CUDA error raises here, not as a later async assert.
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Capture np_per_particle on the first FK-active step (re-indexed on each resample).
                    if state.np_per_particle is None:
                        np_counts = torch.zeros(n_particles, dtype=torch.long, device=cur_r.device)
                        np_counts.scatter_add_(
                            0, pos_batch_idx,
                            torch.ones_like(pos_batch_idx, dtype=torch.long),
                        )
                        state.np_per_particle = np_counts.cpu()

                    # Log trajectory: total + per-component (if WeightedSum exposes it).
                    state.reward_trajectory.setdefault("total", []).append(
                        cur_r.detach().cpu().clone()
                    )
                    components = getattr(state.reward, "last_components", None)
                    if components:
                        for cname, cv in components.items():
                            state.reward_trajectory.setdefault(cname, []).append(
                                cv.detach().cpu().clone()
                            )

                    if not state.just_resampled:
                        if state.potential == "diff" and state.prev_r is not None:
                            G = state.lambda_ * (cur_r - state.prev_r)
                        elif state.potential == "max" and state.prev_r is not None:
                            G = state.lambda_ * torch.maximum(cur_r, state.prev_r)
                        else:
                            G = state.lambda_ * cur_r
                        new_log_w = state.log_w + G
                        clipped = new_log_w.clamp(-state.log_w_clip, state.log_w_clip)
                        state.n_clip_hits += int(
                            (new_log_w != clipped).sum().item()
                        )
                        state.log_w = clipped

                    state.prev_r = cur_r
                    state.just_resampled = False

                    # ── Resample on schedule + ESS gate ──────────────────
                    if (i % state.resample_every) == 0:
                        w = torch.softmax(state.log_w, dim=0)
                        ess = 1.0 / (w * w).sum().clamp_min(1e-12)
                        if ess < n_particles * state.ess_threshold_frac:
                            if state.stratify_resample_by_n_atoms and state.np_per_particle is not None:
                                # Resample within each N_p group so each keeps its population.
                                idx_full = torch.empty(
                                    n_particles, dtype=torch.long, device=w.device
                                )
                                np_arr = state.np_per_particle.to(w.device)
                                groups = torch.unique(np_arr).tolist()
                                cursor = 0
                                for g in groups:
                                    g_mask = (np_arr == g)
                                    g_indices = g_mask.nonzero(as_tuple=False).squeeze(-1)
                                    g_w = w[g_indices]
                                    g_w = g_w / g_w.sum().clamp_min(1e-12)
                                    g_n = int(g_indices.numel())
                                    g_pick = torch.multinomial(g_w, g_n, replacement=True)
                                    idx_full[cursor:cursor + g_n] = g_indices[g_pick]
                                    cursor += g_n
                                idx = idx_full
                            else:
                                idx = torch.multinomial(w, n_particles, replacement=True)
                            batch, mean_batch = _resample_batch(
                                batch, mean_batch, idx
                            )
                            state.log_w = torch.zeros_like(state.log_w)
                            state.prev_r = cur_r[idx]   # re-index to new particles
                            state.np_per_particle = state.np_per_particle[idx.cpu()]
                            state.resample_log.append((int(i), idx.cpu().tolist()))
                            state.just_resampled = True
                            state.n_resamples += 1

        # Optional top-k trim by final cumulative log_w.
        if state.keep_top_k > 0 and state.keep_top_k < n_particles:
            order = torch.argsort(state.log_w, descending=True)
            keep_idx = order[:state.keep_top_k]
            batch, mean_batch = _resample_batch(batch, mean_batch, keep_idx)

        # Post-hoc Hungarian Z-override: reassign atom labels to the target multiset; positions/cell untouched.
        if state.enforce_target_counts and state.target_counts is not None:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
            from fk_rewards import _expand_multiset_to_size  # noqa: E402
            from scipy.optimize import linear_sum_assignment as _linsum  # noqa: E402

            # Final score_fn at eps_t for terminal atomic_numbers probs.
            t_final = torch.full((batch.get_batch_size(),), float(timesteps[-1].item()),
                                 device=sampler._device)
            score_final = sampler._score_fn(batch, t_final)
            probs = torch.softmax(score_final["atomic_numbers"][:, :100], dim=-1)
            batch_idx_t = sampler._multi_corruption._get_batch_indices(batch)
            pos_batch_idx = batch_idx_t.get("pos", None)

            old_z = batch["atomic_numbers"].clone()
            new_z = old_z.clone()
            n_overrides = 0
            for p in range(batch.get_batch_size()):
                mask = pos_batch_idx == p
                idxs = mask.nonzero(as_tuple=False).squeeze(-1)
                n_p = int(idxs.numel())
                if n_p == 0:
                    continue
                target_zs = _expand_multiset_to_size(state.target_counts, n_p)
                target_cols = [z - 1 for z in target_zs]
                cost = -(probs[mask][:, target_cols] + 1e-4).log()
                rows, cols = _linsum(cost.detach().cpu().numpy())
                for atom_local_i, slot_idx in zip(rows, cols):
                    new_global = idxs[int(atom_local_i)].item()
                    target_z = target_zs[int(slot_idx)]
                    if int(old_z[new_global].item()) != target_z:
                        n_overrides += 1
                    new_z[new_global] = target_z
            # Update BOTH: draw_samples_from_sampler builds final structures from mean_batch.atomic_numbers.
            batch["atomic_numbers"] = new_z
            try:
                mean_batch["atomic_numbers"] = new_z.clone()
            except Exception as exc:
                print(f"[fk] WARNING: could not update mean_batch.atomic_numbers ({exc}); "
                      "post-hoc override may not propagate to final structures.",
                      flush=True)
            print(f"[fk] post-hoc Z-override: changed {n_overrides}/{int(old_z.numel())} "
                  f"atomic_numbers to enforce target_counts={state.target_counts}",
                  flush=True)

        clip_frac = (state.n_clip_hits / max(state.n_fk_steps * n_particles, 1))
        print(f"[fk] DONE — n_fk_steps={state.n_fk_steps}, "
              f"n_resamples={state.n_resamples}, "
              f"clip_hits={state.n_clip_hits} "
              f"({100*clip_frac:.1f}% of (step×particle) updates), "
              f"final log_w max={state.log_w.max().item():+.3f} "
              f"min={state.log_w.min().item():+.3f}",
              flush=True)
        return batch, mean_batch, recorded_samples

    sampler._denoise = fk_denoise
    sampler._fk_installed = True


def _parse_target_counts(spec: str | None,
                         from_prompt_json: str | None = None) -> dict[int, int] | None:
    """Parse spec ("V:2,Ga:1,Fe:1") or from_prompt_json ("path:tag") to {Z: count}, or None."""
    from ase.data import atomic_numbers as _ase_z
    if from_prompt_json:
        path, _, tag = from_prompt_json.partition(":")
        if not path or not tag:
            raise ValueError(
                f"--fk_target_counts_from_prompt_json wants 'path:tag', got {from_prompt_json!r}"
            )
        import json as _json
        with open(path) as f:
            data = _json.load(f)
        if tag not in data:
            raise KeyError(
                f"prompt tag {tag!r} not in {path}; have {list(data.keys())}"
            )
        return {int(_ase_z[sym]): int(n) for sym, n in data[tag].items()}
    if spec:
        out = {}
        for entry in spec.split(","):
            entry = entry.strip()
            if not entry:
                continue
            sym, _, n = entry.partition(":")
            sym = sym.strip()
            if sym not in _ase_z:
                raise ValueError(f"unknown element symbol: {sym!r}")
            out[int(_ase_z[sym])] = int(n)
        return out
    return None


def generate_for_prompts(
    prompts,
    alm,
    tokenizer,
    pl_module,
    out_root,
    batch_size: int = 4,
    num_batches: int = 1,
    diffusion_guidance_factor: float = 1.0,
    diffusion_snr: float | None = None,
    num_atoms_distribution: str = "ALEX_MP_20",
    record_trajectories: bool = False,
    save_meta: bool = True,
    prompt_ids=None,
    allowed_elements_per_prompt=None,
    fk_n_particles: int = 0,
    fk_rewards: str = "stoich_match:1.0",
    fk_target_counts_per_prompt=None,
    fk_resample_every: int = 5,
    fk_t_start_frac: float = 0.5,
    fk_lambda: float = 1.0,
    fk_potential: str = "diff",
    fk_ess_threshold_frac: float = 0.5,
    fk_keep_top_k: int = -1,
    fk_log_w_clip: float = 10.0,
    fk_constrain_n_atoms_to_target_multiple: bool = False,
    fk_stratify_resample_by_n_atoms: bool = False,
    fk_n_atoms_exact_sum_target: bool = False,
    fk_enforce_target_counts: bool = False,
    fk_physical_bounds: Mapping[str, float] | None = None,
    fk_target_sg_per_prompt=None,
    diffusion_seed: int | None = None,
    init_types_at_target: bool = False,
    init_types_target_counts_per_prompt=None,
    lock_types_via_mask: bool = False,
    lock_types_target_counts_per_prompt=None,
    bias_types_via_score: bool = False,
    bias_types_target_counts_per_prompt=None,
    type_bias_t_start: float = 0.5,
    type_bias_t_end: float = 0.05,
    chemical_system_per_prompt=None,
    space_group_per_prompt=None,
    composition_count_per_prompt=None,
    json_counts_per_prompt=None,
    cot_tokens: int = 0,
    llm_temperature: float = 1.0,
    cot_top_p: float = 0.9,
    cot_seed: int | None = None,
):
    """Batched generation: load the model once, loop over prompts; each lands in out_root/<id>/.

    fk_n_particles>0 installs the FK hook per prompt (batch dim becomes the particle dim).
    Returns list[list[Structure]]: outer = prompt, inner = its generations.
    """
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: F401  (used downstream)
    device = next(pl_module.parameters()).device
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if prompt_ids is None:
        prompt_ids = [f"prompt_{i:04d}" for i in range(len(prompts))]
    if len(prompt_ids) != len(prompts):
        raise ValueError("prompt_ids must align 1:1 with prompts")
    if allowed_elements_per_prompt is not None and len(allowed_elements_per_prompt) != len(prompts):
        raise ValueError("allowed_elements_per_prompt must align 1:1 with prompts")
    # Lazy-install the element-mask wrapper once; state stays None until a prompt has an allowed set.
    if allowed_elements_per_prompt is not None:
        _ensure_element_mask_installed(pl_module)

    fk_active = fk_n_particles > 0
    if fk_active:
        if fk_target_counts_per_prompt is None or len(fk_target_counts_per_prompt) != len(prompts):
            raise ValueError(
                "fk_target_counts_per_prompt must be a list of {Z: count} dicts "
                "aligned 1:1 with prompts when fk_n_particles > 0"
            )
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
        from fk_rewards import parse_rewards as _fk_parse_rewards  # noqa: E402
        _ensure_fk_hook_installed(pl_module)

    if init_types_at_target:
        if init_types_target_counts_per_prompt is None or \
           len(init_types_target_counts_per_prompt) != len(prompts):
            raise ValueError(
                "init_types_target_counts_per_prompt must be a list of {Z: count} dicts "
                "aligned 1:1 with prompts when init_types_at_target=True"
            )
        _ensure_init_types_state(pl_module)

    if lock_types_via_mask and bias_types_via_score:
        raise ValueError(
            "lock_types_via_mask and bias_types_via_score are mutually exclusive."
        )
    if lock_types_via_mask:
        if (lock_types_target_counts_per_prompt is None or
                len(lock_types_target_counts_per_prompt) != len(prompts)):
            raise ValueError(
                "lock_types_target_counts_per_prompt must be a list of {Z: count} "
                "dicts aligned 1:1 with prompts when lock_types_via_mask=True"
            )
        _ensure_lock_types_state(pl_module)
    if bias_types_via_score:
        if (bias_types_target_counts_per_prompt is None or
                len(bias_types_target_counts_per_prompt) != len(prompts)):
            raise ValueError(
                "bias_types_target_counts_per_prompt must be a list of {Z: count} "
                "dicts aligned 1:1 with prompts when bias_types_via_score=True"
            )
    if json_counts_per_prompt is not None and len(json_counts_per_prompt) != len(prompts):
        raise ValueError(
            "json_counts_per_prompt must be a list of {element_symbol: count} dicts "
            "aligned 1:1 with prompts"
        )

    all_results = []
    n_skipped = 0
    # No-LLM-bridge ckpts lack alm_embedding -> skip the ALM forward and don't pass it as a condition.
    _has_alm_cond = "alm_embedding" in pl_module.diffusion_module.model.property_embeddings_adapt
    if not _has_alm_cond:
        print("[gen] adapter has no alm_embedding cond_field — skipping ALM forward, "
              "running conditional generation from discrete cond_fields only.")
    else:
        _alm_mapper = (
            pl_module.diffusion_module.model.property_embeddings_adapt["alm_embedding"]
            .conditional_embedding_module
        )
        _mapper_name = type(_alm_mapper).__name__
        if (
            _mapper_name == "AtomsMapperProducerConsumer"
            and json_counts_per_prompt is None
            and float(diffusion_guidance_factor) != 0.0
        ):
            print(
                "[gen] WARNING: QFormer alm_embedding guidance is active but "
                "json_counts_per_prompt is None. lm_loss_json/QFormer training "
                "conditioned the producer on a committed composition prefix; this "
                "open-ended path is a context-shift eval unless a planner or caller "
                "supplies counts.",
                flush=True,
            )

    for i, (pid, prompt) in enumerate(zip(prompt_ids, prompts)):
        # Apply (or clear) the per-prompt element mask.
        if allowed_elements_per_prompt is not None:
            ae = allowed_elements_per_prompt[i]
            pl_module._element_mask_state.allowed_z = (
                _z_set_from_elements(ae) if ae is not None else None
            )
        prompt_dir = out_root / str(pid)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Per-prompt seed: reproducible and order-independent.
            if diffusion_seed is not None:
                _s = (int(diffusion_seed) + i) & 0x7FFFFFFF
                torch.manual_seed(_s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(_s)
                np.random.seed(_s)
                random.seed(_s)
            if _has_alm_cond:
                # Per-prompt cot_seed offset for distinct yet reproducible CoT (cot_tokens=0 -> no-op).
                _cot_seed_i = (
                    (int(cot_seed) + i) & 0x7FFFFFFF
                    if (cot_tokens > 0 and cot_seed is not None) else None
                )
                alm_emb = get_alm_embedding(
                    alm, tokenizer, prompt, device,
                    cot_tokens=cot_tokens, llm_temperature=llm_temperature,
                    cot_top_p=cot_top_p, cot_seed=_cot_seed_i,
                    json_counts=(
                        json_counts_per_prompt[i]
                        if json_counts_per_prompt is not None else None
                    ),
                )
                # Log L2 so a collapsed (dead-bridge) channel is visible.
                print(f"[gen] prompt {i} alm_embedding L2={alm_emb.float().norm(p=2).item():.4f}",
                      flush=True)
            else:
                alm_emb = None
            # Per-prompt FK config; sampler batch_size = fk_n_particles.
            sampler_batch_size = fk_n_particles if fk_active else batch_size
            sampler_num_batches = 1 if fk_active else num_batches
            constrain_mult = 0
            constrain_exact = 0
            target_counts = None
            if fk_active:
                target_counts = fk_target_counts_per_prompt[i]
                if target_counts and fk_n_atoms_exact_sum_target:
                    constrain_exact = int(sum(target_counts.values()))
                elif target_counts and fk_constrain_n_atoms_to_target_multiple:
                    constrain_mult = int(sum(target_counts.values()))
            # Env-var override for predictor step count (default None = MG's 1000).
            _diffusion_steps_env = os.environ.get("DIFFUSION_STEPS", "")
            _diffusion_steps = int(_diffusion_steps_env) if _diffusion_steps_env else None
            # Only stamp cond fields the ckpt was trained on (else the generator AssertionErrors).
            _trained_cf = set(pl_module.diffusion_module.model.cond_fields_model_was_trained_on)
            _cs_for_loader = None
            if chemical_system_per_prompt is not None and "chemical_system" in _trained_cf:
                _cs_for_loader = "-".join(sorted(set(chemical_system_per_prompt[i])))
            _sg_for_loader = None
            if space_group_per_prompt is not None and "space_group" in _trained_cf:
                _sg_for_loader = int(space_group_per_prompt[i])
            _cc_for_loader = None
            if composition_count_per_prompt is not None and "composition_count" in _trained_cf:
                _cc_for_loader = composition_count_per_prompt[i]
            sampler, condition_loader = build_sampler_and_loader(
                pl_module=pl_module,
                batch_size=sampler_batch_size,
                num_batches=sampler_num_batches,
                num_atoms_distribution=num_atoms_distribution,
                alm_emb_vec=alm_emb,
                diffusion_guidance_factor=diffusion_guidance_factor,
                diffusion_snr=diffusion_snr,
                diffusion_steps=_diffusion_steps,
                constrain_n_atoms_to_multiple_of=constrain_mult,
                constrain_n_atoms_exact=constrain_exact,
                chemical_system=_cs_for_loader,
                space_group=_sg_for_loader,
                composition_count=_cc_for_loader,
            )
            if fk_active:
                ae = (allowed_elements_per_prompt[i]
                      if allowed_elements_per_prompt else None)
                target_sg_i = (
                    fk_target_sg_per_prompt[i]
                    if fk_target_sg_per_prompt is not None else None
                )
                reward = _fk_parse_rewards(
                    fk_rewards,
                    allowed_elements=ae,
                    target_counts=target_counts,
                    physical_bounds=fk_physical_bounds,
                    target_sg=target_sg_i,
                )
                st = pl_module._fk_state
                st.enabled = True
                st.reward = reward
                st.target_counts = target_counts
                st.enforce_target_counts = bool(fk_enforce_target_counts)
                st.n_particles = fk_n_particles
                st.resample_every = fk_resample_every
                st.t_start_frac = fk_t_start_frac
                st.lambda_ = fk_lambda
                st.potential = fk_potential
                st.ess_threshold_frac = fk_ess_threshold_frac
                st.keep_top_k = fk_keep_top_k
                st.log_w_clip = fk_log_w_clip
                st.stratify_resample_by_n_atoms = fk_stratify_resample_by_n_atoms
                _install_fk_on_sampler(sampler, st)
            if init_types_at_target:
                it_state = pl_module._init_types_state
                it_state.enabled = True
                it_state.target_counts = init_types_target_counts_per_prompt[i]
                # Per-prompt seed offset so each prompt gets a different multiset permutation.
                it_state.seed = (
                    (int(diffusion_seed) + i) & 0x7FFFFFFF
                    if diffusion_seed is not None else None
                )
                _install_init_types_on_sampler(sampler, it_state)
            if lock_types_via_mask:
                lk_state = pl_module._lock_types_state
                lk_state.enabled = True
                lk_state.target_counts = lock_types_target_counts_per_prompt[i]
                lk_state.seed = (
                    (int(diffusion_seed) + i) & 0x7FFFFFFF
                    if diffusion_seed is not None else None
                )
                _install_lock_types_on_sampler(sampler, lk_state)
            if bias_types_via_score:
                _ensure_type_biasing_installed(
                    pl_module, bias_types_target_counts_per_prompt[i],
                    t_start=type_bias_t_start, t_end=type_bias_t_end,
                )
                pl_module._type_bias_state.enabled = True
            # Stamp discrete cond fields (chemical_system / space_group) the ckpt was trained on, one value per prompt.
            # Gating is required: the generator asserts all props are in cond_fields_model_was_trained_on.
            trained_cond_fields = set(
                pl_module.diffusion_module.model.cond_fields_model_was_trained_on
            )
            props = {}
            if _has_alm_cond and alm_emb is not None and "alm_embedding" in trained_cond_fields:
                props["alm_embedding"] = alm_emb.detach().cpu()
            if chemical_system_per_prompt is not None and "chemical_system" in trained_cond_fields:
                cs_i = chemical_system_per_prompt[i]
                # Sorted dashed format matches training-time set_chemical_system_string.
                props["chemical_system"] = "-".join(sorted(set(cs_i)))
            if space_group_per_prompt is not None and "space_group" in trained_cond_fields:
                sg_i = int(space_group_per_prompt[i])
                # sg in [1,230] as a 1-element LongTensor; PyG batching replicates.
                props["space_group"] = torch.tensor([sg_i], dtype=torch.long)
            if composition_count_per_prompt is not None and "composition_count" in trained_cond_fields:
                cc_i = composition_count_per_prompt[i]
                if not torch.is_tensor(cc_i):
                    cc_i = torch.as_tensor(cc_i, dtype=torch.float32)
                props["composition_count"] = cc_i.to(torch.float32).reshape(1, -1)
            structures = draw_samples_from_sampler(
                sampler=sampler,
                condition_loader=condition_loader,
                properties_to_condition_on=props,
                output_path=prompt_dir,
                cfg=OmegaConf.create({}),
                record_trajectories=record_trajectories,
            )
            if fk_active and pl_module._fk_state.reward_trajectory:
                st = pl_module._fk_state
                traj = {k: torch.stack(v) for k, v in st.reward_trajectory.items()}
                torch.save({
                    "np_per_particle_final": st.np_per_particle,
                    "trajectory": traj,
                    "resample_log": st.resample_log,
                    "n_fk_steps": st.n_fk_steps,
                    "n_resamples": st.n_resamples,
                    "n_clip_hits": st.n_clip_hits,
                    "fk_t_start_frac": st.t_start_frac,
                    "fk_lambda": st.lambda_,
                    "fk_resample_every": st.resample_every,
                    "fk_target_counts": target_counts,
                    "fk_rewards_spec": fk_rewards,
                    "stratify_resample_by_n_atoms": st.stratify_resample_by_n_atoms,
                }, prompt_dir / "fk_trajectory.pt")
            if save_meta:
                _meta = {
                    "prompt": prompt,
                    "prompt_id": pid,
                    "diffusion_guidance_factor": diffusion_guidance_factor,
                    "num_atoms_distribution": num_atoms_distribution,
                    "fk_target_counts": target_counts if fk_active else None,
                }
                if alm_emb is not None:
                    _meta["alm_embedding"] = alm_emb.detach().cpu()
                torch.save(_meta, prompt_dir / "stage3a_inference_meta.pt")
        except Exception as exc:
            # MatterGen's sampler can crash mid-denoise on stochastic edge cases; log + skip one prompt.
            n_skipped += 1
            print(f"[gen-batch] {i+1}/{len(prompts)}: id={pid} FAILED — "
                  f"{type(exc).__name__}: {exc}", flush=True)
            # Do NOT `import os` here: it would shadow the module-level import and UnboundLocalError above.
            import traceback
            if os.environ.get("ALM_FULL_TRACEBACK") == "1":
                traceback.print_exc()
            structures = []
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        all_results.append(structures)
        print(f"[gen-batch] {i+1}/{len(prompts)}: id={pid} → {len(structures)} structures"
              + (" (skipped)" if not structures else ""),
              flush=True)
    if n_skipped:
        print(f"[gen-batch] SUMMARY: {n_skipped}/{len(prompts)} prompts FAILED "
              f"(stochastic sampler crashes); rest succeeded.", flush=True)
    return all_results


def load_alm_and_pl_module(
    alm_checkpoint,
    atoms_mapper,
    mattergen_pretrained: str = "mattergen_base",
    device=None,
    use_cached_embeddings: bool = True,
    model_path: str | None = None,
):
    """One-shot load helper for eval scripts; returns (alm, tokenizer, pl_module, K) without freeing alm.

    use_cached_embeddings=False instantiates OrbV3 for live encoding (e.g. eval_polymorph).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # CPU-only pre-read recovers ALM bridge settings; missing keys fall back to load_alm defaults.
    try:
        _meta_ckpt = torch.load(atoms_mapper, map_location="cpu", weights_only=False)
    except TypeError:
        _meta_ckpt = torch.load(atoms_mapper, map_location="cpu")
    _num_output_atom_tokens = _meta_ckpt.get("num_output_atom_tokens", 8)
    _use_last_prompt_token = _meta_ckpt.get("use_last_prompt_token", False)
    _bridge_source = _meta_ckpt.get("bridge_source", "atoms_tokens")
    _init_atoms_tokens_from_eos = _meta_ckpt.get("init_atoms_tokens_from_eos", False)
    # Q-Former: force context_plus_atoms extraction + qformer_n_context so the source matches training.
    _bridge_kind = _norm_bridge(_meta_ckpt.get("bridge_kind", "pool"))
    _qformer_n_context = _meta_ckpt.get("qformer_context_tokens", 128)
    _qformer_input_atoms = _meta_ckpt.get("qformer_input_atoms", 0)
    if _bridge_kind in ("producer-consumer", "producer-consumer-pool"):
        _bridge_source = "context_plus_atoms"
    del _meta_ckpt  # build_pl_module re-reads it
    print(f"[gen] loading ALM from {alm_checkpoint} (cached_embs={use_cached_embeddings}) "
          f"K={_num_output_atom_tokens} bridge_source={_bridge_source!r} "
          f"last_k_prompt={_use_last_prompt_token} eos_init={_init_atoms_tokens_from_eos}"
          + (f" qformer_n_context={_qformer_n_context}" if _bridge_kind == "producer-consumer" else ""),
          flush=True)
    alm, tokenizer = load_alm(
        checkpoint=alm_checkpoint, merge_lora=True,
        use_cached_embeddings=use_cached_embeddings, device=device,
        num_output_atom_tokens=_num_output_atom_tokens,
        use_last_prompt_token=_use_last_prompt_token,
        bridge_source=_bridge_source,
        qformer_n_context=_qformer_n_context,
        qformer_input_atoms=_qformer_input_atoms,
        init_atoms_tokens_from_eos=_init_atoms_tokens_from_eos,
    )
    alm.eval()
    K = len(alm.output_atom_token_ids)
    print(f"[gen] K={K}, hidden_dim={alm.llm_hidden_dim}")
    print(f"[gen] building MatterGen adapter ({mattergen_pretrained}) ...", flush=True)
    print(f"[gen] overlaying AtomsMapper + cond_adapt/mixin from {atoms_mapper}")
    # Auto-detect mid_dim from the ckpt's AtomsMapper shape (sentinel for no-LLM-bridge ckpts).
    _ckpt = torch.load(atoms_mapper, map_location="cpu")
    _am_sd = _ckpt.get("atoms_mapper_state_dict") or {}
    if "proj.0.weight" in _am_sd:
        _mid_dim = int(_am_sd["proj.0.weight"].shape[0])
        print(f"[gen] detected AtomsMapper mid_dim={_mid_dim} from ckpt")
    else:
        _mid_dim = 2048
        print(f"[gen] no AtomsMapper in ckpt (no-LLM-bridge control); using "
              f"placeholder mid_dim={_mid_dim}")
    pl_module = build_pl_module(
        Path(atoms_mapper), mattergen_pretrained,
        hidden_dim=alm.llm_hidden_dim, K=K, mid_dim=_mid_dim, device=device,
        model_path=model_path,
    )
    # Force everything onto device: build_pl_module's internal .to() is unreliable on single-GPU eval paths.
    pl_module = pl_module.to(device)
    if hasattr(pl_module, "diffusion_module"):
        pl_module.diffusion_module = pl_module.diffusion_module.to(device)
        if hasattr(pl_module.diffusion_module, "model"):
            pl_module.diffusion_module.model = pl_module.diffusion_module.model.to(device)
    return alm, tokenizer, pl_module, K


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alm_checkpoint", required=True,
                   help="Stage 2 ckpt dir (lora_adapter/ + projector_and_state.pt). Stage 3a "
                        "doesn't train the LLM, so this is just the Stage 2 ckpt used at training.")
    p.add_argument("--atoms_mapper", required=True,
                   help="Path to atoms_mapper.pt produced by the trainer")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mattergen_pretrained", default="mattergen_base",
                   help="Same value used at training time")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=2)
    p.add_argument("--diffusion_snr", type=float, default=None,
                   help="Sampling temperature analog. Scales Langevin-corrector "
                        "SNR (pos / cell) relative to defaults (0.4 / 0.2). "
                        "1.0 = default, <1.0 = warmer (more exploration), >1.0 = "
                        "cooler (more deterministic). None leaves defaults intact.")
    p.add_argument("--diffusion_guidance_factor", type=float, default=2.0,
                   help="Classifier-free guidance scale. 0 = unconditional, ~2.0 typical")
    p.add_argument("--num_atoms_distribution", default="ALEX_MP_20",
                   help="Empirical distribution for sampling structure size")
    p.add_argument("--record_trajectories", action="store_true",
                   help="Save denoising trajectories (extra disk; nice for figures)")
    p.add_argument("--allowed_elements", type=str, default=None,
                   help="Comma-separated element symbols (e.g. 'Cu,Ni'). When set, the "
                        "score model's atomic-number logits are masked at every denoising "
                        "step so only these elements can be sampled. CFG-safe.")
    # Feynman-Kac steering
    p.add_argument("--fk_n_particles", type=int, default=0,
                   help="Number of FK particles. 0 = disabled (default). When > 0, the "
                        "sampler's batch_size is set to fk_n_particles and the sampler-loop "
                        "is wrapped to evaluate per-step rewards and resample particles.")
    p.add_argument("--fk_rewards", type=str, default="stoich_match:1.0",
                   help="Semicolon-separated 'name:weight' reward spec, e.g. "
                        "'stoich_match:1.0' or 'stoich_match:1.0;in_set_completeness:0.3'.")
    p.add_argument("--fk_target_counts", type=str, default=None,
                   help="Comma-separated 'Sym:n' for stoich_match, e.g. 'V:2,Ga:1,Fe:1'. "
                        "Mutually exclusive with --fk_target_counts_from_prompt_json.")
    p.add_argument("--fk_target_counts_from_prompt_json", type=str, default=None,
                   help="'<path>:<tag>' lookup, e.g. "
                        "'scripts/eval_prompts/id_prompt_targets.json:v2gafe'. "
                        "Single source of truth across the bash wrapper and the CLI.")
    p.add_argument("--fk_resample_every", type=int, default=5,
                   help="Resample cadence (every k denoising steps). Default 5.")
    p.add_argument("--fk_t_start_frac", type=float, default=0.5,
                   help="Defer FK steering until t < T·(1 − t_start_frac). Default 0.5.")
    p.add_argument("--fk_lambda", type=float, default=1.0,
                   help="Weight on log_w accumulation. Default 1.0; lower (e.g. 0.3) if "
                        "diversity collapses or log_w_clip frequency is high.")
    p.add_argument("--fk_potential", type=str, default="diff",
                   choices=["diff", "sum", "max"],
                   help="Per-step potential. 'diff' (default, Singhal et al.) is the safest.")
    p.add_argument("--fk_ess_threshold_frac", type=float, default=0.5,
                   help="Resample only when ESS < N · frac. Default 0.5.")
    p.add_argument("--fk_keep_top_k", type=int, default=-1,
                   help="-1 = keep all N particles (default). >0 = keep K highest-log_w.")
    p.add_argument("--fk_log_w_clip", type=float, default=10.0,
                   help="Clip cumulative log_w to ±clip per step. Default 10.0.")
    p.add_argument("--fk_constrain_n_atoms_to_target_multiple", action="store_true",
                   help="Restrict per-particle N_p sampled from --num_atoms_distribution to "
                        "multiples of sum(target_counts). For V₂GaFe (sum=4) only "
                        "{4,8,12,16,20} stay; for LaClAu (sum=3) only {3,6,9,…}. Without "
                        "this, ALEX_MP_20's mode at 4 atoms forces multiset rounding to "
                        "drop target elements (LiMnPO₅H₂ → 0×Mn) so per-atom Hungarian "
                        "satisfies presence but never strict ratio.")
    p.add_argument("--fk_stratify_resample_by_n_atoms", action="store_true",
                   help="Resample WITHIN each N_p group separately so each group keeps "
                        "its representation. Mitigates N_p collapse where one N_p "
                        "value crowds out the others. Default OFF; "
                        "fk_trajectory.pt's resample_log surfaces collapse without it.")
    p.add_argument("--fk_n_atoms_exact_sum_target", action="store_true",
                   help="STRICT: force N_p = sum(target_counts) exactly (no multiples). "
                        "For SrTiO3 → only N_p=5; for V2GaFe → only N_p=4. Wins over "
                        "--fk_constrain_n_atoms_to_target_multiple if both are set. "
                        "Pairs naturally with --fk_enforce_target_counts to guarantee "
                        "exact-stoichiometry generations.")
    p.add_argument("--fk_enforce_target_counts", action="store_true",
                   help="Post-hoc Hungarian Z-override at end of denoising: for each "
                        "particle, reassign atom labels to the target multiset using the "
                        "model's final atomic_numbers probs. Positions and lattice are "
                        "untouched. Forces reduced_formula_match=True for any cell where "
                        "N_p % sum(target_counts) == 0. Best paired with "
                        "--fk_n_atoms_exact_sum_target so every particle is exactly the "
                        "right size and exactly the right composition.")
    p.add_argument("--fk_physical_bounds_path", type=str, default=None,
                   help="Path to JSON of empirical physical-prior bounds (output of "
                        "archive/data_prep/calibrate_physical_priors.py). Required if "
                        "--fk_rewards includes 'physical_sanity'.")
    # peaked atomic_numbers init at sampler t=T
    p.add_argument("--init_types_at_target", action="store_true",
                   help="Override atomic_numbers immediately after _sample_prior with a "
                        "permuted target multiset (Z values, integer). Lets the denoiser "
                        "see real types from t=T → positions can co-evolve. Requires "
                        "--fk_target_counts (or --fk_target_counts_from_prompt_json) and "
                        "--fk_n_atoms_exact_sum_target. Independent of FK; combine with "
                        "--fk_n_particles 0 to isolate the init-prior effect, or pair "
                        "with FK steering as you like.")
    # hard composition lock via inpainting-style clamp
    p.add_argument("--lock_types_via_mask", action="store_true",
                   help="Hard composition lock: override atomic_numbers after _sample_prior "
                        "with a permuted target multiset and re-clamp at every denoising "
                        "step. Mathematically equivalent to MatterGen's mask+conditioning_data "
                        "interface but works around lerp's int-dtype incompatibility. Requires "
                        "--fk_n_atoms_exact_sum_target. Mutually exclusive with "
                        "--bias_types_via_score and --init_types_at_target.")
    # gradual score-fn type biasing
    p.add_argument("--bias_types_via_score", action="store_true",
                   help="Wrap the score_fn so atomic_numbers logits are biased toward the "
                        "target composition. Schedule: q(z|site,t) is uniform over target "
                        "elements at high t, peaked Hungarian assignment at low t. Mutually "
                        "exclusive with --lock_types_via_mask and --init_types_at_target.")
    p.add_argument("--type_bias_t_start", type=float, default=0.5,
                   help="t at which type biasing begins (uniform-over-target). Default 0.5.")
    p.add_argument("--type_bias_t_end", type=float, default=0.05,
                   help="t at which type biasing reaches full peaked assignment. Default 0.05.")
    p.add_argument("--diffusion_seed", type=int, default=None,
                   help="Optional PRNG seed (torch/cuda/numpy/random). Lets callers retry "
                        "with bumped seeds when MatterGen's GemNet hits the stochastic "
                        "torch.max(empty) crash mid-denoise.")
    # LLM-temperature CoT sampling between the anchor and the [atoms_i] tokens (cot_tokens=0 = deterministic)
    p.add_argument("--cot_tokens", type=int, default=0,
                   help="K' = number of LLM-sampled tokens to splice between the "
                        "assistant anchor and the K=8 [atoms_i] tokens. 0 (default) "
                        "= deterministic, no CoT.")
    p.add_argument("--llm_temperature", type=float, default=1.0,
                   help="Sampling temperature for the CoT (only used when "
                        "--cot_tokens > 0).")
    p.add_argument("--cot_top_p", type=float, default=0.9,
                   help="Nucleus-sampling cutoff for the CoT (only used when "
                        "--cot_tokens > 0).")
    p.add_argument("--cot_seed", type=int, default=None,
                   help="PRNG seed for the CoT sampling. Defaults to --diffusion_seed "
                        "when set, else 1337. Per-prompt offset is added in the "
                        "multi-prompt path so reordering prompts doesn't perturb "
                        "individual outputs.")
    main(p.parse_args())
