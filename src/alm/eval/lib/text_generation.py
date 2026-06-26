"""Batched greedy generation on inputs_embeds for ALM eval."""

import torch

# Markdown-image / URL openers passed as bad_words_ids to block leak paths.
_LEAK_PREFIXES = ["![", "![]", "![](", "https://", "http://"]
_LEAK_CACHE_ATTR = "_alm_eval_bad_words_ids"


def _leak_bad_words_ids(tokenizer):
    cached = getattr(tokenizer, _LEAK_CACHE_ATTR, None)
    if cached is not None:
        return cached
    ids = []
    for s in _LEAK_PREFIXES:
        toks = tokenizer(s, add_special_tokens=False)["input_ids"]
        if toks:
            ids.append(toks)
    setattr(tokenizer, _LEAK_CACHE_ATTR, ids)
    return ids


@torch.no_grad()
def generate_batch(model, batch, max_new_tokens=512, atomistic=True,
                   block_leak_tokens=False):
    """Returns generated text per sample (newly generated tokens only)."""
    device = model.device
    tokenizer = model.tokenizer

    input_ids = [t.squeeze(0).to(device) for t in batch["input_ids"]]
    labels    = [t.squeeze(0).to(device) for t in batch["labels"]]
    attn_mask = [t.squeeze(0).to(device) for t in batch["attention_mask"]]

    # Prompt = labels==-100 positions; fall back to first 50 tokens for raw-LM samples.
    prompt_ids_list = []
    for ids, labs in zip(input_ids, labels):
        mask = labs == -100
        if mask.any():
            n_prompt = int(mask.sum().item())
            prompt = ids[:n_prompt]
        else:
            prompt = ids[:min(50, len(ids))]
        prompt_ids_list.append(prompt)

    embed_layer = model.llm.get_input_embeddings()
    text_embeds = [embed_layer(p) for p in prompt_ids_list]
    dummy_labels = [torch.full((p.shape[0],), -100, dtype=torch.long, device=device)
                    for p in prompt_ids_list]
    prompt_attn = [torch.ones(p.shape[0], dtype=torch.long, device=device)
                   for p in prompt_ids_list]

    if atomistic:
        if "atom_embeds" in batch:
            atom_features, n_atoms = model.encode_cached_atoms(batch["atom_embeds"])
        else:
            atom_features, n_atoms = model.encode_atoms(batch["atom_rows"])
        atom_features = torch.split(atom_features, n_atoms)
    else:
        # Empty per-sample tensors so _merge_embeddings's zero-atoms branch fires.
        embed_dim = text_embeds[0].shape[-1]
        atom_features = [torch.zeros(0, embed_dim, dtype=text_embeds[0].dtype, device=device)
                         for _ in prompt_ids_list]

    inputs_embeds, _, attention_mask, _ = model._merge_embeddings(
        text_embeds, atom_features, prompt_ids_list, dummy_labels, prompt_attn,
    )

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    bad_words_ids = _leak_bad_words_ids(tokenizer) if block_leak_tokens else None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            bad_words_ids=bad_words_ids,
        )
    # With inputs_embeds, generate returns only newly generated tokens (no prefix to strip).
    return [tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in out_ids]
