"""Generate from a Stage 2 LoRA checkpoint and print prompt + ground-truth + model output."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm"))
from utils import (
    ArxivAbstractDataset, CamelAIDataset, MaScQADataset,
    FullAtomisticLanguageDataset, GPTNarrativeDataset,
    describe_tasks_for_dataset, property_tasks_for_dataset,
    describe_tasks_for_narrative, applications_tasks_for_narrative,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
from loader import load_alm


def build_dataset(args, tok):
    b = args.bucket
    if b == "arxiv":
        return ArxivAbstractDataset(tok, args.arxiv_parquet, args.max_num_tokens)
    if b == "camel":
        return CamelAIDataset(tok, args.camel_jsonl, max_num_tokens=args.max_num_tokens)
    if b == "mascqa":
        return MaScQADataset(tok, args.mascqa_json, args.mascqa_xlsx,
                             max_num_tokens=args.max_num_tokens)
    if b in ("describe", "property"):
        tasks_fn = describe_tasks_for_dataset if b == "describe" else property_tasks_for_dataset
        return FullAtomisticLanguageDataset(
            tokenizer=tok, split="train", parent_folder=args.data_parent_path,
            max_num_tokens=args.max_num_tokens,
            cached_embs_parent_path=args.cached_embs_parent_path,
            tasks=tasks_fn,
        )
    if b in ("narrate", "explain"):
        name = args.narrative_name
        task_fn = describe_tasks_for_narrative if b == "narrate" else applications_tasks_for_narrative
        cache = Path(f"{args.narrative_cache_dir}/{name}/embeddings/orb_v3_direct_20_omat_atom.flat.bin")
        return GPTNarrativeDataset(
            tokenizer=tok,
            parquet_path=f"{args.narrative_parquet_dir}/{name}_gpt_narratives.parquet",
            cached_embs_path=str(cache) if cache.exists() else None,
            max_num_tokens=args.max_num_tokens, dataset_name=name, tasks=task_fn(name),
        )
    raise ValueError(f"unknown bucket {b}")


def load_model(args, device):
    """Load Stage 2 (LoRA + projector) or Stage 1 (projector only)."""
    # merge_lora=False keeps adapters live so runtime adapter-swap ablations stay possible.
    model, _ = load_alm(
        checkpoint=args.checkpoint or None,
        stage1_projector=args.stage1_projector or None,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        merge_lora=False, device=device,
    )
    print(f"Loaded {'Stage 2' if args.checkpoint else 'Stage 1'} model")
    return model


@torch.no_grad()
def generate_one(model, sample, device, max_new_tokens, temperature, top_p, repetition_penalty):
    ids = sample["input_ids"].squeeze(0).tolist()
    labs = sample["labels"].squeeze(0).tolist()
    prompt_ids = [i for i, l in zip(ids, labs) if l == -100]
    target_ids = [i for i, l in zip(ids, labs) if l != -100]
    # Arxiv (raw LM) has no masked prompt; seed from the first ~50 tokens.
    if not prompt_ids:
        prompt_ids = ids[:min(50, len(ids))]
        target_ids = ids[len(prompt_ids):]

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    embed_layer = model.llm.get_input_embeddings()
    text_embeds = [embed_layer(prompt_tensor)]

    if "atom_embed" in sample:
        atomistic_features, n_atoms = model.encode_cached_atoms([sample["atom_embed"]])
    else:
        atomistic_features, n_atoms = model.encode_atoms(sample["atom_rows"])
    atomistic_features = torch.split(atomistic_features, n_atoms)

    dummy_labels = [torch.full((prompt_tensor.shape[0],), -100, dtype=torch.long, device=device)]
    attn_mask = [torch.ones(prompt_tensor.shape[0], dtype=torch.long, device=device)]
    inputs_embeds, _, attention_mask, _ = model._merge_embeddings(
        text_embeds, atomistic_features, [prompt_tensor], dummy_labels, attn_mask,
    )
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
        )
    return prompt_ids, target_ids, out_ids[0].tolist()


def main(args):
    if not args.checkpoint and not args.stage1_projector:
        raise SystemExit("Pass either --checkpoint <stage2 dir> or --stage1_projector <.pt>")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)
    tok = model.tokenizer
    ds = build_dataset(args, tok)

    print(f"[{args.bucket}] dataset size = {len(ds):,}")
    g = torch.Generator().manual_seed(args.seed)
    for i in torch.randperm(len(ds), generator=g)[:args.n].tolist():
        sample = ds[i]
        prompt_ids, target_ids, gen_ids = generate_one(
            model, sample, device, args.max_new_tokens,
            args.temperature, args.top_p, args.repetition_penalty,
        )
        print(f"\n{'='*70}\n{args.bucket}[{i}]  id={sample.get('id')}\n{'='*70}")
        print("-- PROMPT --");     print(tok.decode(prompt_ids,  skip_special_tokens=False))
        print("-- GROUND TRUTH --"); print(tok.decode(target_ids, skip_special_tokens=False))
        print("-- MODEL --");      print(tok.decode(gen_ids,    skip_special_tokens=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Path to a Stage 2 step=N/ checkpoint dir")
    p.add_argument("--stage1_projector", default=None,
                   help="Stage 1 projector .pt for baseline (no LoRA). Mutually exclusive with --checkpoint.")
    p.add_argument("--bucket", required=True,
                   choices=["arxiv", "camel", "mascqa", "describe", "property", "narrate", "explain"])
    p.add_argument("--narrative_name", default="dft_3d",
                   choices=["dft_3d", "mp_3d_2020", "aflow2", "oqmd"])
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--arxiv_parquet", default="/tmp/jarvis_arxiv.parquet")
    p.add_argument("--camel_jsonl",   default="/tmp/camel_ai.jsonl")
    p.add_argument("--mascqa_json",   default="/tmp/MaScQA/mascqa-eval.json")
    p.add_argument("--mascqa_xlsx",   default="/tmp/MaScQA/scoresheets/all_questions.xlsx")
    p.add_argument("--data_parent_path",        default="/tmp/LLM4Mat-Bench")
    p.add_argument("--cached_embs_parent_path", default="/tmp/cached_embs")
    p.add_argument("--narrative_parquet_dir",   default="/tmp/GPT-Narratives-for-Materials")
    p.add_argument("--narrative_cache_dir",     default="/tmp/cached_embs_narratives")
    args = p.parse_args()
    main(args)
