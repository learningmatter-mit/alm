import argparse
from pathlib import Path

import torch

from utils import AtomisticLanguageDataset
from model import AtomisticLanguageModel


def generate_from_sample(model, sample, max_new_tokens=512, temperature=0.6, top_p=0.95, repetition_penalty=1.3):
    """Mirror the training forward pass, then autoregressively decode."""
    device = model.device

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if "atom_embed" in sample:
                atomistic_features, n_atoms = model.encode_cached_atoms([sample["atom_embed"]])
            else:
                atomistic_features, n_atoms = model.encode_atoms(sample["atom_rows"])
            atomistic_features = torch.split(atomistic_features, n_atoms)

            messages = [
                {"role": "system", "content": "You are an expert at materials science and atomistic structure."},
                {"role": "user", "content": "<atoms>\nDescribe the structure of this material."},
            ]
            prompt_ids = model.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

            embed_layer = model.llm.get_input_embeddings()
            text_embeds = [embed_layer(prompt_tensor)]
            dummy_labels = [torch.full((prompt_tensor.shape[0],), -100, dtype=torch.long, device=device)]
            attn_mask = [torch.ones(prompt_tensor.shape[0], dtype=torch.long, device=device)]
            inputs_embeds, _, attention_mask, _ = model._merge_embeddings(
                text_embeds, atomistic_features, [prompt_tensor], dummy_labels, attn_mask,
            )

            output_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
            )

    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cached = bool(args.cached_embs_path)
    model = AtomisticLanguageModel(
        llm_name="Qwen/Qwen3-8B",
        atomistic_model_name="orb_v3_direct_20_omat",
        device=device,
        use_cached_embeddings=use_cached,
        max_atoms=max(1, args.max_num_tokens - 256),
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "projector_state_dict" in checkpoint:
        model.projector.load_state_dict(checkpoint["projector_state_dict"])
    else:
        model.projector.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    folder = Path(args.data_folder)
    dataset = AtomisticLanguageDataset(
        tokenizer=model.tokenizer,
        db_path=(folder / f"{args.split}.db") if not use_cached else None,
        csv_path=folder / f"{args.split}.csv",
        thinking=False,
        max_num_tokens=args.max_num_tokens,
        dataset_name=folder.name,
        cached_embs_path=args.cached_embs_path,
    )

    for i in range(min(args.n_samples, len(dataset))):
        sample = dataset[i]
        sample_id = sample["id"]
        ground_truth = dataset._descriptions[i]

        response = generate_from_sample(
            model, sample,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        formula = ""
        if "atom_rows" in sample:
            formula = sample["atom_rows"][0].toatoms().get_chemical_formula()

        print(f"\n{'=' * 60}")
        print(f"Sample {i} | id={sample_id} | {formula}")
        print(f"{'=' * 60}")
        print(f"GROUND TRUTH:\n{ground_truth[:300]}")
        print("---")
        print(f"GENERATED:\n{response[:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Single dataset folder in LLM4Mat-Bench layout, e.g. /tmp/LLM4Mat-Bench/oqmd.")
    parser.add_argument("--cached_embs_path", type=str, default=None,
                        help="Full path to {dataset}/embeddings/{model}_{split}_atom.flat.bin. "
                             "If set, skips DB/live OrbV3 (cached-embedding path, matches training).")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_num_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()
    evaluate(args)
