"""Park et al. Mat2MCQ: 4-way multiple-choice over elements present in a material."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import _atoms_struct_to_ase

from loader import load_alm
from text_generation import generate_batch
from parsers import detect_leak, extract_choice
from runs import run_dir, write_run
from paths import DATA_ROOT


_PERIODIC = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
    "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi",
]


def _synthesize_mcq(parquet_path, max_n, split_seed):
    """Yield (i, atoms, choices, gold_letter) from the parquet's atoms struct."""
    import polars as pl
    df = pl.read_parquet(parquet_path, columns=["atoms"]).head(max_n)
    rng = random.Random(split_seed)
    for i, atoms_struct in enumerate(df["atoms"].to_list()):
        atoms = _atoms_struct_to_ase(atoms_struct)
        elements = sorted(set(atoms.get_chemical_symbols()))
        if not elements:
            continue
        target_el = rng.choice(elements)
        distractor_pool = [e for e in _PERIODIC if e not in set(elements)]
        rng.shuffle(distractor_pool)
        choices = [target_el] + distractor_pool[:3]
        rng.shuffle(choices)
        gold = chr(ord("A") + choices.index(target_el))
        yield i, atoms, choices, gold


def _build_sample(atoms, choices, tokenizer, max_num_tokens):
    user = ("<atoms>\nWhich of the following elements is present in this material?\n"
            f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n"
            "Respond with only the letter.")
    messages = [
        {"role": "system", "content": "You are a materials science expert."},
        {"role": "user", "content": user},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, truncation=True, max_length=max_num_tokens,
    )
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    return {
        "input_ids": ids,
        "labels": torch.full_like(ids, -100),
        "attention_mask": torch.ones_like(ids),
        "atom_rows": [atoms],
        "id": None,
    }


def _collate(batch):
    return {
        "input_ids":      [b["input_ids"].squeeze(0)      for b in batch],
        "labels":         [b["labels"].squeeze(0)         for b in batch],
        "attention_mask": [b["attention_mask"].squeeze(0) for b in batch],
        "atom_rows":      [b["atom_rows"][0]              for b in batch],
        "id":             [b["id"]                        for b in batch],
    }


def _eval_text_mcq(model, tokenizer, mcqs, args):
    """Text-only MCQ eval; returns (predictions, correct, total, n_leaked)."""
    predictions, correct, total, n_leaked = [], 0, 0, 0
    samples_buf, gold_buf, id_buf = [], [], []

    def flush():
        nonlocal correct, total, n_leaked
        if not samples_buf:
            return
        batch = _collate(samples_buf)
        gens = generate_batch(model, batch, max_new_tokens=args.max_new_tokens, atomistic=False,
                              block_leak_tokens=args.block_leak_tokens)
        for sid, gen, gold in zip(batch["id"], gens, gold_buf):
            pred = extract_choice(gen)
            leaked = detect_leak(gen)
            ok = pred == gold and not leaked
            predictions.append({"id": sid, "target": gold, "generated": gen,
                                "parsed": pred, "leaked": leaked, "ok": ok})
            total += 1
            correct += int(ok)
            if leaked:
                n_leaked += 1
        samples_buf.clear(); gold_buf.clear(); id_buf.clear()

    for q in mcqs:
        choices = q["choices"]
        user = (f"{q['question']}\n"
                f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n"
                "Respond with only the letter.")
        messages = [
            {"role": "system", "content": "You are a materials science expert."},
            {"role": "user", "content": user},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, truncation=True, max_length=args.max_num_tokens,
        )
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        samples_buf.append({
            "input_ids": ids,
            "labels": torch.full_like(ids, -100),
            "attention_mask": torch.ones_like(ids),
            "atom_rows": [None],   # unused on atomistic=False
            "id": q.get("id"),
        })
        gold_buf.append(q["gold"])
        id_buf.append(q.get("id"))
        if len(samples_buf) >= args.batch_size:
            flush()
    flush()
    return predictions, correct, total, n_leaked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--narrative_parquet",
                   default=os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials/mp_3d_2020_gpt_narratives.parquet"))
    p.add_argument("--mcq_jsonl", default=None,
                   help="Park et al.'s generated MCQs (one JSON per line: "
                        "{'id', 'question', 'choices': [4], 'gold': 'A'-'D'}). "
                        "When set, --narrative_parquet is ignored.")
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=8)
    p.add_argument("--no_merge_lora", action="store_true")
    p.add_argument("--block_leak_tokens", action="store_true",
                   help="Suppress markdown-image / URL token openers at decode time (off by default).")
    args = p.parse_args()

    # MCQ atoms are in-memory, so load with the live OrbV3 encoder.
    model, tokenizer = load_alm(
        checkpoint=args.checkpoint, merge_lora=not args.no_merge_lora,
        use_cached_embeddings=False,
    )

    if args.mcq_jsonl:
        # Text-only path: no atoms, runs through generate_batch(atomistic=False).
        with open(args.mcq_jsonl) as f:
            mcqs = [json.loads(line) for line in f][: args.max_samples]
        predictions, correct, total, n_leaked = _eval_text_mcq(model, tokenizer, mcqs, args)
        metrics = {"n_total": total, "accuracy": (correct / max(1, total)),
                   "n_leaked": n_leaked,
                   "leak_rate": n_leaked / max(1, total),
                   "source": "mcq_jsonl"}
        write_run(run_dir("mat2mcq", args.checkpoint), metrics, predictions)
        print(metrics)
        return

    samples_buf, gold_buf, ids_buf, predictions = [], [], [], []
    correct, total, n_leaked = 0, 0, 0

    def flush():
        nonlocal correct, total, n_leaked
        if not samples_buf:
            return
        for i in range(len(samples_buf)):
            samples_buf[i]["id"] = ids_buf[i]
        batch = _collate(samples_buf)
        gens = generate_batch(model, batch, max_new_tokens=args.max_new_tokens, atomistic=True,
                              block_leak_tokens=args.block_leak_tokens)
        for sid, gen, gold in zip(batch["id"], gens, gold_buf):
            pred = extract_choice(gen)
            leaked = detect_leak(gen)
            ok = pred == gold and not leaked
            predictions.append({"id": sid, "target": gold, "generated": gen,
                                "parsed": pred, "leaked": leaked, "ok": ok})
            total += 1
            correct += int(ok)
            if leaked:
                n_leaked += 1
        samples_buf.clear()
        gold_buf.clear()
        ids_buf.clear()

    for i, atoms, choices, gold in _synthesize_mcq(
        args.narrative_parquet, args.max_samples, args.split_seed,
    ):
        samples_buf.append(_build_sample(atoms, choices, tokenizer, args.max_num_tokens))
        gold_buf.append(gold)
        ids_buf.append(f"mcq/{i}")
        if len(samples_buf) >= args.batch_size:
            flush()
    flush()

    metrics = {"n_total": total, "accuracy": (correct / max(1, total)),
               "n_leaked": n_leaked, "leak_rate": n_leaked / max(1, total)}
    write_run(run_dir("mat2mcq", args.checkpoint), metrics, predictions)
    print(metrics)


if __name__ == "__main__":
    main()
