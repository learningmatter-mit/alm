"""MatText perovskites / KVRH / GVRH MAE via live OrbV3 from CIF strings."""

import argparse
from io import StringIO

import torch
from ase.io import read as ase_read
from datasets import load_dataset

from utils import _PROPERTY_PREDICTION_SYSTEM

from loader import load_alm
from text_generation import generate_batch
from parsers import detect_leak, extract_number
from metrics import mae
from runs import run_dir, write_run


# task -> (n0w0f/MatText config, property name in the prompt). Every config has `cif_p1` and `labels` columns.
_TASKS = {
    "perovskites": ("perovskites-train-filtered", "heat of formation"),
    "kvrh":        ("kvrh-train-filtered", "log10(bulk modulus)"),
    "gvrh":        ("gvrh-train-filtered", "log10(shear modulus)"),
}


def _build_sample(cif, prop_name, tokenizer, max_num_tokens):
    atoms = ase_read(StringIO(cif), format="cif")
    messages = [
        {"role": "system", "content": _PROPERTY_PREDICTION_SYSTEM},
        {"role": "user", "content": f"<atoms>\nProperty name: {prop_name}."},
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


def _run_task(model, tokenizer, task, args):
    config_name, prop_name = _TASKS[task]
    # n0w0f/MatText ships 5-fold CV splits, no "test" split; default fold_0.
    ds = load_dataset("n0w0f/MatText", config_name, split=args.fold)
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    preds, targets, predictions = [], [], []
    n_leaked = 0
    samples_buf, raw_targets_buf, ids_buf = [], [], []

    def flush():
        nonlocal n_leaked
        if not samples_buf:
            return
        for i in range(len(samples_buf)):
            samples_buf[i]["id"] = ids_buf[i]
        batch = _collate(samples_buf)
        gens = generate_batch(model, batch, max_new_tokens=args.max_new_tokens, atomistic=True,
                              block_leak_tokens=args.block_leak_tokens)
        for sid, gen, raw in zip(batch["id"], gens, raw_targets_buf):
            parsed = extract_number(gen)
            leaked = detect_leak(gen)
            ok = parsed is not None and raw is not None and not leaked
            predictions.append({"task": task, "id": sid, "target": raw,
                                "generated": gen, "parsed": parsed,
                                "leaked": leaked, "ok": ok})
            if leaked:
                n_leaked += 1
            if ok:
                preds.append(parsed)
                targets.append(float(raw))
        samples_buf.clear()
        raw_targets_buf.clear()
        ids_buf.clear()

    for i, row in enumerate(ds):
        target = float(row["labels"])
        samples_buf.append(_build_sample(row["cif_p1"], prop_name, tokenizer, args.max_num_tokens))
        raw_targets_buf.append(target)
        ids_buf.append(f"{task}/{i}")
        if len(samples_buf) >= args.batch_size:
            flush()
    flush()

    n_total = len(predictions)
    metrics = {"n_total": n_total, "n_valid": len(preds), "n_leaked": n_leaked,
               "validity_rate": len(preds) / max(1, n_total),
               "leak_rate":     n_leaked / max(1, n_total)}
    if preds:
        metrics["mae"] = mae(preds, targets)
    return metrics, predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tasks", default="perovskites,kvrh,gvrh")
    p.add_argument("--fold", default="fold_0",
                   choices=["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"],
                   help="MatText CV split to evaluate on; paper averages over all 5.")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--no_merge_lora", action="store_true")
    p.add_argument("--block_leak_tokens", action="store_true",
                   help="Suppress markdown-image / URL token openers at decode time (off by default).")
    args = p.parse_args()

    # Structures come from CIF strings, so load the live OrbV3 encoder.
    model, tokenizer = load_alm(
        checkpoint=args.checkpoint, merge_lora=not args.no_merge_lora,
        use_cached_embeddings=False,
    )

    metrics, predictions = {}, []
    for task in [t.strip() for t in args.tasks.split(",")]:
        if task not in _TASKS:
            print(f"[skip] unknown task {task}")
            continue
        print(f"[run] mattext/{task}")
        m, preds = _run_task(model, tokenizer, task, args)
        metrics[task] = m
        predictions.extend(preds)
        print(f"  {m}")

    write_run(run_dir("mattext", args.checkpoint), metrics, predictions)


if __name__ == "__main__":
    main()
