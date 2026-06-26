"""MaScQA held-out evaluation (validation split: 131 stratified-by-topic Qs)."""

import argparse
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import MaScQADataset, custom_collate_fn

from loader import load_alm
from text_generation import generate_batch
from parsers import detect_leak, extract_choice, extract_number
from metrics import accuracy, mae
from runs import run_dir, write_run
from paths import DATA_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mascqa_json", default=os.path.join(DATA_ROOT, "MaScQA/mascqa-eval.json"))
    p.add_argument("--mascqa_xlsx", default=os.path.join(DATA_ROOT, "MaScQA/scoresheets/all_questions.xlsx"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_num_tokens", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--no_merge_lora", action="store_true")
    p.add_argument("--atom_bidirectional_attention", action="store_true",
                   help="Load under bidirectional atom attention. REQUIRED for bidir-trained "
                        "Stage 2 ckpts — it's a runtime mask flag, not detectable from weights; "
                        "omitting it loads them causal → mismatch.")
    p.add_argument("--block_leak_tokens", action="store_true",
                   help="Suppress markdown-image / URL token openers at decode time (off by default).")
    args = p.parse_args()

    model, tokenizer = load_alm(checkpoint=args.checkpoint, merge_lora=not args.no_merge_lora,
                                atom_bidirectional_attention=args.atom_bidirectional_attention)

    ds = MaScQADataset(
        tokenizer=tokenizer, questions_json=args.mascqa_json,
        scoresheet_xlsx=args.mascqa_xlsx,
        thinking=False, max_num_tokens=args.max_num_tokens, split="validation",
    )
    print(f"[mascqa] held-out size = {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=custom_collate_fn)

    qid_to_idx = {q: i for i, q in enumerate(ds._qids)}
    mcq_pred, mcq_tgt, num_pred, num_tgt, predictions = [], [], [], [], []
    mcq_leaks = num_leaks = 0
    n_mcq_total = n_num_total = 0

    for batch in loader:
        gens = generate_batch(model, batch, max_new_tokens=args.max_new_tokens, atomistic=False,
                              block_leak_tokens=args.block_leak_tokens)
        for sid, gen in zip(batch["id"], gens):
            i = qid_to_idx[sid]
            target_str = ds._answers[i]
            qtype = ds._qtypes[i]
            tgt_letter = extract_choice(target_str)
            leaked = detect_leak(gen)
            row = {"id": sid, "qtype": str(qtype), "topic": ds._topics[i],
                   "target": target_str, "generated": gen, "leaked": leaked}
            if tgt_letter:
                n_mcq_total += 1
                pred_letter = extract_choice(gen)
                row["parsed"] = pred_letter
                row["mode"] = "mcq"
                row["ok"] = pred_letter is not None and not leaked
                if leaked:
                    mcq_leaks += 1
                if row["ok"]:
                    mcq_pred.append(pred_letter)
                    mcq_tgt.append(tgt_letter)
            else:
                n_num_total += 1
                tgt_num = extract_number(target_str)
                pred_num = extract_number(gen)
                row["parsed"] = pred_num
                row["mode"] = "numerical"
                row["ok"] = pred_num is not None and tgt_num is not None and not leaked
                if leaked:
                    num_leaks += 1
                if row["ok"]:
                    num_pred.append(pred_num)
                    num_tgt.append(tgt_num)
            predictions.append(row)

    metrics = {"n_total": len(predictions), "n_mcq": n_mcq_total, "n_numerical": n_num_total,
               "mcq_n_valid": len(mcq_pred), "mcq_n_leaked": mcq_leaks,
               "mcq_leak_rate": mcq_leaks / max(1, n_mcq_total),
               "mcq_validity_rate": len(mcq_pred) / max(1, n_mcq_total),
               "numerical_n_valid": len(num_pred), "numerical_n_leaked": num_leaks,
               "numerical_leak_rate": num_leaks / max(1, n_num_total),
               "numerical_validity_rate": len(num_pred) / max(1, n_num_total)}
    if mcq_pred:
        metrics["mcq_accuracy"] = accuracy(mcq_pred, mcq_tgt)
    if num_pred:
        metrics["numerical_mae"] = mae(num_pred, num_tgt)

    write_run(run_dir("mascqa", args.checkpoint), metrics, predictions)


if __name__ == "__main__":
    main()
