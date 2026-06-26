"""Language retention battery: MMLU, GSM8K, GPQA (Diamond chemistry filter). Run twice (--model alm, --model base); aggregator joins them."""

import argparse
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loader import load_alm
from parsers import detect_leak, extract_choice, extract_number, extract_gsm8k_answer
from text_generation import _leak_bad_words_ids
from metrics import accuracy
from runs import run_dir, write_run


_MMLU_PROMPT_HEADER = ("The following are multiple choice questions (with answers) "
                       "about {subject}.\n\n")


def _format_mmlu_q(q, choices):
    s = q + "\n"
    for letter, choice in zip("ABCD", choices):
        s += f"{letter}. {choice}\n"
    s += "Answer:"
    return s


def _build_mmlu_prompts(args, tokenizer):
    """Yields (prompt_text, gold_letter, subject) per test row."""
    test = load_dataset("cais/mmlu", "all", split="test")
    if args.mmlu_subjects:
        wanted = set(s.strip() for s in args.mmlu_subjects.split(","))
        test = test.filter(lambda r: r["subject"] in wanted)
    if args.mmlu_few_shot > 0:
        dev = load_dataset("cais/mmlu", "all", split="dev")
        by_subj = {}
        for r in dev:
            by_subj.setdefault(r["subject"], []).append(r)
    else:
        by_subj = {}
    for row in test:
        subj = row["subject"]
        prompt = _MMLU_PROMPT_HEADER.format(subject=subj.replace("_", " "))
        for fs in by_subj.get(subj, [])[: args.mmlu_few_shot]:
            prompt += _format_mmlu_q(fs["question"], fs["choices"])
            prompt += f" {chr(ord('A') + fs['answer'])}\n\n"
        prompt += _format_mmlu_q(row["question"], row["choices"])
        yield prompt, chr(ord("A") + row["answer"]), subj


def _build_gpqa_prompts():
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    ds = ds.filter(lambda r: "chem" in (r.get("High-level domain") or "").lower()
                            or "chem" in (r.get("Subdomain") or "").lower())
    rng = random.Random(42)
    for row in ds:
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[i] for i in order]
        gold_letter = chr(ord("A") + order.index(0))
        q = row["Question"] + "\n"
        for letter, choice in zip("ABCD", shuffled):
            q += f"{letter}. {choice}\n"
        q += "Answer:"
        yield q, gold_letter, "chemistry"


def _build_gsm8k_prompts():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for row in ds:
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        q = (f"Question: {row['question']}\n"
             f"Answer: Let's think step by step.")
        yield q, gold, "gsm8k"


def _greedy(model, tokenizer, prompt_text, max_new_tokens, device, block_leak_tokens=False):
    ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                    max_length=2048).to(device)
    bad_words_ids = _leak_bad_words_ids(tokenizer) if block_leak_tokens else None
    with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids,
        )
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["alm", "base"], required=True)
    p.add_argument("--checkpoint", default=None, help="required when --model alm")
    p.add_argument("--base_name", default="Qwen/Qwen3-8B")
    p.add_argument("--task", choices=["mmlu", "gpqa", "gsm8k", "all"], default="all")
    p.add_argument("--mmlu_subjects", default=None,
                   help="comma list, e.g. high_school_chemistry,college_chemistry; default = full MMLU")
    p.add_argument("--mmlu_few_shot", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--max_new_tokens_mcq", type=int, default=8)
    p.add_argument("--max_new_tokens_gsm", type=int, default=512)
    p.add_argument("--block_leak_tokens", action="store_true",
                   help="Suppress markdown-image / URL token openers at decode time (off by default).")
    p.add_argument("--atom_bidirectional_attention", action="store_true",
                   help="Load under bidirectional atom attention. REQUIRED for bidir-trained "
                        "Stage 2 ckpts (runtime mask flag, not weight-detectable).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "alm":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required when --model alm")
        alm, tokenizer = load_alm(checkpoint=args.checkpoint,
                                  atom_bidirectional_attention=args.atom_bidirectional_attention)
        llm = alm.llm
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_name)
        llm = AutoModelForCausalLM.from_pretrained(
            args.base_name, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device).eval()

    metrics = {"model": args.model}
    predictions = []

    def _run_mcq(name, gen):
        preds, golds = [], []
        n_total = n_leaked = 0
        for i, (prompt, gold, subj) in enumerate(gen):
            if args.max_samples and i >= args.max_samples:
                break
            n_total += 1
            out = _greedy(llm, tokenizer, prompt, args.max_new_tokens_mcq, device,
                          block_leak_tokens=args.block_leak_tokens)
            pred = extract_choice(out)
            leaked = detect_leak(out)
            ok = pred is not None and not leaked
            row = {"task": name, "subject": subj, "prompt": prompt[-300:],
                   "generated": out, "target": gold, "parsed": pred,
                   "leaked": leaked, "ok": ok}
            predictions.append(row)
            if leaked:
                n_leaked += 1
            if ok:
                preds.append(pred)
                golds.append(gold)
        return {f"{name}_n_valid": len(preds),
                f"{name}_n_total": n_total,
                f"{name}_n_leaked": n_leaked,
                f"{name}_leak_rate": n_leaked / max(1, n_total),
                f"{name}_validity_rate": len(preds) / max(1, n_total),
                f"{name}_accuracy": accuracy(preds, golds) if preds else None}

    def _run_gsm():
        n_total = n_leaked = n_valid = n_correct = 0
        for i, (prompt, gold, _) in enumerate(_build_gsm8k_prompts()):
            if args.max_samples and i >= args.max_samples:
                break
            n_total += 1
            out = _greedy(llm, tokenizer, prompt, args.max_new_tokens_gsm, device,
                          block_leak_tokens=args.block_leak_tokens)
            leaked = detect_leak(out)
            parsed = extract_gsm8k_answer(out)
            try:
                gold_num = float(gold)
            except ValueError:
                continue
            valid = parsed is not None and not leaked
            ok = valid and abs(parsed - gold_num) < 1e-3
            row = {"task": "gsm8k", "prompt": prompt[-300:], "generated": out,
                   "target": gold, "parsed": parsed, "leaked": leaked, "ok": ok}
            predictions.append(row)
            if leaked:
                n_leaked += 1
            if valid:
                n_valid += 1
            if ok:
                n_correct += 1
        return {"gsm8k_n": n_total,
                "gsm8k_n_valid": n_valid,
                "gsm8k_n_leaked": n_leaked,
                "gsm8k_leak_rate": n_leaked / max(1, n_total),
                "gsm8k_validity_rate": n_valid / max(1, n_total),
                "gsm8k_accuracy": (n_correct / n_valid) if n_valid else None}

    if args.task in ("mmlu", "all"):
        metrics.update(_run_mcq("mmlu", _build_mmlu_prompts(args, tokenizer)))
    if args.task in ("gpqa", "all"):
        metrics.update(_run_mcq("gpqa", _build_gpqa_prompts()))
    if args.task in ("gsm8k", "all"):
        metrics.update(_run_gsm())

    # ALM_EVAL_RUN_ID disambiguates run dirs that share `step=N`.
    import os as _os
    env_rid = _os.environ.get("ALM_EVAL_RUN_ID")
    if env_rid:
        rid = f"{args.model}_{env_rid}"
    else:
        legacy = (Path(args.checkpoint).name if args.checkpoint
                  else args.base_name.replace('/', '_'))
        rid = f"{args.model}_{legacy}"
    out_dir = run_dir("language_retention",
                      args.checkpoint or args.base_name.replace("/", "_"),
                      run_id=rid)
    write_run(out_dir, metrics, predictions)


if __name__ == "__main__":
    main()
