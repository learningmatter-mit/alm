#!/usr/bin/env python3
"""LLM-judge knowledge/language retention probe for ALM bridge checkpoints."""

import argparse
import asyncio
import json
import sys
import os
import time
from collections import Counter
from pathlib import Path
import torch

from paths import CHECKPOINTS, EVAL_RESULTS

_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGE2 = os.path.join(CHECKPOINTS, "alm_checkpoints/stage2_checkpoints/step=12000")

# Factual questions paired with acceptable keywords for the cheap cross-check.
FACTUAL = [
    ("What is the canonical chemical formula for a perovskite, in the form ABXn? Just the formula.", ["abx3", "abo3"]),
    ("What is the canonical chemical formula for a spinel? Just the formula.", ["ab2o4", "ab2x4"]),
    ("Rocksalt (halite) has the chemical formula of which simple ionic compound? Just the formula.", ["nacl"]),
    ("What is the chemical formula of strontium titanate? Just the formula.", ["srtio3", "srtio₃"]),
    ("What is the chemical formula of calcium titanate? Just the formula.", ["catio3", "catio₃"]),
    ("Give the chemical formula of barium titanate. Just the formula.", ["batio3", "batio₃"]),
    ("What is the chemical formula of magnetite? Just the formula.", ["fe3o4", "fe₃o₄"]),
    ("In a perovskite of formula ABX3, where is the B-site cation located in the unit cell? One short sentence.",
     ["center", "octahedral", "centre", "centro"]),
    ("Which space group does the canonical cubic perovskite belong to? Just the symbol.", ["pm-3m", "pm3m", "221"]),
    ("What is 23 + 19? Just the number.", ["42"]),
    ("What is 7 times 8? Just the number.", ["56"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("Water has the chemical formula:", ["h2o", "h₂o"]),
]
# Open-ended (no keyword; judged for knowledge + coherence).
OPEN = [
    "Describe the structure of an ABO3 cubic perovskite: what sits on the A-site, the B-site, and the oxygen sublattice?",
    "Explain in 2-3 sentences why doping a perovskite oxide on the B-site can change its electronic properties.",
    "Name two well-known perovskite-structured oxides and one application of each.",
    "Briefly explain the difference between a rocksalt and a perovskite crystal structure.",
    "In one paragraph, explain what makes a material a good thermoelectric.",
    "Write one clear sentence explaining what a band gap is to a first-year student.",
]

JUDGE_SYSTEM = """You grade a materials-science/chemistry assistant's answer for CORRECTNESS and COHERENCE \
(we are checking whether fine-tuning eroded its knowledge/language). Given the question, optional expected-answer \
hints, and the model's response, output STRICT JSON only:
{"score": 2 | 1 | 0, "reason": "<=20 words"}
2 = correct AND coherent (fluent, on-topic, no repetition loops).
1 = partially correct, or correct-but-rambling/verbose, or minor error.
0 = wrong, hallucinated, incoherent, empty, or stuck in a repetition loop."""


def build_msgs(item):
    hint = f"\nExpected-answer hints (any acceptable): {item['keywords']}" if item.get("keywords") else ""
    user = (f"QUESTION: {item['question']}{hint}\n\nMODEL RESPONSE:\n{item['response']}\n\n"
            f"Grade per the rubric. STRICT JSON only.")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def detect_loop(text, window=4):
    toks = text.split()
    if len(toks) < window * 3:
        return False
    grams = [" ".join(toks[i:i+window]) for i in range(len(toks)-window+1)]
    c = Counter(grams)
    return c.most_common(1)[0][1] >= 4 if c else False


def load_model(mode, bridge_dir, k, device, hf_model_path=None):
    from loader import load_alm
    if mode == "base":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2").to(device).eval()
        return llm, tok, None
    if mode == "hf":
        # External HF causal-LM (e.g. CrystalReasoner) judged with its own tokenizer.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert hf_model_path, "mode=hf requires --hf_model_path"
        print(f"[knowledge-judge] loading external HF model from {hf_model_path}", flush=True)
        tok = AutoTokenizer.from_pretrained(hf_model_path)
        llm = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2").to(device).eval()
        return llm, tok, None
    # Full-FT bridge ckpts store the whole Qwen3 (no lora_adapter/); load it directly, skip the LoRA overlay.
    _is_full_ft = mode == "part" and bridge_dir is not None and \
        (Path(bridge_dir) / "llm_full_ft" / "qwen3_state_dict.pt").exists()
    if _is_full_ft:
        print(f"[knowledge-judge] full-FT checkpoint detected → loading full Qwen3 from "
              f"{bridge_dir}/llm_full_ft (bridge-LoRA overlay skipped)", flush=True)
        alm, tok = load_alm(checkpoint=bridge_dir, merge_lora=True, use_cached_embeddings=True,
                            device=device, num_output_atom_tokens=k)
    else:
        alm, tok = load_alm(checkpoint=STAGE2, merge_lora=True, use_cached_embeddings=True,
                            device=device, num_output_atom_tokens=k)
        if mode == "part":
            from eval_bridge_csp import apply_bridge_lora
            apply_bridge_lora(alm, Path(bridge_dir) / "lora_adapter", device)
    alm.eval()
    return alm.llm, tok, alm


@torch.no_grad()
def generate(llm, tok, question, device, max_new=96):
    msgs = [{"role": "system", "content": "You are a helpful, knowledgeable materials-science assistant."},
            {"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False, return_tensors="pt").to(device)
    out = llm.generate(ids, max_new_tokens=max_new, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "stage2", "part", "hf"], required=True)
    ap.add_argument("--bridge_dir", default=None, help="step=N dir (mode=part)")
    ap.add_argument("--hf_model_path", default=None,
                    help="external HF causal-LM dir (mode=hf, e.g. CrystalReasoner)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--out_dir", default=os.path.join(EVAL_RESULTS, "knowledge_retention"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[kret:{args.tag}] loading model (mode={args.mode}) ...", flush=True)
    llm, tok, _ = load_model(args.mode, args.bridge_dir, args.k, device, hf_model_path=args.hf_model_path)

    items = []
    t0 = time.time()
    for q, kw in FACTUAL:
        r = generate(llm, tok, q, device, max_new=64)
        items.append({"question": q, "keywords": kw, "response": r,
                      "kw_pass": any(k in r.lower() for k in kw), "loop": detect_loop(r)})
    for q in OPEN:
        r = generate(llm, tok, q, device, max_new=128)
        items.append({"question": q, "keywords": None, "response": r,
                      "kw_pass": None, "loop": detect_loop(r)})
    print(f"[kret:{args.tag}] generated {len(items)} responses in {time.time()-t0:.0f}s; judging ...", flush=True)

    from llm_judge import batch_judge, parse_score
    verdicts = asyncio.run(batch_judge(items, build_msgs, model=args.judge_model, concurrency=16))
    scores = [parse_score(v, default=0) for v in verdicts]
    for it, v, s in zip(items, verdicts, scores):
        it["judge_score"] = s
        it["judge_reason"] = (v or {}).get("reason", "")

    n = len(items)
    mean_score = sum(scores) / max(1, n)
    kw_items = [it for it in items if it["kw_pass"] is not None]
    kw_rate = sum(it["kw_pass"] for it in kw_items) / max(1, len(kw_items))
    loop_rate = sum(it["loop"] for it in items) / max(1, n)
    summary = {"tag": args.tag, "mode": args.mode, "n": n,
               "judge_mean_0to2": round(mean_score, 3),
               "judge_frac_score2": round(sum(s == 2 for s in scores) / max(1, n), 3),
               "keyword_pass_rate": round(kw_rate, 3),
               "loop_rate": round(loop_rate, 3)}
    (out_dir / f"{args.tag}.json").write_text(json.dumps({"summary": summary, "items": items}, indent=2))
    print(f"\n===== KNOWLEDGE RETENTION — {args.tag} =====")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir/f'{args.tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
