"""Structure-blind text-only Qwen3-8B baseline for property prediction (vLLM); --bench llm4mat | mat2props."""

import argparse
import json
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import _DATASET_PROPERTIES, _NARRATIVE_PROPERTIES
from parsers import detect_leak, extract_number
from metrics import mae, mad_mae_ratio
from paths import DATA_ROOT
from runs import run_dir


def _flush(out_dir, metrics, pred_fh):
    """Per-property checkpoint so a crash only loses the in-flight property."""
    pred_fh.flush()
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


_SYSTEM_TEXT_ONLY = (
    "You are a material scientist. "
    "Given the chemical formula and a brief description of a crystalline "
    "material, predict its property. "
    'The output must be in JSON format, e.g. {"property_name": predicted_value}. '
    "Answer as precisely and concisely as possible."
)
_SYSTEM_FORMULA_ONLY = (
    "You are a material scientist. "
    "Given the chemical formula of a crystalline material, predict its property. "
    'The output must be in JSON format, e.g. {"property_name": predicted_value}. '
    "Answer as precisely and concisely as possible."
)


def _make_messages(system, user):
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _chat_batch(llm, sampling_params, all_messages):
    """vLLM batched chat; enable_thinking=False to match training-time format."""
    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    return [o.outputs[0].text for o in outputs]


def _run_llm4mat(args, llm, sampling_params, out_dir, pred_fh):
    if args.configs == "all":
        configs = ["mp", "jarvis_dft", "oqmd", "gnome", "snumat",
                   "hmof", "cantor_hea", "jarvis_qetb", "omdb"]
    else:
        configs = [c.strip() for c in args.configs.split(",")]

    all_metrics = {"split": args.split, "max_samples": args.max_samples,
                   "include_description": args.include_description, "by_config": {}}

    for config in configs:
        if config not in _DATASET_PROPERTIES:
            print(f"[skip] unknown config: {config}")
            continue
        csv_path = Path(args.data_root) / config / f"{args.split}.csv"
        if not csv_path.exists():
            print(f"[skip] {config}: CSV missing at {csv_path}")
            continue

        df = pl.read_csv(str(csv_path), infer_schema_length=10000, ignore_errors=True)
        cols = df.columns
        formula_col = "formula_pretty" if "formula_pretty" in cols else "formula"
        if formula_col not in cols:
            print(f"[skip] {config}: no formula column")
            continue
        id_col = next((c for c in cols if c.endswith("_id") or c == "material_id"), None)
        has_desc = "description" in cols
        if args.include_description and not has_desc:
            print(f"[warn] {config}: no description column; falling back to formula-only")

        all_metrics["by_config"][config] = {}
        n_rows = min(len(df), args.max_samples) if args.max_samples > 0 else len(df)
        sub = df.head(n_rows)
        formulas = sub[formula_col].to_list()
        descs = sub["description"].to_list() if (args.include_description and has_desc) else [None] * n_rows
        ids = sub[id_col].to_list() if id_col else list(range(n_rows))
        row_indices = [j for j in range(n_rows) if (j % args.num_shards) == args.shard_id]

        sys_msg = (_SYSTEM_TEXT_ONLY if (args.include_description and has_desc)
                   else _SYSTEM_FORMULA_ONLY)

        for prop in _DATASET_PROPERTIES[config]:
            if prop not in cols:
                continue
            shard_tag = (f" [shard {args.shard_id}/{args.num_shards}: {len(row_indices)} rows]"
                         if args.num_shards > 1 else "")
            print(f"[run] {config}/{prop}  n={n_rows}{shard_tag}")
            raw_targets = sub[prop].to_list()

            messages_batch = []
            meta = []
            for j in row_indices:
                f = formulas[j]
                if f is None:
                    continue
                d = descs[j]
                if args.include_description and d:
                    # Trim or vLLM hard-fails the batch on >max_model_len descriptions.
                    d_trunc = d[: args.max_desc_chars]
                    user = f"Formula: {f}\nDescription: {d_trunc}\nProperty name: {prop}."
                else:
                    user = f"Formula: {f}\nProperty name: {prop}."
                messages_batch.append(_make_messages(sys_msg, user))
                meta.append((ids[j], raw_targets[j]))

            gens = _chat_batch(llm, sampling_params, messages_batch) if messages_batch else []

            preds, targets, n_total, n_valid, n_leaked = [], [], 0, 0, 0
            for (sid, raw), gen in zip(meta, gens):
                parsed = extract_number(gen)
                leaked = detect_leak(gen)
                ok = parsed is not None and raw is not None and not leaked
                row = {"id": sid, "property": prop, "config": config, "target": raw,
                       "generated": gen, "parsed": parsed,
                       "leaked": leaked, "ok": ok}
                pred_fh.write(json.dumps(row) + "\n")
                n_total += 1
                if leaked:
                    n_leaked += 1
                if ok:
                    n_valid += 1
                    preds.append(parsed)
                    targets.append(float(raw))

            m = {"n_total": n_total, "n_valid": n_valid, "n_leaked": n_leaked,
                 "validity_rate": (n_valid / n_total) if n_total else 0.0,
                 "leak_rate": (n_leaked / n_total) if n_total else 0.0}
            if preds:
                m["mae"] = mae(preds, targets)
                m["mad_mae_ratio"] = mad_mae_ratio(preds, targets)
            all_metrics["by_config"][config][prop] = m
            print(f"  → {m}")
            _flush(out_dir, all_metrics, pred_fh)

    return all_metrics


def _run_mat2props(args, llm, sampling_params, out_dir, pred_fh):
    name = args.narrative_name
    parquet = Path(args.narrative_parquet_dir) / f"{name}_gpt_narratives.parquet"
    properties = ([p_.strip() for p_ in args.properties.split(",")]
                  if args.properties else _NARRATIVE_PROPERTIES[name])

    df = pl.read_parquet(str(parquet))
    formula_col = "pretty formula" if "pretty formula" in df.columns else "reduced_formula"
    desc_col = "gpt_text" if "gpt_text" in df.columns else None

    cut = int(0.9 * len(df))
    sub = df.slice(cut, len(df) - cut)
    if args.max_samples > 0:
        sub = sub.head(args.max_samples)

    formulas = sub[formula_col].to_list()
    descs = sub[desc_col].to_list() if (args.include_description and desc_col) else [None] * len(sub)
    ids = list(range(cut, cut + len(sub)))
    n_rows = len(sub)
    row_indices = [j for j in range(n_rows) if (j % args.num_shards) == args.shard_id]

    sys_msg = (_SYSTEM_TEXT_ONLY if (args.include_description and desc_col)
               else _SYSTEM_FORMULA_ONLY)

    all_metrics = {"narrative": name, "max_samples": args.max_samples,
                   "include_description": args.include_description}

    for prop in properties:
        if prop not in df.columns:
            print(f"[skip] {name}/{prop}: column missing")
            continue
        targets_raw = sub[prop].to_list()
        shard_tag = (f" [shard {args.shard_id}/{args.num_shards}: {len(row_indices)} rows]"
                     if args.num_shards > 1 else "")
        print(f"[run] mat2props/{name}/{prop}  n={n_rows}{shard_tag}")

        messages_batch = []
        meta = []
        for j in row_indices:
            f = formulas[j]
            if f is None:
                continue
            d = descs[j]
            if args.include_description and d:
                # Trim or vLLM hard-fails the batch on >max_model_len descriptions.
                d_trunc = d[: args.max_desc_chars]
                user = f"Formula: {f}\nDescription: {d_trunc}\nProperty name: {prop}."
            else:
                user = f"Formula: {f}\nProperty name: {prop}."
            messages_batch.append(_make_messages(sys_msg, user))
            meta.append((ids[j], targets_raw[j]))

        gens = _chat_batch(llm, sampling_params, messages_batch) if messages_batch else []

        preds, targets, n_total, n_valid, n_leaked = [], [], 0, 0, 0
        for (sid, raw), gen in zip(meta, gens):
            parsed = extract_number(gen)
            leaked = detect_leak(gen)
            ok = parsed is not None and raw is not None and not leaked
            row = {"id": sid, "property": prop, "target": raw,
                   "generated": gen, "parsed": parsed,
                   "leaked": leaked, "ok": ok}
            pred_fh.write(json.dumps(row) + "\n")
            n_total += 1
            if leaked:
                n_leaked += 1
            if ok:
                n_valid += 1
                preds.append(parsed)
                targets.append(float(raw))

        m = {"n_total": n_total, "n_valid": n_valid, "n_leaked": n_leaked,
             "validity_rate": (n_valid / n_total) if n_total else 0.0,
             "leak_rate": (n_leaked / n_total) if n_total else 0.0}
        if preds:
            m["mae"] = mae(preds, targets)
            m["mad_mae_ratio"] = mad_mae_ratio(preds, targets)
        all_metrics[prop] = m
        print(f"  → {m}")
        _flush(out_dir, all_metrics, pred_fh)

    return all_metrics


def _merge_shards(args):
    """Concatenate shard predictions.jsonl and recompute metrics into the canonical run dir."""
    shard_dirs = [Path(s.strip()) for s in args.merge_shards_from.split(",") if s.strip()]
    if not shard_dirs:
        raise SystemExit("--merge_shards_from is empty")

    bench_dir = "llm4mat" if args.bench == "llm4mat" else "mat2props"
    rid = f"base_{args.base_name.replace('/', '_')}"
    if not args.include_description:
        rid += "_formula_only"
    out_dir = run_dir(bench_dir, args.base_name.replace("/", "_"), run_id=rid)

    all_rows = []
    for sd in shard_dirs:
        p_path = sd / "predictions.jsonl"
        if not p_path.exists():
            print(f"[merge] WARN: {p_path} missing; skipping")
            continue
        with open(p_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))

    out_pred = out_dir / "predictions.jsonl"
    with open(out_pred, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[merge] concatenated {len(all_rows)} predictions from {len(shard_dirs)} shards "
          f"→ {out_pred}")

    if args.bench == "llm4mat":
        merged = {"split": args.split, "max_samples": args.max_samples,
                  "include_description": args.include_description,
                  "merged_from_shards": [str(s) for s in shard_dirs],
                  "by_config": {}}
        groups = {}
        for r in all_rows:
            cfg = r.get("config")
            prp = r.get("property")
            if cfg is None or prp is None:
                continue
            groups.setdefault(cfg, {}).setdefault(prp, []).append(r)
        for cfg, by_prop in groups.items():
            merged["by_config"][cfg] = {}
            for prp, rows in by_prop.items():
                merged["by_config"][cfg][prp] = _metrics_from_rows(rows)
    else:
        merged = {"narrative": args.narrative_name, "max_samples": args.max_samples,
                  "include_description": args.include_description,
                  "merged_from_shards": [str(s) for s in shard_dirs]}
        groups = {}
        for r in all_rows:
            prp = r.get("property")
            if prp is None:
                continue
            groups.setdefault(prp, []).append(r)
        for prp, rows in groups.items():
            merged[prp] = _metrics_from_rows(rows)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    print(f"[merge] {out_dir}/metrics.json")


def _metrics_from_rows(rows):
    preds, targets, n_total, n_valid, n_leaked = [], [], 0, 0, 0
    for r in rows:
        n_total += 1
        if r.get("leaked"):
            n_leaked += 1
        if r.get("ok") and r.get("parsed") is not None and r.get("target") is not None:
            n_valid += 1
            preds.append(float(r["parsed"]))
            targets.append(float(r["target"]))
    m = {"n_total": n_total, "n_valid": n_valid, "n_leaked": n_leaked,
         "validity_rate": (n_valid / n_total) if n_total else 0.0,
         "leak_rate": (n_leaked / n_total) if n_total else 0.0}
    if preds:
        m["mae"] = mae(preds, targets)
        m["mad_mae_ratio"] = mad_mae_ratio(preds, targets)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["llm4mat", "mat2props"], required=True)
    p.add_argument("--base_name", default="Qwen/Qwen3-8B")
    p.add_argument("--include_description", action=argparse.BooleanOptionalAction, default=True,
                   help="Pass formula + description (LLM4Mat-paper-parity); off = formula only.")

    # llm4mat
    p.add_argument("--configs", default="all",
                   help="comma list (e.g. 'mp,jarvis_dft') or 'all' for the 9 staged configs.")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--data_root", default=os.path.join(DATA_ROOT, "LLM4Mat-Bench"))

    # mat2props
    p.add_argument("--narrative_name", default="mp_3d_2020",
                   choices=["mp_3d_2020", "dft_3d", "aflow2", "oqmd"])
    p.add_argument("--narrative_parquet_dir",
                   default=os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials"))
    p.add_argument("--properties", default=None,
                   help="comma list; default = full _NARRATIVE_PROPERTIES[name].")
    p.add_argument("--max_desc_chars", type=int, default=2000,
                   help="Pre-trim descriptions (LLM4Mat or gpt_text) to this many chars. "
                        "hMOF / qMOF can hit 10k+ chars; default 2000 ≈ 600-800 tokens.")

    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--max_new_tokens", type=int, default=32)

    p.add_argument("--max_model_len", type=int, default=8192,
                   help="Max input+output tokens. 8192 + max_desc_chars=2000 trim covers all "
                        "LLM4Mat & Mat2Props rows. Bump if you raise --max_desc_chars.")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Multi-GPU tensor parallelism. For an 8B model on >1 GPU prefer DP "
                        "(--shard_id/--num_shards) over TP — TP all-reduce kills throughput.")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                   help="Fraction of each GPU vLLM may use for KV cache.")
    p.add_argument("--seed", type=int, default=42)

    # Data-parallel sharding
    p.add_argument("--shard_id", type=int, default=0,
                   help="This shard's index, 0..num_shards-1. Rows are striped: shard S "
                        "takes row i where i %% num_shards == S.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total shard count for data-parallel runs. 1 = no sharding.")
    p.add_argument("--merge_shards_from", default=None,
                   help="Comma-separated list of shard run dirs to merge. Skips engine "
                        "load entirely; concatenates predictions.jsonl, recomputes metrics.")
    args = p.parse_args()

    if args.merge_shards_from:
        return _merge_shards(args)

    # Import inside main so the script can be inspected without booting the GPU.
    from vllm import LLM, SamplingParams

    print(f"[load] vLLM engine: {args.base_name} "
          f"(tp={args.tensor_parallel_size}, max_len={args.max_model_len}, "
          f"util={args.gpu_memory_utilization})")
    llm = LLM(
        model=args.base_name,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    bench_dir = "llm4mat" if args.bench == "llm4mat" else "mat2props"
    rid = f"base_{args.base_name.replace('/', '_')}"
    if not args.include_description:
        rid += "_formula_only"
    if args.num_shards > 1:
        rid += f"__shard{args.shard_id}of{args.num_shards}"
    out_dir = run_dir(bench_dir, args.base_name.replace("/", "_"), run_id=rid)
    print(f"[out] streaming predictions → {out_dir}/predictions.jsonl "
          f"(metrics.json rewritten per property)")

    # Truncate so reruns don't accumulate stale rows from a crashed sweep.
    with open(out_dir / "predictions.jsonl", "w") as pred_fh:
        if args.bench == "llm4mat":
            metrics = _run_llm4mat(args, llm, sampling_params, out_dir, pred_fh)
        else:
            metrics = _run_mat2props(args, llm, sampling_params, out_dir, pred_fh)
        _flush(out_dir, metrics, pred_fh)
    print(f"[done] {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
