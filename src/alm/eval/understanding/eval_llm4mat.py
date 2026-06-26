"""LLM4Mat-Bench held-out eval: per-config per-property regression, MAD:MAE ratio metric."""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))           # alm/eval
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # alm/
from utils import AtomisticLanguageDataset, _DATASET_PROPERTIES, _property_task, custom_collate_fn

from loader import load_alm
from text_generation import generate_batch
from parsers import detect_leak, extract_number
from metrics import mae, mad_mae_ratio
from runs import run_dir, write_run
from paths import DATA_ROOT


def _build_dataset(config, prop, args, tokenizer):
    embs = Path(args.cached_embs_root) / config / "embeddings" / f"{args.atomistic_model_name}_{args.split}_atom.flat.bin"
    csv  = Path(args.data_root) / config / f"{args.split}.csv"
    db   = Path(args.data_root) / config / f"{args.split}.db"
    if not embs.exists():
        print(f"[skip] {config}/{prop}: cached embs missing at {embs}")
        return None
    if not csv.exists():
        print(f"[skip] {config}/{prop}: csv missing at {csv}")
        return None
    return AtomisticLanguageDataset(
        tokenizer=tokenizer, db_path=str(db), csv_path=str(csv),
        thinking=False, max_num_tokens=args.max_num_tokens,
        dataset_name=config, cached_embs_path=str(embs),
        tasks=[_property_task(prop)],
        atomistic_feature_dim=args.atomistic_feature_dim,
    )


def _eval_one(model, dataset, prop, args):
    n = len(dataset)
    if args.max_samples and args.max_samples > 0:
        n = min(n, args.max_samples)
    target_col = prop
    id_to_idx = {sid: i for i, sid in enumerate(dataset._ids)}

    indices = list(range(n))
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=custom_collate_fn,
    )

    preds_num, targets_num, predictions = [], [], []
    n_leaked = 0
    for batch in loader:
        gens = generate_batch(model, batch, max_new_tokens=args.max_new_tokens, atomistic=True,
                              block_leak_tokens=args.block_leak_tokens)
        for sid, gen in zip(batch["id"], gens):
            raw = dataset._column_data[target_col][id_to_idx[sid]]
            parsed = extract_number(gen)
            leaked = detect_leak(gen)
            ok = parsed is not None and raw is not None and not leaked
            row = {"id": sid, "property": prop, "target": raw,
                   "generated": gen, "parsed": parsed, "leaked": leaked, "ok": ok}
            predictions.append(row)
            if leaked:
                n_leaked += 1
            if ok:
                preds_num.append(parsed)
                targets_num.append(float(raw))

    n_total = len(predictions)
    metrics = {
        "n_total": n_total,
        "n_valid": len(preds_num),
        "n_leaked": n_leaked,
        "validity_rate": (len(preds_num) / n_total) if n_total else 0.0,
        "leak_rate":     (n_leaked / n_total) if n_total else 0.0,
    }
    if preds_num:
        metrics["mae"] = mae(preds_num, targets_num)
        metrics["mad_mae_ratio"] = mad_mae_ratio(preds_num, targets_num)
    return metrics, predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Stage 2 step=N/ dir")
    p.add_argument("--configs", default="mp",
                   help="comma list (e.g. 'mp,jarvis_dft,oqmd' or 'all' for the 9 staged configs)")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--data_root", default=os.path.join(DATA_ROOT, "LLM4Mat-Bench"))
    p.add_argument("--cached_embs_root", default=os.path.join(DATA_ROOT, "cached_embs"))
    p.add_argument("--max_samples", type=int, default=1000,
                   help="cap per (config, property); 0 or negative for full split")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--no_merge_lora", action="store_true")
    p.add_argument("--llm_name", type=str, default="Qwen/Qwen3-8B",
                   help="Base LLM HF id. Must match what the checkpoint was trained on.")
    p.add_argument("--block_leak_tokens", action="store_true",
                   help="Suppress markdown-image / URL token openers at decode time (off by default).")
    p.add_argument("--atomistic_model_name", type=str, default=None,
                   help="Cache filename prefix (orb_v3_direct_20_omat / uma-s-1p1 / "
                        "pet-mad-xs / pet-mad-s). Auto-detected from the checkpoint "
                        "projector's input dim when omitted.")
    p.add_argument("--atomistic_feature_dim", type=int, default=None,
                   help="Per-atom feature dim (256 / 128 / 640 / 1280). Auto-"
                        "detected from the checkpoint projector when omitted.")
    p.add_argument("--atom_bidirectional_attention", action="store_true",
                   help="Load under bidirectional atom attention. Required for bidir-trained "
                        "Stage 2 ckpts (runtime mask flag, not weight-detectable).")
    args = p.parse_args()

    if args.configs == "all":
        configs = ["mp", "jarvis_dft", "oqmd", "gnome", "snumat",
                   "hmof", "cantor_hea", "jarvis_qetb", "omdb"]
    else:
        configs = [c.strip() for c in args.configs.split(",")]

    model, tokenizer = load_alm(checkpoint=args.checkpoint,
                                base_model=args.llm_name,
                                merge_lora=not args.no_merge_lora,
                                atom_bidirectional_attention=args.atom_bidirectional_attention)
    # Detect encoder from projector input dim so the cache file matches Stage 2.
    _ATOMISTIC_NAME_BY_DIM = {256: "orb_v3_direct_20_omat", 128: "uma-s-1p1",
                              640: "pet-mad-xs", 1280: "pet-mad-s"}
    detected_dim = int(model.projector[0].in_features)
    if args.atomistic_feature_dim is None:
        args.atomistic_feature_dim = detected_dim
    if args.atomistic_model_name is None:
        args.atomistic_model_name = _ATOMISTIC_NAME_BY_DIM.get(
            args.atomistic_feature_dim, "orb_v3_direct_20_omat"
        )
    print(f"[eval_llm4mat] atomistic_feature_dim={args.atomistic_feature_dim} "
          f"→ atomistic_model_name={args.atomistic_model_name}")

    out = run_dir("llm4mat", args.checkpoint)
    all_metrics = {"split": args.split, "max_samples": args.max_samples, "by_config": {}}
    all_predictions = []
    for config in configs:
        if config not in _DATASET_PROPERTIES:
            print(f"[skip] unknown config: {config}")
            continue
        all_metrics["by_config"][config] = {}
        for prop in _DATASET_PROPERTIES[config]:
            ds = _build_dataset(config, prop, args, tokenizer)
            if ds is None:
                continue
            print(f"[run] {config}/{prop}  n={min(len(ds), args.max_samples or len(ds))}")
            m, preds = _eval_one(model, ds, prop, args)
            all_metrics["by_config"][config][prop] = m
            all_predictions.extend(preds)
            print(f"  → {m}")

    write_run(out, all_metrics, all_predictions)


if __name__ == "__main__":
    main()
