"""Park et al. Mat2Props per-property MAE on the MP-derived held-out subset (last 10%, or --id_list for protocol parity)."""

import argparse
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    GPTNarrativeDataset, _NARRATIVE_PROPERTIES, _property_task, custom_collate_fn,
)

from loader import load_alm
from text_generation import generate_batch
from parsers import detect_leak, extract_number
from metrics import mae
from runs import run_dir, write_run
from paths import DATA_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--narrative_name", default="mp_3d_2020",
                   choices=["mp_3d_2020", "dft_3d", "aflow2", "oqmd"])
    p.add_argument("--narrative_parquet_dir",
                   default=os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials"))
    p.add_argument("--narrative_cache_dir",
                   default=os.path.join(DATA_ROOT, "cached_embs_narratives"))
    p.add_argument("--id_list", default=None,
                   help="newline-separated parquet row ids; default = last 10%%")
    p.add_argument("--properties", default=None,
                   help="comma list; default = full _NARRATIVE_PROPERTIES[name]")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--no_merge_lora", action="store_true")
    p.add_argument("--atomistic_model_name", type=str, default=None)
    p.add_argument("--atomistic_feature_dim", type=int, default=None)
    p.add_argument("--block_leak_tokens", action="store_true",
                   help="Suppress markdown-image / URL token openers at decode time (off by default).")
    args = p.parse_args()

    model, tokenizer = load_alm(checkpoint=args.checkpoint, merge_lora=not args.no_merge_lora)
    _ATOMISTIC_NAME_BY_DIM = {256: "orb_v3_direct_20_omat", 128: "uma-s-1p1",
                              640: "pet-mad-xs", 1280: "pet-mad-s"}
    if args.atomistic_feature_dim is None:
        args.atomistic_feature_dim = int(model.projector[0].in_features)
    if args.atomistic_model_name is None:
        args.atomistic_model_name = _ATOMISTIC_NAME_BY_DIM.get(
            args.atomistic_feature_dim, "orb_v3_direct_20_omat")
    print(f"[eval_mat2props] atomistic_feature_dim={args.atomistic_feature_dim} "
          f"→ atomistic_model_name={args.atomistic_model_name}")

    name = args.narrative_name
    parquet = Path(args.narrative_parquet_dir) / f"{name}_gpt_narratives.parquet"
    cache   = Path(args.narrative_cache_dir) / name / "embeddings" / f"{args.atomistic_model_name}_atom.flat.bin"
    properties = ([p_.strip() for p_ in args.properties.split(",")]
                  if args.properties else _NARRATIVE_PROPERTIES[name])

    all_metrics, all_predictions = {}, []
    for prop in properties:
        ds = GPTNarrativeDataset(
            tokenizer=tokenizer, parquet_path=str(parquet),
            cached_embs_path=str(cache) if cache.exists() else None,
            thinking=False, max_num_tokens=args.max_num_tokens,
            dataset_name=name, tasks=[_property_task(prop)],
            atomistic_feature_dim=args.atomistic_feature_dim,
        )
        if args.id_list:
            wanted = set(open(args.id_list).read().split())
            indices = [i for i, sid in enumerate(ds._ids) if str(sid) in wanted]
        else:
            cut = int(0.9 * len(ds))
            indices = list(range(cut, len(ds)))
        if args.max_samples > 0:
            indices = indices[: args.max_samples]

        loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
        id_to_idx = {sid: i for i, sid in enumerate(ds._ids)}
        preds, targets, predictions = [], [], []
        n_leaked = 0
        for batch in loader:
            gens = generate_batch(model, batch, max_new_tokens=args.max_new_tokens, atomistic=True,
                                  block_leak_tokens=args.block_leak_tokens)
            for sid, gen in zip(batch["id"], gens):
                raw = ds._column_data[prop][id_to_idx[sid]]
                parsed = extract_number(gen)
                leaked = detect_leak(gen)
                ok = parsed is not None and raw is not None and not leaked
                predictions.append({"property": prop, "id": sid, "target": raw,
                                    "generated": gen, "parsed": parsed,
                                    "leaked": leaked, "ok": ok})
                if leaked:
                    n_leaked += 1
                if ok:
                    preds.append(parsed); targets.append(float(raw))

        n_total = len(predictions)
        m = {"n_total": n_total, "n_valid": len(preds), "n_leaked": n_leaked,
             "validity_rate": len(preds) / max(1, n_total),
             "leak_rate":     n_leaked / max(1, n_total)}
        if preds:
            m["mae"] = mae(preds, targets)
        all_metrics[prop] = m
        all_predictions.extend(predictions)
        print(f"[mat2props/{name}/{prop}] {m}")

    write_run(run_dir("mat2props", args.checkpoint), all_metrics, all_predictions)


if __name__ == "__main__":
    main()
