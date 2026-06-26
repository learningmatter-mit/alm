"""GNoME formation-energy MAE on LLM4Mat-Bench's gnome split (MatterChat comparison)."""

import argparse
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import AtomisticLanguageDataset, _property_task, custom_collate_fn

from loader import load_alm
from text_generation import generate_batch
from parsers import detect_leak, extract_number
from metrics import mae, rmse, mad_mae_ratio
from paths import DATA_ROOT
from runs import run_dir, write_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--data_root", default=os.path.join(DATA_ROOT, "LLM4Mat-Bench"))
    p.add_argument("--cached_embs_root", default=os.path.join(DATA_ROOT, "cached_embs"))
    p.add_argument("--id_list", default=None,
                   help="optional newline-separated GNoME ids (MatterChat exact subset)")
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
    print(f"[eval_gnome_fe] atomistic_feature_dim={args.atomistic_feature_dim} "
          f"→ atomistic_model_name={args.atomistic_model_name}")

    embs = Path(args.cached_embs_root) / "gnome" / "embeddings" / f"{args.atomistic_model_name}_{args.split}_atom.flat.bin"
    csv  = Path(args.data_root) / "gnome" / f"{args.split}.csv"
    prop = "Formation_Energy_Per_Atom"
    ds = AtomisticLanguageDataset(
        tokenizer=tokenizer, db_path=None, csv_path=str(csv),
        thinking=False, max_num_tokens=args.max_num_tokens,
        dataset_name="gnome", cached_embs_path=str(embs),
        tasks=[_property_task(prop)],
        atomistic_feature_dim=args.atomistic_feature_dim,
    )

    indices = list(range(len(ds)))
    if args.id_list:
        wanted = set(open(args.id_list).read().split())
        indices = [i for i, sid in enumerate(ds._ids) if str(sid) in wanted]
        print(f"[gnome_fe] filtered to {len(indices)} ids from --id_list")
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
            predictions.append({"id": sid, "target": raw, "generated": gen,
                                "parsed": parsed, "leaked": leaked, "ok": ok})
            if leaked:
                n_leaked += 1
            if ok:
                preds.append(parsed); targets.append(float(raw))

    n_total = len(predictions)
    metrics = {"n_total": n_total, "n_valid": len(preds), "n_leaked": n_leaked,
               "validity_rate": len(preds) / max(1, n_total),
               "leak_rate":     n_leaked / max(1, n_total)}
    if preds:
        metrics["mae"] = mae(preds, targets)
        metrics["rmse"] = rmse(preds, targets)
        metrics["mad_mae_ratio"] = mad_mae_ratio(preds, targets)

    write_run(run_dir("gnome_fe", args.checkpoint), metrics, predictions)
    print(metrics)


if __name__ == "__main__":
    main()
