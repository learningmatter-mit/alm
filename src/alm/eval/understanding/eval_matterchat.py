"""MatterChat 9-task MP benchmark over LLM4Mat-Bench's mp split."""

import argparse
import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import AtomisticLanguageDataset, custom_collate_fn

from loader import load_alm
from text_generation import generate_batch
from parsers import detect_leak, extract_number, extract_choice
from metrics import mae, rmse, mad_mae_ratio, accuracy, weighted_f1
from runs import run_dir, write_run
from paths import DATA_ROOT


_PROPERTY_SYSTEM = (
    "You are a material scientist. "
    "Look at the structure of the given crystalline material and predict its property."
)


def _build_task_dict(target_column, prompt, target_format):
    return {
        "name": f"matterchat_{target_column}",
        "system": _PROPERTY_SYSTEM,
        "user": prompt,
        "target_column": target_column,
        "target_format": target_format,
        "bucket": "matterchat",
    }


def _bool_to_yn(v):
    return "Yes" if bool(v) else "No"


_YN = {True: "A", False: "B", "True": "A", "False": "B", "true": "A", "false": "B"}
_MAG = {"NM": "A", "FM": "B", "AFM": "C", "FiM": "D"}
_XS = {"Cubic": "A", "Tetragonal": "B", "Orthorhombic": "C", "Hexagonal": "D",
       "Trigonal": "E", "Monoclinic": "F", "Triclinic": "G"}

TASK_DEFS = {
    "formation_energy": {
        "col": "formation_energy", "type": "reg",
        "prompt": "<atoms>\nPredict the formation energy per atom (eV/atom).",
        "fmt": lambda v: f"{float(v):.4f} eV/atom",
    },
    "energy_above_hull": {
        "col": "energy_above_hull", "type": "reg",
        "prompt": "<atoms>\nPredict the energy above the convex hull (eV/atom).",
        "fmt": lambda v: f"{float(v):.4f} eV/atom",
    },
    "bandgap": {
        "col": "bandgap", "type": "reg",
        "prompt": "<atoms>\nPredict the band gap (eV).",
        "fmt": lambda v: f"{float(v):.4f} eV",
    },
    "is_metal": {
        "col": "is_metal", "type": "cls",
        "prompt": "<atoms>\nIs this material a metal? Answer A) Yes or B) No.",
        "fmt": lambda v: "A) Yes" if bool(v) else "B) No",
        "label_map": _YN,
    },
    "is_magnetic": {
        "col": "is_magnetic", "type": "cls",
        "prompt": "<atoms>\nIs this material magnetic? Answer A) Yes or B) No.",
        "fmt": lambda v: "A) Yes" if bool(v) else "B) No",
        "label_map": _YN,
    },
    "direct_bandgap": {
        "col": "direct_bandgap", "type": "cls",
        "prompt": "<atoms>\nIs the band gap direct? Answer A) Yes or B) No.",
        "fmt": lambda v: "A) Yes" if bool(v) else "B) No",
        "label_map": _YN,
    },
    "stable": {
        "col": "stable", "type": "cls",
        "prompt": "<atoms>\nIs this material thermodynamically stable? Answer A) Yes or B) No.",
        "fmt": lambda v: "A) Yes" if bool(v) else "B) No",
        "label_map": _YN,
    },
    "magnetic_order": {
        "col": "magnetic_order", "type": "cls",
        "prompt": ("<atoms>\nClassify the magnetic ordering. "
                   "A) NM (non-magnetic) B) FM (ferromagnetic) C) AFM (antiferromagnetic) D) FiM (ferrimagnetic)."),
        "fmt": lambda v: f"{_MAG.get(str(v), 'A')})",
        "label_map": _MAG,
    },
    "crystal_system": {
        "col": "crystal_system", "type": "cls",
        "prompt": ("<atoms>\nWhich crystal system does this material belong to? "
                   "A) Cubic B) Tetragonal C) Orthorhombic D) Hexagonal "
                   "E) Trigonal F) Monoclinic G) Triclinic."),
        "fmt": lambda v: f"{_XS.get(str(v), 'A')})",
        "label_map": _XS,
    },
}


def _eval_task(model, tokenizer, key, defn, args):
    embs = Path(args.cached_embs_root) / args.config / "embeddings" / f"{args.atomistic_model_name}_{args.split}_atom.flat.bin"
    csv  = Path(args.data_csv) if args.data_csv else Path(args.data_root) / args.config / f"{args.split}.csv"
    if not embs.exists():
        print(f"[skip] {key}: cached embs missing at {embs}")
        return None, []

    task = _build_task_dict(defn["col"], defn["prompt"], defn["fmt"])
    ds = AtomisticLanguageDataset(
        tokenizer=tokenizer, db_path=None, csv_path=str(csv),
        thinking=False, max_num_tokens=args.max_num_tokens,
        dataset_name=args.config, cached_embs_path=str(embs), tasks=[task],
        atomistic_feature_dim=args.atomistic_feature_dim,
    )
    n = min(len(ds), args.max_samples) if args.max_samples > 0 else len(ds)
    loader = DataLoader(Subset(ds, list(range(n))), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)

    id_to_idx = {sid: i for i, sid in enumerate(ds._ids)}
    preds_num, tgts_num, preds_cls, tgts_cls, predictions = [], [], [], [], []
    n_leaked = 0
    for batch in loader:
        gens = generate_batch(model, batch, max_new_tokens=args.max_new_tokens, atomistic=True,
                              block_leak_tokens=args.block_leak_tokens)
        for sid, gen in zip(batch["id"], gens):
            raw = ds._column_data[defn["col"]][id_to_idx[sid]]
            leaked = detect_leak(gen)
            row = {"task": key, "id": sid, "target": raw, "generated": gen, "leaked": leaked}
            if leaked:
                n_leaked += 1
            if defn["type"] == "reg":
                parsed = extract_number(gen)
                ok = parsed is not None and raw is not None and not leaked
                row["parsed"], row["ok"] = parsed, ok
                if ok:
                    preds_num.append(parsed); tgts_num.append(float(raw))
            else:
                tgt_letter = defn["label_map"].get(raw)
                # Pass the task's choice set so multi-class (A-G) parses too.
                pred = extract_choice(gen, choices=tuple(defn["label_map"].values()))
                ok = pred is not None and tgt_letter is not None and not leaked
                row["parsed"], row["ok"], row["target_letter"] = pred, ok, tgt_letter
                if ok:
                    preds_cls.append(pred); tgts_cls.append(tgt_letter)
            predictions.append(row)

    n_total = len(predictions)
    metrics = {"n_total": n_total,
               "n_valid": len(preds_num) + len(preds_cls),
               "n_leaked": n_leaked,
               "validity_rate": (len(preds_num) + len(preds_cls)) / max(1, n_total),
               "leak_rate":     n_leaked / max(1, n_total)}
    if defn["type"] == "reg" and preds_num:
        metrics["mae"] = mae(preds_num, tgts_num)
        metrics["rmse"] = rmse(preds_num, tgts_num)
        metrics["mad_mae_ratio"] = mad_mae_ratio(preds_num, tgts_num)
    elif defn["type"] == "cls" and preds_cls:
        metrics["accuracy"] = accuracy(preds_cls, tgts_cls)
        metrics["weighted_f1"] = weighted_f1(preds_cls, tgts_cls)
    return metrics, predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="mp")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--data_root", default=os.path.join(DATA_ROOT, "LLM4Mat-Bench"))
    p.add_argument("--cached_embs_root", default=os.path.join(DATA_ROOT, "cached_embs"))
    p.add_argument("--data_csv", default=None,
                   help="override CSV path (use this when pointing at MatterChat's Zenodo CSV)")
    p.add_argument("--tasks", default=",".join(TASK_DEFS.keys()))
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--no_merge_lora", action="store_true")
    p.add_argument("--block_leak_tokens", action="store_true",
                   help="Suppress markdown-image / URL token openers at decode time (off by default).")
    p.add_argument("--atomistic_model_name", type=str, default=None,
                   help="Cache filename prefix; auto-detected from the checkpoint when omitted.")
    p.add_argument("--atomistic_feature_dim", type=int, default=None,
                   help="Per-atom feature dim; auto-detected from the checkpoint when omitted.")
    args = p.parse_args()

    model, tokenizer = load_alm(checkpoint=args.checkpoint, merge_lora=not args.no_merge_lora)
    # Auto-detect encoder from the projector in_dim (same map as eval_llm4mat).
    _ATOMISTIC_NAME_BY_DIM = {256: "orb_v3_direct_20_omat", 128: "uma-s-1p1",
                              640: "pet-mad-xs", 1280: "pet-mad-s"}
    detected_dim = int(model.projector[0].in_features)
    if args.atomistic_feature_dim is None:
        args.atomistic_feature_dim = detected_dim
    if args.atomistic_model_name is None:
        args.atomistic_model_name = _ATOMISTIC_NAME_BY_DIM.get(
            args.atomistic_feature_dim, "orb_v3_direct_20_omat"
        )
    print(f"[eval_matterchat] atomistic_feature_dim={args.atomistic_feature_dim} "
          f"→ atomistic_model_name={args.atomistic_model_name}")

    all_metrics, all_predictions = {}, []
    for key in [t.strip() for t in args.tasks.split(",")]:
        if key not in TASK_DEFS:
            print(f"[skip] unknown task {key}")
            continue
        print(f"[run] matterchat/{key}")
        m, preds = _eval_task(model, tokenizer, key, TASK_DEFS[key], args)
        if m is not None:
            all_metrics[key] = m
            all_predictions.extend(preds)
            print(f"  → {m}")

    write_run(run_dir("matterchat", args.checkpoint), all_metrics, all_predictions)


if __name__ == "__main__":
    main()
