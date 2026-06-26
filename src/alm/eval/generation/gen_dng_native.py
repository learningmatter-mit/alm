"""DNG-style generation for csp_backbone: oracle-composition CSP (K=1) over sampled prompts, CIFs out."""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ALM_ROOT)

from eval_dng import _sample_prompts_from_parquet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mg_ckpt_dir", type=Path, required=True,
                    help="MG-CSP training dir (csp_backbone).")
    ap.add_argument("--pairs_parquet", type=Path, required=True,
                    help="pairs.parquet (or other stage3a parquet) for prompt sampling.")
    ap.add_argument("--n_prompts", type=int, default=1000)
    ap.add_argument("--prompts_seed", type=int, default=42,
                    help="Prompt RNG seed.")
    ap.add_argument("--K", type=int, default=1,
                    help="Generations per prompt. DNG convention K=1.")
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--max_n_atoms", type=int, default=30,
                    help="Filter to comps with ≤30 atoms (csp_backbone training distrib).")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen-dng] writing → {args.out_dir}", flush=True)
    t0 = time.time()

    prompts, prompt_ids, elements_per_prompt = _sample_prompts_from_parquet(
        args.pairs_parquet,
        n_prompts=args.n_prompts,
        seed=args.prompts_seed,
        parent_filter=None,
    )
    kept = [(p, pid, els) for p, pid, els in zip(prompts, prompt_ids, elements_per_prompt)
            if len(els) <= args.max_n_atoms]
    print(f"  sampled {len(prompts)} prompts, {len(kept)} after ≤{args.max_n_atoms}-atom filter",
          flush=True)

    if args.num_shards > 1:
        kept = [k for i, k in enumerate(kept)
                if (i % args.num_shards) == args.shard_idx]
        print(f"  shard {args.shard_idx}/{args.num_shards}: {len(kept)} rows",
              flush=True)

    if not kept:
        print("[gen-dng] no rows to process; exiting", flush=True)
        return

    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.generator import CrystalGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  loading MG-CSP-fs from {args.mg_ckpt_dir} (device={device}) ...", flush=True)
    ckpt_info = MatterGenCheckpointInfo(model_path=str(args.mg_ckpt_dir), load_epoch="last")

    generator = CrystalGenerator(
        checkpoint_info=ckpt_info,
        batch_size=args.K,
        num_batches=1,
        sampling_config_name="csp",
        diffusion_guidance_factor=args.guidance_factor,
    )

    cif_root = args.out_dir / "cifs"
    cif_root.mkdir(parents=True, exist_ok=True)
    records = []

    for i, (prompt, pid, elems) in enumerate(kept):
        if i % 25 == 0:
            print(f"  [{i}/{len(kept)}] (t={time.time()-t0:.0f}s) ...", flush=True)
        target_comp = dict(Counter(elems))
        try:
            structures = generator.generate(
                target_compositions_dict=[target_comp] * args.K,
            )
        except Exception as e:
            records.append({"prompt_id": pid, "err": str(e)[:160]})
            continue
        for k, s in enumerate(structures[:args.K]):
            try:
                cif_str = s.to(fmt="cif")
                p = cif_root / f"{pid}__k{k}.cif"
                p.write_text(cif_str)
            except Exception:
                pass
        records.append({
            "prompt_id": pid,
            "n_atoms_target": sum(target_comp.values()),
            "formula_target": "".join(f"{el}{n}" for el, n in sorted(target_comp.items())),
            "prompt": prompt[:200],
            "k_saved": min(args.K, len(structures)),
        })

    (args.out_dir / f"records_shard{args.shard_idx}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records))
    print(f"\n[gen-dng] shard{args.shard_idx} done: {len(records)} rows, "
          f"{sum(r.get('k_saved', 0) for r in records)} CIFs saved in {cif_root}")


if __name__ == "__main__":
    main()
