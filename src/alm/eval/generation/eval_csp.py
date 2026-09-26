"""CSP eval (M@1, M@K) for a CSP-mode MatterGen checkpoint trained from scratch.

Per-row RMSD: rmsd_n1 is the matched RMSD of candidate 0 and rmsd_nK the minimum matched RMSD over
the K candidates (CDVAE/CrystaLLM convention; compute_csp_rmse.py uses the same definition).
Match rates divide by every targeted row; timed-out and failed rows count as misses.

Usage:
  python -m alm.eval.generation.eval_csp \\
      --mattergen_model_path <csp_backbone_dir> \\
      --max_rows 100 --K 64 \\
      --out_dir <results_dir>/mg_csp_from_scratch_K64
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from paths import DATA_ROOT

MP20_CSV = Path(os.path.join(DATA_ROOT, "eval_data/csp/mp_20/test.csv"))
TOL = dict(ltol=0.3, stol=0.5, angle_tol=10.0)


def load_targets(csv_path):
    targets = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mp_id, cif = row.get("material_id"), row.get("cif")
            if mp_id and cif:
                try:
                    targets.append((mp_id, Structure.from_str(cif, fmt="cif")))
                except Exception:
                    pass
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mattergen_model_path", "--ckpt_dir", dest="ckpt_dir", metavar="DIR", type=Path, required=True,
                    help="MatterGen CSP checkpoint dir (contains checkpoints/ + config.yaml).")
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--num_steps", type=int, default=1000,
                    help="Number of diffusion steps (sampler_partial.N).")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Number of stride shards for parallel GPU launches.")
    ap.add_argument("--shard_idx", type=int, default=0,
                    help="This worker's shard index; keeps rows where row_idx %% num_shards == shard_idx.")
    ap.add_argument("--test_csv", type=Path, default=MP20_CSV,
                    help="Benchmark test CSV with material_id and cif columns (default: MP-20).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Torch RNG seed.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[mattergen-csp] writing to {args.out_dir}", flush=True)
    t0 = time.time()

    all_targets = load_targets(args.test_csv)[:args.max_rows]
    if args.num_shards > 1:
        targets = [t for i, t in enumerate(all_targets) if (i % args.num_shards) == args.shard_idx]
        print(f"  shard {args.shard_idx}/{args.num_shards}: {len(targets)}/{len(all_targets)} rows", flush=True)
    else:
        targets = all_targets
        print(f"  {len(targets)} test rows from {args.test_csv}", flush=True)

    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.generator import CrystalGenerator

    print(f"  loading MatterGen CSP checkpoint from {args.ckpt_dir} ...", flush=True)
    ckpt_info = MatterGenCheckpointInfo(model_path=str(args.ckpt_dir), load_epoch="last")

    matcher = StructureMatcher(**TOL)
    results = []
    n_match_n1 = 0
    n_match_nK = 0

    # Build the generator once; constructing it loads the checkpoint.
    gen = CrystalGenerator(
        checkpoint_info=ckpt_info,
        batch_size=args.K, num_batches=1,
        target_compositions_dict=[dict(Counter([str(s.specie.symbol) for s in targets[0][1]]))],
        diffusion_guidance_factor=args.guidance_factor,
        # CSP config drops the atomic_numbers predictor (types fixed by target_compositions_dict).
        sampling_config_name="csp",
        sampling_config_overrides=[f"sampler_partial.N={args.num_steps}"],
        # record_trajectories defaults True and writes a large .zip/row that fills the shared volume.
        record_trajectories=False,
    )

    # Per-row watchdog: some compositions hang in gen.generate() without raising; SIGALRM raises
    # _RowTimeout so the row is recorded as a miss. A normal row takes ~60-120 s.
    class _RowTimeout(Exception):
        pass
    def _on_alarm(signum, frame):
        raise _RowTimeout("gen.generate exceeded ROW_TIMEOUT_S")
    signal.signal(signal.SIGALRM, _on_alarm)
    ROW_TIMEOUT_S = int(os.environ.get("ROW_TIMEOUT_S", "360"))
    n_timeout = 0

    for i, (mp_id, target) in enumerate(targets):
        target_comp = dict(Counter([str(s.specie.symbol) for s in target]))
        formula = str(target.composition.reduced_formula)
        if (i % 5) == 0:
            print(f"  [{i:3d}/{len(targets)}] {mp_id} {formula} "
                  f"t={time.time()-t0:.0f}s", flush=True)
        try:
            prompt_gens_dir = args.out_dir / "gens" / mp_id
            prompt_gens_dir.mkdir(parents=True, exist_ok=True)
            signal.alarm(ROW_TIMEOUT_S)
            samples = gen.generate(
                batch_size=args.K, num_batches=1,
                target_compositions_dict=[target_comp],
                output_dir=str(prompt_gens_dir),
            )
            signal.alarm(0)
        except _RowTimeout as e:
            signal.alarm(0)
            n_timeout += 1
            print(f"    [{mp_id}] timed out after {ROW_TIMEOUT_S}s ({formula}); counted as a miss "
                  f"(n_timeout={n_timeout})", flush=True)
            results.append({"row_id": mp_id, "formula": formula, "matched_n1": False,
                            "matched_nK": False, "first_match_idx": -1, "n_gen": 0, "timed_out": True})
            continue
        except Exception as e:
            signal.alarm(0)
            print(f"    [{mp_id}] gen failed: {e}", flush=True)
            results.append({"row_id": mp_id, "formula": formula, "matched_n1": False,
                            "matched_nK": False, "first_match_idx": -1, "n_gen": 0, "gen_failed": True})
            continue

        matched_n1 = False
        matched_nK = False
        first_match_idx = -1
        rmsd_n1 = None      # matched RMSD of candidate 0 (CDVAE/CrystaLLM rms[0] convention)
        rmsd_nK = None      # min matched RMSD over K candidates
        for j, s in enumerate(samples):
            if not isinstance(s, Structure):
                try: s = AseAtomsAdaptor.get_structure(s)
                except Exception: continue
            try:
                rd = matcher.get_rms_dist(target, s)   # (rms, max_dist) if match else None
                if rd is not None:
                    matched_nK = True
                    rmsd = float(rd[0])
                    if rmsd_nK is None or rmsd < rmsd_nK: rmsd_nK = rmsd
                    if j == 0:
                        matched_n1 = True
                        rmsd_n1 = rmsd
                    if first_match_idx < 0: first_match_idx = j
            except Exception:
                pass
        if matched_n1: n_match_n1 += 1
        if matched_nK: n_match_nK += 1
        results.append({
            "row_id": mp_id, "formula": formula,
            "matched_n1": matched_n1, "matched_nK": matched_nK,
            "first_match_idx": first_match_idx, "n_gen": len(samples),
            "rmsd_n1": rmsd_n1, "rmsd_nK": rmsd_nK,
        })

    n_completed = len(results)
    n_targeted = len(targets)   # denominator: all targeted rows; timed-out and failed rows are misses
    n_gen_failed = sum(1 for r in results if r.get("gen_failed"))
    headline = {
        "n_rows": n_targeted, "n_completed": n_completed, "n_timeout": n_timeout,
        "n_gen_failed": n_gen_failed,
        "gen_failed_rate": (n_gen_failed + n_timeout) / n_targeted if n_targeted else 0.0,
        "K": args.K, "guidance_factor": args.guidance_factor,
        "num_steps": args.num_steps,
        "match_rate_n1": n_match_n1 / max(1, n_targeted),
        "match_rate_nK": n_match_nK / max(1, n_targeted),
        "ckpt_dir": str(args.ckpt_dir),
        "note": "MatterGen CSP-mode baseline (no ALM bridge, no FK); timed-out and failed rows count as misses",
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(headline, indent=2))
    with (args.out_dir / "predictions.jsonl").open("w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    print(f"\n[mattergen-csp] {args.test_csv.parent.name} ({n_targeted} rows, K={args.K}):")
    print(f"  M@1   = {headline['match_rate_n1']:.3f}")
    print(f"  M@K   = {headline['match_rate_nK']:.3f}")
    print(f"  total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
