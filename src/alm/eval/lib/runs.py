"""Per-run output: metrics.json + predictions.jsonl under a run dir."""

import json
import os
from pathlib import Path

from paths import EVAL_RESULTS

DEFAULT_RESULTS_ROOT = EVAL_RESULTS


def run_dir(benchmark, checkpoint_path, run_id=None):
    # run_id precedence: explicit arg > ALM_EVAL_RUN_ID env (avoids step=N basename collisions) > checkpoint dir name.
    rid = run_id or os.environ.get("ALM_EVAL_RUN_ID") or Path(checkpoint_path).name
    results_root = os.environ.get("ALM_EVAL_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)
    d = Path(results_root) / benchmark / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run(run_path, metrics, predictions):
    run_path = Path(run_path)
    with open(run_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(run_path / "predictions.jsonl", "w") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")
    print(f"[eval] wrote {run_path}/metrics.json ({len(predictions)} predictions)")
