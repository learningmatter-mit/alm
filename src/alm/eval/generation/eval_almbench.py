#!/usr/bin/env python
"""ALM-Bench: run the per-task editing eval (eval_edit.py) across all tasks and roll up one summary table."""

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_EDIT = os.path.join(_HERE, "eval_edit.py")

# task -> (primary metric key, human label)
PRIMARY = {
    "atomtxt":   ("direction_correct_rate", "direction-correct"),
    "polymorph": ("polymorph_lower_energy_rate", "lower-E polymorph"),
    "doping":    ("doping_correct_rate", "doping-correct"),
    "app":       ("app_consistency", "app-consistency (judge)"),
}
DIAGNOSTICS = ["structurally_valid", "gen_failed_rate", "n_scored"]


def run_task(task, args):
    out_dir = os.path.join(args.out_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, _EVAL_EDIT, "--task", task,
        "--alm_checkpoint", args.alm_checkpoint,
        "--atoms_mapper", args.atoms_mapper,
        "--guidance_factor", str(args.guidance_factor),
        "--diffusion_steps", str(args.diffusion_steps),
        "--max_rows", str(args.max_rows),
        "--out_dir", out_dir,
    ]
    if args.mattergen_model_path:
        cmd += ["--mattergen_model_path", args.mattergen_model_path]
    if task == "app":
        cmd += ["--judge_model", args.judge_model]
    print(f"\n{'='*72}\n[almbench] task={task}\n  $ {' '.join(cmd)}\n{'='*72}", flush=True)
    rc = subprocess.run(cmd).returncode
    # metrics.json may nest under a run subdir
    found = None
    for dp, _, fs in os.walk(out_dir):
        if "metrics.json" in fs:
            found = os.path.join(dp, "metrics.json")
            break
    metrics = {}
    if found:
        try:
            metrics = json.load(open(found))
        except Exception as e:  # noqa: BLE001
            print(f"[almbench] WARN: could not parse {found}: {e}")
    return {"task": task, "returncode": rc, "metrics_path": found, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True, help="Stage-2 step=N dir")
    ap.add_argument("--atoms_mapper", required=True, help="Stage-3 step=N/atoms_mapper.pt (edit/synthesis model)")
    ap.add_argument("--mattergen_model_path", default=None,
                    help="from-scratch CSP-mode backbone dir (csp_backbone); editing conditions on the input structure")
    ap.add_argument("--tasks", default="atomtxt,polymorph,doping,app",
                    help="comma list; default = all four ALM-Bench tasks")
    ap.add_argument("--guidance_factor", type=float, default=0.5, help="CFG g (operating point = 0.5)")
    ap.add_argument("--diffusion_steps", type=int, default=1000)
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--judge_model", default="gpt-4o-mini", help="app task only (needs OPENAI_API_KEY)")
    ap.add_argument("--out_dir", default="eval_results/almbench")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if "app" in tasks and not os.environ.get("OPENAI_API_KEY"):
        print("[almbench] NOTE: OPENAI_API_KEY not set — skipping the 'app' task (LM-judge). "
              "Set the key to include it.")
        tasks = [t for t in tasks if t != "app"]

    os.makedirs(args.out_dir, exist_ok=True)
    results = [run_task(t, args) for t in tasks]

    summary = {"out_dir": args.out_dir, "guidance_factor": args.guidance_factor,
               "max_rows": args.max_rows, "tasks": {}}
    rows = []
    for r in results:
        m = r["metrics"]
        key, label = PRIMARY.get(r["task"], (None, r["task"]))
        primary = m.get(key) if key else None
        summary["tasks"][r["task"]] = {
            "primary_metric": key, "primary_value": primary,
            "returncode": r["returncode"],
            **{d: m.get(d) for d in DIAGNOSTICS},
        }
        rows.append((r["task"], label, primary,
                     m.get("structurally_valid"), m.get("gen_failed_rate"), m.get("n_scored"),
                     r["returncode"]))

    sp = os.path.join(args.out_dir, "almbench_summary.json")
    json.dump(summary, open(sp, "w"), indent=2)

    print(f"\n{'='*78}\nALM-Bench summary  (g={args.guidance_factor}, max_rows={args.max_rows})\n{'='*78}")
    print(f"{'task':<11}{'metric':<26}{'value':>8}{'valid':>8}{'gen_fail':>10}{'n':>6}{'rc':>4}")
    for task, label, primary, valid, genfail, n, rc in rows:
        pv = f"{primary:.4f}" if isinstance(primary, (int, float)) else "  n/a"
        vv = f"{valid:.3f}" if isinstance(valid, (int, float)) else " n/a"
        gf = f"{genfail:.3f}" if isinstance(genfail, (int, float)) else " n/a"
        nn = str(n) if n is not None else "n/a"
        print(f"{task:<11}{label:<26}{pv:>8}{vv:>8}{gf:>10}{nn:>6}{rc:>4}")
    print(f"\nwrote {sp}")
    if any(rc != 0 for *_, rc in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
