"""Walk eval_results/, join with cited baselines, emit CSV + LaTeX."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "alm" / "eval"))
from baselines import BASELINES


def _read_metrics(p):
    with open(p) as f:
        return json.load(f)


def _flatten(d, prefix=()):
    """Yield (path_tuple, value) for every numeric leaf in a nested dict."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flatten(v, prefix + (str(k),))
    elif isinstance(d, (int, float)):
        yield prefix, d


def _tex_table(rows, benchmark):
    bench_rows = [r for r in rows if r["benchmark"] == benchmark]
    if not bench_rows:
        return ""
    cols = sorted({(r["task"], r["metric"]) for r in bench_rows})
    models = sorted({r["model"] for r in bench_rows})
    out = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule"]
    out.append("Model & " + " & ".join(f"{t}/{m}" for t, m in cols) + " \\\\ \\midrule")
    for model in models:
        cells = []
        for task, metric in cols:
            match = [r for r in bench_rows if r["model"] == model
                     and r["task"] == task and r["metric"] == metric]
            cells.append(f"{match[0]['value']:.4f}" if match else "—")
        out.append(f"{model} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.environ.get(
        "ALM_EVAL_RESULTS_ROOT", "./eval_results"))
    p.add_argument("--out_dir", default="evals")
    p.add_argument("--run_id", default=None,
                   help="filter to one run subdir per benchmark (e.g. step=12000)")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="optional list of benchmark dir names; defaults to all under --root")
    args = p.parse_args()

    rows = []
    root = Path(args.root)
    for benchmark_dir in sorted(p_ for p_ in root.glob("*") if p_.is_dir()):
        if args.benchmarks and benchmark_dir.name not in args.benchmarks:
            continue
        for run in sorted(benchmark_dir.iterdir()):
            if not (run.is_dir() and (run / "metrics.json").exists()):
                continue
            if args.run_id and run.name != args.run_id:
                continue
            metrics = _read_metrics(run / "metrics.json")
            for path, value in _flatten(metrics):
                # path leaf is the metric; everything before it joins into "task"
                task = ".".join(path[:-1]) or "_"
                rows.append({"benchmark": benchmark_dir.name, "run_id": run.name,
                             "model": "ALM", "task": task,
                             "metric": path[-1], "value": float(value)})

    # Inject cited baselines; skip None cells (some report match-rate but no RMSE).
    for bench, models in BASELINES.items():
        if args.benchmarks and bench not in args.benchmarks:
            continue
        for model, tasks in models.items():
            for task, metrics in tasks.items():
                for metric, value in metrics.items():
                    if value is None:
                        continue
                    try:
                        v = float(value)
                    except (TypeError, ValueError):
                        continue
                    rows.append({"benchmark": bench, "run_id": "cited",
                                 "model": model, "task": str(task),
                                 "metric": metric, "value": v})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "headline_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "run_id", "model", "task", "metric", "value"])
        w.writeheader()
        w.writerows(rows)

    tex_path = out_dir / "headline_table.tex"
    with open(tex_path, "w") as f:
        for bench in sorted({r["benchmark"] for r in rows}):
            f.write(f"% --- {bench} ---\n")
            f.write(_tex_table(rows, bench) + "\n\n")

    print(f"wrote {csv_path} ({len(rows)} rows) and {tex_path}")


if __name__ == "__main__":
    main()
