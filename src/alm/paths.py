"""Filesystem roots resolved from env vars (override per cluster; defaults repo-relative)."""
import os

DATA_ROOT = os.environ.get("ALM_DATA_ROOT", "./data")
CHECKPOINTS = os.environ.get("ALM_CHECKPOINTS", "./checkpoints")
RUNS = os.environ.get("ALM_RUNS", "./runs")
EVAL_RESULTS = os.environ.get("ALM_EVAL_RESULTS_ROOT", "./eval_results")
