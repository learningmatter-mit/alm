"""ALM eval metrics. mad_mae_ratio = MAD(targets)/MAE (LLM4Mat-Bench; >5 = good)."""

import numpy as np
from sklearn.metrics import f1_score


def _finite_pair(pred, target):
    p, t = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    m = np.isfinite(p) & np.isfinite(t)
    return p[m], t[m]


def mae(pred, target):
    p, t = _finite_pair(pred, target)
    if len(p) == 0:
        return float("nan")
    return float(np.mean(np.abs(p - t)))


def rmse(pred, target):
    p, t = _finite_pair(pred, target)
    if len(p) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mad_mae_ratio(pred, target):
    p, t = _finite_pair(pred, target)
    if len(p) == 0:
        return float("nan")
    mad = float(np.mean(np.abs(t - np.mean(t))))
    err = float(np.mean(np.abs(p - t)))
    return mad / err if err > 0 else float("nan")


def accuracy(pred, target):
    p, t = np.asarray(pred), np.asarray(target)
    return float(np.mean(p == t))


def weighted_f1(pred, target):
    return float(f1_score(target, pred, average="weighted", zero_division=0))
