"""Phase 1 success metrics, decided before any model is trained, and used by everyone.
Owner: Rishik."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)


def ece(y, p, n_bins: int = 15) -> float:
    """Expected calibration error: average |observed rate - mean predicted| over probability bins."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def at_threshold(y, p, thr: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p)
    flag = p >= thr
    tp = int((flag & (y == 1)).sum())
    return {"threshold": float(thr), "flag_rate": float(flag.mean()),
            "precision": tp / max(int(flag.sum()), 1), "recall": tp / max(int(y.sum()), 1)}


def threshold_for_precision(y, p, target: float = 0.5) -> float:
    """Lowest threshold whose precision >= target (i.e. the most recall at that precision)."""
    prec, _, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= target)[0]
    return float(thr[ok[0]]) if len(ok) else 1.0


def evaluate(y, p) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p)), "ece": ece(y, p),
            "base_rate": float(y.mean()), "lift": float(average_precision_score(y, p) / y.mean())}


def day_block_bootstrap(y, p, dates, n_boot: int = 200, seed: int = 0) -> tuple:
    """95% CI of PR AUC by resampling whole DAYS (flights on one day share weather, so they are not independent)."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y).astype(int), np.asarray(p)
    _, inv = np.unique(np.asarray(dates), return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(inv.max() + 1)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        vals.append(average_precision_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
