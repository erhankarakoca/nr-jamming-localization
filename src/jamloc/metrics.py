"""Cell-level detection metrics.

Every detector gives one score per cell (higher = more likely jammed). A threshold is chosen on
non-jammed validation cells so that a fraction ``fa`` of them is flagged (per-cell false-alarm
rate); the detection probability is the fraction of jammed cells above that threshold.

Detectors defined on only part of the grid (e.g. the residual, only on DMRS cells) give ``-inf``
elsewhere. Those cells are never flagged and are left out of the false-alarm rate, so every
detector flags 1% of the non-jammed cells *it looks at*. (Counting them would let a detector that
scores few cells spend the whole 1% budget there, e.g. ~26% false alarm on DMRS cells.)
"""

import numpy as np


def threshold_at_fa(neg_scores: np.ndarray, fa: float = 0.01) -> float:
    """Threshold such that a fraction ``fa`` of the scored negative (non-jammed) cells lie above it.

    Cells the detector cannot score (``-inf``) are ignored.
    """
    neg = np.ravel(neg_scores)
    neg = neg[neg > -np.inf]
    if neg.size == 0:
        return float("-inf")
    return float(np.quantile(neg, 1 - fa, method="higher"))


def detection_rate(pos_scores: np.ndarray, thr: float) -> float:
    """Fraction of jammed cells flagged (``nan`` if there are none)."""
    pos_scores = np.ravel(pos_scores)
    return float((pos_scores > thr).mean()) if pos_scores.size else float("nan")


def f1(pred: np.ndarray, label: np.ndarray) -> float:
    """Cell-level F1 score. Flagging the wrong OFDM symbols counts as false alarms and misses."""
    pred, label = np.asarray(pred, bool), np.asarray(label, bool)
    tp = np.sum(pred & label)
    fp = np.sum(pred & ~label)
    fn = np.sum(~pred & label)
    return float(2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else 1.0


def wilson_interval(errors, trials, z: float = 1.96):
    """95% confidence interval of an error rate ``errors / trials`` (Wilson score interval).

    Unlike the naive "rate +- 2 sigma" it stays inside [0, 1] and still gives a useful upper bound
    when no error was observed (about 3.8 / trials).
    """
    k, n = np.asarray(errors, float), np.asarray(trials, float)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)
