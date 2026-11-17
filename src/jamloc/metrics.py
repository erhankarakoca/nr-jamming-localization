"""Cell-level detection metrics.

Every detector gives one score per cell (higher = more likely jammed). A threshold is chosen on
non-jammed validation cells so that a fraction ``fa`` of them is flagged (per-cell false-alarm
rate); the detection probability is the fraction of jammed cells above that threshold.

Detectors defined on only part of the grid (e.g. the residual, only on DMRS cells) give ``-inf``
elsewhere: those cells are never flagged but still count in the false-alarm denominator, so all
detectors are compared at the same number of false flags per slot.
"""

import numpy as np


def threshold_at_fa(neg_scores: np.ndarray, fa: float = 0.01) -> float:
    """Threshold such that a fraction ``fa`` of the negative (non-jammed) scores lie above it."""
    return float(np.quantile(np.ravel(neg_scores), 1 - fa, method="higher"))


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
