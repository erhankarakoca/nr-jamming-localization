import numpy as np

from jamloc.baselines import evm_scores, residual_scores
from jamloc.features import FEATURE_NAMES
from jamloc.metrics import detection_rate, f1, threshold_at_fa


def test_threshold_at_fa():
    scores = np.random.default_rng(0).uniform(size=100_000)
    thr = threshold_at_fa(scores, fa=0.01)
    assert abs(thr - 0.99) < 0.005
    assert abs((scores > thr).mean() - 0.01) < 0.002


def test_f1():
    label = np.zeros((4, 24, 14), dtype=bool)
    label[:, 3:6, 2] = True
    assert f1(label, label) == 1.0
    assert f1(np.zeros_like(label), label) == 0.0


def test_masked_cells_never_flagged():
    features = np.zeros((2, len(FEATURE_NAMES), 24, 14), dtype=np.float32)
    names = list(FEATURE_NAMES)
    features[:, names.index("dmrs_mask"), 4:10, 2] = 1
    features[:, names.index("data_mask"), 4:10, :] = 1
    features[:, names.index("data_mask"), 4:10, 2] = 0
    features[:, names.index("residual_db"), 4:10, 2] = 20.0
    features[:, names.index("evm_db"), 4:10] = 10.0

    r, e = residual_scores(features), evm_scores(features)
    assert np.isneginf(r[:, :4]).all() and np.isneginf(r[:, 4:10, 3]).all()
    assert np.isneginf(e[:, 4:10, 2]).all()
    assert np.allclose(r[:, 4:10, 2], 100.0) and np.allclose(e[:, 4:10, 5], 10.0)
    assert detection_rate(r[np.isneginf(r)], thr=-1e30) == 0.0


def test_threshold_counts_only_cells_the_detector_scores():
    # A detector that scores 5% of the cells: 1% false alarm is set over those cells only
    rng = np.random.default_rng(1)
    neg = np.full(100_000, -np.inf)
    neg[:5_000] = rng.uniform(size=5_000)
    thr = threshold_at_fa(neg, fa=0.01)
    assert abs(thr - 0.99) < 0.01
    assert detection_rate(np.array([0.995, 0.5, -np.inf]), thr) == 1 / 3


def test_threshold_when_detector_scores_no_negative_cell():
    assert threshold_at_fa(np.full(10, -np.inf)) == -np.inf


def test_wilson_interval():
    from jamloc.metrics import wilson_interval

    lo, hi = wilson_interval(np.array([0, 50, 100]), np.array([100, 100, 100]))
    assert lo[0] == 0 and abs(hi[0] - 0.0370) < 1e-3  # no error in 100: below ~3.7%
    assert lo[1] < 0.5 < hi[1] and abs((0.5 - lo[1]) - (hi[1] - 0.5)) < 1e-9
    assert abs(hi[2] - 1) < 1e-12 and 0.95 < lo[2] < 1
