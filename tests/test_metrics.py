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


def test_threshold_when_detector_covers_few_cells():
    # 0.5% of the negative cells are defined: the threshold must be -inf, not nan
    neg = np.full(1000, -np.inf)
    neg[:5] = np.arange(5.0)
    thr = threshold_at_fa(neg, fa=0.01)
    assert thr == -np.inf
    assert detection_rate(np.array([0.1, -np.inf]), thr) == 0.5
