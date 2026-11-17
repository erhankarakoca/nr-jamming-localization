import numpy as np
from scipy import stats

from jamloc import theory


def test_tail_of_weighted_exponentials_matches_closed_forms():
    t = np.array([0.5, 1.0, 3.0])
    assert np.allclose(theory.ccdf(t, [1.0]), np.exp(-t), atol=1e-9)
    # equal weights: a Gamma distribution
    assert np.allclose(theory.ccdf(t, np.full(6, 1 / 6)), stats.gamma(6, scale=1 / 6).sf(t))


def test_weights_sum_to_one_and_edges_have_heavier_tails():
    for b in range(12):
        assert abs(theory.weights(12, b).sum() - 1) < 1e-9
    thr = [theory.threshold(12, b) for b in range(12)]
    assert np.allclose(thr, thr[::-1], atol=1e-6)  # mirror symmetry of the band
    assert thr[0] > thr[5] > 2.0  # band edges: fewer effective degrees of freedom


def test_threshold_gives_the_design_false_alarm_rate():
    rng = np.random.default_rng(0)
    p = theory.projector(12)
    w = (rng.standard_normal((200_000, 72)) + 1j * rng.standard_normal((200_000, 72))) / 2**0.5
    r = w @ p.T  # residual of noise alone (the channel is removed by the projection)
    kept = np.real(np.diag(p))
    for b in (0, 5):
        cells = slice(6 * b, 6 * b + 6)
        stat = (np.abs(r[:, cells]) ** 2).sum(1) / kept[cells].sum()
        assert abs(np.mean(stat > theory.threshold(12, b)) - 0.01) < 0.0009


def test_detection_probability_grows_with_jnr_and_leaks_to_neighbours():
    block = [4, 5, 6, 7]
    pd = theory.detection_probability([-10, 0, 10, 20], 12, 5, block, draws=500)
    assert np.all(np.diff(pd) > 0) and pd[0] < 0.05 and pd[-1] > 0.95
    flat = theory.detection_probability([0.0], 12, 6, block, fading="flat")
    assert abs(flat[0] - 0.3096) < 0.002  # independent derivation and Monte Carlo agree
    leak = theory.detection_probability([-20, 15], 12, 8, block, fading="flat")
    assert abs(leak[0] - 0.01) < 0.002 and leak[1] > 0.3  # unjammed PRB next to the block


def test_pvalue_is_consistent_with_threshold():
    t = theory.threshold(12, 3)
    assert abs(theory.log10_pvalue(t, 12, 3) - (-2)) < 1e-3
    assert theory.log10_pvalue(100.0, 12, 3) < -12  # extrapolated tail
