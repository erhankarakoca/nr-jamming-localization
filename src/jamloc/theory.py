"""Statistics of the DMRS residual: an exact false-alarm threshold and the detection probability.

Notation (one allocation of ``n`` PRBs, ``N = 6n`` DMRS pilots on one comb):

* LS estimate at pilot k: ``h_ls,k = h_k + w_k / x_k``; the noise term has variance
  ``s2 = N0 / Pp`` (``Pp``: pilot power).
* ``U`` ``[N, L]``: the delay subspace of ``jamloc.features`` and ``P = I - U U^H`` the projector
  onto its complement. The residual is ``r = P h_ls``.
* Statistic of PRB b (pilot set B): ``R_b = ||r_B||^2 / (s2 * tr P_BB)`` (``features.residual_map``).

**No jamming.** If the UE channel lies in the delay subspace (in practice: if its energy outside
the subspace is small compared with the noise; for a path near ``tau_max`` at the band edge this
holds up to about 25 dB SNR, beyond which a larger ``tau_max`` is needed), ``P h = 0`` and ``r_B = (P w/x)_B``
is complex Gaussian with covariance ``s2 * P_BB`` (a 6 x 6 block of ``P``). Rotating it to the
eigenvectors of ``P_BB`` gives

    R_b = sum_i lambda_i E_i,   E_i ~ Exp(1) independent,   lambda_i = mu_i / sum_j mu_j,

with ``mu_i`` the eigenvalues of ``P_BB``. Neither the channel nor the noise level appears: the
false-alarm rate of a fixed threshold is the same at every SNR and on every channel (a CFAR test),
and the threshold for a false-alarm rate ``alpha`` can be computed instead of learned. It depends
only on the allocation size and on where the PRB sits in it (``P`` is not uniform near the band
edges).

**Gaussian jamming.** A jammer with energy ``S_J`` per resource element, seen through its channel
``g_k``, adds ``g_k z_k / x_k`` with variance ``|g_k|^2 S_J / Pp`` on the jammed pilots. Then
``r_B`` has covariance ``s2 * (P (I + rho G) P)_BB`` with ``rho = S_J / N0`` (jammer-to-noise ratio
per resource element) and ``G = diag(|g_k|^2 on jammed pilots, 0 elsewhere)``. ``R_b`` is again a
weighted sum of exponentials, with the eigenvalues of that matrix divided by ``tr P_BB``. The
projection also moves jamming energy onto unjammed pilots next to the block (leakage), which the
same formula covers.

The tail probability of a weighted sum of exponentials is computed exactly as that of a chain of
exponential stages (a phase-type distribution), with a matrix exponential.
"""

from functools import cache

import numpy as np
from scipy import linalg

from jamloc.config import SimConfig
from jamloc.features import _subspace

PILOTS_PER_PRB = 6


CFG = SimConfig()


@cache
def _projector(n_prb: int, scs: float, tau_max: float, step: int = 2) -> np.ndarray:
    known_sc = tuple(range(0, 12 * n_prb, step))
    u = _subspace(known_sc, scs, tau_max, 1 - 1e-6)
    return np.eye(len(known_sc)) - u @ u.conj().T


def projector(n_prb: int, cfg: SimConfig = CFG, step: int = 2) -> np.ndarray:
    """``P = I - U U^H`` for an allocation of ``n_prb`` PRBs whose known symbols sit on every
    ``step``-th subcarrier: ``[6n, 6n]`` for the DMRS comb (``step=2``), ``[12n, 12n]`` for a
    decoded data symbol (``step=1``)."""
    return _projector(n_prb, cfg.scs, cfg.tau_max, step)


def _block(prb: int, step: int = 2) -> slice:
    per = 12 // step
    return slice(per * prb, per * (prb + 1))


def weights(n_prb: int, prb: int, jnr=None, cfg: SimConfig = CFG, step: int = 2) -> np.ndarray:
    """Weights ``lambda_i`` of ``R_b = sum_i lambda_i E_i`` for PRB ``prb`` of the allocation.

    ``jnr``: optional jammer-to-noise ratio per known symbol ``[12n / step]`` (linear, already
    multiplied by the jammer's channel gain ``|g_k|^2``; 0 where there is no jamming). Without it:
    no jamming. ``step``: 2 for the DMRS comb, 1 for a decoded data symbol.
    """
    p = projector(n_prb, cfg, step)
    b = _block(prb, step)
    norm = np.trace(p[b, b]).real
    if jnr is None:
        cov = p[b, b]
    else:
        cov = (p * (1 + np.asarray(jnr, float))[None, :]) @ p.conj().T
        cov = cov[b, b]
    return np.clip(np.linalg.eigvalsh(cov), 0, None) / norm


def _generator(lam: np.ndarray) -> np.ndarray:
    """Phase-type generator of ``sum_i lam_i E_i``: exponential stages in series, rates 1/lam_i."""
    lam = lam[lam > 1e-9 * lam.max()]  # zero modes contribute nothing
    rates = 1 / lam
    return np.diag(-rates) + np.diag(rates[:-1], 1)


def ccdf(t, lam) -> np.ndarray:
    """``P(sum_i lam_i E_i > t)`` for ``E_i ~ Exp(1)`` independent: exact, for any weights.

    The sum is the time to pass through exponential stages in series, so its tail probability is
    ``e_1^T expm(S t) 1`` with the stage generator ``S``. Unlike the textbook partial-fraction
    formula this stays accurate when weights are (nearly) equal.
    ``t``: thresholds (any shape). ``lam``: weights ``[k]``, or ``[M, k]`` for M sums at once;
    the result then has shape ``[M, *t.shape]``.
    """
    t = np.asarray(t, float)
    lam_in = np.asarray(lam, float)
    out = np.array(
        [
            [linalg.expm(_generator(row) * ti)[0].sum() for ti in t.reshape(-1)]
            for row in np.atleast_2d(lam_in)
        ]
    )
    out = np.where(t.reshape(-1) < 0, 1.0, np.clip(out, 0.0, 1.0)).reshape((-1, *t.shape))
    return out[0] if lam_in.ndim == 1 else out


def slot_average(lam: np.ndarray, slots: int) -> np.ndarray:
    """Weights of the residual averaged over ``slots`` slots with independent noise.

    With the same allocation in every slot, ``(R_1 + ... + R_K) / K`` is again a weighted sum of
    unit exponentials: every weight divided by K and repeated K times. Its mean stays 1 and its
    variance shrinks by K. (Under jamming this holds when the jammer's channel stays the same over
    the K slots; the gains of ``detection_probability`` are drawn once per K slots.)
    """
    return np.tile(np.asarray(lam, float) / slots, slots)


def threshold(
    n_prb: int, prb: int, fa: float = 0.01, cfg: SimConfig = CFG, slots: int = 1, step: int = 2
) -> float:
    """Threshold on ``R_b`` (averaged over ``slots`` slots) exceeded with probability ``fa``
    without jamming.

    Read from the tabulated tail probability (linear interpolation of its logarithm on a 0.05
    grid; error below 1e-4 against an exact root search).
    """
    grid, logp = _log_ccdf_table(n_prb, prb, cfg.scs, cfg.tau_max, slots, step)
    return float(np.interp(np.log10(fa), logp[::-1], grid[::-1]))


@cache
def _log_ccdf_table(
    n_prb: int, prb: int, scs: float, tau_max: float, slots: int = 1, step: int = 2
):
    # On a uniform grid, expm(S t_k) = expm(S dt)^k: one matrix exponential for the whole table
    grid = np.linspace(0.0, 40.0, 801)
    cfg = SimConfig(scs=scs, tau_max=tau_max)
    lam = slot_average(weights(n_prb, prb, cfg=cfg, step=step), slots)
    step = linalg.expm(_generator(lam) * (grid[1] - grid[0]))
    row = np.eye(len(step))[0]
    p = np.empty_like(grid)
    for k in range(len(grid)):
        p[k] = row.sum()
        row = row @ step
    return grid, np.log10(np.clip(p, 1e-300, 1.0))


def log10_pvalue(r, n_prb: int, prb: int, cfg: SimConfig = CFG, step: int = 2) -> np.ndarray:
    """``log10 P(R_b > r | no jamming)`` for observed residuals ``r`` of PRB ``prb``.

    Values beyond the table (r > 40, p far below 1e-12) are extrapolated with the exponential tail
    of the largest weight. ``step``: 2 for the DMRS comb, 1 for a decoded data symbol.
    """
    grid, logp = _log_ccdf_table(n_prb, prb, cfg.scs, cfg.tau_max, 1, step)
    r = np.asarray(r, float)
    out = np.interp(r, grid, logp)
    beyond = r > grid[-1]
    if beyond.any():
        slope = -1 / (weights(n_prb, prb, cfg=cfg, step=step).max() * np.log(10))
        out = np.where(beyond, logp[-1] + slope * (r - grid[-1]), out)
    return out


def detection_probability(
    jnr_db,
    n_prb: int,
    prb: int,
    jammed_prbs,
    fa: float = 0.01,
    fading: str = "rayleigh",
    draws: int = 2000,
    seed: int = 0,
    cfg: SimConfig = CFG,
    slots: int = 1,
) -> np.ndarray:
    """Probability that ``R_b`` of PRB ``prb`` (averaged over ``slots`` slots, see
    ``slot_average``) exceeds the ``fa`` threshold, per JNR.

    ``jammed_prbs``: PRB indices (inside the allocation) whose DMRS is jammed. ``fading``:
    ``"flat"`` (jammer channel gain 1) or ``"rayleigh"`` (gain constant within a PRB, independent
    Rayleigh across PRBs; averaged over ``draws`` draws). ``prb`` need not be jammed: then this is
    the false-alarm rate caused by leakage from the block.
    """
    t = threshold(n_prb, prb, fa, cfg, slots)
    rng = np.random.default_rng(seed)
    jammed = np.zeros(n_prb)
    jammed[list(jammed_prbs)] = 1
    # The same gain draws for every JNR, so the curve is smooth
    gains = jammed[None, :] if fading == "flat" else rng.exponential(1.0, (draws, n_prb)) * jammed
    out = []
    for rho in 10 ** (np.atleast_1d(np.asarray(jnr_db, float)) / 10):
        lam = np.stack(
            [
                slot_average(weights(n_prb, prb, np.repeat(rho * g, PILOTS_PER_PRB), cfg), slots)
                for g in gains
            ]
        )
        out.append(float(ccdf(t, lam).mean()))
    return np.array(out)
