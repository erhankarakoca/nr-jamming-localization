"""Threshold detectors used as baselines.

* EVM threshold (Örnek & Kartal): flags data cells whose EVM is high. It cannot look at the DMRS
  symbol, and pilot jamming raises the EVM of the *data* cells, so it reports such attacks at the
  wrong symbols.
* Residual threshold (OFDM counterpart of pilot-subspace detection, Mehrabian & Kaddoum): flags
  DMRS cells whose residual is high. It is blind to data-only jamming.
* Residual, exact p-value ("residual CFAR"): the same residual, scored by its exact tail
  probability without jamming (``jamloc.theory``), ``-log10 p``. In benign slots its threshold for
  a false-alarm rate ``alpha`` is ``-log10 alpha`` on every PRB, at every SNR and on every channel,
  so it needs no training data; it also accounts for the PRBs at the band edges, whose residual
  has a heavier tail. (``evaluate`` still calibrates it on validation like every detector, where
  leakage from jammed PRBs raises the threshold; the analytic threshold is checked in
  ``experiments/residual_theory``.)

* Power threshold (Li et al., MILCOM 2012): flags scheduled cells whose received power is high
  relative to the slot's typical scheduled cell, the classic per-tone power test. Fading of the
  UE's own signal makes it insensitive to weak jamming.

Scores are the linear feature values; cells where the feature is not defined get ``-inf``.
"""

import numpy as np

from jamloc.features import FEATURE_NAMES

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _masked_linear(features: np.ndarray, feature: str, mask: str) -> np.ndarray:
    values = 10 ** (features[:, _IDX[feature]] / 10)
    return np.where(features[:, _IDX[mask]] > 0, values, -np.inf)


def evm_scores(features: np.ndarray) -> np.ndarray:
    """``[N, 24, 14]``: linear EVM on data cells, ``-inf`` elsewhere."""
    return _masked_linear(features, "evm_db", "data_mask")


def power_scores(features: np.ndarray) -> np.ndarray:
    """``[N, 24, 14]``: received power of each scheduled cell relative to the slot's median
    scheduled cell, ``-inf`` elsewhere.

    The test compares the power with the level the UE's signal is expected to have; the median
    over the slot's scheduled cells is a robust estimate of that level as long as the jammer
    covers fewer than half of them. Without it, one threshold for slots of 5 to 25 dB SNR would be
    set by the strongest slots.
    """
    scores = _masked_linear(features, "power_db", "alloc_mask")
    scheduled = np.where(np.isfinite(scores), scores, np.nan)
    level = np.nanmedian(scheduled.reshape(len(scores), -1), axis=1)
    return scores / level[:, None, None]


def residual_scores(features: np.ndarray) -> np.ndarray:
    """``[N, 24, 14]``: linear DMRS residual on DMRS cells, ``-inf`` elsewhere."""
    return _masked_linear(features, "residual_db", "dmrs_mask")


def residual_cfar_scores(d: dict) -> np.ndarray:
    """``[N, 24, 14]``: ``-log10`` of the no-jamming tail probability of each DMRS cell's residual.

    ``d`` is a loaded split (needs ``features``, ``alloc_start`` and ``alloc_size``).
    """
    from jamloc.config import SimConfig
    from jamloc.theory import log10_pvalue

    dmrs = SimConfig().dmrs_symbol
    features = d["features"]
    out = np.full((len(features), features.shape[2], features.shape[3]), -np.inf)
    r = 10 ** (features[:, _IDX["residual_db"], :, dmrs] / 10)  # [N, 24]
    for n in np.unique(d["alloc_size"]):
        for start in np.unique(d["alloc_start"][d["alloc_size"] == n]):
            rows = np.flatnonzero((d["alloc_size"] == n) & (d["alloc_start"] == start))
            for pos in range(int(n)):
                prb = int(start) + pos
                out[rows, prb, dmrs] = -log10_pvalue(r[rows, prb], int(n), pos)
    return out
