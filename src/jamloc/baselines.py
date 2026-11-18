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

These four score only the cells where their feature is defined (``-inf`` elsewhere), so none of
them can see jamming outside the allocation. The fifth is the strongest classical detector we
could build from the same features:

* Block test (training-free): every cell of the carrier gets the ``-log10`` p-value of the
  statistic that matches the attack structure. Jammers hit whole PRBs on the DMRS symbol, on the
  13 other symbols, or on both, so the statistic of a data cell is averaged over the 13 data
  symbols of its PRB:

  ===============  ==========================================  =====================================
  cell             statistic                                   null distribution (no jamming)
  ===============  ==========================================  =====================================
  unscheduled DMRS power P                                     mean of 12 unit exponentials (exact)
  unscheduled data P averaged over the PRB's 13 data symbols   mean of 156 unit exponentials (exact)
  scheduled DMRS   residual R                                  weighted exponentials (exact, theory)
  scheduled data   EVM averaged over the PRB's data symbols    benign ``val`` PRBs, same SNR bin
  ===============  ==========================================  =====================================

  The EVM needs an empirical null because its level without jamming grows with the SNR
  (``jamloc.features``) and is higher at the allocation edges; the gNB knows its SNR and its
  allocation, so the null is taken per 4 dB SNR bin, separately for edge PRBs, and its tail is
  extrapolated as exponential beyond the 95th percentile (so is the power ratio's below).

  Pilot jamming raises the EVM of the scheduled data symbols through the corrupted channel
  estimate. A rule separates it from partial-band jamming, which also raises the residual: the
  DMRS cell and the data cells of a PRB see the same channel, so their power ratio is near 1
  without jamming and under partial-band jamming, and above 1 when only the pilots are jammed.
  When the residual and this ratio are both significant at 1% (thresholds fixed in advance, not
  tuned), the PRB's data cells are not flagged.

Scores are linear feature values (block test: ``-log10`` p-values).
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


# --- Block test -------------------------------------------------------------------------------

EVM_SNR_EDGES_DB = (9.0, 13.0, 17.0, 21.0)  # bins of the EVM null: <9, 9-13, ..., >=21 dB
_TAIL = 0.05  # the EVM null is empirical up to its 95th percentile, exponential beyond
_RULE = 2.0  # -log10 p of both tests of the pilot-only rule (1%, fixed in advance)


def _log10_gamma_sf(x: np.ndarray, num: int) -> np.ndarray:
    """``log10 P(X >= x)`` for X the mean of ``num`` unit exponentials (noise-only power)."""
    from scipy.stats import gamma

    return gamma.logsf(x, a=num, scale=1.0 / num) / np.log(10)


def prb_data_evm(features: np.ndarray) -> np.ndarray:
    """``[N, 24]``: linear EVM averaged over each PRB's data cells (nan if not scheduled)."""
    data = features[:, _IDX["data_mask"]] > 0
    evm = 10 ** (features[:, _IDX["evm_db"]] / 10)
    count = data.sum(-1)
    return np.where(count > 0, (evm * data).sum(-1) / np.maximum(count, 1), np.nan)


def _evm_groups(d: dict) -> np.ndarray:
    """``[N, 24]``: null group of each PRB, the SNR bin and whether it is an allocation edge
    (the edge PRBs' channel estimates are extrapolated, so their EVM has a heavier tail)."""
    pos = np.arange(d["features"].shape[2])[None] - d["alloc_start"][:, None]
    edge = (pos == 0) | (pos == d["alloc_size"][:, None] - 1)
    return 2 * np.digitize(d["snr_db"], EVM_SNR_EDGES_DB)[:, None] + edge


def _pilot_to_data_power(features: np.ndarray, dmrs: int) -> np.ndarray:
    """``[N, 24]``: log of the DMRS cell's power over the PRB's mean data-symbol power.

    Both see the same channel at low mobility (the DMRS comb carries twice the energy on half the
    subcarriers, so the cell means match), so the ratio is near 0 without jamming and under
    partial-band jamming, and positive when only the pilots are jammed."""
    power = 10 ** (features[:, _IDX["power_db"]] / 10)
    data_symbols = np.arange(features.shape[3]) != dmrs
    return np.log(power[:, :, dmrs] / power[:, :, data_symbols].mean(-1))


def _empirical_scores(d: dict, null: dict, statistic) -> np.ndarray:
    """``[N, 24]``: ``-log10`` of the fraction of benign scheduled ``null`` PRBs of the same group
    (SNR bin, edge or not) whose ``statistic`` is at least as high as this PRB's.

    Beyond the group's 95th percentile x0 the null is extrapolated as exponential (peaks over
    threshold), so the score keeps growing past the largest reference value instead of
    saturating at log10 of the number of reference PRBs. A group with fewer than 20 reference
    PRBs uses all of them."""
    benign = {k: v[null["attack"] == 0] for k, v in null.items()}
    keep = benign["features"][:, _IDX["alloc_mask"]].max(-1) > 0
    ref, ref_group = statistic(benign["features"])[keep], _evm_groups(benign)[keep]

    values, group = statistic(d["features"]), _evm_groups(d)
    out = np.zeros(values.shape)
    for k in np.unique(group):
        sorted_ref = np.sort(ref[ref_group == k] if (ref_group == k).sum() >= 20 else ref)
        sel, x = group == k, values[group == k]
        at_least = len(sorted_ref) - np.searchsorted(sorted_ref, x, side="left")
        score = -np.log10((at_least + 1) / (len(sorted_ref) + 1))
        x0 = np.quantile(sorted_ref, 1 - _TAIL)
        excess = sorted_ref[sorted_ref > x0] - x0
        tail = -np.log10(_TAIL) + (x - x0) / (max(excess.mean(), 1e-12) * np.log(10))
        out[sel] = np.where(x > x0, tail, score)
    return np.nan_to_num(out, nan=0.0)


def block_test_scores(d: dict, null: dict) -> np.ndarray:
    """``[N, 24, 14]``: ``-log10`` p-value of every cell under the block test (see above).

    ``d`` is a loaded split; ``null`` the split whose benign slots give the EVM and power-ratio
    nulls (``val``).
    """
    from jamloc.config import SimConfig

    dmrs = SimConfig().dmrs_symbol
    features = d["features"]
    data_symbols = np.arange(features.shape[3]) != dmrs
    scheduled = features[:, _IDX["alloc_mask"], :, dmrs] > 0  # [N, 24]
    power = 10 ** (features[:, _IDX["power_db"]] / 10)

    # One score for the DMRS cell and one for the data cells of every PRB: [N, 24] each
    pilot = np.where(
        scheduled,
        residual_cfar_scores(d)[:, :, dmrs],
        -_log10_gamma_sf(power[:, :, dmrs], 12),
    )
    data = np.where(
        scheduled,
        _empirical_scores(d, null, prb_data_evm),
        -_log10_gamma_sf(power[:, :, data_symbols].mean(-1), 12 * int(data_symbols.sum())),
    )

    # Pilot-only rule: if the residual and the DMRS-to-data power ratio are both significant, the
    # PRB's EVM rise comes from the corrupted channel estimate, not from jammed data symbols
    ratio = _empirical_scores(d, null, lambda f: _pilot_to_data_power(f, dmrs))
    pilot_only = scheduled & (pilot > _RULE) & (ratio > _RULE)
    data = np.where(pilot_only, -np.inf, data)

    out = np.empty(power.shape)
    out[:, :, dmrs] = pilot
    out[:, :, data_symbols] = data[:, :, None]
    return out
