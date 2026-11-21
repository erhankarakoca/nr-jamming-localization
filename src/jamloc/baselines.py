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

* Empty-comb test: type-1 DMRS leaves every second subcarrier of the DMRS symbol empty. Without
  jamming the received energy there, divided by the noise power, is the mean of 6 unit
  exponentials on every PRB, inside the allocation too, at any SNR and on any channel. A jammer on
  the whole DMRS symbol raises it; a jammer on the pilot comb only, or a spoofer, does not. Scored
  by ``-log10`` of that exact tail on the scheduled DMRS cells.

These five score only the cells where their feature is defined (``-inf`` elsewhere), so none of
them can see jamming outside the allocation. The block test is the strongest classical detector
we could build from the same features:

* Block test (training-free): every cell of the carrier gets the ``-log10`` p-value of the
  statistic that matches the attack structure. The test scores DMRS cells separately and
  averages the statistic of a data cell over the PRB's ``m`` data symbols:

  ===============  ==========================================  =====================================
  cell             statistic                                   null distribution (no jamming)
  ===============  ==========================================  =====================================
  unscheduled DMRS power P                                     mean of 12 unit exponentials (exact)
  unscheduled data P averaged over the PRB's data symbols      mean of 12 m unit exponentials (exact)
  scheduled DMRS   residual R                                  weighted exponentials (exact, theory)
  scheduled data   EVM averaged over the PRB's data symbols    benign ``val`` PRBs, same SNR bin
  ===============  ==========================================  =====================================

  The EVM needs an empirical null because its level without jamming grows with the SNR
  (``jamloc.features``) and is higher at the allocation edges. The null is taken from benign
  ``val`` PRBs per 4 dB bin of the SNR the gNB estimates from the slot (``snr_db_est``,
  ``jamloc.noise``; jamming on the data raises that estimate, as in a real receiver), separately
  for edge PRBs, and its tail is extrapolated as exponential beyond the 95th percentile (so is
  the power ratio's below). The test has no learned parameters, but it does need these benign
  validation slots.

  Pilot jamming raises the EVM of the scheduled data symbols through the corrupted channel
  estimate. A rule separates it from partial-band jamming, which also raises the residual: the
  DMRS and data cells of a PRB see the same channel, so the ratio of their mean powers is near 1
  without jamming and under partial-band jamming, and above 1 when only the pilots are jammed.
  When the residual and this ratio are both significant at 1% (thresholds fixed in advance, not
  tuned), the PRB's data cells get p = 1 (score 0): they stay among the scored cells but are not
  flagged.

* Block test with the empty comb: the same, but a scheduled DMRS cell is scored by the smaller of
  the residual's and the empty comb's p-values (Bonferroni-corrected for the two tests), and the
  pilot-only rule also fires when the empty comb is significant.

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


def comb_scores(features: np.ndarray) -> np.ndarray:
    """``[N, 24, 14]``: ``-log10`` p-value of the empty-comb power on scheduled DMRS cells."""
    comb = 10 ** (features[:, _IDX["comb_db"]] / 10)
    return np.where(features[:, _IDX["dmrs_mask"]] > 0, -_log10_gamma_sf(comb, 6), -np.inf)


def _dmrs_symbols(features: np.ndarray) -> np.ndarray:
    """DMRS rows of a split, including the unallocated PRBs on those same rows."""
    return np.flatnonzero(features[:, _IDX["dmrs_mask"]].any(axis=(0, 1)))


def residual_cfar_scores(d: dict) -> np.ndarray:
    """``[N, 24, 14]``: ``-log10`` of the no-jamming tail probability of each DMRS cell's residual.

    ``d`` is a loaded split (needs ``features``, ``alloc_start`` and ``alloc_size``).
    """
    from jamloc.theory import log10_pvalue

    features = d["features"]
    dmrs = _dmrs_symbols(features)
    out = np.full((len(features), features.shape[2], features.shape[3]), -np.inf)
    r = 10 ** (features[:, _IDX["residual_db"]][..., dmrs] / 10)  # [N, 24, num_dmrs]
    for n in np.unique(d["alloc_size"]):
        for start in np.unique(d["alloc_start"][d["alloc_size"] == n]):
            rows = np.flatnonzero((d["alloc_size"] == n) & (d["alloc_start"] == start))
            for pos in range(int(n)):
                prb = int(start) + pos
                out[rows[:, None], prb, dmrs] = -log10_pvalue(r[rows, prb], int(n), pos)
    return out


# --- Block test -------------------------------------------------------------------------------

EVM_SNR_EDGES_DB = (9.0, 13.0, 17.0, 21.0)  # bins of the EVM null: <9, 9-13, ..., >=21 dB
_TAIL = 0.05  # the EVM null is empirical up to its 95th percentile, exponential beyond
_RULE = 2.0  # -log10 p of both tests of the pilot-only rule (1%, fixed in advance)


def _log10_gamma_sf(x: np.ndarray, num: int) -> np.ndarray:
    """``log10 P(X >= x)`` for X the mean of ``num`` unit exponentials (noise-only power).

    Far in the tail SciPy's value underflows to ``-inf``; there the leading term of the
    asymptotic expansion, ``z^(num-1) e^-z / Gamma(num)`` with ``z = num x``, keeps the scores
    finite and ordered.
    """
    from scipy.special import gammaln
    from scipy.stats import gamma

    out = gamma.logsf(x, a=num, scale=1.0 / num)
    z = num * np.asarray(x, float)
    tail = (num - 1) * np.log(np.maximum(z, 1e-300)) - z - gammaln(num)
    return np.where(np.isfinite(out), out, tail) / np.log(10)


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
    snr = d.get("snr_db_est", d["snr_db"])  # older datasets have only the true SNR
    return 2 * np.digitize(snr, EVM_SNR_EDGES_DB)[:, None] + edge


def _pilot_to_data_power(features: np.ndarray, dmrs: np.ndarray) -> np.ndarray:
    """``[N, 24]``: log of the ratio of mean DMRS power to mean data power in a PRB.

    Both see the same channel at low mobility (the DMRS comb carries twice the energy on half the
    subcarriers, so the cell means match), so the ratio is near 0 without jamming and under
    partial-band jamming, and positive when only the pilots are jammed."""
    power = 10 ** (features[:, _IDX["power_db"]] / 10)
    data_symbols = ~np.isin(np.arange(features.shape[3]), dmrs)
    return np.log(power[:, :, dmrs].mean(-1) / power[:, :, data_symbols].mean(-1))


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


def block_test_scores(d: dict, null: dict, use_comb: bool = False) -> np.ndarray:
    """``[N, 24, 14]``: ``-log10`` p-value of every cell under the block test (see above).

    ``d`` is a loaded split; ``null`` the split whose benign slots give the EVM and power-ratio
    nulls (``val``). ``use_comb`` adds the empty-comb test to the scheduled DMRS cells.
    """
    features = d["features"]
    dmrs = _dmrs_symbols(features)
    data_symbols = ~np.isin(np.arange(features.shape[3]), dmrs)
    scheduled = features[:, _IDX["alloc_mask"], :, dmrs[0]] > 0  # [N, 24]
    power = 10 ** (features[:, _IDX["power_db"]] / 10)

    # One score per DMRS cell [N, 24, num_dmrs], one for the data cells of each PRB [N, 24]
    pilot_scheduled = residual_cfar_scores(d)[:, :, dmrs]
    if use_comb:  # smaller of the two p-values, times 2 (Bonferroni)
        comb = comb_scores(features)[:, :, dmrs]
        pilot_scheduled = np.maximum(pilot_scheduled, comb) - np.log10(2)
    pilot = np.where(scheduled[..., None], pilot_scheduled, -_log10_gamma_sf(power[:, :, dmrs], 12))
    data = np.where(
        scheduled,
        _empirical_scores(d, null, prb_data_evm),
        -_log10_gamma_sf(power[:, :, data_symbols].mean(-1), 12 * int(data_symbols.sum())),
    )

    # Pilot-only rule: if the residual and the DMRS-to-data power ratio are both significant, the
    # PRB's EVM rise comes from the corrupted channel estimate, not from jammed data symbols
    ratio = _empirical_scores(d, null, lambda f: _pilot_to_data_power(f, dmrs))
    pilot_only = scheduled & (pilot > _RULE).any(-1) & (ratio > _RULE)
    data = np.where(pilot_only, 0.0, data)  # p = 1: scored (counted as tested), never flagged

    out = np.empty(power.shape)
    out[:, :, dmrs] = pilot
    out[:, :, data_symbols] = data[:, :, None]
    return out
