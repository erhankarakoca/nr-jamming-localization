"""Threshold detectors used as baselines.

* EVM threshold (Örnek & Kartal): flags data cells whose EVM is high. It cannot look at the DMRS
  symbol, and pilot jamming raises the EVM of the *data* cells, so it reports such attacks at the
  wrong symbols.
* Residual threshold (OFDM counterpart of pilot-subspace detection, Mehrabian & Kaddoum): flags
  DMRS cells whose residual is high. It is blind to data-only jamming.

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


def residual_scores(features: np.ndarray) -> np.ndarray:
    """``[N, 24, 14]``: linear DMRS residual on DMRS cells, ``-inf`` elsewhere."""
    return _masked_linear(features, "residual_db", "dmrs_mask")
