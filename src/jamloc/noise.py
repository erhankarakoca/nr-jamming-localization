"""The noise power the gNB assumes, and how it can estimate it from one slot.

Every feature is divided by the receiver's noise power N0, and the LMMSE channel estimator and
equalizer use it too. In the nominal experiments the receiver knows N0 exactly. A real gNB has to
estimate it, and an estimate that is 1 dB too low already makes the learned detectors flag many
clean cells. This module provides what the link needs to run the whole receiver with a wrong or an
estimated N0:

* **A set error** (``SimConfig.noise_estimate = "true"``): the receiver uses N0 times
  ``10^(e/10)``, with ``e`` the per-slot ``noise_error_db`` of the scenario.
* **A per-slot estimate** (``"comb"``): type-1 DMRS with two CDM groups without data leaves the
  other comb of the DMRS symbol empty, so every scheduled PRB has 6 resource elements that carry
  only noise, plus jamming if the jammer hits them. Without jamming, ``|y|^2 / N0`` of such an
  element is a unit exponential. The estimate is the median of ``|y|^2`` over the empty comb of
  the allocation, divided by the expected median of as many unit exponentials. The median ignores
  jammed elements as long as they are fewer than half of the comb; a jammer on the whole DMRS
  symbol of the whole allocation biases it upward. Only elements inside the allocation are used,
  because in a loaded cell the other PRBs belong to other UEs.

The received SNR that the block test and the SNR-binned thresholds need is estimated from the
same slot (``estimated_snr_db``).
"""

from functools import cache

import numpy as np
import torch
from sionna.phy.nr import PUSCHTransmitter

from jamloc.config import SimConfig

NOISE_ESTIMATES = ("true", "comb")


def empty_comb(tx: PUSCHTransmitter) -> tuple[torch.Tensor, torch.Tensor]:
    """OFDM symbol and subcarrier indices (within the allocation) of the empty DMRS comb."""
    pp = tx.resource_grid.pilot_pattern
    sym, sc = torch.nonzero(pp.mask[0, 0], as_tuple=True)  # DMRS REs, both combs
    empty = pp.pilots[0, 0].abs() == 0  # the comb without pilots
    return sym[empty], sc[empty]


@cache
def _median_of_exponentials(m: int) -> float:
    """Expected value of the median (linear interpolation, as ``torch.quantile``) of ``m`` unit
    exponentials. The k-th smallest of m has mean 1/m + 1/(m-1) + ... + 1/(m-k+1)."""
    order_means = np.cumsum(1.0 / np.arange(m, 0, -1))  # k = 1 ... m
    pos = 0.5 * (m - 1)
    lo, frac = int(np.floor(pos)), pos - np.floor(pos)
    hi = min(lo + 1, m - 1)
    return float(order_means[lo] + frac * (order_means[hi] - order_means[lo]))


def estimate_noise_comb(y_alloc: torch.Tensor, tx: PUSCHTransmitter) -> torch.Tensor:
    """Per-slot noise power ``[B]`` from the empty DMRS comb of the allocation (see above)."""
    sym, sc = empty_comb(tx)
    energy = y_alloc[:, 0, 0, sym, sc].abs() ** 2  # [B, 6 n]
    return torch.quantile(energy, 0.5, dim=1) / _median_of_exponentials(energy.shape[1])


def receiver_noise(
    cfg: SimConfig,
    no: torch.Tensor,
    noise_error_db: torch.Tensor | None,
    y_alloc: torch.Tensor,
    tx: PUSCHTransmitter,
) -> torch.Tensor:
    """Noise power ``[B]`` the receiver assumes, given the true one ``no``."""
    if cfg.noise_estimate == "comb":
        return estimate_noise_comb(y_alloc, tx)
    if cfg.noise_estimate != "true":
        raise ValueError(f"noise_estimate must be one of {NOISE_ESTIMATES}")
    if noise_error_db is None:
        return no
    return no * 10 ** (noise_error_db.to(no.device, no.dtype) / 10)


def estimated_snr_db(y_alloc: torch.Tensor, no: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Received SNR per data RE ``[B]`` as the gNB can estimate it from one slot.

    Data REs carry unit energy through a unit-power channel, so their mean received energy is
    1 + N0, and divided by N0 it is the SNR plus one: the estimate is ``mean |y|^2 / N0 - 1``.
    Jamming on the data symbols raises it, as it would in a real receiver.
    """
    data_symbols = [s for s in range(cfg.num_symbols) if s != cfg.dmrs_symbol]
    energy = (y_alloc[:, 0, 0, data_symbols].abs() ** 2).mean(dim=(1, 2))
    return 10 * torch.log10((energy / no - 1).clamp_min(1e-3))
