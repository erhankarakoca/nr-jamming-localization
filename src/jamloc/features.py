"""Per-cell jamming features from quantities the gNB receiver already has.

A cell is one PRB x one OFDM symbol, so every feature is a 24 x 14 map per slot. All three are
divided by the noise level:

* **Power P** - received energy in the cell / noise variance. Defined everywhere on the carrier.
  About 1 on empty cells and about 1 + SNR on scheduled cells without jamming.
* **DMRS residual R** - the LS channel estimates at the pilots, minus their projection onto the
  set of channels that are physically possible (delays up to ``tau_max``). The user's own signal
  lies in that set, so what remains is interference plus noise. Defined on the DMRS symbol of the
  scheduled PRBs. About 1 without jamming, at any SNR and on every PRB.
* **EVM E** - distance from each equalized symbol to the nearest QPSK point, divided by the noise
  variance the LMMSE equalizer expects. Defined on the data symbols of the scheduled PRBs. About 1
  without jamming at moderate SNR, but it grows at high SNR (mean ~2 at 25 dB and ~4 at 30 dB on
  TDL-C): the LS estimator's error variance leaves out interpolation error and channel aging, so
  the equalizer underestimates the true error. Thresholds and the U-Net are calibrated on data
  that contains this effect.
"""

from functools import cache

import numpy as np
import torch
from sionna.phy.nr import PUSCHTransmitter

from jamloc.config import SimConfig
from jamloc.link import SlotBatch
from jamloc.receiver import data_positions

FEATURE_NAMES = ("power_db", "residual_db", "evm_db", "alloc_mask", "dmrs_mask", "data_mask")
DB_RANGE = (-10.0, 40.0)


def _to_carrier(values: torch.Tensor, batch: SlotBatch, cfg: SimConfig) -> torch.Tensor:
    """Place an allocation map ``[B, n, 14]`` on the carrier map ``[B, 24, 14]`` (zeros outside)."""
    out = values.new_zeros(values.shape[0], cfg.num_prb, cfg.num_symbols)
    start = batch.scenario.alloc_start
    out[:, start : start + values.shape[1]] = values
    return out


# --- Power ------------------------------------------------------------------------------------


def power_map(y: torch.Tensor, no: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Mean received energy per cell divided by the noise variance. ``y``: ``[B, 14, 288]``."""
    energy = (y.abs() ** 2).reshape(y.shape[0], cfg.num_symbols, cfg.num_prb, 12).mean(-1)
    return energy.transpose(1, 2) / no[:, None, None]


# --- DMRS residual ----------------------------------------------------------------------------


def dmrs_ls(y_alloc: torch.Tensor, tx: PUSCHTransmitter):
    """Raw LS channel estimates ``y / x`` at the DMRS pilots of the allocation.

    Returns ``h_ls`` ``[B, N_p]``, the pilot subcarrier indices ``[N_p]`` and the pilot power.
    (Sionna's PUSCH estimator averages neighbouring pilots; here every pilot stays separate.)
    """
    pp = tx.resource_grid.pilot_pattern
    sym, sc = torch.nonzero(pp.mask[0, 0], as_tuple=True)  # DMRS REs, row-major order
    pilots = pp.pilots[0, 0]
    used = pilots.abs() > 0  # type-1 DMRS: the other comb is left empty
    sym, sc, pilots = sym[used], sc[used], pilots[used]
    h_ls = y_alloc[:, 0, 0, sym, sc] / pilots
    pilot_power = (pilots.abs() ** 2).mean().item()
    return h_ls, sc, pilot_power


@cache
def _subspace(pilot_sc: tuple, scs: float, tau_max: float, energy: float) -> np.ndarray:
    # Frequency correlation of a channel whose delays are spread uniformly over [0, tau_max]
    df = (np.subtract.outer(pilot_sc, pilot_sc)) * scs
    corr = np.exp(-1j * np.pi * df * tau_max) * np.sinc(df * tau_max)
    eigval, eigvec = np.linalg.eigh(corr)
    eigval, eigvec = eigval[::-1], eigvec[:, ::-1]  # largest first
    num = int(np.searchsorted(np.cumsum(eigval) / eigval.sum(), energy)) + 1
    return eigvec[:, :num]


def delay_subspace(
    pilot_sc: torch.Tensor, scs: float, tau_max: float, energy: float = 1 - 1e-6
) -> torch.Tensor:
    """Orthonormal basis ``[N_p, L]`` of all channel responses with delays up to ``tau_max``."""
    basis = _subspace(tuple(pilot_sc.tolist()), scs, tau_max, energy)
    return torch.as_tensor(basis.copy(), dtype=torch.complex64, device=pilot_sc.device)


def residual_per_prb(
    h_ls: torch.Tensor, pilot_sc: torch.Tensor, no: torch.Tensor, pilot_power: float, cfg: SimConfig
) -> torch.Tensor:
    """DMRS residual ``[B, n]`` of each PRB of the allocation, from the LS pilot estimates."""
    u = delay_subspace(pilot_sc, cfg.scs, cfg.tau_max)

    # Remove the part of the pilot estimates that a real channel can explain
    residual = h_ls - (h_ls @ u.conj()) @ u.T

    # Noise alone would leave N0/|x_p|^2 * (1 - ||u_k||^2) on pilot k: less near the band edges,
    # where the subspace vectors are concentrated
    kept = 1 - (u.abs() ** 2).sum(dim=1)  # [N_p]
    noise_only = no[:, None] / pilot_power * kept  # [B, N_p]

    # Residual power per PRB (6 pilots) divided by what noise alone would leave there
    b = h_ls.shape[0]
    return (residual.abs() ** 2).reshape(b, -1, 6).sum(-1) / noise_only.reshape(b, -1, 6).sum(-1)


def residual_map(batch: SlotBatch, cfg: SimConfig) -> torch.Tensor:
    """DMRS residual per PRB, on the DMRS symbol of the scheduled PRBs; 0 elsewhere."""
    h_ls, pilot_sc, pilot_power = dmrs_ls(batch.y_alloc, batch.tx)
    r = residual_per_prb(h_ls, pilot_sc, batch.no, pilot_power, cfg)
    cells = r.new_zeros(r.shape[0], r.shape[1], cfg.num_symbols)
    cells[:, :, cfg.dmrs_symbol] = r
    return _to_carrier(cells, batch, cfg)


# --- EVM --------------------------------------------------------------------------------------


def evm_map(batch: SlotBatch, cfg: SimConfig) -> torch.Tensor:
    """Decision-directed EVM per cell, on the data symbols of the scheduled PRBs; 0 elsewhere."""
    x_hat, no_eff = batch.rx["x_hat"], batch.rx["no_eff"]

    # Nearest QPSK point (Sionna's QPSK constellation is (+-1 +-1j)/sqrt(2))
    decision = (
        torch.where(x_hat.real >= 0, 1.0, -1.0) + 1j * torch.where(x_hat.imag >= 0, 1.0, -1.0)
    ) / np.sqrt(2)
    error = (x_hat - decision).abs() ** 2

    # Put error and expected noise back on the resource grid, then sum the 12 subcarriers of each
    # cell (every RE is written once, so this is deterministic on GPU as well)
    sym, sc = data_positions(batch.tx)
    num_prb = batch.scenario.alloc_size
    sums = []
    for v in (error, no_eff):
        grid = v.new_zeros(v.shape[0], cfg.num_symbols, 12 * num_prb)
        grid[:, sym, sc] = v
        sums.append(grid.reshape(-1, cfg.num_symbols, num_prb, 12).sum(-1).transpose(1, 2))
    err_sum, no_sum = sums
    e = torch.where(no_sum > 0, err_sum / no_sum.clamp_min(1e-12), 0.0)
    return _to_carrier(e, batch, cfg)


# --- Masks and the stacked input --------------------------------------------------------------


def masks(batch: SlotBatch, cfg: SimConfig):
    """``alloc_mask``, ``dmrs_mask``, ``data_mask``, each ``[B, 24, 14]`` float (1 = defined)."""
    b = batch.scenario.batch_size
    alloc = torch.zeros(b, cfg.num_prb, cfg.num_symbols, device=batch.y.device)
    alloc[
        :, batch.scenario.alloc_start : batch.scenario.alloc_start + batch.scenario.alloc_size
    ] = 1
    dmrs = torch.zeros_like(alloc)
    dmrs[:, :, cfg.dmrs_symbol] = alloc[:, :, cfg.dmrs_symbol]
    return alloc, dmrs, alloc - dmrs


def _db(x: torch.Tensor) -> torch.Tensor:
    return (10 * torch.log10(x.clamp_min(1e-10))).clamp(*DB_RANGE)


def extract(batch: SlotBatch, cfg: SimConfig) -> torch.Tensor:
    """All features stacked as channels ``[B, 6, 24, 14]`` in the order of ``FEATURE_NAMES``."""
    alloc, dmrs, data = masks(batch, cfg)
    p = _db(power_map(batch.y, batch.no, cfg))
    r = _db(residual_map(batch, cfg)) * dmrs
    e = _db(evm_map(batch, cfg)) * data
    return torch.stack([p, r, e, alloc, dmrs, data], dim=1).float()
