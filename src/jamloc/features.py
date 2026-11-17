"""Per-cell jamming features from quantities the gNB receiver already has.

A cell is one PRB x one OFDM symbol, so every feature is a 24 x 14 map per slot. All three are
normalized by the noise so that a slot without jamming gives values around 1 (0 dB):

* **Power P** - received energy in the cell / noise variance. Defined everywhere on the carrier.
* **DMRS residual R** - the LS channel estimates at the pilots, minus their projection onto the
  set of channels that are physically possible (delays up to ``tau_max``). The user's own signal
  lies in that set, so what remains is interference plus noise. Defined on the DMRS symbol of the
  scheduled PRBs.
* **EVM E** - distance from each equalized symbol to the nearest QPSK point, divided by the noise
  variance the LMMSE equalizer expects. Defined on the data symbols of the scheduled PRBs.
"""

from functools import lru_cache

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


@lru_cache(maxsize=None)
def _subspace(pilot_sc: tuple, scs: float, tau_max: float, energy: float) -> np.ndarray:
    # Frequency correlation of a channel whose delays are spread uniformly over [0, tau_max]
    df = (np.subtract.outer(pilot_sc, pilot_sc)) * scs
    corr = np.exp(-1j * np.pi * df * tau_max) * np.sinc(df * tau_max)
    eigval, eigvec = np.linalg.eigh(corr)
    eigval, eigvec = eigval[::-1], eigvec[:, ::-1]  # largest first
    num = int(np.searchsorted(np.cumsum(eigval) / eigval.sum(), energy)) + 1
    return eigvec[:, :num]


def delay_subspace(pilot_sc: torch.Tensor, scs: float, tau_max: float,
                   energy: float = 1 - 1e-6) -> torch.Tensor:
    """Orthonormal basis ``[N_p, L]`` of all channel responses with delays up to ``tau_max``."""
    basis = _subspace(tuple(pilot_sc.tolist()), scs, tau_max, energy)
    return torch.as_tensor(basis.copy(), dtype=torch.complex64, device=pilot_sc.device)


def residual_map(batch: SlotBatch, cfg: SimConfig) -> torch.Tensor:
    """DMRS residual per PRB, on the DMRS symbol of the scheduled PRBs; 0 elsewhere."""
    h_ls, pilot_sc, pilot_power = dmrs_ls(batch.y_alloc, batch.tx)
    u = delay_subspace(pilot_sc, cfg.scs, cfg.tau_max)
    n_p, n_basis = u.shape

    # Remove the part of the pilot estimates that a real channel can explain
    residual = h_ls - (h_ls @ u.conj()) @ u.T

    # Mean residual power per PRB, divided by what noise alone would leave
    per_prb = (residual.abs() ** 2).reshape(h_ls.shape[0], -1, 6).mean(-1)  # 6 pilots per PRB
    noise_only = batch.no[:, None] / pilot_power * (n_p - n_basis) / n_p
    r = per_prb / noise_only

    cells = r.new_zeros(r.shape[0], r.shape[1], cfg.num_symbols)
    cells[:, :, cfg.dmrs_symbol] = r
    return _to_carrier(cells, batch, cfg)


# --- EVM --------------------------------------------------------------------------------------


def evm_map(batch: SlotBatch, cfg: SimConfig) -> torch.Tensor:
    """Decision-directed EVM per cell, on the data symbols of the scheduled PRBs; 0 elsewhere."""
    x_hat, no_eff = batch.rx["x_hat"], batch.rx["no_eff"]

    # Nearest QPSK point (Sionna's QPSK constellation is (+-1 +-1j)/sqrt(2))
    decision = (torch.where(x_hat.real >= 0, 1.0, -1.0)
                + 1j * torch.where(x_hat.imag >= 0, 1.0, -1.0)) / np.sqrt(2)
    error = (x_hat - decision).abs() ** 2

    # Sum error and expected noise over the resource elements of each cell
    sym, sc = data_positions(batch.tx)
    num_prb = batch.scenario.alloc_size
    cell = ((sc // 12) * cfg.num_symbols + sym).to(x_hat.device)
    sums = []
    for v in (error, no_eff):
        s = v.new_zeros(v.shape[0], num_prb * cfg.num_symbols)
        sums.append(s.index_add_(1, cell, v).reshape(-1, num_prb, cfg.num_symbols))
    err_sum, no_sum = sums
    e = torch.where(no_sum > 0, err_sum / no_sum.clamp_min(1e-12), 0.0)
    return _to_carrier(e, batch, cfg)


# --- Masks and the stacked input --------------------------------------------------------------


def masks(batch: SlotBatch, cfg: SimConfig):
    """``alloc_mask``, ``dmrs_mask``, ``data_mask``, each ``[B, 24, 14]`` float (1 = defined)."""
    b = batch.scenario.batch_size
    alloc = torch.zeros(b, cfg.num_prb, cfg.num_symbols, device=batch.y.device)
    alloc[:, batch.scenario.alloc_start : batch.scenario.alloc_start + batch.scenario.alloc_size] = 1
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
