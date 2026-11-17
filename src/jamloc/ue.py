"""The scheduled UE: PUSCH transmitters and placement of the allocation on the carrier.

Sionna's PUSCH always fills its whole bandwidth part, so a transmitter for ``n`` PRBs produces a
grid that is ``12 n`` subcarriers wide. To schedule the UE on PRBs ``start .. start+n-1`` of the
24-PRB carrier, we place that grid into the carrier grid at the right offset.
"""

import torch
import torch.nn.functional as F
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter
from sionna.phy.ofdm import ResourceGrid

from jamloc.config import SimConfig


def make_pusch_config(cfg: SimConfig, num_prb: int) -> PUSCHConfig:
    """PUSCH configuration for an allocation of ``num_prb`` PRBs."""
    pc = PUSCHConfig()
    pc.carrier.subcarrier_spacing = cfg.scs / 1e3  # Sionna expects kHz
    pc.carrier.n_size_grid = num_prb
    pc.n_size_bwp = num_prb
    pc.dmrs.config_type = 1  # comb-2
    pc.dmrs.length = 1  # single-symbol DMRS
    pc.dmrs.additional_position = 0  # front-loaded only
    pc.tb.mcs_index = cfg.mcs_index
    return pc


class UEPool:
    """One PUSCH transmitter per allocation size, created on first use."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._transmitters = {}

    def transmitter(self, num_prb: int) -> PUSCHTransmitter:
        if num_prb not in self._transmitters:
            pc = make_pusch_config(self.cfg, num_prb)
            self._transmitters[num_prb] = PUSCHTransmitter(pc, output_domain="freq")
        return self._transmitters[num_prb]


def check_allocation(start_prb: int, num_prb: int, cfg: SimConfig) -> None:
    """Raise ``ValueError`` unless PRBs ``start_prb .. start_prb + num_prb - 1`` fit on the carrier."""
    if num_prb < 1 or start_prb < 0 or start_prb + num_prb > cfg.num_prb:
        raise ValueError(
            f"allocation of {num_prb} PRBs starting at PRB {start_prb} does not fit on the "
            f"{cfg.num_prb}-PRB carrier"
        )


def place_on_carrier(x: torch.Tensor, start_prb: int, cfg: SimConfig) -> torch.Tensor:
    """Put an allocation grid ``[..., 12 n]`` on PRBs ``start_prb ..`` of the carrier.

    Returns ``[..., 288]`` with zeros on the subcarriers outside the allocation.
    """
    check_allocation(start_prb, x.shape[-1] // 12, cfg)
    left = 12 * start_prb
    right = cfg.num_subcarriers - left - x.shape[-1]
    return F.pad(x, (left, right))


def take_allocation(y: torch.Tensor, start_prb: int, num_prb: int) -> torch.Tensor:
    """Cut the allocated subcarriers ``[..., 12 n]`` out of a carrier grid ``[..., 288]``."""
    return y[..., 12 * start_prb : 12 * (start_prb + num_prb)]


def carrier_resource_grid(cfg: SimConfig) -> ResourceGrid:
    """Resource grid of the whole carrier, used to generate the channels."""
    return ResourceGrid(
        num_ofdm_symbols=cfg.num_symbols,
        fft_size=cfg.num_subcarriers,
        subcarrier_spacing=cfg.scs,
    )
