"""Resource-selective Gaussian jammer.

The jammer knows the slot timing and the DMRS position but not the allocation. It puts complex
Gaussian noise on a block of PRBs, on the OFDM symbols chosen by the attack pattern:

* ``none``    - no jamming (benign slot)
* ``pilot``   - only the DMRS symbol
* ``data``    - every symbol except the DMRS symbol
* ``partial`` - all 14 symbols (partial-band jamming)

A *cell* is one PRB x one OFDM symbol. The jammed cells are also the labels the detector learns.
"""

import torch
from sionna.phy import Block
from sionna.phy.utils import complex_normal

from jamloc.config import SimConfig

ATTACKS = ("none", "pilot", "data", "partial")


def cell_mask(
    attack: torch.Tensor, jam_start: torch.Tensor, jam_size: torch.Tensor, cfg: SimConfig
) -> torch.Tensor:
    """Jammed cells ``[B, num_prb, num_symbols]`` for each slot of the batch.

    ``attack`` holds indices into ``ATTACKS``; the jammed PRBs are
    ``jam_start .. jam_start + jam_size - 1``.
    """
    device = attack.device
    prb = torch.arange(cfg.num_prb, device=device)
    sym = torch.arange(cfg.num_symbols, device=device)

    # Which PRBs are jammed: [B, num_prb]
    in_block = (prb >= jam_start[:, None]) & (prb < (jam_start + jam_size)[:, None])

    # Which OFDM symbols are jammed for each attack: [num_attacks, num_symbols]
    is_dmrs = sym == cfg.dmrs_symbol
    symbols_per_attack = torch.stack(
        [torch.zeros_like(is_dmrs), is_dmrs, ~is_dmrs, torch.ones_like(is_dmrs)]
    )
    jammed_symbols = symbols_per_attack[attack]  # [B, num_symbols]

    return in_block[:, :, None] & jammed_symbols[:, None, :]


class Jammer(Block):
    """Gaussian jamming signal on the resource grid of the carrier.

    Input: jammed cells ``[B, num_prb, num_symbols]`` (bool) and the jammer-to-signal ratio per
    jammed resource element ``jsr_db`` ``[B]``. The UE sends unit energy per data resource
    element, so the jammer energy per jammed resource element is ``10**(jsr_db/10)``.

    Output: jamming grid ``[B, 1, 1, num_symbols, 12 * num_prb]`` (before the jammer's channel).
    """

    def __init__(self, cfg: SimConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def call(self, cells: torch.Tensor, jsr_db: torch.Tensor) -> torch.Tensor:
        cells, jsr_db = cells.to(self.device), jsr_db.to(self.device)

        # Cells -> resource elements: every subcarrier of a jammed PRB is jammed
        re_mask = cells.repeat_interleave(12, dim=1)  # [B, 288, 14]
        re_mask = re_mask.transpose(1, 2)[:, None, None]  # [B, 1, 1, 14, 288]

        # Unit-variance complex Gaussian noise, scaled to the jammer energy per RE
        z = complex_normal(
            re_mask.shape, precision=self.precision, device=self.device, generator=self.torch_rng
        )
        amplitude = torch.sqrt(10 ** (jsr_db.to(self.dtype) / 10))
        return z * amplitude[:, None, None, None, None] * re_mask
