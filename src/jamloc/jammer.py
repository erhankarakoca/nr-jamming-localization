"""Resource-selective Gaussian jammer.

The jammer knows the slot timing and the DMRS position but not the allocation. It puts complex
Gaussian noise on a block of PRBs, on the OFDM symbols chosen by the attack pattern (a QPSK
waveform can be chosen in ``SimConfig`` to test generalization):

* ``none``    - no jamming (benign slot)
* ``pilot``   - only the DMRS symbol
* ``data``    - every symbol except the DMRS symbol
* ``partial`` - all 14 symbols (partial-band jamming)

A *cell* is one PRB x one OFDM symbol. The jammed cells are also the labels the detector learns.

Other jammers test generalization (all set in ``SimConfig``, all off in the threat model). Each
changes one thing: the waveform, the time-frequency support, or what the attacker knows.

* ``jammer_pilot_comb`` jams only the pilot subcarriers of the DMRS symbol (half its resource
  elements; the empty comb stays clean, so the same damage to the channel estimate costs half the
  energy).
* ``jammer_hopping`` moves the jammed PRB block to a new random position on every OFDM symbol.
* ``jammer_duty`` < 1 gives a pulsed (partial-time) jammer: each OFDM symbol of the attack pattern
  is jammed only with that probability (``pulse_mask``).
* ``jammer_tones`` > 0 jams only that many subcarriers of every jammed PRB (a sparse, multi-tone
  jammer); with ``jammer_waveform="tone"`` each of them carries a continuous tone.
* ``jammer_waveform="dmrs"`` is pilot spoofing. The DMRS sequence follows from parameters the
  network broadcasts, so a jammer can compute it and send it (scaled, with one random phase per
  slot) on the pilot subcarriers. The gNB's least-squares estimate at pilot p then becomes
  ``h_p + a g_p`` (``a``: the jammer's scale and phase, ``g``: its channel). That is again a
  physically possible channel, so it lies in the delay subspace and the DMRS residual cannot see
  it by construction; only power and EVM can. The jammer does not know the pilot values outside
  the UE's allocation, so it sends random QPSK symbols of the same energy there.
* ``jammer_waveform="ue"`` models a legitimate, synchronized neighbor-cell PUSCH on a
  partial-band block (all 14 symbols). It sends unit-energy random QPSK on the data symbols and
  independent QPSK of energy 2 on the even subcarriers of the DMRS symbol, as for a different
  scrambling identity. The other comb stays empty. Mean energy per cell is the same on DMRS and
  data, and JSR controls the interference-to-signal ratio.

Labels never change: a cell is jammed when the jammer transmits on any of its resource elements.
"""

import torch
from sionna.phy import Block
from sionna.phy.utils import complex_normal

from jamloc.config import SimConfig

ATTACKS = ("none", "pilot", "data", "partial")


def cell_mask(
    attack: torch.Tensor,
    jam_start: torch.Tensor,
    jam_size: torch.Tensor,
    cfg: SimConfig,
    hop_start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jammed cells ``[B, num_prb, num_symbols]`` for each slot of the batch.

    ``attack`` holds indices into ``ATTACKS``; the jammed PRBs are
    ``jam_start .. jam_start + jam_size - 1``. A hopping jammer gives one start per OFDM symbol in
    ``hop_start`` ``[B, num_symbols]``, which then replaces ``jam_start``.
    """
    device = attack.device
    prb = torch.arange(cfg.num_prb, device=device)
    sym = torch.arange(cfg.num_symbols, device=device)

    # Which PRBs are jammed on each symbol: [B, num_prb, 1] (or [B, num_prb, num_symbols])
    if hop_start is None:
        start = jam_start[:, None, None]
    else:
        start = hop_start[:, None, :]
    stop = start + jam_size[:, None, None]
    in_block = (prb[None, :, None] >= start) & (prb[None, :, None] < stop)

    # Which OFDM symbols are jammed for each attack: [num_attacks, num_symbols]
    is_dmrs = sym == cfg.dmrs_symbol
    symbols_per_attack = torch.stack(
        [torch.zeros_like(is_dmrs), is_dmrs, ~is_dmrs, torch.ones_like(is_dmrs)]
    )
    jammed_symbols = symbols_per_attack[attack]  # [B, num_symbols]

    return in_block & jammed_symbols[:, None, :]


def pulse_mask(
    cells: torch.Tensor, duty: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Pulsed jammer: keep each jammed OFDM symbol of a slot only with probability ``duty``.

    ``cells`` ``[B, num_prb, num_symbols]`` are the cells of the continuous attack. Every symbol
    the attack uses is switched on independently; if a slot would end up with no symbol at all, one
    of its symbols is switched back on, so a jammed slot stays jammed (a pilot attack, which uses
    a single symbol, is therefore unchanged).
    """
    used = cells.any(dim=1)  # [B, num_symbols]: symbols the continuous attack would jam
    draw = torch.rand(used.shape, generator=generator, device=cells.device)
    on = used & (draw < duty)
    # Slots left without any jammed symbol: switch on the used symbol with the smallest draw
    empty = used.any(dim=1) & ~on.any(dim=1)
    if empty.any():
        first = torch.where(used, draw, torch.full_like(draw, 2.0)).argmin(dim=1)
        on[empty, first[empty]] = True
    return cells & on[:, None, :]


class Jammer(Block):
    """Jamming signal on the resource grid of the carrier.

    Input: jammed cells ``[B, num_prb, num_symbols]`` (bool), the jammer-to-signal ratio per
    jammed resource element ``jsr_db`` ``[B]``, and (only for the ``"dmrs"`` waveform) the UE's
    transmitted carrier grid ``x`` ``[B, 1, 1, num_symbols, 12 * num_prb]``, whose pilots the
    jammer replays. The UE sends unit energy per data resource element, so the jammer energy per
    jammed resource element is ``10**(jsr_db/10)`` on average for the Gaussian waveform. For
    the neighbor UE the same scale sets the data energy, and its pilots carry twice that energy.

    Output: jamming grid ``[B, 1, 1, num_symbols, 12 * num_prb]`` (before the jammer's channel).
    """

    def __init__(self, cfg: SimConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def call(
        self, cells: torch.Tensor, jsr_db: torch.Tensor, x: torch.Tensor | None = None
    ) -> torch.Tensor:
        cfg = self.cfg
        cells, jsr_db = cells.to(self.device), jsr_db.to(self.device)
        batch = cells.shape[0]
        num_sc = 12 * cfg.num_prb

        # Cells -> resource elements: every subcarrier of a jammed PRB is jammed
        re_mask = cells.repeat_interleave(12, dim=1)  # [B, 288, 14]
        re_mask = re_mask.transpose(1, 2)[:, None, None]  # [B, 1, 1, 14, 288]
        if cfg.jammer_pilot_comb or cfg.jammer_waveform in ("dmrs", "ue"):
            # Leave the empty DMRS comb (odd subcarriers) alone: only the pilots are hit
            keep = torch.ones(re_mask.shape[-2:], dtype=torch.bool, device=self.device)
            keep[cfg.dmrs_symbol, 1::2] = False
            re_mask = re_mask & keep
        if cfg.jammer_tones:
            # Sparse jammer: in every PRB keep only `jammer_tones` random subcarriers, the same
            # ones on all OFDM symbols of the slot (ranks of random numbers give a random subset)
            draw = torch.rand(batch, cfg.num_prb, 12, generator=self.torch_rng, device=self.device)
            tones = draw.argsort(dim=-1).argsort(dim=-1) < cfg.jammer_tones  # [B, num_prb, 12]
            re_mask = re_mask & tones.reshape(batch, 1, 1, 1, num_sc)

        # Draw the waveform before applying the JSR scale and the resource mask.
        z = complex_normal(
            re_mask.shape, precision=self.precision, device=self.device, generator=self.torch_rng
        )
        if cfg.jammer_waveform in ("qpsk", "ue"):
            one = torch.ones_like(z.real)
            z = torch.complex(torch.copysign(one, z.real), torch.copysign(one, z.imag)) / 2**0.5
            if cfg.jammer_waveform == "ue":
                # Type-1 DMRS puts the cell's energy on half of its subcarriers.
                z[..., cfg.dmrs_symbol, :] *= 2**0.5
        elif cfg.jammer_waveform == "tone":
            # A continuous tone per subcarrier: random phase, constant over the OFDM symbols
            phase = torch.rand(batch, num_sc, generator=self.torch_rng, device=self.device)
            tone = torch.exp(2j * torch.pi * phase).to(z.dtype)
            z = tone[:, None, None, None, :].expand_as(z)
        elif cfg.jammer_waveform == "dmrs":
            z = z.clone()
            z[..., cfg.dmrs_symbol, :] = self._spoofed_pilots(x, batch, num_sc).to(z.dtype)
        elif cfg.jammer_waveform != "gaussian":
            raise ValueError(f"unknown jammer waveform {cfg.jammer_waveform!r}")
        amplitude = torch.sqrt(10 ** (jsr_db.to(self.dtype) / 10))
        return z * amplitude[:, None, None, None, None] * re_mask

    def _spoofed_pilots(self, x: torch.Tensor | None, batch: int, num_sc: int) -> torch.Tensor:
        """Unit-energy pilot replay on the DMRS symbol ``[B, 1, 1, num_sc]``.

        Inside the UE's allocation the jammer sends the UE's own pilots, normalized to unit energy
        and rotated by one random phase per slot; elsewhere random QPSK symbols. Received through
        the jammer's channel ``g``, this adds ``a g_k`` to the gNB's LS estimate at every jammed
        pilot: a smooth, channel-like term that the delay-subspace projection removes.
        """
        if x is None:
            raise ValueError('jammer_waveform="dmrs" needs the UE grid x to replay its pilots')
        pilots = x[..., self.cfg.dmrs_symbol, :].to(self.device)  # [B, 1, 1, num_sc]
        energy = (pilots.abs() ** 2).sum(-1) / (pilots.abs() > 0).sum(-1).clamp_min(1)
        replay = pilots / torch.sqrt(energy.clamp_min(1e-12))[..., None]
        bits = torch.rand(2, batch, 1, 1, num_sc, generator=self.torch_rng, device=self.device)
        qpsk = torch.where(bits[0] < 0.5, 1.0, -1.0) + 1j * torch.where(bits[1] < 0.5, 1.0, -1.0)
        replay = torch.where(pilots.abs() > 0, replay, qpsk / 2**0.5)
        phase = torch.rand(batch, 1, 1, 1, generator=self.torch_rng, device=self.device)
        return replay * torch.exp(2j * torch.pi * phase)
