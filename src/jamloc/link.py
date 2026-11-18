"""End-to-end uplink slot with a jammer, written like the model classes in the Sionna tutorials.

One call simulates a batch of slots::

    UE transmitter -> place on carrier -> UE channel ----+
                                                          +--> AWGN -> take allocation -> receiver
    jammer         ------------------> jammer channel ---+

The UE allocation is the same for all slots of a batch; the attack type, jammer PRBs, JSR and SNR
are drawn per slot.
"""

from dataclasses import dataclass

import numpy as np
import torch
from sionna.phy import config
from sionna.phy.channel import AWGN, OFDMChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.nr import PUSCHTransmitter

from jamloc.config import SimConfig
from jamloc.jammer import ATTACKS, Jammer, cell_mask, pulse_mask
from jamloc.noise import receiver_noise
from jamloc.receiver import PUSCHReceiverChain
from jamloc.ue import (
    UEPool,
    carrier_resource_grid,
    check_allocation,
    place_on_carrier,
    take_allocation,
)


@dataclass
class Scenario:
    """What happens in a batch of slots. Per-slot values are tensors of length ``B``."""

    channel: str  # "A" (TDL-A) or "C" (TDL-C)
    alloc_start: int  # first scheduled PRB
    alloc_size: int  # number of scheduled PRBs
    attack: torch.Tensor  # index into ATTACKS
    jam_start: torch.Tensor  # first jammed PRB
    jam_size: torch.Tensor  # number of jammed PRBs
    jsr_db: torch.Tensor  # jammer-to-signal ratio per jammed RE [dB]
    snr_db: torch.Tensor  # UE energy per data RE / noise variance [dB]
    # error of the receiver's noise power [dB] (jamloc.noise); None: the receiver knows it
    noise_error_db: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return len(self.attack)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Sionna/PyTorch and return a NumPy generator for drawing scenarios."""
    config.seed = seed
    return np.random.default_rng(seed)


def _draw(rng, value, size):
    """A fixed value, or a uniform draw from a ``(low, high)`` range."""
    if isinstance(value, (tuple, list)):
        return rng.uniform(value[0], value[1], size)
    return np.full(size, float(value))


def draw_scenario(
    rng: np.random.Generator,
    batch_size: int,
    cfg: SimConfig,
    channel: str,
    attacks=(0, 1, 2, 3),
    jsr_db=(-15.0, 10.0),
    snr_db=(5.0, 25.0),
    jam_size=(2, 12),
    scheduling_aware: bool = False,
    alloc=None,
    noise_error_db=0.0,
) -> Scenario:
    """Random scenario for one batch. Ranges are uniform; a single number fixes the value.

    ``alloc=(start, size)`` fixes the allocation. With ``scheduling_aware=True`` the jammer
    hits exactly the scheduled PRBs; otherwise it jams a random block anywhere on the carrier.
    ``noise_error_db`` is the error of the noise power the receiver assumes (0: exact).
    """
    if alloc is None:
        size = int(rng.integers(cfg.min_alloc_prb, cfg.num_prb + 1))
        start = int(rng.integers(0, cfg.num_prb - size + 1))
    else:
        start, size = alloc
        check_allocation(start, size, cfg)

    attack = rng.choice(np.asarray(attacks), batch_size)
    if scheduling_aware:
        j_size = np.full(batch_size, size)
        j_start = np.full(batch_size, start)
    else:
        if isinstance(jam_size, (tuple, list)):
            j_size = rng.integers(jam_size[0], jam_size[1] + 1, batch_size)
        else:
            j_size = np.full(batch_size, int(jam_size))
        j_start = rng.integers(0, cfg.num_prb - j_size + 1)

    return Scenario(
        channel=channel,
        alloc_start=start,
        alloc_size=size,
        attack=torch.as_tensor(attack, dtype=torch.long),
        jam_start=torch.as_tensor(j_start, dtype=torch.long),
        jam_size=torch.as_tensor(j_size, dtype=torch.long),
        jsr_db=torch.as_tensor(_draw(rng, jsr_db, batch_size), dtype=torch.float32),
        snr_db=torch.as_tensor(_draw(rng, snr_db, batch_size), dtype=torch.float32),
        # drawn last, so that the other draws stay the same as without it
        noise_error_db=torch.as_tensor(_draw(rng, noise_error_db, batch_size), dtype=torch.float32),
    )


@dataclass
class SlotBatch:
    """Everything one simulated batch produces."""

    y: torch.Tensor  # received carrier grid [B, 14, 288]
    no: torch.Tensor  # noise variance the receiver assumes [B] (the true one unless set otherwise)
    labels: torch.Tensor  # jammed cells [B, 24, 14] (bool)
    scenario: Scenario
    y_alloc: torch.Tensor  # received allocation grid [B, 1, 1, 14, 12 n]
    rx: dict  # receiver outputs (see PUSCHReceiverChain)
    b: torch.Tensor  # transmitted information bits
    tx: PUSCHTransmitter  # transmitter used for this batch
    no_true: torch.Tensor | None = None  # true noise variance [B]


class JammedPUSCHLink:
    """The whole link. A plain class (not a Sionna ``Block``) because it takes a ``Scenario``,
    which Sionna's automatic input-to-tensor conversion does not accept."""

    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg = cfg or SimConfig()
        self.device = config.device
        self.ue = UEPool(cfg)
        self._receivers = {}

        # Independent fading channels for the UE and for the jammer (same TDL model)
        rg = carrier_resource_grid(cfg)
        self.ue_channel, self.jammer_channel = {}, {}
        for model in ("A", "C"):
            for channels in (self.ue_channel, self.jammer_channel):
                tdl = TDL(
                    model,
                    cfg.delay_spread[model],
                    cfg.carrier_frequency,
                    min_speed=0.0,
                    max_speed=cfg.max_speed,
                )
                channels[model] = OFDMChannel(tdl, rg, normalize_channel=cfg.normalize_channel)

        self.jammer = Jammer(cfg)
        self.awgn = AWGN()

    def receiver(self, num_prb: int) -> PUSCHReceiverChain:
        if num_prb not in self._receivers:
            self._receivers[num_prb] = PUSCHReceiverChain(self.ue.transmitter(num_prb))
        return self._receivers[num_prb]

    def __call__(self, scenario: Scenario, decode: bool = False) -> SlotBatch:
        cfg, sc = self.cfg, scenario
        tx = self.ue.transmitter(sc.alloc_size)

        # UE: PUSCH slot on its allocation, placed on the carrier, through its channel
        x, b = tx(sc.batch_size)
        x = place_on_carrier(x, sc.alloc_start, cfg)
        y_ue = self.ue_channel[sc.channel](x)

        # Jammer: Gaussian noise on the attacked cells, through its own channel
        hop_start = None
        if cfg.jammer_hopping:  # a new block position for every OFDM symbol
            u = torch.rand(
                sc.batch_size,
                cfg.num_symbols,
                generator=self.jammer.torch_rng,
                device=self.device,
            )
            positions = (cfg.num_prb - sc.jam_size.to(self.device) + 1)[:, None]
            hop_start = (u * positions).long()
        labels = cell_mask(
            sc.attack.to(self.device),
            sc.jam_start.to(self.device),
            sc.jam_size.to(self.device),
            cfg,
            hop_start,
        )
        if cfg.jammer_duty < 1.0:  # pulsed jammer: only some of the OFDM symbols
            labels = pulse_mask(labels, cfg.jammer_duty, self.jammer.torch_rng)
        # (the pilot-spoofing waveform replays the UE's pilots, so the jammer gets the UE grid)
        z = self.jammer(labels, sc.jsr_db, x)
        y_jam = self.jammer_channel[sc.channel](z)

        # Receiver noise (UE data REs have unit energy, so no = 10^(-SNR/10))
        no = 10 ** (-sc.snr_db.to(self.device) / 10)
        y = self.awgn(y_ue + y_jam, no)

        # gNB: cut out the scheduled PRBs and run the receiver chain with the noise power it
        # assumes (the true one, one off by a set error, or an estimate: jamloc.noise)
        y_alloc = take_allocation(y, sc.alloc_start, sc.alloc_size)
        no_rx = receiver_noise(cfg, no, sc.noise_error_db, y_alloc, tx)
        rx = self.receiver(sc.alloc_size)(y_alloc, no_rx, decode=decode)

        return SlotBatch(
            y=y[:, 0, 0],
            no=no_rx,
            labels=labels,
            scenario=sc,
            y_alloc=y_alloc,
            rx=rx,
            b=b,
            tx=tx,
            no_true=no,
        )


__all__ = [
    "ATTACKS",
    "JammedPUSCHLink",
    "Scenario",
    "SlotBatch",
    "draw_scenario",
    "seed_everything",
]
