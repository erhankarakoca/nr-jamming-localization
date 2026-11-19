"""Simulation parameters (proposal Table 1) in one place."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimConfig:
    # Carrier and numerology
    num_prb: int = 24  # PRBs on the carrier
    num_symbols: int = 14  # OFDM symbols per slot
    scs: float = 30e3  # subcarrier spacing [Hz]

    # PUSCH
    mcs_index: int = 5  # QPSK, code rate ~0.37 (MCS table 1)
    min_alloc_prb: int = 6  # smallest allocation
    dmrs_symbol: int = 2  # one front-loaded DMRS symbol (mapping type A)

    # Channel: 3GPP TDL-A / TDL-C, low mobility
    carrier_frequency: float = 3.5e9  # [Hz]
    delay_spread: dict = field(default_factory=lambda: {"A": 100e-9, "C": 300e-9})  # [s]
    max_speed: float = 3.0  # [m/s]
    # True: Sionna scales every channel realization to unit energy over the carrier grid, so each
    # slot has exactly the nominal SNR and JSR on average over the carrier (all nominal
    # experiments). False: the TDL model's own normalization, unit power only on average over
    # realizations, so the received SNR and JSR also vary from slot to slot.
    normalize_channel: bool = True

    # Noise power the receiver assumes (jamloc.noise). "true": the simulator's noise variance,
    # off by the per-slot error ``noise_error_db`` of the scenario (0 unless an experiment sets
    # it). "comb": estimated in every slot from the empty comb of the DMRS symbol.
    noise_estimate: str = "true"

    # Largest channel delay assumed by the DMRS residual feature
    tau_max: float = 3e-6  # [s]

    # Jamming waveform per resource element (see jamloc.jammer):
    # "gaussian" - complex Gaussian noise, the threat model of the paper;
    # "qpsk"     - random constant-envelope QPSK symbols;
    # "tone"     - a continuous tone on each jammed subcarrier (random phase, the same on every
    #              OFDM symbol), usually combined with jammer_tones below;
    # "dmrs"     - pilot spoofing: on the DMRS symbol the jammer replays the UE's pilot sequence
    #              (scaled, one random phase per slot), elsewhere it sends Gaussian noise;
    # "ue"       - synchronized neighbor-cell PUSCH (partial-band): unit-energy QPSK on data
    #              symbols and double-energy QPSK on the even DMRS comb, with the odd comb empty.
    # All but "gaussian" are only used to test generalization.
    jammer_waveform: str = "gaussian"

    # Smarter jammers, only used to test generalization (both off in the threat model):
    # on the DMRS symbol, jam only the comb of subcarriers that carries the pilots (type-1 DMRS,
    # port 0: the even subcarriers), not the empty comb next to it
    jammer_pilot_comb: bool = False
    # draw a new block of PRBs (same size) for every OFDM symbol: hopping from symbol to symbol
    jammer_hopping: bool = False
    # pulsed (partial-time) jammer: each OFDM symbol of the attack pattern is jammed with this
    # probability, drawn per slot (at least one symbol stays jammed); 1.0 = always on
    jammer_duty: float = 1.0
    # sparse (multi-tone) jammer: jam only this many subcarriers of every jammed PRB, at random
    # positions drawn per slot and PRB and kept for the whole slot; 0 = all 12 subcarriers
    jammer_tones: int = 0

    @property
    def num_subcarriers(self) -> int:
        return 12 * self.num_prb
