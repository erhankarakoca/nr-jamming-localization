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

    # Largest channel delay assumed by the DMRS residual feature
    tau_max: float = 3e-6  # [s]

    # Jamming waveform per resource element: "gaussian" (the threat model) or "qpsk" (random
    # constant-envelope symbols, only used to test generalization)
    jammer_waveform: str = "gaussian"

    @property
    def num_subcarriers(self) -> int:
        return 12 * self.num_prb
