"""The showcase examples. PRB and symbol indices start at zero."""

from dataclasses import dataclass, replace

from jamloc.config import SimConfig
from jamloc.experiments.recovery import ALLOC, JAMMERS, NEST_DETECTORS, TRAINED_ON

NOMINAL = ("fused", "no_residual", "power", "block_test", "evm_threshold", "residual_threshold")
DETECTORS = (*NOMINAL, "power_threshold", *TRAINED_ON, "fused_atk2")
AIDED_RECEIVER = "fused_soft_da"


@dataclass(frozen=True)
class Case:
    """One reproducible batch and the detector maps passed to the protected receiver."""

    name: str
    title: str
    attack: str
    jsr_db: float | None
    channel: str = "C"
    snr_db: float = 15.0
    jammer: str = "nominal"
    noise_estimate: str = "true"
    seed: int = 7
    detector_seed: int = 0
    alloc: tuple[int, int] = ALLOC
    jam: tuple[int, int] = (10, 4)
    decode_maps: tuple[str, ...] = ("fused",)
    selection_rule: str = "median_hit_rate"
    batch_size: int = 64
    include_channel: bool = False
    iterations: int = 0

    @property
    def detectors(self):
        """Layers available with this noise estimate and these training sets."""
        if self.noise_estimate == "comb":
            return tuple(NEST_DETECTORS)
        return tuple(dict.fromkeys((*NOMINAL, *self.decode_maps)))

    def config(self):
        options = {"jammer_waveform": "dmrs"} if self.jammer == "spoof" else JAMMERS[self.jammer]
        return SimConfig(noise_estimate=self.noise_estimate, **options)


_PILOT = Case("pilot", "Pilot jamming, +3 dB", "pilot", 3, include_channel=True)
_PARTIAL = Case("partial", "Partial-band jamming, +3 dB", "partial", 3)
CASES = {
    case.name: case
    for case in (
        Case("benign", "Benign reference", "none", None),
        _PILOT,
        Case("pilot_weak", "Weak pilot jamming, −12 dB", "pilot", -12),
        Case("data", "Data jamming, +3 dB", "data", 3),
        _PARTIAL,
        Case(
            "hopping",
            "Hopping jammer, +9 dB",
            "partial",
            9,
            jammer="hopping",
            decode_maps=("fused", "fused_atk"),
        ),
        Case("pulsed", "Pulsed jammer, +9 dB", "partial", 9, jammer="pulsed"),
        Case(
            "spoof",
            "Pilot spoofing, −9 dB",
            "pilot",
            -9,
            jam=ALLOC,
            jammer="spoof",
            decode_maps=("fused", "fused_atk2"),
        ),
        Case(
            "pilot_whole_comb",
            "Whole-allocation pilot jamming with estimated noise, +3 dB",
            "pilot",
            3,
            jam=ALLOC,
            noise_estimate="comb",
            selection_rule="first_true_soft_failure",
        ),
        Case("jsr_sweep", "Pilot jamming across JSR", "pilot", 0),
        # Seed 0 is the first CPU realization passing the paper's no-deep-fade test.
        Case(
            "fig3",
            "Fig. 3 · Paired benign and jammed pilots",
            "pilot",
            0,
            snr_db=20,
            seed=0,
            batch_size=1,
            selection_rule="paper_realization",
        ),
        # The overview's walkthrough (and the paper's Fig. 2): the jammer hits pilots and data, so
        # both uses of the map show; the same batch as "partial", first slot the map recovers
        replace(
            _PARTIAL,
            name="partial_flip",
            title="Partial-band jamming with a recovered CRC",
            selection_rule="first_crc_flip",
            include_channel=True,
        ),
        Case(
            "aided_flip",
            "Whole-allocation pilot jamming recovered with data-aided passes",
            "pilot",
            9,
            jam=ALLOC,
            selection_rule="first_aided_crc_flip",
            include_channel=True,
            iterations=3,
        ),
    )
}
