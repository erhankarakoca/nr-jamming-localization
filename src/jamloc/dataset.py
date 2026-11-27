"""Labelled datasets of feature maps, generated from the link with fixed seeds.

Splits (batch ``i`` of a split uses seed ``base + i``; the seed ranges of the splits are disjoint,
so channel realizations and jammer draws are independent between splits. Allocations are drawn
independently too, but only 190 exist on a 24-PRB carrier, so they do repeat):

* ``train`` / ``val`` - random attack (incl. benign), JSR in [-15, 10] dB, SNR in [5, 25] dB,
  TDL-A and TDL-C alternating per batch, jammer anywhere on the carrier.
* ``test`` - grid over channel x attack x JSR (-15 ... 9 dB in 3 dB steps), plus benign slots.
* ``sched`` - same grid, but the jammer hits exactly the scheduled PRBs (never seen in training).
* Generalization (same grid, conditions never seen in training, see ``GENERALIZATION``):
  ``longdelay`` - TDL-C with 1000 ns delay spread, so channel delays reach 8.7 us, far beyond the
  ``tau_max`` = 3 us the DMRS residual assumes; ``fast`` - UE and jammer moving at up to 30 m/s
  instead of 3 m/s; ``qpskjam`` - the jammer sends random QPSK symbols instead of Gaussian noise;
  ``combjam`` - on the DMRS symbol only the pilot subcarriers are jammed; ``hopjam`` - the jammed
  PRB block moves to a new position on every OFDM symbol; ``spoofjam`` - pilot spoofing, the
  jammer replays the UE's DMRS on the scheduled PRBs (pilot attack only); ``pulsejam`` - each OFDM
  symbol of a data or partial-band attack is jammed with probability 0.5; ``tonejam`` - one
  continuous tone per jammed PRB instead of all 12 subcarriers; ``neighbor`` - interference
  from a synchronized neighbor-cell UE on a partial-band block.
* ``train_mix`` / ``val_mix`` - like ``train`` / ``val``, but the batches cycle through the nominal
  channels, long delay spreads and high speeds (``MIXED_CONDITIONS``).
* ``train_atk`` / ``val_atk`` - like ``train`` / ``val``, but the batches cycle through the
  nominal, hopping and pilot-comb jammers (``ATTACK_CONDITIONS``).
* ``train_atk2`` / ``val_atk2`` - like ``train_atk``, plus batches with pilot spoofing over the
  scheduled PRBs (``ATTACK_CONDITIONS_SPOOF``).
* ``train_gen`` / ``val_gen`` - every combination of those jammers, those channels and two
  modulations, QPSK and 16-QAM (``GENERAL_CONDITIONS``, 18 conditions).
* ``train_all`` / ``val_all`` - the same with 64-QAM as a third modulation, each with one and with
  two DMRS symbols (``ALL_CONDITIONS``, 54 conditions); ``test_qam64`` / ``hopjam_qam64`` and
  ``<grid>_dmrs2`` (``DMRS2_GRIDS``) are the matching test grids.
* ``margin`` - the jammer hits exactly the scheduled PRBs, as in the BLER experiment, at a fixed
  SNR of 15 dB and down to JSR -30 dB (``experiments/detection_margin``); ``margin_snr<x>`` the
  same at another SNR.
* Noise power (``experiments/calibration``): ``noise_m1``, ``noise_p0.5`` ... - test grids
  where the receiver's noise power is off by a fixed error (minus or plus, in dB) in the whole
  receiver chain; ``train_nest`` / ``val_nest`` / ``test_nest`` / ``sched_nest`` - the receiver
  estimates the noise power in every slot from the empty DMRS comb (``jamloc.noise``).
* ``train_dmrs2`` / ``val_dmrs2`` / ``test_dmrs2`` / ``sched_dmrs2`` - nominal splits with two
  DMRS symbols per slot.

Grid splits use batches of 16 slots (8 for ``margin``), and the allocation is drawn once per
batch, so a grid point of 500 slots mixes about 31 allocations. Each split is one ``.npz`` file
in ``data_dir()``: ``features`` ``[N, 7, 24, 14]`` (stored as float16), ``labels``
``[N, 24, 14]`` and one value per slot for every key in ``META_KEYS``: among them ``batch``, the
index of the batch the slot came from (the unit that shares an allocation), ``snr_db_est``, the
SNR the gNB estimates from the slot, ``noise_error_db``, the error of the noise power the
receiver used, and ``mcs_index``, the UE's modulation and coding scheme.

    python -m jamloc.dataset --split train
"""

import argparse
import dataclasses
import math
import time

import numpy as np
import torch

from jamloc.config import SimConfig
from jamloc.features import extract
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.noise import estimated_snr_db
from jamloc.paths import data_dir

CHANNELS = ("A", "C")
TEST_JSR_DB = tuple(float(j) for j in range(-15, 10, 3))
META_KEYS = (
    "attack",
    "jsr_db",
    "snr_db",
    "channel",
    "alloc_start",
    "alloc_size",
    "jam_start",
    "jam_size",
    "scheduling_aware",
    "batch",
    "snr_db_est",
    "noise_error_db",
    "mcs_index",
)
GRID_BATCH_SIZE = 16  # small batches: every grid point mixes many allocations

SPLITS = {
    "train": {"seed": 1_000_000, "num_slots": 40_000},
    "val": {"seed": 2_000_000, "num_slots": 5_000},
    "test": {"seed": 3_000_000, "per_point": 500, "benign_per_channel": 2_000, "sched": False},
    "sched": {"seed": 4_000_000, "per_point": 300, "benign_per_channel": 0, "sched": True},
}
CORE_SPLITS = tuple(SPLITS)

# Test conditions the detectors were not trained on: changes to SimConfig and the channel models
GENERALIZATION = {
    "longdelay": {
        "seed": 5_000_000,
        "channels": ("C",),
        "cfg": {"delay_spread": {"A": 100e-9, "C": 1000e-9}},
        "title": "TDL-C with 1000 ns delay spread (unseen)",
    },
    "fast": {
        "seed": 6_000_000,
        "cfg": {"max_speed": 30.0},
        "title": "speeds up to 30 m/s (unseen)",
    },
    "qpskjam": {
        "seed": 7_000_000,
        "cfg": {"jammer_waveform": "qpsk"},
        "title": "QPSK jamming waveform (unseen)",
    },
    "combjam": {
        "seed": 12_000_000,
        "cfg": {"jammer_pilot_comb": True},
        "title": "jammer only on the pilot subcarriers of the DMRS symbol (unseen)",
    },
    "hopjam": {
        "seed": 13_000_000,
        "cfg": {"jammer_hopping": True},
        "title": "jammer hops to a new PRB block every OFDM symbol (unseen)",
    },
    # Pilot spoofing needs the allocation (the jammer replays the UE's pilots on its PRBs), so it
    # is scheduling-aware, and only the pilot attack is of interest
    "spoofjam": {
        "seed": 16_000_000,
        "cfg": {"jammer_waveform": "dmrs"},
        "sched": True,
        "attacks": (1,),
        "title": "pilot spoofing: the jammer replays the UE's DMRS on its PRBs (unseen)",
    },
    # A pulsed pilot attack would be the plain pilot attack (one symbol), so data and partial only
    "pulsejam": {
        "seed": 17_000_000,
        "cfg": {"jammer_duty": 0.5},
        "attacks": (2, 3),
        "title": "pulsed jammer: each OFDM symbol jammed with probability 0.5 (unseen)",
    },
    "tonejam": {
        "seed": 18_000_000,
        "cfg": {"jammer_waveform": "tone", "jammer_tones": 1},
        "title": "multi-tone jammer: one tone per jammed PRB (unseen)",
    },
    # Pilot spoofing on a block of PRBs anywhere on the carrier, so usually on part of the
    # allocation only: the replayed pilots then end inside the allocation, which the residual sees
    "spoofpart": {
        "seed": 29_000_000,
        "cfg": {"jammer_waveform": "dmrs"},
        "attacks": (1,),
        "title": "pilot spoofing on part of the band (unseen)",
    },
    # 16-QAM (MCS 13) at the SNRs where link adaptation would choose it, with the nominal and the
    # hopping jammer (experiments/general_training)
    "test_qam16": {
        "seed": 52_000_000,
        "cfg": {"mcs_index": 13},
        "snr_db": (10.0, 25.0),
        "title": "16-QAM (MCS 13), SNR 10-25 dB",
    },
    "hopjam_qam16": {
        "seed": 53_000_000,
        "cfg": {"mcs_index": 13, "jammer_hopping": True},
        "snr_db": (10.0, 25.0),
        "title": "16-QAM with the hopping jammer, SNR 10-25 dB",
    },
    # The same with 64-QAM (MCS 17), at the SNRs where link adaptation would choose it
    # (experiments/one_unet)
    "test_qam64": {
        "seed": 62_000_000,
        "cfg": {"mcs_index": 17},
        "snr_db": (16.0, 30.0),
        "title": "64-QAM (MCS 17), SNR 16-30 dB",
    },
    "hopjam_qam64": {
        "seed": 63_000_000,
        "cfg": {"mcs_index": 17, "jammer_hopping": True},
        "snr_db": (16.0, 30.0),
        "title": "64-QAM with the hopping jammer, SNR 16-30 dB",
    },
    # Legitimate interference: a synchronized UE of a neighboring cell (experiments/neighbor)
    "neighbor": {
        "seed": 19_000_000,
        "cfg": {"jammer_waveform": "ue"},
        "attacks": (3,),
        "title": "synchronized neighbor-cell PUSCH interference",
    },
    # Channels with the TDL model's own normalization (unit power on average over realizations)
    # instead of Sionna's per-realization scaling to unit energy over the carrier
    "unnorm": {
        "seed": 20_000_000,
        "cfg": {"normalize_channel": False},
        "title": "channels without per-slot normalization (unseen)",
    },
}
for _name, _spec in GENERALIZATION.items():
    SPLITS[_name] = {"per_point": 200, "benign_per_channel": 1_000, "sched": False, **_spec}

# Training on mixed conditions: batches cycle through the nominal setup, long delays (TDL-A 300 ns,
# TDL-C 1000 ns) and speeds up to 30 m/s. Seeds differ from every other split.
MIXED_CONDITIONS = (
    {},
    {"delay_spread": {"A": 300e-9, "C": 1000e-9}},
    {"max_speed": 30.0},
)
SPLITS["train_mix"] = {"seed": 10_000_000, "num_slots": 40_000, "mix": MIXED_CONDITIONS}
SPLITS["val_mix"] = {"seed": 11_000_000, "num_slots": 5_000, "mix": MIXED_CONDITIONS}

# Training on mixed attacks: batches cycle through the nominal jammer, the hopping jammer and the
# pilot-comb jammer (``experiments/attack_training``). Seeds differ from every other split.
ATTACK_CONDITIONS = (
    {},
    {"jammer_hopping": True},
    {"jammer_pilot_comb": True},
)
SPLITS["train_atk"] = {"seed": 14_000_000, "num_slots": 40_000, "mix": ATTACK_CONDITIONS}
SPLITS["val_atk"] = {"seed": 15_000_000, "num_slots": 5_000, "mix": ATTACK_CONDITIONS}

# The same, plus pilot spoofing: the spoofer replays the UE's pilots, so it needs the schedule.
# Its batches jam exactly the scheduled PRBs and contain pilot attacks and benign slots. Keys
# starting with "_" change the batch (scenario), not SimConfig.
ATTACK_CONDITIONS_SPOOF = (
    *ATTACK_CONDITIONS,
    {"jammer_waveform": "dmrs", "_scheduling_aware": True, "_attacks": (0, 1)},
)
SPLITS["train_atk2"] = {"seed": 25_000_000, "num_slots": 40_000, "mix": ATTACK_CONDITIONS_SPOOF}

# General training: every combination of the jammers of ATTACK_CONDITIONS, the channels of
# MIXED_CONDITIONS and two modulations (18 conditions). 16-QAM (MCS 13) batches draw their SNR from
# 10-25 dB, where link adaptation would choose it. The pulsed jammer stays out as an unseen test.
MODULATION_CONDITIONS = ({}, {"mcs_index": 13, "_snr_db": (10.0, 25.0)})
GENERAL_CONDITIONS = tuple(
    {**jammer, **channel, **modulation}
    for jammer in ATTACK_CONDITIONS
    for channel in MIXED_CONDITIONS
    for modulation in MODULATION_CONDITIONS
)
SPLITS["train_gen"] = {"seed": 50_000_000, "num_slots": 120_000, "mix": GENERAL_CONDITIONS}
SPLITS["val_gen"] = {"seed": 51_000_000, "num_slots": 15_000, "mix": GENERAL_CONDITIONS}
# The same without the long delay spreads (12 conditions): channels whose delays exceed tau_max
# raise the DMRS residual without a jammer, so a network trained on them may trust it less
GENERAL_SHORT_CONDITIONS = tuple(c for c in GENERAL_CONDITIONS if "delay_spread" not in c)
SPLITS["train_gen_short"] = {
    "seed": 54_000_000,
    "num_slots": 120_000,
    "mix": GENERAL_SHORT_CONDITIONS,
}
SPLITS["val_gen_short"] = {"seed": 55_000_000, "num_slots": 15_000, "mix": GENERAL_SHORT_CONDITIONS}

# One U-Net for every configuration (paper v11, experiments/one_unet): the general conditions with
# 64-QAM (MCS 17, SNR 16-30 dB) as a third modulation, each with one and with two DMRS symbols
# (54 conditions). Twice the slots of train_gen, so each condition keeps about 4 400 of them.
QAM64_MCS = 17
QAM64_SNR_DB = (16.0, 30.0)
ALL_MODULATIONS = (*MODULATION_CONDITIONS, {"mcs_index": QAM64_MCS, "_snr_db": QAM64_SNR_DB})
DMRS_CONDITIONS = ({}, {"dmrs_additional_position": 1})
ALL_CONDITIONS = tuple(
    {**jammer, **channel, **modulation, **dmrs}
    for jammer in ATTACK_CONDITIONS
    for channel in MIXED_CONDITIONS
    for modulation in ALL_MODULATIONS
    for dmrs in DMRS_CONDITIONS
)
SPLITS["train_all"] = {"seed": 60_000_000, "num_slots": 240_000, "mix": ALL_CONDITIONS}
SPLITS["val_all"] = {"seed": 61_000_000, "num_slots": 30_000, "mix": ALL_CONDITIONS}
# Test grids with two DMRS symbols: the grids of the general training and the 64-QAM ones, each
# with a second DMRS symbol (``test_dmrs2`` already exists among the nominal two-DMRS splits)
DMRS2_GRIDS = (
    "hopjam",
    "combjam",
    "longdelay",
    "fast",
    "test_qam16",
    "hopjam_qam16",
    "test_qam64",
    "hopjam_qam64",
    "pulsejam",
)


def dmrs2_split(split: str) -> str:
    """Name of a test grid with two DMRS symbols, e.g. ``hopjam_dmrs2``."""
    return f"{split}_dmrs2"


for _k, _split in enumerate(DMRS2_GRIDS):
    SPLITS[dmrs2_split(_split)] = {
        **SPLITS[_split],
        "seed": 64_000_000 + 1_000_000 * _k,
        "cfg": {**SPLITS[_split].get("cfg", {}), "dmrs_additional_position": 1},
        "title": SPLITS[_split]["title"] + ", two DMRS symbols",
    }
SPLITS["val_atk2"] = {"seed": 26_000_000, "num_slots": 5_000, "mix": ATTACK_CONDITIONS_SPOOF}

# Noise power: the whole receiver chain with a fixed error of the noise power it assumes
# (smaller test grids), and with a per-slot estimate from the empty DMRS comb
NOISE_ERRORS_DB = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def noise_split(error_db: float) -> str:
    """Name of the test grid with a fixed noise-power error, e.g. ``noise_m0.5``."""
    return f"noise_{'m' if error_db < 0 else 'p'}{abs(error_db):g}"


for _k, _e in enumerate(NOISE_ERRORS_DB):
    SPLITS[noise_split(_e)] = {
        "seed": 27_000_000 + 100_000 * _k,
        "per_point": 100,
        "benign_per_channel": 1_000,
        "sched": False,
        "noise_error_db": _e,
        "title": f"receiver noise power off by {_e:+g} dB",
    }
_NEST = {"noise_estimate": "comb"}
SPLITS["train_nest"] = {"seed": 21_000_000, "num_slots": 40_000, "cfg": _NEST}
SPLITS["val_nest"] = {"seed": 22_000_000, "num_slots": 5_000, "cfg": _NEST}
SPLITS["test_nest"] = {**SPLITS["test"], "seed": 23_000_000, "cfg": _NEST}
SPLITS["sched_nest"] = {**SPLITS["sched"], "seed": 24_000_000, "cfg": _NEST}

for _k, _split in enumerate(CORE_SPLITS):
    SPLITS[f"{_split}_dmrs2"] = {
        **SPLITS[_split],
        "seed": (30 + _k) * 1_000_000,
        "cfg": {"dmrs_additional_position": 1},
    }

# Detection before damage: the setting of the BLER curves (jammer on the scheduled PRBs, SNR
# 15 dB), with JSRs low enough to find where detection starts
SPLITS["margin"] = {
    "seed": 8_000_000,
    # small batches, so every grid point mixes 25 allocations (a batch shares one allocation)
    "batch_size": 8,
    "per_point": 200,
    # many benign slots, so that the slot threshold can be set for 0.1% false alarm as well
    "benign_per_channel": 5_000,
    "sched": True,
    "snr_db": 15.0,
    "jsr_grid": tuple(float(j) for j in range(-30, 1, 3)),
    "title": "jammer on the scheduled PRBs, SNR 15 dB",
}
# The same at lower SNRs, where MCS 5 has less margin to spare (experiments/detection_margin)
MARGIN_SNRS_DB = (5.0, 8.0, 11.0)


def margin_split(snr_db: float) -> str:
    """Name of the margin grid at an SNR: ``margin`` at 15 dB, else e.g. ``margin_snr8``."""
    return "margin" if snr_db == 15.0 else f"margin_snr{snr_db:g}"


for _k, _snr in enumerate(MARGIN_SNRS_DB):
    SPLITS[margin_split(_snr)] = {
        **SPLITS["margin"],
        "seed": 9_000_000 + 10_000 * _k,
        "snr_db": _snr,
        "jsr_grid": tuple(float(j) for j in range(-36, 1, 3)),
        "title": f"jammer on the scheduled PRBs, SNR {_snr:g} dB",
    }


def _random_jobs(num_slots: int, batch_size: int) -> list[dict]:
    jobs = []
    for i in range(math.ceil(num_slots / batch_size)):
        n = min(batch_size, num_slots - i * batch_size)
        jobs.append(
            {
                "n": n,
                "channel": CHANNELS[i % 2],
                "attacks": (0, 1, 2, 3),
                "jsr_db": (-15.0, 10.0),
                "snr_db": (5.0, 25.0),
                "scheduling_aware": False,
            }
        )
    return jobs


def _grid_jobs(spec: dict, num_slots, batch_size: int) -> list[dict]:
    per_point, benign = spec["per_point"], spec["benign_per_channel"]
    channels = spec.get("channels", CHANNELS)
    attacks = spec.get("attacks", (1, 2, 3))
    jsr_grid = spec.get("jsr_grid", TEST_JSR_DB)
    snr = spec.get("snr_db", (5.0, 25.0))
    if num_slots is not None:  # smaller version of the same grid
        scale = num_slots / (len(channels) * len(attacks) * len(jsr_grid) * per_point)
        per_point = max(1, int(per_point * scale))
        benign = max(1, round(benign * scale)) if benign else 0

    def batches(total, **kw):
        return [dict(n=min(batch_size, total - s), **kw) for s in range(0, total, batch_size)]

    jobs = []
    for channel in channels:
        for attack in attacks:
            for jsr in jsr_grid:
                jobs += batches(
                    per_point,
                    channel=channel,
                    attacks=(attack,),
                    jsr_db=jsr,
                    snr_db=snr,
                    scheduling_aware=spec["sched"],
                    noise_error_db=spec.get("noise_error_db", 0.0),
                )
        jobs += batches(
            benign,
            channel=channel,
            attacks=(0,),
            jsr_db=0.0,
            snr_db=snr,
            scheduling_aware=False,
            noise_error_db=spec.get("noise_error_db", 0.0),
        )
    return jobs


def generate(split: str, num_slots: int | None = None, batch_size: int = 64):
    """Simulate a split and save it to ``data_dir()/{split}.npz``. Returns the file path."""
    spec = SPLITS[split]
    if "per_point" in spec:  # grid splits: small batches, so that allocations vary
        batch_size = spec.get("batch_size", GRID_BATCH_SIZE)
        jobs = _grid_jobs(spec, num_slots, batch_size)
    else:
        batch_size = spec.get("batch_size", batch_size)
        jobs = _random_jobs(num_slots or spec["num_slots"], batch_size)

    # One link per condition; batch i uses condition i // 2 (both TDL models per condition). Keys
    # starting with "_" change the batch (attacks, scheduling-aware jammer), the others SimConfig.
    conditions = spec.get("mix", (spec.get("cfg", {}),))
    links = [
        JammedPUSCHLink(
            dataclasses.replace(SimConfig(), **{k: v for k, v in c.items() if k[0] != "_"})
        )
        for c in conditions
    ]
    cfg = links[0].cfg
    features, labels, meta = [], [], {k: [] for k in META_KEYS}
    start = time.perf_counter()
    for i, job in enumerate(jobs):
        cond = conditions[(i // 2) % len(conditions)]
        job = {**job, **{k[1:]: v for k, v in cond.items() if k[0] == "_"}}
        rng = seed_everything(spec["seed"] + i)
        sc = draw_scenario(
            rng,
            job["n"],
            cfg,
            job["channel"],
            attacks=job["attacks"],
            jsr_db=job["jsr_db"],
            snr_db=job["snr_db"],
            scheduling_aware=job["scheduling_aware"],
            noise_error_db=job.get("noise_error_db", 0.0),
        )
        link = links[(i // 2) % len(links)]
        batch = link(sc)
        features.append(extract(batch, link.cfg).cpu().numpy().astype(np.float16))
        labels.append(batch.labels.cpu().numpy().astype(np.uint8))

        n = sc.batch_size
        meta["attack"].append(sc.attack.numpy())
        meta["jsr_db"].append(sc.jsr_db.numpy())
        meta["snr_db"].append(sc.snr_db.numpy())
        meta["channel"].append(np.full(n, CHANNELS.index(sc.channel)))
        meta["alloc_start"].append(np.full(n, sc.alloc_start))
        meta["alloc_size"].append(np.full(n, sc.alloc_size))
        meta["jam_start"].append(sc.jam_start.numpy())
        meta["jam_size"].append(sc.jam_size.numpy())
        meta["scheduling_aware"].append(np.full(n, job["scheduling_aware"]))
        meta["batch"].append(np.full(n, i))
        meta["snr_db_est"].append(estimated_snr_db(batch.y_alloc, batch.no, link.cfg).cpu().numpy())
        meta["noise_error_db"].append(
            (10 * torch.log10(batch.no / batch.no_true)).cpu().numpy().astype(np.float32)
        )
        meta["mcs_index"].append(np.full(n, link.cfg.mcs_index))

        if (i + 1) % 50 == 0 or i + 1 == len(jobs):
            done = sum(len(f) for f in features)
            rate = done / (time.perf_counter() - start)
            print(f"[{split}] {done} slots, {rate:.0f} slots/s", flush=True)

    path = data_dir() / f"{split}.npz"
    np.savez_compressed(
        path,
        features=np.concatenate(features),
        labels=np.concatenate(labels),
        **{k: np.concatenate(v) for k, v in meta.items()},
    )
    return path


def load(split: str) -> dict[str, np.ndarray]:
    """Load a split; features are returned as float32."""
    with np.load(data_dir() / f"{split}.npz") as f:
        d = {k: f[k] for k in f.files}
    d["features"] = d["features"].astype(np.float32)
    return d


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate a dataset split.")
    p.add_argument("--split", choices=list(SPLITS), required=True)
    p.add_argument("--num-slots", type=int, default=None, help="smaller split (for testing)")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    print("Wrote", generate(args.split, args.num_slots, args.batch_size))


if __name__ == "__main__":
    main()
