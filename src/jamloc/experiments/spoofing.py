"""Pilot spoofing: is the attack noticed, where is it placed, and how much does it hurt the link?

The spoofing jammer (``jammer_waveform="dmrs"``, test grid ``spoofjam``) replays the UE's DMRS on
the scheduled PRBs. Its contribution to the least-squares pilot estimates is the jammer's channel
times a constant, so when that channel stays within the delay subspace, the DMRS residual cannot
see the attack (``jamloc.jammer``). Two questions remain, answered here:

1. **Slot alarm and placement.** Does the fused U-Net still raise a slot alarm, and on which cells?
   A slot is flagged when its largest logit exceeds the level that 1% of the benign slots of the
   grid exceed; every other benign slot sets that level and the rest check it. Per JSR we report
   the share of spoofed slots flagged and the share of DMRS and data cells flagged at the per-cell
   threshold from ``val``.
2. **Damage.** The BLER of the unprotected link under spoofing and under Gaussian jamming of the
   DMRS symbol at the same JSR per jammed resource element (TDL-C, 15 dB SNR, jammer on all 12
   scheduled PRBs). The spoofer jams only the pilot comb, so it spends half the energy.

    python -m jamloc.experiments.spoofing
"""

import argparse
import csv
import dataclasses

import numpy as np
import torch

from jamloc.config import SimConfig
from jamloc.dataset import generate, load
from jamloc.evaluate import _logits
from jamloc.features import FEATURE_NAMES
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.metrics import threshold_at_fa
from jamloc.paths import data_dir, results_dir
from jamloc.train import load_model

FA = 0.01
BLER_JSR_DB = (-15.0, -9.0, -3.0, 3.0)
BLER_SEED = 9_100_000  # seed range of the BLER batches, unused by the dataset splits


def slot_alarms(seeds, split: str = "spoofjam") -> list[dict]:
    """Slot-level detection and the share of DMRS / data cells flagged, per seed and JSR."""
    if not (data_dir() / f"{split}.npz").is_file():
        generate(split, batch_size=128)
    val, d = load("val"), load(split)
    jammed, benign = d["attack"] == 1, np.flatnonzero(d["attack"] == 0)
    calibrate, check = benign[::2], benign[1::2]  # alternate benign slots: set and check the level
    dmrs = d["features"][:, FEATURE_NAMES.index("dmrs_mask")] > 0
    data = d["features"][:, FEATURE_NAMES.index("data_mask")] > 0

    rows = []
    for seed in seeds:
        model = load_model("fused", seed)
        s_val = _logits(model, "fused", torch.from_numpy(val["features"]))
        s = _logits(model, "fused", torch.from_numpy(d["features"]))
        flags = s > threshold_at_fa(s_val[val["labels"] == 0], FA)  # per-cell decisions
        largest = s.reshape(len(s), -1).max(axis=1)  # slot statistic
        level = np.quantile(largest[calibrate], 1 - FA, method="higher")
        slot_fa = float((largest[check] > level).mean())
        for jsr in np.unique(d["jsr_db"][jammed]):
            sel = jammed & (d["jsr_db"] == jsr)
            rows.append(
                {
                    "seed": seed,
                    "jsr_db": float(jsr),
                    "slot_fa": slot_fa,
                    "slot_pd": float((largest[sel] > level).mean()),
                    "dmrs_flagged": float((flags[sel] & dmrs[sel]).sum() / dmrs[sel].sum()),
                    "data_flagged": float((flags[sel] & data[sel]).sum() / data[sel].sum()),
                }
            )
    return rows


def bler(batches: int = 8, batch_size: int = 64) -> list[dict]:
    """Unprotected BLER under Gaussian pilot jamming and under spoofing, same slots per JSR."""
    rows = []
    for waveform in ("gaussian", "dmrs"):
        cfg = dataclasses.replace(SimConfig(), jammer_waveform=waveform)
        link = JammedPUSCHLink(cfg)
        for jsr in BLER_JSR_DB:
            errors = blocks = 0
            for i in range(batches):
                rng = seed_everything(BLER_SEED + i)
                sc = draw_scenario(
                    rng,
                    batch_size,
                    cfg,
                    "C",
                    attacks=(1,),
                    jsr_db=jsr,
                    snr_db=15.0,
                    scheduling_aware=True,
                    alloc=(6, 12),
                )
                crc_ok = link(sc, decode=True).rx["tb_crc"].bool()
                errors += int((~crc_ok).sum())
                blocks += crc_ok.numel()
            rows.append(
                {
                    "jammer": "spoofing" if waveform == "dmrs" else "gaussian",
                    "jsr_db": jsr,
                    "block_errors": errors,
                    "blocks": blocks,
                    "bler": errors / blocks,
                }
            )
            print(f"{waveform:8s} JSR {jsr:+.0f} dB: BLER {errors / blocks:.3f}", flush=True)
    return rows


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batches", type=int, default=8, help="BLER batches of 64 slots per JSR")
    args = p.parse_args(argv)

    out = results_dir() / "spoofing"
    out.mkdir(parents=True, exist_ok=True)
    alarms = slot_alarms(args.seeds)
    _write(alarms, out / "slot_alarms.csv")
    for r in alarms:
        print(
            f"seed {r['seed']} JSR {r['jsr_db']:+.0f} dB: slot Pd {r['slot_pd']:.2f} "
            f"(slot FA {r['slot_fa']:.3f}), DMRS cells flagged {r['dmrs_flagged']:.2f}, "
            f"data cells flagged {r['data_flagged']:.2f}"
        )
    _write(bler(args.batches), out / "bler.csv")
    print(f"Wrote {out}/slot_alarms.csv and bler.csv")


if __name__ == "__main__":
    main()
