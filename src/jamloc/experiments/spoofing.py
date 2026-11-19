"""Pilot spoofing: is the attack noticed, where is it placed, and how much does it hurt the link?

The spoofing jammer (``jammer_waveform="dmrs"``, test grid ``spoofjam``) replays the UE's DMRS on
the scheduled PRBs. Its contribution to the least-squares pilot estimates is the jammer's channel
times a constant, so when that channel stays within the delay subspace, the DMRS residual cannot
see the attack (``jamloc.jammer``). Two questions remain, answered here:

1. **Slot alarm and placement.** Does the fused U-Net still raise a slot alarm, and on which cells?
   A slot is flagged when its largest logit exceeds the level that 1% of the benign slots of the
   grid exceed; the benign batches alternate between setting that level and checking it. Per JSR
   we report the share of spoofed slots flagged, the share of DMRS and data cells flagged at the
   per-cell threshold from validation, and the cell-level F1 and false-alarm rate of the slots at
   that JSR. This is done for the nominally trained network and, if trained
   (``experiments/attack_training``), for the one trained with spoofing in its mix of attacks.
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


TRAININGS = {"": "nominal training", "atk2": "attack-mixed training with spoofing"}


def slot_alarms(seeds, split: str = "spoofjam", data_tag: str = "") -> list[dict]:
    """Slot-level detection and the share of DMRS / data cells flagged, per seed and JSR, for the
    fused U-Net trained on ``train`` (``data_tag=""``) or on ``train_<data_tag>``."""
    from jamloc.metrics import f1

    if not (data_dir() / f"{split}.npz").is_file():
        generate(split, batch_size=128)
    val, d = load(f"val_{data_tag}" if data_tag else "val"), load(split)
    jammed, benign = d["attack"] == 1, np.flatnonzero(d["attack"] == 0)
    # alternate benign batches (whole allocations) between setting and checking the level; with
    # a single benign batch (tiny test splits), alternate slots instead
    batch = d["batch"][benign] if "batch" in d else np.arange(len(benign))
    if len(np.unique(batch)) < 2:
        batch = np.arange(len(benign))
    rank = np.unique(batch, return_inverse=True)[1]  # 0, 1, 2, ... over the benign batches
    calibrate, check = benign[rank % 2 == 0], benign[rank % 2 == 1]
    dmrs = d["features"][:, FEATURE_NAMES.index("dmrs_mask")] > 0
    data = d["features"][:, FEATURE_NAMES.index("data_mask")] > 0
    labels = d["labels"].astype(bool)

    rows = []
    for seed in seeds:
        model = load_model("fused", seed, data=data_tag)
        s_val = _logits(model, "fused", torch.from_numpy(val["features"]))
        s = _logits(model, "fused", torch.from_numpy(d["features"]))
        flags = s > threshold_at_fa(s_val[val["labels"] == 0], FA)  # per-cell decisions
        largest = s.reshape(len(s), -1).max(axis=1)  # slot statistic
        level = np.quantile(largest[calibrate], 1 - FA, method="higher")
        slot_fa = float((largest[check] > level).mean())
        for jsr in np.unique(d["jsr_db"][jammed]):
            sel = jammed & (d["jsr_db"] == jsr)
            neg = ~labels[sel]
            rows.append(
                {
                    "training": TRAININGS[data_tag],
                    "seed": seed,
                    "jsr_db": float(jsr),
                    "slot_fa": slot_fa,
                    "slot_pd": float((largest[sel] > level).mean()),
                    "dmrs_flagged": float((flags[sel] & dmrs[sel]).sum() / dmrs[sel].sum()),
                    "data_flagged": float((flags[sel] & data[sel]).sum() / data[sel].sum()),
                    # cell-level F1 and false alarms of the spoofed slots at this JSR only
                    "f1": f1(flags[sel], labels[sel]),
                    "fa": float((flags[sel] & neg).sum() / neg.sum()),
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
    p.add_argument("--no-bler", action="store_true", help="skip the BLER part")
    args = p.parse_args(argv)

    from jamloc.train import checkpoint_path

    out = results_dir() / "spoofing"
    out.mkdir(parents=True, exist_ok=True)
    alarms = []
    for tag in TRAININGS:
        if all(checkpoint_path("fused", s, data=tag).is_file() for s in args.seeds):
            alarms += slot_alarms(args.seeds, data_tag=tag)
    _write(alarms, out / "slot_alarms.csv")
    for r in alarms:
        print(
            f"{r['training']:36s} seed {r['seed']} JSR {r['jsr_db']:+.0f} dB: slot Pd "
            f"{r['slot_pd']:.2f} (slot FA {r['slot_fa']:.3f}), DMRS cells flagged "
            f"{r['dmrs_flagged']:.2f}, data cells flagged {r['data_flagged']:.2f}, "
            f"F1 {r['f1']:.3f}, FA {r['fa']:.3f}"
        )
    if not args.no_bler:
        _write(bler(args.batches), out / "bler.csv")
    print(f"Wrote {out}/slot_alarms.csv" + ("" if args.no_bler else " and bler.csv"))


if __name__ == "__main__":
    main()
