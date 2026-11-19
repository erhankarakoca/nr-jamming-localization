"""Neighbor-cell interference: how often do the detectors report it as jamming?

A synchronized UE in a loaded neighboring cell can occupy the same PRB block and symbols as a
partial-band jammer. We score the ``neighbor`` split with the detectors in ``jamloc.evaluate``,
using the same thresholds from ``val`` at 1% false alarm on scored non-jammed cells. At each ISR
and across all ISR points, we measure the flagged share of interfered cells inside and outside
the scheduled allocation and the false-alarm rate on clean cells in those same slots. When
``test.npz`` is available, its Gaussian partial-band slots give a comparison at the same power
levels. The scheduled-cell figure uses the fused U-Net's Gaussian curve as the reference.

    python -m jamloc.experiments.neighbor
    python -m jamloc.experiments.neighbor --num-slots 20  # small split if missing
"""

import argparse

import numpy as np

from jamloc.dataset import generate, load
from jamloc.evaluate import (
    DETECTOR_STYLE,
    FA,
    PLOT_DETECTORS,
    _write_csv,
    aggregate,
    available_seeds,
    score_split,
)
from jamloc.features import FEATURE_NAMES
from jamloc.jammer import ATTACKS
from jamloc.metrics import detection_rate, threshold_at_fa
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save

SHARES = ("scheduled", "unscheduled", "fa", "fa_scored")


def _partial_band(split: dict, isr: np.ndarray | None = None) -> dict:
    keep = split["attack"] == ATTACKS.index("partial")
    if isr is not None:
        keep &= np.isin(split["jsr_db"], isr)
    return {key: values[keep] for key, values in split.items()}


def share_rows(scores: np.ndarray, split: dict, threshold: float) -> list[dict]:
    """Flagged shares by ISR and pooled over the cells of all supplied partial-band slots.

    A detector's unscored cells (``-inf``) can never be flagged. The clean-cell false-alarm rate
    is given both over all clean cells and over scored clean cells. Empty groups yield NaN.
    """
    labels = split["labels"].astype(bool, copy=False)
    scheduled = split["features"][:, FEATURE_NAMES.index("alloc_mask")] > 0
    isr = np.asarray(split["jsr_db"])
    groups = [(float(db), isr == db) for db in np.unique(isr)]

    rows = []
    for db, keep in [*groups, ("all", slice(None))]:
        s, hit, alloc = scores[keep], labels[keep], scheduled[keep]
        inside, outside, clean = s[hit & alloc], s[hit & ~alloc], s[~hit]
        rows.append(
            {
                "isr_db": db,
                "num_scheduled": inside.size,
                "num_unscheduled": outside.size,
                "num_clean": clean.size,
                "scheduled": detection_rate(inside, threshold),
                "unscheduled": detection_rate(outside, threshold),
                "fa": detection_rate(clean, threshold),
                "fa_scored": detection_rate(clean[clean > -np.inf], threshold),
            }
        )
    return rows


def plot_shares(rows: list[dict]):
    fig, ax = new_figure(5.6, 3.2)
    for detector in PLOT_DETECTORS:
        points = [
            r
            for r in rows
            if r["source"] == "neighbor" and r["detector"] == detector and r["isr_db"] != "all"
        ]
        if points:
            style = DETECTOR_STYLE[detector]
            x = [r["isr_db"] for r in points]
            mean = np.array([r["scheduled_mean"] for r in points])
            std = np.array([r["scheduled_std"] for r in points])
            ax.plot(x, mean, **style)
            ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15, lw=0)

    reference = [
        r
        for r in rows
        if r["source"] == "gaussian" and r["detector"] == "fused" and r["isr_db"] != "all"
    ]
    if reference:
        ax.plot(
            [r["isr_db"] for r in reference],
            [r["scheduled_mean"] for r in reference],
            color="0.35",
            linestyle="--",
            label="Gaussian partial-band (fused U-Net)",
        )
    ax.set(
        xlabel="ISR (JSR for Gaussian) per data RE [dB]",
        ylabel="Flagged share of\ninterfered scheduled cells",
        ylim=(-0.02, 1.02),
    )
    fig.legend(loc="outside lower center", ncols=3, frameon=False)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Measure jamming alarms on neighbor-cell interference.")
    p.add_argument(
        "--num-slots", type=int, help="generate a smaller grid if neighbor.npz is missing"
    )
    args = p.parse_args(argv)
    if args.num_slots is not None and args.num_slots < 1:
        p.error("--num-slots must be positive")

    seeds = available_seeds()
    if not seeds:
        raise SystemExit("No trained models in results/models. Run python -m jamloc.train first.")
    val = load("val")
    if not (data_dir() / "neighbor.npz").is_file():
        generate("neighbor", num_slots=args.num_slots)

    neighbor = _partial_band(load("neighbor"))
    if not len(neighbor["labels"]):
        raise SystemExit("The neighbor split must contain partial-band slots.")
    splits = {"neighbor": neighbor}
    if (data_dir() / "test.npz").is_file():
        gaussian = _partial_band(load("test"), np.unique(neighbor["jsr_db"]))
        if len(gaussian["labels"]):
            splits["gaussian"] = gaussian

    rows = []
    negatives = val["labels"] == 0
    for seed in seeds:
        thresholds = {
            detector: threshold_at_fa(scores[negatives], FA)
            for detector, scores in score_split(val, seed, null=val).items()
        }
        for source, split in splits.items():
            for detector, scores in score_split(split, seed, null=val).items():
                rows.extend(
                    {"source": source, "detector": detector, "seed": seed, **r}
                    for r in share_rows(scores, split, thresholds[detector])
                )
    mean = aggregate(rows, SHARES)

    out = results_dir() / "neighbor"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out / "cell_shares_per_seed.csv")
    _write_csv(mean, out / "cell_shares.csv")
    with paper_style():
        fig = plot_shares(mean)
    save(fig, out / "scheduled_vs_isr")

    print(f"Training seeds {seeds}")
    for source, split in splits.items():
        print(f"{source}: {len(split['labels'])} partial-band slots")
    for r in mean:
        if r["isr_db"] == "all":
            print(
                f"{r['source']:8s} {r['detector']:20s} "
                f"scheduled {r['scheduled_mean']:.1%} +- {r['scheduled_std']:.1%}, "
                f"unscheduled {r['unscheduled_mean']:.1%} +- {r['unscheduled_std']:.1%}, "
                f"clean FA {r['fa_mean']:.2%} +- {r['fa_std']:.2%}"
            )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
