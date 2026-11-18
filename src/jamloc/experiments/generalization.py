"""Conditions the detectors never saw in training: where does the method break, and why?

The trained detectors (thresholds unchanged, from the ``val`` split) are evaluated on three test
grids, each changing one thing (see ``jamloc.dataset.GENERALIZATION``):

* ``longdelay`` - TDL-C with 1000 ns delay spread. Channel delays reach 8.7 us, beyond the
  ``tau_max`` = 3 us of the delay subspace, so part of the UE's own channel is left in the DMRS
  residual: expected to raise the residual's false alarms.
* ``fast`` - speeds up to 30 m/s. The channel changes within the slot, the single front-loaded DMRS
  no longer describes the last symbols well, and the EVM grows without jamming.
* ``qpskjam`` - a constant-envelope QPSK jammer instead of Gaussian noise (same energy).

    python -m jamloc.experiments.generalization
"""

import argparse
import csv

import numpy as np

from jamloc import evaluate
from jamloc.dataset import GENERALIZATION, generate
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save

CONDITIONS = {"test": "matched (test)", **{k: v["title"] for k, v in GENERALIZATION.items()}}
SHORT = {"test": "matched", "longdelay": "1000 ns", "fast": "30 m/s", "qpskjam": "QPSK"}
DETECTORS = (
    "fused",
    "no_residual",
    "power",
    "evm",
    "block_test",
    "evm_threshold",
    "residual_threshold",
)


def _f1_rows(split: str) -> list[dict]:
    with open(results_dir() / "eval" / f"{split}_f1.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summary(splits) -> list[dict]:
    rows = []
    for split in splits:
        for r in _f1_rows(split):
            if r["detector"] in DETECTORS:
                rows.append(
                    {
                        "condition": split,
                        "detector": r["detector"],
                        "attack": r["attack"],
                        "f1": float(r["f1_mean"]),
                        "f1_std": float(r["f1_std"]),
                        "fa_scored": float(r["fa_scored_mean"]),
                    }
                )
    return rows


def plot(rows, splits):
    fig, axes = new_figure(7.16, 2.5, ncols=3)
    width = 0.8 / len(DETECTORS)
    style = evaluate.DETECTOR_STYLE
    for ax, (attack, key, title) in zip(
        axes,
        (
            ("pilot", "f1", "F1, pilot jamming"),
            ("all", "f1", "F1, all slots"),
            ("all", "fa_scored", "False-alarm rate, all slots"),
        ),
        strict=True,
    ):
        for i, det in enumerate(DETECTORS):
            vals = []
            for split in splits:
                (r,) = [
                    r
                    for r in rows
                    if r["condition"] == split and r["detector"] == det and r["attack"] == attack
                ]
                vals.append(r[key])
            x = np.arange(len(splits)) + (i - (len(DETECTORS) - 1) / 2) * width
            ax.bar(x, vals, width, color=style[det]["color"], label=style[det]["label"])
        ax.set_xticks(np.arange(len(splits)), [SHORT[s] for s in splits])
        ax.set_title(title)
        if key == "fa_scored":
            ax.axhline(evaluate.FA, color="k", lw=0.8, ls=":")
            ax.set_yscale("log")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=6, frameon=False)
    fig.suptitle(
        "Unseen conditions: TDL-C at 1000 ns delay spread, speeds up to 30 m/s, QPSK jammer\n"
        "(thresholds from the matched validation set)"
    )
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--splits", nargs="+", default=list(GENERALIZATION), choices=list(GENERALIZATION)
    )
    p.add_argument("--num-slots", type=int, default=None, help="smaller grids (for testing)")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)

    if not args.plot_only:
        if not (results_dir() / "eval" / "test_f1.csv").is_file():
            evaluate.main(["--split", "test"])  # the matched reference
        for split in args.splits:
            if not (data_dir() / f"{split}.npz").is_file():
                generate(split, args.num_slots, batch_size=128)
            evaluate.main(["--split", split])

    splits = ["test", *args.splits]
    rows = summary(splits)
    out = results_dir() / "generalization"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "generalization.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paper_style():
        fig = plot(rows, splits)
    save(fig, out / "generalization")
    print(f"Wrote {out / 'generalization'}.csv/.pdf/.png")


if __name__ == "__main__":
    main()
