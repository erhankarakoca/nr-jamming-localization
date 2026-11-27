"""Where the detectors find the jammed cells: Pd against the JSR (paper v12).

The v11 U-Net (``experiments.one_unet``, three training seeds, threshold from ``val_all``) and the
EVM threshold (threshold from ``val``, as in the decoding runs) mark the cells of the test grids.
Pd is the share of the jammed cells inside the allocation that a detector marks, for the DMRS cells
under pilot-only jamming and for the data cells under data-only jamming, at each JSR of the grid.

    python -m jamloc.experiments.detection_curves            # CSV and figure
    python -m jamloc.experiments.detection_curves --plot-only

Output: ``results/detection_curves/detection_curves.csv`` and ``detection_curves.pdf/png``.
"""

import argparse
import csv

import numpy as np

from jamloc.baselines import evm_scores
from jamloc.dataset import load
from jamloc.evaluate import _logits, evaluate_scores
from jamloc.paths import results_dir
from jamloc.plotting import BLUE, COLUMN_WIDTH, GREEN, MAGENTA, RED, new_figure, paper_style, save
from jamloc.train import load_model

SEEDS = (0, 1, 2)
VARIANT, DATA, VAL = "fused", "all", "val_all"  # the v11 U-Net
# (detector, grid): label and line style of a curve
CURVES = {
    ("unet", "test"): ("U-Net, block", {"color": BLUE, "marker": "o", "ls": "-"}),
    ("unet", "test_dmrs2"): ("U-Net, block, 2 DMRS", {"color": BLUE, "marker": "s", "ls": "--"}),
    ("unet", "hopjam"): ("U-Net, hopping", {"color": GREEN, "marker": "^", "ls": "-"}),
    ("unet", "pulsejam"): ("U-Net, pulsed (unseen)", {"color": MAGENTA, "marker": "v", "ls": ":"}),
    ("evm", "test"): ("EVM, block", {"color": RED, "marker": "x", "ls": "-."}),
}
# Panel: (attack, title)
PANELS = (("pilot", "(a) Pilot-only, DMRS cells"), ("data", "(b) Data-only, data cells"))


def pd_rows(seeds=SEEDS) -> list[dict]:
    """Pd of every curve, attack of ``PANELS`` and JSR, per seed for the U-Net."""
    val_all, val = load(VAL), load("val")
    rows = []
    for grid in sorted({g for _, g in CURVES}):
        test = load(grid)
        runs = (
            [
                (
                    "evm",
                    None,
                    {"evm": (evm_scores(val["features"]), evm_scores(test["features"]))},
                    val,
                )
            ]
            if ("evm", grid) in CURVES
            else []
        )
        for seed in seeds:
            model = load_model(VARIANT, seed, data=DATA)
            scores = (
                _logits(model, VARIANT, _tensor(val_all)),
                _logits(model, VARIANT, _tensor(test)),
            )
            runs.append(("unet", seed, {"unet": scores}, val_all))
        for detector, seed, scores, v in runs:
            for r in evaluate_scores(scores, v, test)[0]:
                keep = r["channel"] == "both" and r["cells"] == "scheduled"
                if keep and r["attack"] in dict(PANELS):
                    rows.append({"detector": detector, "grid": grid, "seed": seed, **r})
    return rows


def _tensor(d: dict):
    import torch

    return torch.from_numpy(d["features"])


def mean_rows(rows: list[dict]) -> list[dict]:
    """Pd averaged over the seeds of each curve, attack and JSR."""
    groups = {}
    for r in rows:
        key = (r["detector"], r["grid"], r["attack"], float(r["jsr_db"]))
        groups.setdefault(key, []).append(float(r["pd"]))
    return [
        {
            "detector": d,
            "grid": g,
            "attack": a,
            "jsr_db": j,
            "pd": float(np.mean(v)),
            "runs": len(v),
        }
        for (d, g, a, j), v in sorted(groups.items())
    ]


def figure(mean: list[dict]):
    with paper_style():
        fig, axes = new_figure(COLUMN_WIDTH, 2.05, ncols=2, sharey=True)
        for ax, (attack, title) in zip(axes, PANELS, strict=True):
            for (detector, grid), (label, style) in CURVES.items():
                pts = [
                    m
                    for m in mean
                    if (m["detector"], m["grid"], m["attack"]) == (detector, grid, attack)
                ]
                if pts:
                    x = [m["jsr_db"] for m in pts]
                    y = [m["pd"] for m in pts]
                    ax.plot(x, y, ms=3, mfc="none", lw=0.9, label=label, **style)
            ax.set(title=title, xlabel="JSR (dB)", ylim=(-0.02, 1.02), xticks=[-15, -9, -3, 3, 9])
        axes[0].set_ylabel(r"$P_\mathrm{d}$")
        # Every curve once, in the order of CURVES (the pulsed grid has no pilot-only attack)
        found = {}
        for ax in axes:
            for h, label in zip(*ax.get_legend_handles_labels(), strict=True):
                found.setdefault(label, h)
        labels = [label for label, _ in CURVES.values() if label in found]
        fig.legend(
            [found[label] for label in labels],
            labels,
            loc="outside lower center",
            ncols=3,
            fontsize=6,
            handlelength=2.0,
            columnspacing=0.8,
        )
    return fig


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [
            {**r, "jsr_db": float(r["jsr_db"]), "pd": float(r["pd"])} for r in csv.DictReader(f)
        ]


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)
    out = results_dir() / "detection_curves"
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        mean = _read(out / "detection_curves.csv")
    else:
        rows = pd_rows(args.seeds)
        _write(rows, out / "detection_curves_per_seed.csv")
        mean = mean_rows(rows)
        _write(mean, out / "detection_curves.csv")
    save(figure(mean), out / "detection_curves")
    for m in mean:
        if m["jsr_db"] in (-15.0, -9.0, -3.0, 3.0, 9.0):
            print(
                f"{m['detector']:5s} {m['grid']:11s} {m['attack']:6s} {m['jsr_db']:+5.0f} dB "
                f"Pd {m['pd']:.3f}"
            )


if __name__ == "__main__":
    main()
