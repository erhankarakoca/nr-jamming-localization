"""Can training on more diverse channels fix the unseen-condition failures?

The generalization test showed that delays beyond ``tau_max`` and fast-moving users raise the
false alarms of the detectors that use the receiver features (residual, EVM). Here every U-Net
variant is trained again on ``train_mix`` (batches cycling through the nominal channels, long
delay spreads and speeds up to 30 m/s; thresholds from ``val_mix``) and compared with the
nominally trained models (thresholds from ``val``) on the matched test grid and on the
``longdelay`` and ``fast`` grids. The mixed training data uses its own seeds, so no test slot is
seen in training; the test grids' exact settings (TDL-C at 1000 ns, 30 m/s) are part of the mix.

    python -m jamloc.experiments.robust_training --seeds 0 1 2
"""

import argparse
import csv

import numpy as np
import torch
from sionna.phy import config

from jamloc.dataset import generate, load
from jamloc.evaluate import aggregate, evaluate_scores
from jamloc.model import VARIANTS, select_inputs
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.train import checkpoint_path, load_model, train

CONDITIONS = {"test": "matched", "longdelay": "1000 ns delay spread", "fast": "30 m/s"}
TRAININGS = {"": ("nominal training", "val"), "mix": ("mixed training", "val_mix")}


def _logits(model, variant, features: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(features)
    with torch.no_grad():
        out = [
            model(select_inputs(x[i].to(config.device), variant)).cpu()
            for i in torch.arange(len(x)).split(2048)
        ]
    return torch.cat(out).numpy()


def run(seeds, epochs: int) -> list[dict]:
    for split in ("train_mix", "val_mix", *CONDITIONS):
        if not (data_dir() / f"{split}.npz").is_file():
            generate(split, batch_size=128)
    for variant in VARIANTS:
        for seed in seeds:
            if not checkpoint_path(variant, seed, data="mix").is_file():
                train(variant, epochs=epochs, seed=seed, data="mix")

    rows = []
    vals = {name: load(name) for name in ("val", "val_mix")}
    for split, condition in CONDITIONS.items():
        test = load(split)
        for data, (training, val_split) in TRAININGS.items():
            val = vals[val_split]
            for variant in VARIANTS:
                for seed in seeds:
                    model = load_model(variant, seed, data=data)
                    scores = {
                        variant: (
                            _logits(model, variant, val["features"]),
                            _logits(model, variant, test["features"]),
                        )
                    }
                    _, f1_rows = evaluate_scores(scores, val, test)
                    by = {r["attack"]: r for r in f1_rows}
                    rows.append(
                        {
                            "condition": condition,
                            "training": training,
                            "detector": variant,
                            "seed": seed,
                            "f1": by["all"]["f1"],
                            "f1_pilot": by["pilot"]["f1"],
                            "fa": by["all"]["fa_scored"],
                        }
                    )
            print(f"{condition}, {training}: done", flush=True)
    return rows


def plot(mean_rows):
    fig, axes = new_figure(7.16, 2.4, ncols=2)
    bars = [
        ("fused", "nominal training", "#0072B2", None),
        ("fused", "mixed training", "#0072B2", "///"),
        ("power", "nominal training", "#E69F00", None),
        ("power", "mixed training", "#E69F00", "///"),
    ]
    width = 0.8 / len(bars)
    conditions = list(CONDITIONS.values())
    for ax, key, title in ((axes[0], "f1", "F1, all slots"), (axes[1], "fa", "False-alarm rate")):
        for i, (det, training, color, hatch) in enumerate(bars):
            vals, errs = [], []
            for cond in conditions:
                (r,) = [
                    r
                    for r in mean_rows
                    if r["condition"] == cond and r["detector"] == det and r["training"] == training
                ]
                vals.append(r[f"{key}_mean"])
                errs.append(r[f"{key}_std"])
            x = np.arange(len(conditions)) + (i - (len(bars) - 1) / 2) * width
            label = f"{'P+R+E' if det == 'fused' else 'power only'}, {training}"
            ax.bar(
                x,
                vals,
                width,
                yerr=errs,
                color=color if hatch is None else "white",
                edgecolor=color,
                hatch=hatch,
                label=label,
                capsize=1.5,
            )
        ax.set_xticks(np.arange(len(conditions)), conditions)
        ax.set_title(title)
    axes[0].set_ylim(0.6, 1.0)
    axes[1].axhline(0.01, color="k", lw=0.8, ls=":")
    axes[1].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle("Unseen channel conditions: nominal vs mixed training (bars: std over seeds)")
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)

    out = results_dir() / "robust_training"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "robust_training.csv"
    if not args.plot_only:
        rows = aggregate(run(args.seeds, args.epochs), ("f1", "f1_pilot", "fa"))
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(path, encoding="utf-8", newline="") as f:
        text = ("condition", "training", "detector")
        rows = [{k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    with paper_style():
        fig = plot(rows)
    save(fig, out / "robust_training")
    for r in rows:
        print(
            f"{r['condition']:22s} {r['training']:17s} {r['detector']:12s} "
            f"F1 {r['f1_mean']:.3f}  FA {r['fa_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
