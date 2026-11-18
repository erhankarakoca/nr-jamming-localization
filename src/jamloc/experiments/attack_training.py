"""Smarter jammers: does training on more varied attacks restore the detector?

Two attackers that use their knowledge of the DMRS configuration were never seen in training
(``jamloc.jammer``): one jams only the pilot subcarriers of the DMRS symbol (``combjam``), the other
moves its PRB block to a new position on every OFDM symbol (``hopjam``). The hopping jammer breaks
the whole-PRB structure the U-Net learned from the training attacks.

Here the fused U-Net is trained again on ``train_atk`` (batches cycling through the nominal,
hopping and pilot-comb jammers; thresholds from ``val_atk``) and compared with the nominally
trained one (thresholds from ``val``) on the matched test grid and on both new grids. The training
data uses its own seeds, but the attack types of the test grids are part of the mix.

    python -m jamloc.experiments.attack_training --seeds 0 1 2
"""

import argparse
import csv

import numpy as np
import torch

from jamloc.dataset import generate, load
from jamloc.evaluate import _logits, aggregate, evaluate_scores
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.train import checkpoint_path, load_model, train

CONDITIONS = {"test": "matched", "combjam": "pilot comb", "hopjam": "hopping"}
TRAININGS = {"": ("nominal training", "val"), "atk": ("attack-mixed training", "val_atk")}
VARIANT = "fused"


def run(seeds, epochs: int) -> list[dict]:
    for split in ("train_atk", "val_atk", *CONDITIONS):
        if not (data_dir() / f"{split}.npz").is_file():
            generate(split, batch_size=128)
    for seed in seeds:
        if not checkpoint_path(VARIANT, seed, data="atk").is_file():
            train(VARIANT, epochs=epochs, seed=seed, data="atk")

    rows = []
    vals = {name: load(name) for name in ("val", "val_atk")}
    for split, condition in CONDITIONS.items():
        test = load(split)
        for data, (training, val_split) in TRAININGS.items():
            val = vals[val_split]
            for seed in seeds:
                model = load_model(VARIANT, seed, data=data)
                scores = {
                    VARIANT: (
                        _logits(model, VARIANT, torch.from_numpy(val["features"])),
                        _logits(model, VARIANT, torch.from_numpy(test["features"])),
                    )
                }
                _, f1_rows = evaluate_scores(scores, val, test)
                by = {r["attack"]: r for r in f1_rows}
                rows.append(
                    {
                        "condition": condition,
                        "training": training,
                        "seed": seed,
                        "f1": by["all"]["f1"],
                        "f1_pilot": by["pilot"]["f1"],
                        "f1_data": by["data"]["f1"],
                        "f1_partial": by["partial"]["f1"],
                        "fa": by["all"]["fa_scored"],
                    }
                )
            print(f"{condition}, {training}: done", flush=True)
    return rows


def plot(mean_rows):
    fig, axes = new_figure(7.16, 2.5, ncols=2)
    bars = (("nominal training", None), ("attack-mixed training", "///"))
    conditions = list(CONDITIONS.values())
    width = 0.35
    for ax, key, title in ((axes[0], "f1", "F1, all slots"), (axes[1], "fa", "False-alarm rate")):
        for i, (training, hatch) in enumerate(bars):
            vals, errs = [], []
            for cond in conditions:
                (r,) = [
                    r for r in mean_rows if r["condition"] == cond and r["training"] == training
                ]
                vals.append(r[f"{key}_mean"])
                errs.append(r[f"{key}_std"])
            x = np.arange(len(conditions)) + (i - 0.5) * width
            ax.bar(
                x,
                vals,
                width,
                yerr=errs,
                color="#0072B2" if hatch is None else "white",
                edgecolor="#0072B2",
                hatch=hatch,
                label=f"U-Net P+R+E, {training}",
                capsize=1.5,
            )
        ax.set_xticks(np.arange(len(conditions)), conditions)
        ax.set_title(title)
    axes[0].set_ylim(0.4, 1.0)
    axes[1].axhline(0.01, color="k", lw=0.8, ls=":")
    axes[1].set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("Smarter jammers: nominal vs attack-mixed training (bars: std over seeds)")
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)

    out = results_dir() / "attack_training"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "attack_training.csv"
    if not args.plot_only:
        values = ("f1", "f1_pilot", "f1_data", "f1_partial", "fa")
        rows = aggregate(run(args.seeds, args.epochs), values)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(path, encoding="utf-8", newline="") as f:
        text = ("condition", "training")
        rows = [{k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    with paper_style():
        fig = plot(rows)
    save(fig, out / "attack_training")
    for r in rows:
        print(
            f"{r['condition']:12s} {r['training']:23s} F1 {r['f1_mean']:.3f} "
            f"(pilot {r['f1_pilot_mean']:.3f}, data {r['f1_data_mean']:.3f}, "
            f"partial {r['f1_partial_mean']:.3f})  FA {r['fa_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
