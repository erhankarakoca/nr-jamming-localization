"""Compare all detectors: Pd at 1% per-cell false alarm vs JSR, and cell-level F1.

Thresholds always come from the ``val`` split (1% of its non-jammed cells flagged) and are then
applied unchanged to the evaluated split. Pd is reported over all jammed cells and over the
jammed cells inside the allocation ("scheduled"), where the UE's own signal is present.

Every training seed found in ``results/models`` is evaluated separately (with its own
thresholds); the CSVs give the mean and the sample standard deviation over seeds, and the
``*_per_seed.csv`` files keep the individual runs.

    python -m jamloc.evaluate --split test
    python -m jamloc.evaluate --split sched                 # scheduling-aware jammer
    python -m jamloc.evaluate --split test --snr-db 5 15    # only slots with 5 <= SNR < 15 dB
"""

import argparse
import csv

import numpy as np
import torch
from sionna.phy import config

from jamloc.baselines import (
    block_test_scores,
    comb_scores,
    evm_scores,
    power_scores,
    residual_cfar_scores,
    residual_scores,
)
from jamloc.dataset import CHANNELS, SPLITS, load
from jamloc.features import FEATURE_NAMES
from jamloc.jammer import ATTACKS
from jamloc.metrics import detection_rate, f1, threshold_at_fa
from jamloc.model import EXTRA_VARIANTS, VARIANTS, select_inputs
from jamloc.paths import results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.train import PIXEL_HIDDEN_MATCHED, checkpoint_path, load_model

FA = 0.01
DETECTOR_STYLE = {
    "fused": {"label": "U-Net, P+R+E", "color": "#0072B2", "marker": "o", "linestyle": "-"},
    "no_residual": {"label": "U-Net, P+E", "color": "#56B4E9", "marker": "v", "linestyle": "--"},
    "power": {"label": "U-Net, power only", "color": "#E69F00", "marker": "s", "linestyle": "-."},
    "evm": {"label": "U-Net, EVM only", "color": "#D55E00", "marker": "^", "linestyle": ":"},
    "evm_threshold": {
        "label": "EVM threshold",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": ":",
    },
    "residual_threshold": {
        "label": "Residual threshold",
        "color": "#009E73",
        "marker": "x",
        "linestyle": "--",
    },
    "power_threshold": {
        "label": "Power threshold (Li et al.)",
        "color": "#999999",
        "marker": "*",
        "linestyle": ":",
    },
    "residual_cfar": {
        "label": "Residual, exact p-value",
        "color": "#000000",
        "marker": "+",
        "linestyle": "-.",
    },
    "block_test": {
        "label": "Block test (no training)",
        "color": "#882255",
        "marker": "P",
        "linestyle": "-",
    },
    "fused_nj1": {
        "label": "U-Net, P+R+E, noise jitter",
        "color": "#332288",
        "marker": "d",
        "linestyle": ":",
    },
    "fused_pixel": {
        "label": "Per-cell MLP, P+R+E",
        "color": "#44AA99",
        "marker": "h",
        "linestyle": "--",
    },
    "fused_pixel_wide": {
        "label": "Per-cell MLP (U-Net size), P+R+E",
        "color": "#117733",
        "marker": "H",
        "linestyle": "--",
    },
    "fused_comb": {
        "label": "U-Net, P+R+E+C",
        "color": "#AA4499",
        "marker": "p",
        "linestyle": "-",
    },
    "comb_threshold": {
        "label": "Empty-comb test",
        "color": "#DDCC77",
        "marker": "1",
        "linestyle": "--",
    },
    "block_test_comb": {
        "label": "Block test + empty comb",
        "color": "#661100",
        "marker": "X",
        "linestyle": "-.",
    },
}


# Detectors drawn in the Pd figures (the CSVs keep all of them). The raw residual threshold is
# left out (it is almost identical to its exact p-value), and so are the per-cell MLP and the
# jittered U-Net, which have their own experiments.
PLOT_DETECTORS = (
    "fused",
    "no_residual",
    "power",
    "evm",
    "block_test",
    "evm_threshold",
    "residual_cfar",
    "power_threshold",
)


def available_seeds(data: str = "") -> list[int]:
    """Training seeds for which every U-Net variant has a checkpoint."""
    seeds = None
    suffix = f"_{data}" if data else ""
    for variant in VARIANTS:
        found = {
            int(path.stem.rsplit("_seed", 1)[1])
            for path in (results_dir() / "models").glob(f"{variant}{suffix}_seed*.pt")
        }
        seeds = found if seeds is None else seeds & found
    return sorted(seeds or [])


def _logits(model, variant: str, features: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = [
            model(select_inputs(features[i].to(config.device), variant)).cpu()
            for i in torch.arange(len(features)).split(2048)
        ]
    return torch.cat(logits).numpy()


def score_split(d: dict, seed: int, null: dict, *, data: str = "") -> dict[str, np.ndarray]:
    """Scores ``[N, 24, 14]`` of every detector on one loaded split.

    ``null`` is the validation split whose benign slots calibrate the block test's EVM and
    power-ratio statistics. ``data`` selects the corresponding models. The per-cell MLPs, the U-Net
    trained with noise jitter and the extra U-Net variants are included when their checkpoints
    exist.
    """
    features = torch.from_numpy(d["features"])
    scores = {
        variant: _logits(load_model(variant, seed, data=data), variant, features)
        for variant in VARIANTS
    }
    for variant in EXTRA_VARIANTS:
        if checkpoint_path(variant, seed, data=data).is_file():
            scores[variant] = _logits(load_model(variant, seed, data=data), variant, features)
    optional = {
        "fused_pixel": {"arch": "pixel"},
        "fused_pixel_wide": {"arch": "pixel", "hidden": PIXEL_HIDDEN_MATCHED},
        "fused_nj1": {"jitter": 1.0},
    }
    for name, kw in optional.items():
        if checkpoint_path("fused", seed, data=data, **kw).is_file():
            scores[name] = _logits(load_model("fused", seed, data=data, **kw), "fused", features)
    scores["evm_threshold"] = evm_scores(d["features"])
    scores["residual_threshold"] = residual_scores(d["features"])
    scores["residual_cfar"] = residual_cfar_scores(d)
    scores["power_threshold"] = power_scores(d["features"])
    scores["comb_threshold"] = comb_scores(d["features"])
    scores["block_test"] = block_test_scores(d, null)
    scores["block_test_comb"] = block_test_scores(d, null, use_comb=True)
    return scores


def evaluate_scores(scores: dict, val: dict, test: dict, fa: float = FA):
    """``scores[name] = (val_scores, test_scores)``. Returns Pd rows and F1 rows."""
    labels = test["labels"].astype(bool)
    in_alloc = test["features"][:, FEATURE_NAMES.index("alloc_mask")] > 0
    channel_sel = {"both": np.ones(len(labels), bool)}
    channel_sel.update({ch: test["channel"] == i for i, ch in enumerate(CHANNELS)})

    pd_rows, f1_rows = [], []
    for name, (s_val, s_test) in scores.items():
        thr = threshold_at_fa(s_val[val["labels"] == 0], fa)
        pred = s_test > thr
        scored = s_test > -np.inf

        for attack in (1, 2, 3):
            for channel, ch_sel in channel_sel.items():
                sel = (test["attack"] == attack) & ch_sel
                for jsr in np.unique(test["jsr_db"][sel]):
                    s = sel & (test["jsr_db"] == jsr)
                    for cells, mask in (("all", labels), ("scheduled", labels & in_alloc)):
                        pos = s_test[s][mask[s]]
                        pd_rows.append(
                            {
                                "detector": name,
                                "attack": ATTACKS[attack],
                                "channel": channel,
                                "jsr_db": float(jsr),
                                "cells": cells,
                                "pd": detection_rate(pos, thr),
                                "num_cells": pos.size,
                            }
                        )

        groups = {ATTACKS[a]: test["attack"] == a for a in (1, 2, 3)}
        groups["all"] = np.ones(len(labels), bool)
        for group, sel in groups.items():
            if not sel.any():  # some test grids contain only some attack types
                continue
            neg = ~labels[sel]
            f1_rows.append(
                {
                    "detector": name,
                    "attack": group,
                    "f1": f1(pred[sel], labels[sel]),
                    # false alarms over all non-jammed cells, and over those the detector scores
                    "fa": float((pred[sel] & neg).sum() / neg.sum()),
                    "fa_scored": float((pred[sel] & neg).sum() / max((neg & scored[sel]).sum(), 1)),
                }
            )
    return pd_rows, f1_rows


def aggregate(rows: list[dict], values) -> list[dict]:
    """Mean and sample standard deviation over seeds of the column(s) ``values``.

    Rows that agree on every other column (except ``seed``) form one group.
    """
    values = (values,) if isinstance(values, str) else tuple(values)
    groups = {}
    for r in rows:
        key = tuple((k, v) for k, v in r.items() if k != "seed" and k not in values)
        groups.setdefault(key, []).append(r)
    out = []
    for key, group in groups.items():
        row = dict(key)
        for v in values:
            x = np.array([r[v] for r in group], dtype=float)
            row[f"{v}_mean"] = float(np.mean(x))
            row[f"{v}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        row["n_seeds"] = len(group)
        out.append(row)
    return out


def plot_pd(pd_rows, cells: str, title: str):
    fig, axes = new_figure(7.16, 2.9, ncols=3, sharey=True)
    for ax, attack in zip(axes, ("pilot", "data", "partial"), strict=True):
        for name in PLOT_DETECTORS:
            style = DETECTOR_STYLE[name]
            rows = [
                r
                for r in pd_rows
                if r["detector"] == name
                and r["attack"] == attack
                and r["channel"] == "both"
                and r["cells"] == cells
            ]
            if rows:
                x = np.array([r["jsr_db"] for r in rows])
                mean = np.array([r["pd_mean"] for r in rows])
                std = np.array([r["pd_std"] for r in rows])
                ax.plot(x, mean, **style)
                # band: +-1 standard deviation over training seeds
                ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.15, lw=0)
        ax.set(title=f"{attack} jamming", xlabel="JSR per jammed RE [dB]", ylim=(-0.02, 1.02))
    axes[0].set_ylabel(f"Pd at {FA:.0%} per-cell false alarm")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle(title)
    return fig


def _write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate all detectors on a split.")
    splits = [s for s in SPLITS if not s.startswith(("train", "val"))]
    p.add_argument("--split", choices=splits, default="test")
    p.add_argument(
        "--snr-db",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="only evaluate slots with MIN <= SNR < MAX",
    )
    args = p.parse_args(argv)

    data = "dmrs2" if args.split.endswith("_dmrs2") else ""
    seeds = available_seeds(data)
    if not seeds:
        train_cmd = "python -m jamloc.train --variant all"
        if data:
            train_cmd += f" --data {data}"
        raise SystemExit(f"No complete set of trained models. Run {train_cmd} first.")

    val, test = load(f"val_{data}" if data else "val"), load(args.split)
    titles = {
        "test": "test set",
        "sched": "scheduling-aware jammer",
        "test_dmrs2": "test set with two DMRS symbols",
        "sched_dmrs2": "scheduling-aware jammer with two DMRS symbols",
    }
    name, what = args.split, titles.get(args.split) or SPLITS[args.split]["title"]
    if args.snr_db:
        lo, hi = args.snr_db
        keep = (test["snr_db"] >= lo) & (test["snr_db"] < hi)
        test = {k: v[keep] for k, v in test.items()}
        name += f"_snr{lo:g}-{hi:g}"
        what += f", SNR {lo:g}-{hi:g} dB"

    pd_rows, f1_rows = [], []
    for seed in seeds:
        s_val = score_split(val, seed, null=val, data=data)
        s_test = score_split(test, seed, null=val, data=data)
        scores = {det: (s_val[det], s_test[det]) for det in s_val}
        pd_seed, f1_seed = evaluate_scores(scores, val, test)
        pd_rows += [{**r, "seed": seed} for r in pd_seed]
        f1_rows += [{**r, "seed": seed} for r in f1_seed]
    pd_mean = aggregate(pd_rows, "pd")
    f1_mean = aggregate(f1_rows, ("f1", "fa", "fa_scored"))

    out = results_dir() / "eval"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(pd_rows, out / f"{name}_pd_vs_jsr_per_seed.csv")
    _write_csv(f1_rows, out / f"{name}_f1_per_seed.csv")
    _write_csv(pd_mean, out / f"{name}_pd_vs_jsr.csv")
    _write_csv(f1_mean, out / f"{name}_f1.csv")
    for cells in ("all", "scheduled"):
        channels = " and ".join(f"TDL-{CHANNELS[c]}" for c in np.unique(test["channel"]))
        title = f"{what}, {cells} jammed cells, {channels}, {len(seeds)} seed(s)"
        with paper_style():
            fig = plot_pd(pd_mean, cells, title)
        save(fig, out / f"{name}_pd_vs_jsr_{cells}")

    print(f"{name}: seeds {seeds}")
    for r in f1_mean:
        if r["attack"] == "all":
            print(
                f"{r['detector']:20s} F1 {r['f1_mean']:.3f} +- {r['f1_std']:.3f}  "
                f"FA (all cells) {r['fa_mean']:.4f}  FA (scored cells) {r['fa_scored_mean']:.4f}"
            )


if __name__ == "__main__":
    main()
