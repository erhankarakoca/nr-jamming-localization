"""Detection before damage: does the map see a jammer before it hurts the link?

The BLER experiment (``experiments/ber_bler``) gives the block error rate of the unprotected link
when the jammer covers the user's PRBs at 15 dB SNR. That is all a detector built on CRC failures
or link KPIs can see. Here the same attacks (``margin`` split: jammer on the scheduled PRBs,
SNR 15 dB, JSR -30 ... 0 dB) are given to the detectors. A slot is declared jammed when its
largest cell score exceeds the level that 1% of benign slots exceed (half of the split's benign
slots set that level, the other half check it).

The *detection margin* of an attack is how far below the JSR of 10% BLER the slot-level Pd
reaches 90%: the range of jamming powers in which the attack is already seen but does not yet
hurt. A negative margin would mean the attacker can damage the link unseen.

    python -m jamloc.experiments.detection_margin
"""

import argparse
import csv

import numpy as np
import torch

from jamloc.baselines import block_test_scores
from jamloc.dataset import CHANNELS, SPLITS, generate, load
from jamloc.evaluate import FA, _logits, aggregate, available_seeds
from jamloc.jammer import ATTACKS
from jamloc.metrics import threshold_at_fa
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.train import load_model

DETECTORS = {
    "fused": ("U-Net, P+R+E", "#0072B2", "o", "-"),
    "no_residual": ("U-Net, P+E", "#56B4E9", "v", "--"),
    "block_test": ("Block test (no training)", "#882255", "P", "-"),
}
TARGET_PD, TARGET_BLER = 0.9, 0.1


def _scores(name: str, d: dict, seed: int, val: dict) -> np.ndarray:
    if name == "block_test":
        return block_test_scores(d, val)
    return _logits(load_model(name, seed), name, torch.from_numpy(d["features"]))


def crossing(x, y, target: float) -> float:
    """Smallest ``x`` at which ``y`` reaches ``target`` (linear interpolation).

    ``-inf`` if ``y`` is already there at the first point, ``nan`` if it never gets there.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    above = np.flatnonzero(y >= target)
    if not len(above):
        return float("nan")
    i = above[0]
    if i == 0:
        return float("-inf")
    return float(x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def _batch_ids(d: dict) -> np.ndarray:
    """Index of the simulated batch of every slot (consecutive slots of one batch share their
    channel model, allocation, attack and JSR)."""
    keys = np.stack(
        [d[k].astype(float) for k in ("channel", "alloc_start", "alloc_size", "attack", "jsr_db")]
    )
    new = np.any(np.diff(keys, axis=1) != 0, axis=0)
    return np.concatenate([[0], np.cumsum(new)])


def run(seeds) -> list[dict]:
    if not (data_dir() / "margin.npz").is_file():
        generate("margin")
    d, val = load("margin"), load("val")
    labels = d["labels"].astype(bool)
    # Benign slots: whole batches (one allocation each) alternate between setting the slot
    # threshold and checking it, so the check uses allocations the threshold never saw
    benign = np.flatnonzero(d["attack"] == 0)
    batch = _batch_ids(d)[benign]
    calib, check = benign[batch % 2 == 0], benign[batch % 2 == 1]
    channels = {"both": np.ones(len(labels), bool)}
    channels.update({ch: d["channel"] == i for i, ch in enumerate(CHANNELS)})

    rows = []
    for name in DETECTORS:
        for seed in [seeds[0]] if name == "block_test" else seeds:
            s = _scores(name, d, seed, val)
            s_val = _scores(name, val, seed, val)
            cell_thr = threshold_at_fa(s_val[val["labels"] == 0], FA)
            slot = s.reshape(len(s), -1).max(1)
            slot_thr = threshold_at_fa(slot[calib], FA)
            slot_fa = float((slot[check] > slot_thr).mean())
            for channel, ch_sel in channels.items():
                for attack in (1, 2, 3):
                    for jsr in SPLITS["margin"]["jsr_grid"]:
                        sel = ch_sel & (d["attack"] == attack) & (d["jsr_db"] == jsr)
                        rows.append(
                            {
                                "detector": name,
                                "seed": seed,
                                "channel": channel,
                                "attack": ATTACKS[attack],
                                "jsr_db": jsr,
                                "slot_pd": float((slot[sel] > slot_thr).mean()),
                                "cell_pd": float((s[sel][labels[sel]] > cell_thr).mean()),
                                "slot_fa": slot_fa,
                                "slots": int(sel.sum()),
                            }
                        )
            print(f"{name}, seed {seed}: slot false alarm {slot_fa:.4f}", flush=True)
    return rows


def read_bler(channel: str) -> list[dict]:
    """BLER curves of the unprotected link at the SNR of the ``margin`` split (an error if the
    BLER experiment was last run at another SNR)."""
    snr = SPLITS["margin"]["snr_db"]
    with open(results_dir() / "ber_bler" / f"ber_bler_{channel}.csv", encoding="utf-8") as f:
        rows = [
            {"attack": r["attack"], "jsr_db": float(r["jsr_db"]), "bler": float(r["bler"])}
            for r in csv.DictReader(f)
            if r["attack"] != "none" and float(r["snr_db"]) == snr
        ]
    if not rows:
        raise SystemExit(
            f"No BLER rows at {snr:g} dB SNR: run experiments.ber_bler --snr-db {snr:g}"
        )
    return rows


def summarize(mean_rows) -> list[dict]:
    out = []
    for channel in CHANNELS:
        bler = read_bler(channel)
        for attack in ("pilot", "data", "partial"):
            b = sorted((r for r in bler if r["attack"] == attack), key=lambda r: r["jsr_db"])
            jsr_bler = crossing([r["jsr_db"] for r in b], [r["bler"] for r in b], TARGET_BLER)
            for name in DETECTORS:
                rows = sorted(
                    (
                        r
                        for r in mean_rows
                        if r["detector"] == name
                        and r["channel"] == channel
                        and r["attack"] == attack
                    ),
                    key=lambda r: r["jsr_db"],
                )
                x = [r["jsr_db"] for r in rows]
                jsr_slot = crossing(x, [r["slot_pd_mean"] for r in rows], TARGET_PD)
                jsr_cell = crossing(x, [r["cell_pd_mean"] for r in rows], TARGET_PD)
                out.append(
                    {
                        "channel": channel,
                        "attack": attack,
                        "detector": name,
                        "jsr_bler10_db": jsr_bler,
                        "jsr_slot_pd90_db": jsr_slot,
                        "jsr_cell_pd90_db": jsr_cell,
                        "margin_db": jsr_bler - jsr_slot,
                    }
                )
    return out


def plot(mean_rows):
    fig, axes = new_figure(7.16, 2.6, ncols=3, sharey=True)
    for ax, attack in zip(axes, ("pilot", "data", "partial"), strict=True):
        for name, (label, color, marker, ls) in DETECTORS.items():
            rows = sorted(
                (
                    r
                    for r in mean_rows
                    if r["detector"] == name and r["channel"] == "both" and r["attack"] == attack
                ),
                key=lambda r: r["jsr_db"],
            )
            x = [r["jsr_db"] for r in rows]
            ax.plot(
                x,
                [r["slot_pd_mean"] for r in rows],
                color=color,
                marker=marker,
                ls=ls,
                label=f"slot Pd, {label}",
            )
            if name == "fused":
                ax.plot(
                    x,
                    [r["cell_pd_mean"] for r in rows],
                    color=color,
                    ls=":",
                    lw=1.0,
                    label=f"cell Pd, {label}",
                )
        for channel, color in (("A", "#555555"), ("C", "#000000")):
            b = sorted(
                (r for r in read_bler(channel) if r["attack"] == attack), key=lambda r: r["jsr_db"]
            )
            ax.plot(
                [r["jsr_db"] for r in b],
                [r["bler"] for r in b],
                color=color,
                ls=(0, (1, 1)) if channel == "A" else "-",
                lw=1.6,
                label=f"BLER without protection, TDL-{channel}",
            )
        ax.set(
            title=f"{attack} jamming",
            xlabel="JSR per jammed RE [dB]",
            xlim=(-30, 5),
            ylim=(-0.02, 1.02),
        )
    axes[0].set_ylabel("probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("Jammer on the user's PRBs, SNR 15 dB: detection (1% false alarm) vs damage")
    return fig


def plot_compact(mean_rows, channel: str = "C"):
    """One column for the paper: slot Pd of the fused U-Net (solid) and BLER without protection
    (dashed) per attack, on one channel."""
    from jamloc.plotting import ATTACK_STYLE

    fig, ax = new_figure(3.5, 2.8)
    bler = read_bler(channel)
    for attack in ("pilot", "data", "partial"):
        style = ATTACK_STYLE[attack]
        rows = sorted(
            (
                r
                for r in mean_rows
                if r["detector"] == "fused" and r["channel"] == channel and r["attack"] == attack
            ),
            key=lambda r: r["jsr_db"],
        )
        ax.plot(
            [r["jsr_db"] for r in rows],
            [r["slot_pd_mean"] for r in rows],
            color=style["color"],
            marker=style["marker"],
            ls="-",
            label=f"{attack}, slot $P_d$",
        )
        b = sorted((r for r in bler if r["attack"] == attack), key=lambda r: r["jsr_db"])
        ax.plot(
            [r["jsr_db"] for r in b],
            [r["bler"] for r in b],
            color=style["color"],
            ls="--",
            lw=1.0,
            label=f"{attack}, BLER",
        )
    ax.set(
        xlabel="JSR per jammed RE [dB]",
        ylabel="probability",
        xlim=(-30, 5),
        ylim=(-0.02, 1.02),
        title=f"Jammer on the user's PRBs, TDL-{channel}, SNR 15 dB",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=6)
    return fig


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all trained")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)

    out = results_dir() / "margin"
    out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        rows = run(args.seeds or available_seeds())
        _write(rows, out / "margin_per_seed.csv")
        _write(aggregate(rows, ("slot_pd", "cell_pd", "slot_fa")), out / "margin.csv")
    with open(out / "margin.csv", encoding="utf-8", newline="") as f:
        text = ("detector", "channel", "attack")
        mean_rows = [
            {k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)
        ]
    summary = summarize(mean_rows)
    _write(summary, out / "margin_summary.csv")
    with paper_style():
        fig = plot(mean_rows)
        compact = plot_compact(mean_rows)
    save(fig, out / "margin")
    save(compact, out / "margin_compact")
    for r in summary:
        print(
            f"TDL-{r['channel']} {r['attack']:8s} {r['detector']:12s} "
            f"BLER 10% at {r['jsr_bler10_db']:6.1f} dB, slot Pd 90% at "
            f"{r['jsr_slot_pd90_db']:6.1f} dB, cell Pd 90% at {r['jsr_cell_pd90_db']:6.1f} dB: "
            f"margin {r['margin_db']:5.1f} dB"
        )


if __name__ == "__main__":
    main()
