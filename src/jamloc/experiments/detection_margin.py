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

The margin depends on how much margin the link itself has: MCS 5 decodes without errors at
15 dB SNR, so a jammer must first eat that spare SNR before blocks fail. ``--snr-db`` repeats the
experiment closer to the MCS's working point (splits ``margin_snr<x>``; the BLER curves must
exist at that SNR, ``experiments.ber_bler --snr-db <x>``). Slot alarms are evaluated at 1% and
0.1% slot false alarm (1% of 2000 slots per second is 20 false alarms per second and UE), and
margins are also given per training seed.

With ``--boot N``, each draw resamples the jammed batches within channel, attack and JSR, and
the pooled benign calibration batches separately. Held-out batches estimate false alarm
separately. The same draws are used for every detector, seed and slot false-alarm target.
Pd is averaged over seeds within a draw before taking its crossing. Inference is conditional
on the trained checkpoints and the block test's validation fit.

The saved BLER counts give a binomial plug-in draw at each JSR with the observed number of
blocks fixed. BLER simulations share an allocation across slots in a batch and stop at a
block-error target or batch limit. The CSV totals cannot recover within-allocation correlation
or reproduce that stopping rule, so the binomial approximation can understate BLER uncertainty.

Crossings beyond the simulated grid contribute bounds starting at the relevant grid edge.
The lower 2.5th percentile of draw lower bounds and upper 97.5th percentile of draw upper
bounds give the reported interval, including one-sided limits when one endpoint is infinite.
``censored_fraction`` is the share of draws whose bounds differ. The minimum
also reports the fraction with any censored attack, including attacks that cannot set the minimum.
Bootstrap runs write the CI CSVs separately from the legacy point CSVs. Supply all SNRs in one
run to include independent-SNR differences in ``margin_paired_ci.csv``.
When matching legacy CSVs exist, draw zero uses their observed per-seed Pd curves. The other
draws use newly computed scores, whose numerical precision can differ across devices.

    python -m jamloc.experiments.detection_margin
    python -m jamloc.experiments.detection_margin --snr-db 8
    python -m jamloc.experiments.detection_margin --snr-db 15 11 8 5 --boot 1000
"""

import argparse
import csv

import numpy as np
import torch

from jamloc.baselines import block_test_scores
from jamloc.dataset import CHANNELS, MARGIN_SNRS_DB, SPLITS, generate, load, margin_split
from jamloc.evaluate import FA, _logits, aggregate, available_seeds
from jamloc.jammer import ATTACKS
from jamloc.metrics import threshold_at_fa
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.train import load_model

DETECTORS = {
    "fused": ("U-Net P+R+E", "#CC0000", "o", "-"),
    "no_residual": ("U-Net P+E", "#0033CC", "s", "--"),
    "block_test": ("Block test", "#008000", "d", "-"),
}
TARGET_PD, TARGET_BLER = 0.9, 0.1
SLOT_FAS = (0.01, 0.001)  # slot false-alarm targets
ATTACK_NAMES = ("pilot", "data", "partial")
SNRS = (15.0, *MARGIN_SNRS_DB)
SNR_PAIRS = ((15.0, 11.0), (15.0, 8.0), (8.0, 5.0))


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
    if "batch" in d:
        return d["batch"]
    keys = np.stack(
        [d[k].astype(float) for k in ("channel", "alloc_start", "alloc_size", "attack", "jsr_db")]
    )
    new = np.any(np.diff(keys, axis=1) != 0, axis=0)
    return np.concatenate([[0], np.cumsum(new)])


class SlotBootstrap:
    """Resample whole batches, keeping draw zero at the original batch multiplicities."""

    def __init__(self, d, calib, grid, n_boot, rng):
        batch = _batch_ids(d)
        self.grid = grid
        self.n_draws = n_boot + 1

        def stratum(slots):
            _, group = np.unique(batch[slots], return_inverse=True)
            count = int(group.max()) + 1
            batches = [slots[group == i] for i in range(count)]
            weights = np.ones((self.n_draws, count), dtype=np.int64)
            weights[1:] = rng.multinomial(count, np.full(count, 1 / count), size=n_boot)
            return batches, weights

        if not len(calib):
            raise ValueError("Slot threshold bootstrapping requires benign calibration batches.")
        self.calib = stratum(calib)
        self.jammed = {}
        for i, channel in enumerate(CHANNELS):
            for attack in ATTACK_NAMES:
                sel = (d["channel"] == i) & (d["attack"] == ATTACKS.index(attack))
                for j, jsr in enumerate(grid):
                    slots = np.flatnonzero(sel & (d["jsr_db"] == jsr))
                    if len(slots):
                        self.jammed[channel, attack, j] = stratum(slots)

    def thresholds(self, slot):
        """Weighted version of the higher quantile in ``threshold_at_fa``."""
        batches, weights = self.calib
        values = slot[np.concatenate(batches)]
        batch = np.repeat(np.arange(len(batches)), [len(slots) for slots in batches])
        scored = np.flatnonzero(values > -np.inf)
        order = scored[np.argsort(values[scored])]
        values, batch = values[order], batch[order]
        thresholds = np.full((len(SLOT_FAS), self.n_draws), -np.inf)
        q = 1 - np.asarray(SLOT_FAS)
        for i, draw in enumerate(weights):
            cumulative = np.cumsum(draw[batch])
            if cumulative.size and cumulative[-1]:
                rank = np.ceil((cumulative[-1] - 1) * q).astype(int) + 1
                thresholds[:, i] = values[np.searchsorted(cumulative, rank)]
        return thresholds

    def curves(self, slot):
        """Return slot Pd indexed by false-alarm target, draw and JSR grid point."""
        thresholds = self.thresholds(slot)
        curves = {
            (ch, attack): np.full((*thresholds.shape, len(self.grid)), np.nan)
            for ch in CHANNELS
            for attack in ATTACK_NAMES
        }
        for (ch, attack, j), (batches, weights) in self.jammed.items():
            sizes = np.array([len(slots) for slots in batches])
            # Count scores strictly above each draw's threshold, then weight by batch.
            scores = [np.sort(slot[slots][~np.isnan(slot[slots])]) for slots in batches]
            hits = np.stack(
                [
                    len(values) - np.searchsorted(values, thresholds, side="right")
                    for values in scores
                ],
                axis=-1,
            )
            curves[ch, attack][:, :, j] = (hits * weights).sum(axis=-1) / (weights @ sizes)
        return curves


def run(seeds, snr_db: float = 15.0) -> list[dict]:
    return _evaluate(seeds, snr_db)[0]


def _evaluate(seeds, snr_db, n_boot=0, rng=None):
    split = margin_split(snr_db)
    if not (data_dir() / f"{split}.npz").is_file():
        generate(split)
    d, val = load(split), load("val")
    labels = d["labels"].astype(bool)
    # Benign slots: whole batches (one allocation each) alternate between setting the slot
    # threshold and checking it, so the check uses allocations the threshold never saw
    benign = np.flatnonzero(d["attack"] == 0)
    rank = np.unique(_batch_ids(d)[benign], return_inverse=True)[1]  # 0, 1, ... per benign batch
    calib, check = benign[rank % 2 == 0], benign[rank % 2 == 1]
    grid = SPLITS[split]["jsr_grid"]
    bootstrap = SlotBootstrap(d, calib, grid, n_boot, rng) if n_boot else None
    channels = {"both": np.ones(len(labels), bool)}
    channels.update({ch: d["channel"] == i for i, ch in enumerate(CHANNELS)})

    rows, curves = [], {}
    for name in DETECTORS:
        for seed in [seeds[0]] if name == "block_test" else seeds:
            s = _scores(name, d, seed, val)
            s_val = _scores(name, val, seed, val)
            cell_thr = threshold_at_fa(s_val[val["labels"] == 0], FA)
            slot = s.reshape(len(s), -1).max(1)
            if bootstrap is not None:
                curves[name, seed] = bootstrap.curves(slot)
            for alpha in SLOT_FAS:
                slot_thr = threshold_at_fa(slot[calib], alpha)
                slot_fa = float((slot[check] > slot_thr).mean())
                for channel, ch_sel in channels.items():
                    for attack in (1, 2, 3):
                        for jsr in grid:
                            sel = ch_sel & (d["attack"] == attack) & (d["jsr_db"] == jsr)
                            rows.append(
                                {
                                    "detector": name,
                                    "seed": seed,
                                    "slot_fa_target": alpha,
                                    "channel": channel,
                                    "attack": ATTACKS[attack],
                                    "jsr_db": jsr,
                                    "slot_pd": float((slot[sel] > slot_thr).mean()),
                                    "cell_pd": float((s[sel][labels[sel]] > cell_thr).mean()),
                                    "slot_fa": slot_fa,
                                    "slots": int(sel.sum()),
                                }
                            )
                print(
                    f"{name}, seed {seed}: slot false alarm {slot_fa:.4f} "
                    f"(target {alpha:g}, {len(check)} check slots)",
                    flush=True,
                )
    return rows, curves


def read_bler(channel: str, snr: float = 15.0, *, counts=False) -> list[dict]:
    """BLER curves of the unprotected link at the SNR of the margin split."""
    from jamloc.experiments.ber_bler import path_for

    path = path_for(channel, snr)
    if not path.is_file():
        raise SystemExit(
            f"No BLER curves at {snr:g} dB SNR: run experiments.ber_bler --snr-db {snr:g}"
        )
    with open(path, encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            if r["attack"] == "none" or float(r["snr_db"]) != snr:
                continue
            row = {"attack": r["attack"], "jsr_db": float(r["jsr_db"]), "bler": float(r["bler"])}
            if counts:
                try:
                    row.update({k: int(r[k]) for k in ("block_errors", "blocks")})
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Bootstrap requires integer block counts in {path}") from exc
            rows.append(row)
    if not rows:
        raise SystemExit(
            f"No BLER rows at {snr:g} dB SNR: run experiments.ber_bler --snr-db {snr:g}"
        )
    return rows


def summarize(mean_rows, snr: float = 15.0, suffix: str = "_mean") -> list[dict]:
    """Margins from the seed-averaged curves (``suffix="_mean"``), or from the rows of one seed
    (``suffix=""``, the rows already selected for that seed)."""
    out = []
    for channel in CHANNELS:
        bler = read_bler(channel, snr)
        for attack in ATTACK_NAMES:
            b = sorted((r for r in bler if r["attack"] == attack), key=lambda r: r["jsr_db"])
            jsr_bler = crossing([r["jsr_db"] for r in b], [r["bler"] for r in b], TARGET_BLER)
            for name in DETECTORS:
                for alpha in SLOT_FAS:
                    rows = sorted(
                        (
                            r
                            for r in mean_rows
                            if r["detector"] == name
                            and r["channel"] == channel
                            and r["attack"] == attack
                            and r["slot_fa_target"] == alpha
                        ),
                        key=lambda r: r["jsr_db"],
                    )
                    if not rows:
                        continue
                    x = [r["jsr_db"] for r in rows]
                    jsr_slot = crossing(x, [r[f"slot_pd{suffix}"] for r in rows], TARGET_PD)
                    jsr_cell = crossing(x, [r[f"cell_pd{suffix}"] for r in rows], TARGET_PD)
                    out.append(
                        {
                            "snr_db": snr,
                            "channel": channel,
                            "attack": attack,
                            "detector": name,
                            "slot_fa_target": alpha,
                            "jsr_bler10_db": jsr_bler,
                            "jsr_slot_pd90_db": jsr_slot,
                            "jsr_cell_pd90_db": jsr_cell,
                            "margin_db": jsr_bler - jsr_slot,
                        }
                    )
    return out


def _crossing_samples(x, curves, target):
    """Return the usual crossings and the lower and upper endpoint for each draw."""
    points = np.array([crossing(x, y, target) for y in curves])
    bounds = np.repeat(points[:, None], 2, axis=1)
    bounds[np.isneginf(points), 1] = x[0]
    for i in np.flatnonzero(np.isnan(points)):
        y = curves[i]
        observed = np.flatnonzero(np.isfinite(y))
        reached = observed[y[observed] >= target]
        if reached.size:
            # The previous grid value is missing, so interpolation cannot locate the crossing.
            previous = observed[observed < reached[0]]
            bounds[i] = (x[previous[-1]] if previous.size else -np.inf, x[reached[0]])
        else:
            bounds[i] = (x[observed[-1]] if observed.size else -np.inf, np.inf)
    return points, bounds


def _bler_samples(snr, n_boot, rng):
    samples = {}
    for channel in CHANNELS:
        rows = read_bler(channel, snr, counts=True)
        for attack in ATTACK_NAMES:
            curve = sorted((r for r in rows if r["attack"] == attack), key=lambda r: r["jsr_db"])
            if not curve:
                raise ValueError(f"No BLER grid for TDL-{channel}, {attack}, {snr:g} dB SNR")
            blocks = np.array([r["blocks"] for r in curve])
            errors = np.array([r["block_errors"] for r in curve])
            if np.any((blocks <= 0) | (errors < 0) | (errors > blocks)):
                raise ValueError(f"Invalid BLER counts for TDL-{channel}, {attack}, {snr:g} dB SNR")
            x = [r["jsr_db"] for r in curve]
            original = [r["bler"] for r in curve]
            draws = rng.binomial(blocks, errors / blocks, size=(n_boot, len(x))) / blocks
            samples[channel, attack] = _crossing_samples(
                x, np.vstack((original, draws)), TARGET_BLER
            )
    return samples


def _interval(bounds, name="margin"):
    """Use outward order statistics to retain censored and infinite endpoints."""
    lo, hi = bounds[1:].T
    return {
        "boot_draws": len(lo),
        f"{name}_lo_db": np.quantile(lo, 0.025, method="lower"),
        f"{name}_hi_db": np.quantile(hi, 0.975, method="higher"),
        "censored_fraction": np.mean(lo != hi),
        "point_lo_db": bounds[0, 0],
        "point_hi_db": bounds[0, 1],
    }


def _margin_key(row):
    return row["channel"], row["attack"], row["detector"], row["slot_fa_target"]


def _per_seed_summary(rows, snr):
    summary = []
    for seed in sorted({r["seed"] for r in rows}):
        one_seed = [r for r in rows if r["seed"] == seed]
        summary.extend({**r, "seed": int(seed)} for r in summarize(one_seed, snr, suffix=""))
    return summary


def _margin_samples(snr, summary, per_seed, curves, bler):
    grid = SPLITS[margin_split(snr)]["jsr_grid"]
    ci, mean_margins, mean_pd_bounds = [], {}, {}
    for row in [{**r, "seed": "mean"} for r in summary] + per_seed:
        channel, attack, name, alpha = key = _margin_key(row)
        seed = row["seed"]
        use = [
            c[channel, attack][SLOT_FAS.index(alpha)]
            for (detector, s), c in curves.items()
            if detector == name and (seed == "mean" or s == seed)
        ]
        pd_point, pd_bounds = _crossing_samples(grid, np.mean(use, axis=0), TARGET_PD)
        bler_point, bler_bounds = bler[channel, attack]
        with np.errstate(invalid="ignore"):
            margin_point = bler_point[0] - pd_point[0]
        np.testing.assert_array_equal(
            [bler_point[0], pd_point[0], margin_point],
            [row[k] for k in ("jsr_bler10_db", "jsr_slot_pd90_db", "margin_db")],
            err_msg=f"Draw zero disagrees with the legacy point calculation for {key}, seed {seed}",
        )
        bounds = bler_bounds - pd_bounds[:, ::-1]
        ci.append({**row, **_interval(bounds)})
        if seed == "mean":
            mean_margins[key] = bounds
            mean_pd_bounds[key] = pd_bounds
    return ci, mean_margins, mean_pd_bounds


def _minimum_samples(summary, margins):
    """Take the minimum over attacks within each draw, including censored attacks."""
    minima, rows = {}, []
    points = {_margin_key(r): r["margin_db"] for r in summary}
    for channel in CHANNELS:
        for name in DETECTORS:
            for alpha in SLOT_FAS:
                keys = [(channel, attack, name, alpha) for attack in ATTACK_NAMES]
                finite = [points[k] for k in keys if np.isfinite(points[k])]
                point = min(finite, default=np.nan)
                bounds = np.stack([margins[k] for k in keys])
                any_censored = np.any(bounds[:, :, 0] != bounds[:, :, 1], axis=0)
                minimum = bounds.min(axis=0)
                minima[channel, name, alpha] = point, minimum
                rows.append(
                    {
                        "snr_db": summary[0]["snr_db"],
                        "channel": channel,
                        "detector": name,
                        "slot_fa_target": alpha,
                        "margin_db": point,
                        **_interval(minimum),
                        "point_any_attack_censored": bool(any_censored[0]),
                        "any_attack_censored_fraction": any_censored[1:].mean(),
                    }
                )
    return minima, rows


def _difference(channel, attack, comparison, lhs, rhs, points, bounds, *, nonnegative=False):
    left_snr, left_name, left_fa = lhs
    right_snr, right_name, right_fa = rhs
    left, right = bounds
    delta = left - right[:, ::-1]
    if nonnegative:
        # The relaxed slot threshold is always at most the strict one on the same calibration draw.
        delta[:, 0] = np.maximum(0, delta[:, 0])
    with np.errstate(invalid="ignore"):
        point = np.subtract(*points)
    return {
        "channel": channel,
        "attack": attack,
        "comparison": comparison,
        "left_snr_db": left_snr,
        "right_snr_db": right_snr,
        "left_detector": left_name,
        "right_detector": right_name,
        "left_slot_fa_target": left_fa,
        "right_slot_fa_target": right_fa,
        "delta_db": point,
        **_interval(delta, "delta"),
    }


def _paired_rows(snr, summary, minima, pd_bounds):
    rows = []
    pd_points = {_margin_key(r): r["jsr_slot_pd90_db"] for r in summary}
    pairs = [
        ("detector", "fused", alpha, other, alpha)
        for alpha in SLOT_FAS
        for other in ("no_residual", "block_test")
    ] + [("slot_fa", name, SLOT_FAS[0], name, SLOT_FAS[1]) for name in DETECTORS]
    for channel in CHANNELS:
        for comparison, name, alpha, other, beta in pairs:
            lhs, rhs = (snr, name, alpha), (snr, other, beta)
            for attack in ATTACK_NAMES:
                a, b = (channel, attack, name, alpha), (channel, attack, other, beta)
                # For a single attack the common BLER crossing cancels, even if censored.
                rows.append(
                    _difference(
                        channel,
                        attack,
                        comparison,
                        lhs,
                        rhs,
                        (pd_points[b], pd_points[a]),
                        (pd_bounds[b], pd_bounds[a]),
                        nonnegative=comparison == "slot_fa",
                    )
                )
            a, b = minima[channel, name, alpha], minima[channel, other, beta]
            rows.append(
                _difference(
                    channel,
                    "minimum",
                    comparison,
                    lhs,
                    rhs,
                    (a[0], b[0]),
                    (a[1], b[1]),
                    nonnegative=comparison == "slot_fa",
                )
            )
    return rows


def _independent_rows(by_snr):
    rows = []
    for hi, lo in SNR_PAIRS:
        if hi not in by_snr or lo not in by_snr:
            continue
        for (channel, name, alpha), (point, bounds) in by_snr[hi].items():
            other, other_bounds = by_snr[lo][channel, name, alpha]
            rows.append(
                _difference(
                    channel,
                    "minimum",
                    "snr",
                    (hi, name, alpha),
                    (lo, name, alpha),
                    (point, other),
                    (bounds, other_bounds),
                )
            )
    return rows


def plot(mean_rows, snr: float = 15.0):
    mean_rows = [r for r in mean_rows if r["slot_fa_target"] == FA]
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
                (r for r in read_bler(channel, snr) if r["attack"] == attack),
                key=lambda r: r["jsr_db"],
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
            xlim=(min(r["jsr_db"] for r in mean_rows), 5),
            ylim=(-0.02, 1.02),
        )
    axes[0].set_ylabel("probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle(
        f"Jammer on the user's PRBs, SNR {snr:g} dB: detection (1% slot false alarm) vs damage"
    )
    return fig


def plot_compact(mean_rows, channel: str = "C", snr: float = 15.0):
    """One column for the paper: slot Pd of the fused U-Net (solid) and BLER without protection
    (dashed) per attack, on one channel."""
    from jamloc.plotting import ATTACK_STYLE

    mean_rows = [r for r in mean_rows if r["slot_fa_target"] == FA]
    fig, ax = new_figure(3.5, 2.8)
    bler = read_bler(channel, snr)
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
        xlim=(min(r["jsr_db"] for r in mean_rows), 5),
        ylim=(-0.02, 1.02),
        title=f"Jammer on the user's PRBs, TDL-{channel}, SNR {snr:g} dB",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=6)
    return fig


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _saved_summary(path):
    with path.open(encoding="utf-8", newline="") as f:
        return [
            {
                k: v
                if k in ("channel", "attack", "detector")
                else int(v)
                if k == "seed"
                else float(v)
                for k, v in row.items()
            }
            for row in csv.DictReader(f)
        ]


def _use_saved_pd(path, rows, curves, snr):
    """Keep the original curves for draw zero and diagnose any rescoring differences."""
    grid = SPLITS[margin_split(snr)]["jsr_grid"]

    def key(row):
        return (
            row["detector"],
            int(row["seed"]),
            row["channel"],
            row["attack"],
            float(row["slot_fa_target"]),
            float(row["jsr_db"]),
        )

    with path.open(encoding="utf-8", newline="") as f:
        saved = {
            key(r): (float(r["slot_pd"]), int(r["slots"]))
            for r in csv.DictReader(f)
            if r["channel"] in CHANNELS
        }
    live = {key(r): r for r in rows if r["channel"] in CHANNELS}
    if saved.keys() != live.keys() or any(saved[k][1] != r["slots"] for k, r in live.items()):
        raise ValueError(f"Saved slot Pd grids or slot counts differ from the evaluation in {path}")

    differences = []
    for k, (observed, _) in saved.items():
        name, seed, channel, attack, alpha, jsr = k
        pd = curves[name, seed][channel, attack]
        a, j = SLOT_FAS.index(alpha), grid.index(jsr)
        rerun = live[k]["slot_pd"]
        np.testing.assert_array_equal(pd[a, 0, j], rerun)
        pd[a, 0, j] = observed
        if np.isfinite(observed) and np.isfinite(rerun) and observed != rerun:
            differences.append(abs(observed - rerun))
    if differences:
        print(
            f"{snr:g} dB SNR: rescoring differs from saved Pd at "
            f"{len(differences)}/{len(live)} per-seed grid values "
            f"(largest Pd difference {max(differences):.4g}); draw zero uses saved Pd",
            flush=True,
        )


def _bootstrap_run(out, snrs, seeds, n_boot):
    paired, smallest, by_snr = [], [], {}
    for snr in snrs:
        # Anchor both streams to the SNR so draws agree with a standalone run at that SNR.
        slot_rng, bler_rng = map(
            np.random.default_rng, np.random.SeedSequence([0, SNRS.index(snr)]).spawn(2)
        )
        bler = _bler_samples(snr, n_boot, bler_rng)
        rows, curves = _evaluate(seeds, snr, n_boot, slot_rng)
        summary, per_seed = [], []
        stem = margin_split(snr)
        legacy = out / f"{stem}_summary.csv"
        legacy_per_seed = out / f"{stem}_summary_per_seed.csv"
        legacy_pd = out / f"{stem}_per_seed.csv"
        if legacy.is_file() and legacy_per_seed.is_file() and legacy_pd.is_file():
            saved, saved_per_seed = _saved_summary(legacy), _saved_summary(legacy_per_seed)
            if saved and {(r["detector"], r["seed"]) for r in saved_per_seed} == set(curves):
                _use_saved_pd(legacy_pd, rows, curves, snr)
                summary, per_seed = saved, saved_per_seed
        if not summary:
            summary = summarize(aggregate(rows, ("slot_pd", "cell_pd", "slot_fa")), snr)
            per_seed = _per_seed_summary(rows, snr)
        ci, margins, pd_bounds = _margin_samples(snr, summary, per_seed, curves, bler)
        _write(ci, out / f"{stem}_summary_ci.csv")
        minima, minimum_rows = _minimum_samples(summary, margins)
        by_snr[snr] = minima
        smallest.extend(minimum_rows)
        paired.extend(_paired_rows(snr, summary, minima, pd_bounds))
        print(f"Wrote {stem}_summary_ci.csv ({n_boot} draws; verified draw zero)", flush=True)
    _write(smallest, out / "margin_vs_snr_ci.csv")
    _write(paired + _independent_rows(by_snr), out / "margin_paired_ci.csv")
    print(f"Wrote {out}/margin_vs_snr_ci.csv and margin_paired_ci.csv", flush=True)


def benign_snr_at_bler(channel: str, target: float = TARGET_BLER) -> float:
    """SNR at which the unjammed link reaches ``target`` BLER (``ber_bler --benign``)."""
    path = results_dir() / "ber_bler" / f"benign_{channel}.csv"
    with open(path, encoding="utf-8") as f:
        rows = sorted(
            ((float(r["snr_db"]), float(r["bler"])) for r in csv.DictReader(f)),
            key=lambda r: r[0],
        )
    # BLER falls with SNR: the crossing of 1 - BLER with 1 - target
    return crossing([s for s, _ in rows], [1 - b for _, b in rows], 1 - target)


def plot_vs_snr(detector: str = "fused"):
    """Margin of the smallest-margin attack against the SNR, for both channels and both slot
    false-alarm targets, with the link's own spare margin (SNR minus the SNR of 10% BLER without
    a jammer) on a second axis. One column, for the paper."""
    fig, ax = new_figure(3.5, 2.6)
    out = results_dir() / "margin"
    ci_file = out / "margin_vs_snr_ci.csv"
    intervals = {}
    if ci_file.is_file():
        with ci_file.open(encoding="utf-8", newline="") as f:
            intervals = {
                (float(r["snr_db"]), r["channel"], float(r["slot_fa_target"])): r
                for r in csv.DictReader(f)
                if r["detector"] == detector
            }
    snrs = sorted(SNRS)
    styles = {"C": ("#CC0000", "o"), "A": ("#0033CC", "s")}
    for channel, (color, marker) in styles.items():
        for alpha, ls in ((FA, "-"), (0.001, "--")):
            ys, censored = [], []
            for snr in snrs:
                path = out / f"{margin_split(snr)}_summary.csv"
                r = intervals.get((snr, channel, alpha))
                if not path.is_file():
                    ys.append(np.nan if r is None else float(r["margin_db"]))
                    censored.append(r is not None and r["point_any_attack_censored"] == "True")
                    continue
                with open(path, encoding="utf-8") as f:
                    margins = [
                        float(r["margin_db"])
                        for r in csv.DictReader(f)
                        if r["detector"] == detector
                        and r["channel"] == channel
                        and float(r["slot_fa_target"]) == alpha
                    ]
                # An attack that never reaches 90% Pd within the JSR grid has a margin below
                # the grid; the smallest margin is then at most the smallest finite one (open
                # marker)
                finite = [m for m in margins if np.isfinite(m)]
                ys.append(min(finite) if finite else np.nan)
                censored.append(len(finite) < len(margins))
            label = f"TDL-{channel}, {alpha:.1%} false alarm".replace(".0%", "%")
            ax.plot(snrs, ys, color=color, ls=ls, label=label)
            for x, y, c in zip(snrs, ys, censored, strict=True):
                r = intervals.get((x, channel, alpha))
                if r is not None and np.isclose(float(r["margin_db"]), y, equal_nan=True):
                    lo, hi = float(r["margin_lo_db"]), float(r["margin_hi_db"])
                    if np.isfinite(lo) and np.isfinite(hi):
                        ax.errorbar(
                            x, (lo + hi) / 2, yerr=(hi - lo) / 2, color=color, lw=0.8, capsize=2
                        )
                    elif np.isfinite(hi):
                        ax.errorbar(x, hi, yerr=0.7, uplims=True, color=color, lw=0.8, capsize=2)
                    elif np.isfinite(lo):
                        ax.errorbar(x, lo, yerr=0.7, lolims=True, color=color, lw=0.8, capsize=2)
                face = "white" if c else color
                ax.plot(x, y, marker=marker, color=color, markerfacecolor=face, ls="none")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="SNR (dB)", ylabel="Smallest margin (dB)")
    spare = ax.secondary_xaxis(
        "top",
        functions=(
            lambda s: s - benign_snr_at_bler("C"),
            lambda m: m + benign_snr_at_bler("C"),
        ),
    )
    spare.set_xlabel("Spare link margin, TDL-C (dB)")
    ax.legend(loc="lower right", fontsize=6.5)
    return fig


def _legacy_run(out, snr, seeds, plot_only):
    tag = "" if snr == 15.0 else f"_snr{snr:g}"
    text = ("detector", "channel", "attack")
    if not plot_only:
        rows = run(seeds, snr)
        _write(rows, out / f"margin{tag}_per_seed.csv")
        _write(aggregate(rows, ("slot_pd", "cell_pd", "slot_fa")), out / f"margin{tag}.csv")
    with open(out / f"margin{tag}.csv", encoding="utf-8", newline="") as f:
        mean_rows = [
            {k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)
        ]
    with open(out / f"margin{tag}_per_seed.csv", encoding="utf-8", newline="") as f:
        seed_rows = [
            {k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)
        ]
    summary = summarize(mean_rows, snr)
    _write(summary, out / f"margin{tag}_summary.csv")
    _write(_per_seed_summary(seed_rows, snr), out / f"margin{tag}_summary_per_seed.csv")
    with paper_style():
        fig = plot(mean_rows, snr)
        compact = plot_compact(mean_rows, snr=snr)
    save(fig, out / f"margin{tag}")
    save(compact, out / f"margin{tag}_compact")
    for r in summary:
        if r["slot_fa_target"] != FA:
            continue
        print(
            f"TDL-{r['channel']} {r['attack']:8s} {r['detector']:12s} "
            f"BLER 10% at {r['jsr_bler10_db']:6.1f} dB, slot Pd 90% at "
            f"{r['jsr_slot_pd90_db']:6.1f} dB, cell Pd 90% at {r['jsr_cell_pd90_db']:6.1f} dB: "
            f"margin {r['margin_db']:5.1f} dB"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: all trained")
    p.add_argument("--snr-db", type=float, nargs="+", default=[15.0])
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--vs-snr", action="store_true", help="only the margin-vs-SNR figure")
    p.add_argument("--boot", type=int, default=0, help="number of bootstrap draws (0 disables)")
    args = p.parse_args(argv)
    if args.boot < 0 or (args.boot and (args.plot_only or args.vs_snr)):
        p.error("--boot requires a positive draw count for an evaluation, without plot-only flags")

    out = results_dir() / "margin"
    if args.vs_snr:
        with paper_style():
            fig = plot_vs_snr()
        save(fig, out / "margin_vs_snr")
        print(f"Wrote {out}/margin_vs_snr.pdf/.png")
        return
    out.mkdir(parents=True, exist_ok=True)
    snrs = list(dict.fromkeys(args.snr_db))
    seeds = args.seeds or ([] if args.plot_only else available_seeds())
    if args.boot:
        if any(snr not in SNRS for snr in snrs):
            p.error(f"--boot requires SNRs drawn from {SNRS}")
        _bootstrap_run(out, snrs, seeds, args.boot)
        return
    for snr in snrs:
        _legacy_run(out, snr, seeds, args.plot_only)


if __name__ == "__main__":
    main()
