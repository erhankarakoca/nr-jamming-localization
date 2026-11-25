"""Figures that explain *why* things behave as they do, each with a caption.

Every figure function returns ``(figure, caption)``. The caption says what the figure shows, how
to read it and why it matters; numbers quoted in a caption are computed from the same data as
the figure, so they cannot drift apart.

    python -m jamloc.experiments.figures                  # all figures + gallery
    python -m jamloc.experiments.figures --only constellations

Output: ``results/figures/{name}.pdf`` and ``.png``, ``captions.json`` and ``index.html``.
"""

import argparse
import csv
import html
import json

import numpy as np
import torch
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from sionna.phy import config

from jamloc.baselines import evm_scores
from jamloc.config import SimConfig
from jamloc.dataset import load
from jamloc.evaluate import FA
from jamloc.features import FEATURE_NAMES, delay_subspace, dmrs_ls, power_map, residual_map
from jamloc.jammer import ATTACKS, cell_mask
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.metrics import threshold_at_fa
from jamloc.model import VARIANTS, select_inputs
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import COLUMN_WIDTH, PARULA, new_figure, paper_style, save
from jamloc.receiver import data_positions
from jamloc.theory import threshold
from jamloc.train import checkpoint_path, load_model

CFG = SimConfig()
ALLOC = (6, 12)  # example allocation used by the explanatory figures: PRBs 6-17
JAM = (10, 4)  # example jammer block: PRBs 10-13 (inside the allocation)

# Colours (Okabe-Ito) shared by the explanatory figures
# Resource-grid colors in the plain style of NR figures: gray allocation, dark DMRS, red jamming
ALLOC_COLOR, DMRS_COLOR, JAM_COLOR = "#D9D9D9", "#1F3F8F", "#CC0000"
CLEAN_COLOR = "#0033CC"


def _example_batch(
    attack: str, jsr_db: float, snr_db: float, channel: str, seed: int = 7, decode: bool = False
):
    """One slot with the example allocation and jammer block.

    Re-seeding before every call gives the same UE channel and noise for every attack, so the
    figures compare attacks on otherwise identical slots.
    """
    seed_everything(seed)
    sc = Scenario(
        channel=channel,
        alloc_start=ALLOC[0],
        alloc_size=ALLOC[1],
        attack=torch.tensor([ATTACKS.index(attack)]),
        jam_start=torch.tensor([JAM[0]]),
        jam_size=torch.tensor([JAM[1]]),
        jsr_db=torch.tensor([jsr_db]),
        snr_db=torch.tensor([snr_db]),
    )
    return JammedPUSCHLink(CFG)(sc, decode=decode)


def _fmt_db(x: float) -> str:
    """Rounded dB value for captions, without a misleading "-0"."""
    if not np.isfinite(x):
        return "n/a"
    v = round(float(x))
    return f"{v} dB" if v else "0 dB"


def _block_power_ok(batch, max_drop_db: float = 3.0) -> bool:
    """True if no PRB of the example jammer block (PRBs 10-13) is in a deep fade: each receives at
    least the allocation's mean power minus ``max_drop_db``."""
    p_db = 10 * np.log10(power_map(batch.y, batch.no, CFG)[0].mean(dim=1).cpu().numpy())
    alloc = p_db[ALLOC[0] : ALLOC[0] + ALLOC[1]]
    block = p_db[JAM[0] : JAM[0] + JAM[1]]
    return bool(block.min() >= 10 * np.log10(np.mean(10 ** (alloc / 10))) - max_drop_db)


def _example_seed(channel: str) -> int:
    """First seed whose slot has no deep fade on the jammer block.

    The explanatory figures compare attacks on one slot; a fade on PRBs 10-13 would make even the
    clean slot look jammed. The seed is chosen by this rule when the figure is drawn, not by hand,
    because CPU and GPU draw different channels for the same seed.
    """
    for seed in range(200):
        if _block_power_ok(_example_batch("none", 0.0, 20.0, channel, seed)):
            return seed
    return 0


# --- 1. What the attacker does ---------------------------------------------------------------


# The jammer does not know the allocation, so its block (2-12 PRBs) lands anywhere: blocks of
# different widths inside the allocation, across its edges, or (scheduling-aware) exactly on it.
PATTERN_EXAMPLES = (
    ("Pilot", "pilot", 10, 4),
    ("Data", "data", 15, 6),
    ("Partial-band", "partial", 2, 8),
    ("All pilots", "pilot", ALLOC[0], ALLOC[1]),
)
EMPTY_COLOR = "#FFFFFF"


def _cell_grid(ax, codes, colors):
    """Draw a PRB x symbol grid of category codes as separate cells with thin white gaps."""
    ax.pcolormesh(
        np.arange(CFG.num_symbols + 1) - 0.5,
        np.arange(CFG.num_prb + 1) - 0.5,
        codes,
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=len(colors) - 0.5,
        edgecolors="#B3B3B3",
        linewidth=0.2,
    )
    ax.set(
        xlim=(-0.5, CFG.num_symbols - 0.5),
        ylim=(-0.5, CFG.num_prb - 0.5),
        xticks=[0, CFG.dmrs_symbol, CFG.num_symbols - 1],
        yticks=[0, 6, 12, 18, CFG.num_prb - 1],
    )
    ax.grid(False)
    ax.tick_params(direction="out", length=2, width=0.6, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def attack_patterns():
    """The attack patterns on one slot, one row of grids in the style of the slot figure."""
    width, height = 3.5, 1.72
    fig = Figure(figsize=(width, height))
    colors = [EMPTY_COLOR, ALLOC_COLOR, DMRS_COLOR, JAM_COLOR]  # empty, data, DMRS, attacked
    data_symbols = [s for s in range(CFG.num_symbols) if s != CFG.dmrs_symbol]
    for i, (name, attack, start, size) in enumerate(PATTERN_EXAMPLES):
        ax = fig.add_axes(((0.32 + 0.8 * i) / width, 0.66 / height, 0.7 / width, 1.0 / height))
        codes = np.zeros((CFG.num_prb, CFG.num_symbols), dtype=int)
        codes[ALLOC[0] : sum(ALLOC)] = 1
        codes[ALLOC[0] : sum(ALLOC), CFG.dmrs_symbol] = 2
        attacked = cell_mask(
            torch.tensor([ATTACKS.index(attack)]), torch.tensor([start]), torch.tensor([size]), CFG
        )[0].numpy()
        codes[attacked] = 3
        _cell_grid(ax, codes, colors)
        if attack == "pilot":
            # Unjammed data cells of the attacked, scheduled PRBs: equalized with a corrupted
            # channel estimate, so their EVM rises although their label stays 0.
            low, high = max(start, ALLOC[0]), min(start + size, sum(ALLOC))
            for first, last in ((data_symbols[0], CFG.dmrs_symbol - 1), (CFG.dmrs_symbol + 1, 13)):
                ax.add_patch(
                    Rectangle(
                        (first - 0.5, low - 0.5),
                        last - first + 1,
                        high - low,
                        facecolor="none",
                        edgecolor="#000000",
                        hatch="//////",
                        linewidth=0,
                        zorder=3,
                    )
                )
        ax.tick_params(direction="in", length=2, width=0.5, pad=1.5)
        ax.set_xlabel("Symbol", labelpad=1)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("PRB", labelpad=1)
        box = ax.get_position()
        fig.text(
            (box.x0 + box.x1) / 2, 0.24 / height, f"({'abcd'[i]}) {name}", ha="center", fontsize=7.5
        )
    legend = [
        Patch(facecolor=ALLOC_COLOR, label="Allocation"),
        Patch(facecolor=DMRS_COLOR, label="DMRS"),
        Patch(facecolor=JAM_COLOR, label="Jammed"),
        Patch(facecolor="white", edgecolor="#000000", hatch="//////", lw=0.4, label="EVM affected"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.0),
        ncols=4,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    caption = (
        "Attack patterns on a slot with the UE on PRBs 6-17 and the DMRS on symbol 2: (a) pilot "
        "jamming on 4 PRBs, (b) data jamming on 6 PRBs, (c) partial-band jamming on 8 PRBs, (d) "
        "pilot jamming of the whole allocation. Hatched: unjammed cells whose EVM rises."
    )
    return fig, caption


# --- 2. The receiver and detector ------------------------------------------------------------

SYSTEM_DIAGRAM_CAPTION = (
    "The standard receiver passes the received grid through DMRS channel estimation, LMMSE "
    "equalization, demapping and LDPC decoding. The detector computes power from the grid, "
    "the DMRS residual from raw LS pilot estimates, and EVM from the equalized symbols and "
    "their noise variance. A U-Net and threshold produce a 24 by 14 jamming map. The dashed "
    "paths show how the receiver uses the map to set pilot and data variances on a second "
    "pass through the same slot."
)


# --- 3. How the DMRS residual works ----------------------------------------------------------


@paper_style()
def residual_explained():
    """Show the residual as the gap between the pilot estimates and the smoothest channel."""
    fig, (fit, prbs) = new_figure(3.5, 2.85, nrows=2, sharex=True)
    fig.set_layout_engine(None)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.145, top=0.885, hspace=0.36)
    numbers = np.arange(ALLOC[0], sum(ALLOC))
    edges = np.arange(ALLOC[0], sum(ALLOC) + 1) - 0.5
    seed = _example_seed("C")
    limits = 10 * np.log10([threshold(ALLOC[1], pos, fa=0.01) for pos in range(ALLOC[1])])
    for ax in (fit, prbs):
        ax.axvspan(JAM[0] - 0.5, sum(JAM) - 0.5, color="0.9", lw=0)
        ax.grid(False)
        ax.grid(axis="y", color="0.89", alpha=1, lw=0.5)
        ax.tick_params(length=2, pad=2)
    for edge in edges[1:-1]:
        fit.axvline(edge, color="0.85", lw=0.5, zorder=0)

    # (a) The jammed slot: LS estimate of every pilot and its projection onto the delay subspace,
    # which is the smoothest frequency response a channel with delays up to tau_max can have.
    batch = _example_batch("pilot", 0.0, 20.0, "C", seed)
    h, sc, _ = dmrs_ls(batch.y_alloc, batch.tx)
    basis = delay_subspace(sc, CFG.scs, CFG.tau_max).to(h.dtype)
    smooth = ((h @ basis.conj()) @ basis.T)[0].abs().cpu().numpy()
    estimates = h[0].abs().cpu().numpy()
    sc = sc.cpu().numpy()
    x = edges[0] + (sc + 0.5) / 12
    jammed = (ALLOC[0] + sc // 12 >= JAM[0]) & (ALLOC[0] + sc // 12 < sum(JAM))
    fit.plot(x, smooth, color="black", lw=0.9, label="Subspace fit", zorder=3)
    fit.plot(
        x[~jammed],
        estimates[~jammed],
        ls="none",
        marker="o",
        ms=2.4,
        mfc="none",
        mec=CLEAN_COLOR,
        mew=0.5,
        label="Clean PRBs",
    )
    fit.plot(
        x[jammed],
        estimates[jammed],
        ls="none",
        marker="s",
        ms=2.4,
        mfc=JAM_COLOR,
        mec="none",
        label="Jammed PRBs",
    )
    fit.set_ylabel(r"$|\tilde{h}_p|$", fontsize=8)
    fit.set_ylim(bottom=0)

    # (b) The residual of every PRB in the benign and the jammed slot, against its threshold.
    leak_note = ""
    for attack, label, mark, edge_color, face in (
        ("none", "Benign slot", "o", CLEAN_COLOR, "none"),
        ("pilot", "Jammed slot", "s", JAM_COLOR, JAM_COLOR),
    ):
        batch = _example_batch(attack, 0.0, 20.0, "C", seed)
        db = 10 * np.log10(residual_map(batch, CFG)[0, numbers, CFG.dmrs_symbol].cpu().numpy())
        prbs.plot(
            numbers,
            db,
            ls="none",
            marker=mark,
            ms=3.5,
            mec=edge_color,
            mfc=face,
            mew=0.55,
            label=label,
        )
        if attack == "pilot":
            neighbors = (numbers == JAM[0] - 1) | (numbers == sum(JAM))
            crossings = np.flatnonzero(neighbors & (db > limits))
            if crossings.size:
                k = crossings[np.argmax(db[crossings] - limits[crossings])]
                leak_note = (
                    f" Leakage through the projection takes the clean PRB {numbers[k]} above the "
                    "threshold."
                )
    prbs.stairs(limits, edges, baseline=None, color="0.2", lw=0.9, ls="--", label="1% threshold")
    prbs.yaxis.set_major_locator(MultipleLocator(10))
    prbs.margins(y=0.12)
    prbs.set(
        xlabel="PRB", ylabel=r"$R_b$ (dB)", xticks=numbers, xlim=(edges[0] - 0.07, edges[-1] + 0.07)
    )
    fit.tick_params(bottom=False)
    fit.set_title("(a)", loc="left", fontsize=8, pad=2)
    prbs.set_title("(b)", loc="left", fontsize=8, pad=2)
    legend = {"frameon": False, "fontsize": 7, "handlelength": 1.3, "columnspacing": 0.9}
    fig.legend(
        *fit.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.56, 1.0),
        ncols=3,
        handletextpad=0.4,
        **legend,
    )
    prbs.legend(loc="upper right", ncols=1, handletextpad=0.4, borderaxespad=0.3, **legend)
    caption = (
        "The DMRS residual in two TDL-C slots at 20 dB SNR that share their channel and noise, "
        "with a 12-PRB allocation (PRBs 6-17) and, in one of them, pilot jamming at 0 dB JSR on "
        "PRBs 10-13 (shaded). (a) Magnitude of the LS channel estimates at the 72 pilots of the "
        "jammed slot, and of their projection onto the delay subspace: the smoothest frequency "
        "response that a channel with delays up to 3 us can have. On clean PRBs the estimates "
        "follow the curve; on jammed PRBs they scatter around it. The residual is the complex "
        "difference between estimate and curve. (b) Residual R_b of each PRB in both slots: the "
        "six squared differences summed and divided by their expected value without jamming, "
        "with the analytic 1% threshold for each PRB position (dashed)." + leak_note
    )
    return fig, caption


# --- 4. Why pilot jamming shows up in the data symbols -----------------------------------------


def constellations():
    fig, axes = new_figure(7.16, 2.6, ncols=3, sharex=True, sharey=True)
    titles = {"none": "No jammer", "pilot": "Pilot jamming", "data": "Data jamming"}
    spread = {}
    seed = _example_seed("A")
    for ax, attack in zip(axes, ("none", "pilot", "data"), strict=True):
        # Same slot in every panel, chosen by _example_seed (no deep fade on the jammer block)
        batch = _example_batch(attack, 0.0, 20.0, "A", seed=seed)
        x_hat = batch.rx["x_hat"][0].cpu().numpy()
        _, sc = data_positions(batch.tx)
        prb = ALLOC[0] + sc.cpu().numpy() // 12
        on_jammed = (prb >= JAM[0]) & (prb < JAM[0] + JAM[1])
        ax.scatter(
            x_hat[~on_jammed].real,
            x_hat[~on_jammed].imag,
            s=1.5,
            alpha=0.35,
            color=CLEAN_COLOR,
            marker="o",
            label="PRBs the jammer did not touch",
        )
        ax.scatter(
            x_hat[on_jammed].real,
            x_hat[on_jammed].imag,
            s=4,
            alpha=0.6,
            color=JAM_COLOR,
            marker="x",
            lw=0.6,
            label="PRBs 10-13 (jammed block)",
        )
        ax.set(
            title=titles[attack],
            xlim=(-2.2, 2.2),
            ylim=(-2.2, 2.2),
            aspect="equal",
            xlabel="in-phase",
        )
        ideal = (np.abs(x_hat.real) + 1j * np.abs(x_hat.imag)) - (1 + 1j) / np.sqrt(2)
        spread[attack] = np.mean(np.abs(ideal[on_jammed]) ** 2)
    axes[0].set_ylabel("quadrature")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False, markerscale=4)
    caption = (
        "Equalized data symbols of one slot (QPSK, TDL-A, SNR 20 dB, jammer on PRBs 10-13 of the "
        "12 scheduled PRBs, JSR 0 dB). Blue: PRBs the jammer did not touch; orange: the jammed "
        "block. With pilot jamming, the data symbols of the jammed PRBs scatter (mean squared "
        f"error {spread['pilot']:.2f} against {spread['none']:.2f} without jamming) although the "
        "jammer never transmitted on them: the corrupted DMRS spoils their channel estimate. "
        "This is why EVM reports pilot jamming at the wrong symbols, and why the labels mark "
        "only the DMRS cells. With data jamming the scatter comes from the jammer itself."
    )
    return fig, caption


# --- Figures that need datasets or trained models ----------------------------------------------


class MissingInput(Exception):
    """A figure's inputs do not exist yet; the message says how to produce them."""


def _load_split(split: str) -> dict:
    if not (data_dir() / f"{split}.npz").is_file():
        raise MissingInput(f"needs the {split} split: python -m jamloc.dataset --split {split}")
    return load(split)


def _require_models(variants, seed: int = 0):
    for variant in variants:
        if not checkpoint_path(variant, seed).is_file():
            raise MissingInput("needs trained models: python -m jamloc.train --variant all")


# --- 5. Which feature separates which attack -------------------------------------------------


def _feature_groups(d: dict, attack: str, feature: str, mask: str, jsr_db=(-6.0, 0.0)):
    """Values of one feature in the scheduled cells where it is defined: (not jammed, jammed).

    Power is compared cell by cell: jammed cells against all others. The residual and EVM are
    compared by PRB: cells of the PRBs the jammer hit against cells of untouched PRBs. That
    grouping shows their key behaviour: under pilot jamming EVM rises in data cells that were
    never jammed (the wrong-symbol effect), and under data jamming the residual does not move.
    """
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    slots = (
        (d["attack"] == ATTACKS.index(attack))
        & (d["jsr_db"] >= jsr_db[0])
        & (d["jsr_db"] <= jsr_db[1])
    )
    defined = d["features"][slots, idx[mask]] > 0
    values = d["features"][slots, idx[feature]]
    labels = d["labels"][slots].astype(bool)
    if feature == "power_db":
        jammed = labels
    else:
        jammed = np.broadcast_to(labels.any(axis=2, keepdims=True), labels.shape)
    return values[defined & ~jammed], values[defined & jammed]


def _median(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else float("nan")


def feature_distributions(split: str = "test"):
    d = _load_split(split)
    rows = [
        ("power_db", "alloc_mask", "Power P [dB]\n(per cell)"),
        ("residual_db", "dmrs_mask", "Residual R [dB]\n(per PRB)"),
        ("evm_db", "data_mask", "EVM E [dB]\n(per PRB)"),
    ]
    bins = np.linspace(-10, 40, 51)
    fig, axes = new_figure(7.16, 4.8, nrows=3, ncols=3, sharex=True)
    gap, spread = {}, {}
    for col, attack in enumerate(("pilot", "data", "partial")):
        for row, (feature, mask, label) in enumerate(rows):
            clean, jammed = _feature_groups(d, attack, feature, mask)
            ax = axes[row, col]
            ax.hist(
                clean,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.35,
                color=CLEAN_COLOR,
                label="not jammed",
            )
            ax.hist(
                jammed,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.2,
                color=JAM_COLOR,
                label="jammed (power: the cell itself; R and E: the cell's PRB)",
            )
            gap[(attack, feature)] = _median(jammed) - _median(clean)
            if clean.size:
                spread[(attack, feature)] = float(np.subtract(*np.percentile(clean, [75, 25])))
            if row == 0:
                ax.set_title(f"{attack} jamming")
            if col == 0:
                ax.set_ylabel(label)
            if row == 2:
                ax.set_xlabel("feature value [dB]")
            ax.set_yticks([])
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    shifts = [gap[(a, "power_db")] for a in ("pilot", "data", "partial")]
    lo, hi = _fmt_db(min(shifts)), _fmt_db(max(shifts))
    power_shift = f"about {lo}" if lo == hi else f"{lo} to {hi} depending on the attack"
    fig.legend(handles, labels_, loc="outside lower center", ncols=2, frameon=False)
    caption = (
        "Distributions of the three features over the scheduled cells of the test set (JSR -6 to "
        "0 dB). Blue: not jammed; orange: jammed - for power the cell itself, for the residual "
        "and EVM the cell's PRB, for the reason given below. Where orange moves away from blue, "
        "the feature can see the attack. The residual separates pilot jamming clearly (median "
        f"shift {_fmt_db(gap[('pilot', 'residual_db')])}) but does not move at all under data "
        f"jamming ({_fmt_db(gap[('data', 'residual_db')])}). EVM moves under data jamming "
        f"({_fmt_db(gap[('data', 'evm_db')])}) and also in the data cells of pilot-jammed PRBs "
        f"({_fmt_db(gap[('pilot', 'evm_db')])}), although none of those cells was jammed - the "
        "wrong-symbol effect, which is why these rows are grouped by PRB. The power of a jammed "
        f"cell shifts by only {power_shift}, small against the "
        f"{_fmt_db(spread.get(('data', 'power_db'), float('nan')))} interquartile spread that the "
        "UE's own fading causes, so power alone separates the attacks poorly. No single feature "
        "covers every attack, which is why the detector fuses them."
    )
    return fig, caption


# --- 6. What each detector outputs -----------------------------------------------------------


PD_DETECTORS = {
    "fused": "U-Net P+R+E",
    "no_residual": "U-Net P+E",
    "power": "U-Net power",
    "evm": "U-Net EVM",
    "block_test": "Block test",
    "evm_threshold": "EVM threshold",
    "residual_cfar": "Residual p-value",
}


def _pilot_intervals(split):
    """Read the seed-mean estimates and their paired batch-bootstrap limits in display order."""
    path = results_dir() / "intervals" / f"{split}_pd_ci.csv"
    if not path.is_file():
        raise MissingInput(
            f"needs {path.name}: python -m jamloc.experiments.intervals --split {split}"
        )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["seed"] == "mean" and r["attack"] == "pilot"]
    if not rows:
        raise MissingInput(f"{path.name} has no seed means for pilot jamming")
    jsr = min(float(r["jsr_db"]) for r in rows)
    lowest = {r["detector"]: r for r in rows if float(r["jsr_db"]) == jsr}
    missing = PD_DETECTORS.keys() - lowest.keys()
    if missing:
        raise MissingInput(f"{path.name} lacks {', '.join(sorted(missing))} at {jsr:g} dB")
    values = [[float(lowest[d][key]) for key in ("pd", "pd_lo", "pd_hi")] for d in PD_DETECTORS]
    return jsr, np.array(values), int(lowest["fused"]["n_seeds"])


def _pilot_example(d, val, model):
    """Select the original +3 dB example by the EVM false-alarm count, retaining its tie rule."""
    from jamloc.experiments.recovery import val_threshold

    idx = np.flatnonzero(
        (d["attack"] == ATTACKS.index("pilot")) & (d["jsr_db"] == 3) & (d["alloc_size"] >= 10)
    )
    labels = d["labels"][idx].astype(bool)
    allocated = d["features"][idx, FEATURE_NAMES.index("alloc_mask")] > 0
    inside = labels.any(axis=(1, 2)) & ~(labels & ~allocated).any(axis=(1, 2))
    idx = idx[inside]
    if not idx.size:
        raise MissingInput(
            "needs a +3 dB pilot-jamming slot with the block inside at least 10 allocated PRBs"
        )
    features = d["features"][idx]
    truth = labels[inside]
    evm_thr = threshold_at_fa(evm_scores(val["features"])[val["labels"] == 0], FA)
    evm = evm_scores(features) > evm_thr
    fused_thr = val_threshold("fused", model)
    x = select_inputs(torch.from_numpy(features).to(config.device), "fused")
    with torch.no_grad():
        fused = model(x).cpu().numpy() > fused_thr

    counts = (evm & ~truth).sum(axis=(1, 2))
    pool = np.flatnonzero(counts > 0)
    if pool.size:
        statistic = counts[pool]
        rule = (
            "Among eligible slots with EVM false alarms, the example has the median "
            "false-alarm count."
        )
    else:
        pool = np.arange(len(idx))
        statistic = (fused & truth).sum(axis=(1, 2)) / truth.sum(axis=(1, 2))
        rule = (
            "No eligible slot has an EVM false alarm. The example has the median "
            "U-Net hit rate on jammed cells."
        )
    k = pool[np.argsort(statistic, kind="stable")[len(pool) // 2]]
    return idx[k], evm[k], fused[k], rule


@paper_style()
def detector_outputs(split: str = "test", seed: int | None = None):
    """Compare alarm placement in one slot with detection across all jammed scheduled pilots."""
    from jamloc.evaluate import available_seeds

    jsr, intervals, n_seeds = _pilot_intervals(split)
    if seed is None:
        seed = (available_seeds() or [0])[0]
    _require_models(("fused",), seed)
    d = _load_split(split)
    i, evm, fused, rule = _pilot_example(d, _load_split("val"), load_model("fused", seed))
    truth = d["labels"][i].astype(bool)
    jammed_prbs = np.flatnonzero(truth[:, CFG.dmrs_symbol])
    allocation = d["features"][i, FEATURE_NAMES.index("alloc_mask")] > 0
    lo, size = int(d["alloc_start"][i]), int(d["alloc_size"][i])
    dmrs = CFG.dmrs_symbol

    fig = Figure(figsize=(7.1, 2.8))
    grid = fig.add_gridspec(1, 3, left=0.066, right=0.602, wspace=0.085, bottom=0.29, top=0.775)
    maps = grid.subplots(sharex=True, sharey=True).ravel()
    cmap = ListedColormap(["white", ALLOC_COLOR, DMRS_COLOR])
    hatch = {
        "facecolor": "none",
        "edgecolor": "0.55",
        "hatch": "////",
        "lw": 0,
        "hatch_linewidth": 0.45,
    }
    for ax, title, flags in zip(
        maps,
        ("Injected interference", "EVM threshold flags", "U-Net P+R+E flags"),
        (np.zeros_like(truth), evm, fused),
        strict=True,
    ):
        ax.imshow(
            np.where(flags, 2, allocation),
            cmap=cmap,
            vmin=0,
            vmax=2,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        if ax is maps[1]:
            # EVM exists only on scheduled data. Hatch contiguous areas so adjacent cells
            # share a pattern without interior borders or double hatching at the DMRS.
            for x0, y0, width, height in (
                (0, 0, CFG.num_symbols, lo),
                (0, lo + size, CFG.num_symbols, CFG.num_prb - lo - size),
                (dmrs, lo, 1, size),
            ):
                if height:
                    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), width, height, **hatch))
        ax.add_patch(
            Rectangle(
                (dmrs - 0.5, jammed_prbs[0] - 0.5),
                1,
                len(jammed_prbs),
                fill=False,
                ec="black",
                lw=1.15,
            )
        )
        ax.set(title=title, xticks=[0, dmrs, 7, 13], yticks=[0, 6, 12, 18, 23])
        ax.title.set_fontsize(7.5)
        ax.grid(False)
        ax.tick_params(length=2, width=0.6, pad=2)
        tick = ax.get_xticklabels()[1]
        tick.set_color(DMRS_COLOR)
        tick.set_weight("bold")
    maps[0].set_ylabel("PRB")
    maps[1].set_xlabel("OFDM symbol (2 = DMRS)")
    fig.text(0.066, 0.94, f"(a)  One {split} slot at +3 dB JSR", weight="bold", fontsize=8.5)
    fig.legend(
        handles=[
            Patch(fill=False, edgecolor="black", label="Injected"),
            Patch(facecolor=DMRS_COLOR, label="Flagged"),
            Patch(facecolor=ALLOC_COLOR, label="Allocated"),
            Patch(**hatch, label="Unscored"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.337, 0.015),
        frameon=False,
        ncols=4,
        handlelength=1.25,
        handletextpad=0.4,
        columnspacing=1,
        fontsize=7,
    )

    ax = fig.add_axes([0.790, 0.29, 0.187, 0.485])
    y = np.arange(len(intervals))
    mean, lower, upper = intervals.T
    ax.hlines(y, lower, upper, color="0.25", lw=0.9)
    for end in (lower, upper):
        ax.vlines(end, y - 0.085, y + 0.085, color="0.25", lw=0.7)
    ax.plot(mean, y, "o", color=DMRS_COLOR, ms=2.5, mew=0)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_yticks(y, PD_DETECTORS.values())
    ax.set(
        ylim=(len(y) - 0.5, -0.5),
        xlim=(-0.02, min(1.02, max(0.2, upper.max()) + 0.04)),
        xlabel=r"Detection probability $P_d$",
    )
    ax.grid(False)
    ax.grid(axis="x", color="0.85", lw=0.6, alpha=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2, pad=2)
    fig.text(0.635, 0.94, f"(b)  Pilot jamming at {jsr:g} dB JSR", weight="bold", fontsize=8.5)
    fig.text(0.635, 0.84, "Jammed scheduled DMRS cells", fontsize=8)
    fig.text(0.635, 0.056, "Seed mean with 95% bootstrap interval", fontsize=7)
    caption = (
        f"(a) Pilot jamming in one {split} slot at +3 dB JSR. The "
        "jammed block lies within an allocation of at least 10 PRBs. " + rule + " The black "
        "outline marks the injected cells on all three maps. Hatching marks cells the detector "
        f"cannot score. The EVM threshold flags {evm.sum()} data cells and cannot score the DMRS. "
        f"The U-Net flags {(fused & truth).sum()} of {truth.sum()} jammed cells. "
        f"(b) At {jsr:g} dB JSR, each dot gives the fraction of jammed scheduled DMRS cells "
        f"flagged across the {split} split, averaged over {n_seeds} seeds. Bars give 95% "
        "batch-bootstrap intervals. Both panels use thresholds calibrated for a 1% false-alarm rate "
        "on scored, unjammed validation cells."
    )
    return fig, caption


# --- 7. Training curves ------------------------------------------------------------------------


def training_curves():
    histories = sorted((results_dir() / "models").glob("*_seed*_history.csv"))
    if not histories:
        raise MissingInput(
            "needs training histories: train again (python -m jamloc.train "
            "--variant all, or the full run); histories are saved since this "
            "version"
        )
    fig, axes = new_figure(7.16, 2.4, ncols=len(VARIANTS), sharey=True)
    seed_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    best_epochs = []
    best_loss = {}
    titles = {
        "fused": "U-Net P+R+E",
        "no_residual": "U-Net P+E",
        "power": "power only",
        "evm": "EVM only",
    }
    for ax, variant in zip(axes, VARIANTS, strict=True):
        for path in histories:
            name, seed = path.stem.removesuffix("_history").rsplit("_seed", 1)
            if name != variant:
                continue
            with open(path, encoding="utf-8", newline="") as fh:
                h = list(csv.DictReader(fh))
            epoch = [int(r["epoch"]) for r in h]
            val_loss = [float(r["val_loss"]) for r in h]
            color = seed_colors[int(seed) % len(seed_colors)]
            ax.plot(epoch, [float(r["train_loss"]) for r in h], ls="--", color=color, lw=0.9)
            ax.plot(epoch, val_loss, ls="-", color=color, lw=1.2, label=f"seed {seed}")
            best_epochs.append(epoch[int(np.argmin(val_loss))])
            best_loss.setdefault(variant, []).append(min(val_loss))
        ax.set(title=titles[variant], xlabel="epoch")
    axes[0].set_ylabel("loss (weighted BCE)")
    axes[-1].legend(loc="upper right")
    worst = max(best_loss, key=lambda v: np.mean(best_loss[v]))
    spreads = [(max(v) - min(v)) / np.mean(v) for v in best_loss.values() if len(v) > 1]
    seeds_text = (
        f"Across seeds the best validation loss of a detector differs by at most "
        f"{max(spreads):.0%}. "
        if spreads
        else ""
    )
    caption = (
        "Training (dashed) and validation (solid) loss per epoch for each detector, one colour "
        "per training seed. The validation loss is lowest at epoch "
        f"{int(np.median(best_epochs))} in the median run, and the weights of that epoch are the "
        "ones kept, so later epochs, where the training loss may keep falling, cannot overfit "
        f"the reported models. {seeds_text}{titles[worst]} ends with the highest loss, as it "
        "sees the least information."
    )
    return fig, caption


# --- 8. What the receiver sees once it knows the jamming map ---------------------------------


def constellations_with_map(seed: int | None = None, jsr_db: float = 6.0):
    from jamloc.evaluate import available_seeds
    from jamloc.experiments.recovery import detector_cells, val_threshold
    from jamloc.recovery import MapAwareReceiver

    if seed is None:  # the first training seed that has every model
        seed = (available_seeds() or [0])[0]
    _require_models(("fused",), seed)
    _load_split("val")  # needed for the detector's threshold
    model = load_model("fused", seed)
    thr = val_threshold("fused", model)
    example = _example_seed("A")
    alloc = slice(ALLOC[0], ALLOC[0] + ALLOC[1])

    fig, axes = new_figure(5.0, 5.8, nrows=2, ncols=2, sharex=True, sharey=True)
    stats = {}
    for col, attack in enumerate(("pilot", "data")):
        batch = _example_batch(attack, jsr_db, 20.0, "A", seed=example, decode=True)
        # The map the fused U-Net predicts for this slot, cut to the scheduled PRBs
        cells = detector_cells(batch, CFG, "fused", model, thr)[:, alloc]
        protected = MapAwareReceiver(batch.tx, CFG)(batch.y_alloc, batch.no, cells, mode="hard")
        sym, sc = (t.cpu().numpy() for t in data_positions(batch.tx))
        prb = ALLOC[0] + sc // 12
        in_block = (prb >= JAM[0]) & (prb < JAM[0] + JAM[1])
        erased = cells[0, sc // 12, sym].cpu().numpy()  # data symbols the decoder will ignore

        for row, (rx, name) in enumerate(
            ((batch.rx, "standard receiver"), (protected, "with the jamming map"))
        ):
            ax = axes[row, col]
            x_hat = rx["x_hat"][0].cpu().numpy()
            decoded = bool(rx["tb_crc"][0])
            hide = erased if row == 1 else np.zeros_like(erased)
            ax.scatter(
                x_hat[~in_block & ~hide].real,
                x_hat[~in_block & ~hide].imag,
                s=1.5,
                alpha=0.35,
                color=CLEAN_COLOR,
                label="PRBs the jammer did not touch",
            )
            ax.scatter(
                x_hat[in_block & ~hide].real,
                x_hat[in_block & ~hide].imag,
                s=4,
                alpha=0.6,
                color=JAM_COLOR,
                marker="x",
                lw=0.6,
                label="PRBs 10-13 (jammed block)",
            )
            ax.scatter(
                x_hat[hide].real,
                x_hat[hide].imag,
                s=3,
                alpha=0.25,
                color="#999999",
                label="erased: flagged by the map, ignored by the decoder",
            )
            ax.set(
                xlim=(-2.2, 2.2),
                ylim=(-2.2, 2.2),
                aspect="equal",
                title=f"{attack} jamming, {name}:\ntransport block "
                f"{'decoded' if decoded else 'lost'}",
            )
            ideal = (np.abs(x_hat.real) + 1j * np.abs(x_hat.imag)) - (1 + 1j) / np.sqrt(2)
            stats[(attack, row)] = {
                "mse_block": float(np.mean(np.abs(ideal[in_block]) ** 2)),
                "decoded": decoded,
            }
            if row == 1:
                stats[(attack, "erased")] = int(erased.sum())
                stats[(attack, "erased_in_block")] = int((erased & in_block).sum())
                stats[(attack, "block")] = int(in_block.sum())
            if col == 0:
                ax.set_ylabel("quadrature")
            if row == 1:
                ax.set_xlabel("in-phase")
    handles = [axes[1, 0].collections[i] for i in range(3)]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, markerscale=4)

    def outcome(attack, row):
        return "decoded" if stats[(attack, row)]["decoded"] else "lost"

    copes = ""
    if stats[("pilot", 0)]["decoded"]:
        copes = (
            " (here the decoder copes even without the map, because only a third of the block "
            "is affected; the recovery curves show how often it does not)"
        )
    caption = (
        "The same two slots before and after the receiver uses the jamming map predicted by the "
        f"fused U-Net (QPSK, TDL-A, SNR 20 dB, jammer on PRBs 10-13 of 12 scheduled PRBs, JSR "
        f"{jsr_db:+.0f} dB). Pilot jamming (left): the standard receiver trusts the jammed DMRS, "
        "so the data symbols of the jammed PRBs scatter (mean squared error "
        f"{stats[('pilot', 0)]['mse_block']:.2f}) and the transport block is "
        f"{outcome('pilot', 0)}. With the map, those pilots are dropped and the channel is "
        "interpolated from the clean ones: the same symbols pull back to the four QPSK points "
        f"(mean squared error {stats[('pilot', 1)]['mse_block']:.2f}) and the block is "
        f"{outcome('pilot', 1)}{copes}. Data jamming (right): no channel estimate can remove the jammer's "
        "noise from the symbols themselves, so the map instead tells the decoder to ignore them - "
        f"{stats[('data', 'erased_in_block')]} of the {stats[('data', 'block')]} data symbols of "
        f"the jammed PRBs are erased (grey) - and the block, {outcome('data', 0)} without the "
        f"map, is {outcome('data', 1)} with it. Any grey symbols outside the jammed block are "
        "the map's false alarms."
    )
    return fig, caption


# --- 9. The receiver at work, one slot ------------------------------------------------------


def _case_payload(case) -> dict:
    """Simulate one example slot with the exporter of the showcase site (seeded, and the slot is
    picked by the case's stated rule)."""
    from jamloc.site.export import payload, simulate_case

    _require_models(("fused", "no_residual", "power"))
    return payload(simulate_case(case))


def _outline(ax, mask, **style):
    """Draw the outer boundary of the True cells of a [PRB, symbol] mask."""
    padded = np.pad(mask, 1)
    for b, s in np.argwhere(mask):
        for db, ds, (x0, y0, x1, y1) in (
            (-1, 0, (s - 0.5, b - 0.5, s + 0.5, b - 0.5)),
            (1, 0, (s - 0.5, b + 0.5, s + 0.5, b + 0.5)),
            (0, -1, (s - 0.5, b - 0.5, s - 0.5, b + 0.5)),
            (0, 1, (s + 0.5, b - 0.5, s + 0.5, b + 0.5)),
        ):
            if not padded[b + 1 + db, s + 1 + ds]:
                ax.plot([x0, x1], [y0, y1], solid_capstyle="projecting", **style)


def _constellation(ax, points, jammed_prb, title, reliability=None):
    """Equalized symbols, jammed ones in red; with ``reliability`` (|LLR| per point), points whose
    bits the decoder is told to trust less are drawn fainter (the site uses the same scale)."""
    alpha = (
        np.ones(len(points))
        if reliability is None
        else 0.12 + 0.83 * (1 - np.exp(-reliability / 14))
    )
    for sel, color in ((~jammed_prb, CLEAN_COLOR), (jammed_prb, JAM_COLOR)):
        rgba = np.tile(to_rgba(color), (int(sel.sum()), 1))
        rgba[:, 3] = alpha[sel]
        ax.scatter(*points[sel].T, s=0.6, c=rgba, lw=0, rasterized=True)
    ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), xticks=[], yticks=[], aspect="equal")
    ax.axhline(0, color="0.85", lw=0.4, zorder=0)
    ax.axvline(0, color="0.85", lw=0.4, zorder=0)
    ax.grid(False)
    ax.set_title(title, fontsize=7, pad=2.5)


@paper_style()
def system_views():
    """One partial-band-jammed slot, from the received grid to the equalized symbols with the map.

    The slot of the site's walkthrough: partial-band jamming at +3 dB JSR on 4 of the UE's 12 PRBs,
    so both the pilots and the data cells of those PRBs are hit; the first slot of its batch that
    the map recovers (the caption gives the batch's counts)."""
    from jamloc.site.cases import CASES

    d = _case_payload(CASES["partial_flip"])
    meta = d["metadata"]
    a0, n = meta["alloc"]
    j0, jn = meta["jam"]
    l0 = meta["config"]["dmrs_symbol"]
    grid = {k: np.array(v, dtype=float) for k, v in d["grids"].items()}
    truth, alloc = grid["truth"] > 0, grid["allocation"] > 0
    flags = np.array(d["detectors"]["fused"]["decisions"], dtype=float) > 0

    # Every equalized point belongs to one allocation subcarrier: divmod(i, 12 n) -> (symbol, sc)
    num_data = meta["config"]["num_symbols"] - 1
    sc_of_point = np.arange(12 * n * num_data) % (12 * n)
    jammed_point = (a0 + sc_of_point // 12 >= j0) & (a0 + sc_of_point // 12 < j0 + jn)

    # One row in processing order: (a) grid -> (b) channel estimate -> (c) equalized symbols ->
    # (d) features -> (e) U-Net map -> (f) the same slot equalized again with the map
    width, height = 7.16, 1.6
    fig = Figure(figsize=(width, height))

    def axes(left, w, bottom=0.44, h=0.95):  # inches
        return fig.add_axes((left / width, bottom / height, w / width, h / height))

    a, b, e = axes(0.34, 0.58), axes(1.4, 0.95), axes(2.68, 0.95)
    c, cbar, dd = axes(3.95, 0.58), axes(4.58, 0.045, bottom=0.53, h=0.77), axes(5.12, 0.58)
    f = axes(6.02, 0.95)
    arrows = ((0.97, 1.19), (2.4, 2.62), (3.68, 3.9), (4.87, 5.07), (5.75, 5.97))  # inches

    # (a) received power, (c) residual on the DMRS cells and EVM on the data cells, same scale
    scale = {"cmap": PARULA, "vmin": 0, "vmax": 25, "origin": "lower", "aspect": "auto"}
    image = a.imshow(grid["power_db"], interpolation="nearest", **scale)
    features = np.full(grid["power_db"].shape, np.nan)
    features[:, l0] = grid["residual_db"][:, l0]
    data_cols = [s for s in range(features.shape[1]) if s != l0]
    features[:, data_cols] = grid["evm_db"][:, data_cols]
    c.imshow(np.where(alloc, features, np.nan), interpolation="nearest", **scale)
    c.set_facecolor(EMPTY_COLOR)
    fig.colorbar(image, cax=cbar, ticks=[0, 10, 20])
    cbar.tick_params(labelsize=6.5, length=2, width=0.5, pad=1)
    cbar.set_ylabel("dB", labelpad=1)
    # (d) the U-Net map
    _cell_grid(dd, np.where(flags, 2, np.where(alloc, 1, 0)), [EMPTY_COLOR, ALLOC_COLOR, JAM_COLOR])
    for ax in (a, c, dd):
        _outline(ax, truth, color="black", lw=0.7)
        ax.set(xticks=[0, l0, 13], yticks=[0, 6, 12, 18, 23])
        ax.set_xlabel("Symbol", labelpad=1)
        ax.tick_params(direction="in", length=2, width=0.5, pad=1.5)
        ax.grid(False)
    for ax in (a, c):
        _outline(ax, alloc, color="white", lw=0.5, ls=(0, (2, 1.5)))
    a.set_ylabel("PRB", labelpad=1)
    c.tick_params(labelleft=False)
    dd.tick_params(labelleft=False)

    # (b) LS estimates at the pilots and their projection onto the delay subspace
    pil = d["pilots"]
    sc = np.array(pil["subcarrier"])
    ls = np.hypot(*np.array(pil["ls"], dtype=float).T)
    smooth = np.hypot(*np.array(pil["projection"], dtype=float).T)
    x = a0 - 0.5 + (sc + 0.5) / 12
    hit = (a0 + sc // 12 >= j0) & (a0 + sc // 12 < j0 + jn)
    b.plot(x, smooth, color="0.15", lw=0.8)
    b.plot(x[~hit], ls[~hit], ls="none", marker="o", ms=1.8, mfc="none", mec=CLEAN_COLOR, mew=0.4)
    b.plot(x[hit], ls[hit], ls="none", marker="s", ms=1.7, mfc=JAM_COLOR, mec="none")
    b.set(xlim=(a0 - 0.5, a0 + n - 0.5), ylim=(0, None), xticks=[a0, j0, j0 + jn - 1, a0 + n - 1])
    b.set_xlabel("PRB", labelpad=1)
    b.tick_params(length=2, width=0.5, pad=1.5)
    b.grid(False)

    # (e), (f) equalized data symbols without and with the map, jammed PRBs in color
    for ax, receiver in ((e, "standard"), (f, "fused_soft")):
        rx = d["receivers"][receiver]
        xy = np.array(d["x_hat"][rx["x_hat"]], dtype=float)
        llr = np.array(d["abs_llr"][rx["abs_llr"]], dtype=float)
        _constellation(ax, xy, jammed_point, "", reliability=llr)

    from matplotlib.patches import FancyArrowPatch

    names = (
        "Received grid",
        "Channel estimate",
        "Equalized symbols",
        "Features",
        "U-Net map",
        "With the map",
    )
    order = (a, b, e, c, dd, f)
    for i, (ax, name) in enumerate(zip(order, names, strict=True)):
        box = ax.get_position()
        fig.text(
            (box.x0 + box.x1) / 2,
            0.02 / height,
            f"({'abcdef'[i]}) {name}",
            ha="center",
            va="bottom",
        )
    # Arrows between consecutive steps, at mid-height of the panels
    mid = (0.44 + 0.95 / 2) / height
    for x0, x1 in arrows:
        fig.add_artist(
            FancyArrowPatch(
                (x0 / width, mid),
                (x1 / width, mid),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=6,
                lw=0.7,
                color="0.2",
            )
        )
    crc = meta["batch_crc"]
    caption = (
        f"One slot with partial-band jamming at {meta['jsr_db']:+g} dB JSR on PRBs "
        f"{j0}-{j0 + jn - 1}, all 14 symbols ({jn} of the UE's {n} PRBs, so {jn}/{n} of its pilots "
        f"and data cells; TDL-C, {meta['snr_db']} dB SNR, UE on PRBs {a0}-{a0 + n - 1}), in "
        "processing order: (a) received power, (b) magnitude of the LS pilot estimates and their "
        "subspace fit, (c) equalized symbols, (d) residual on the DMRS and EVM on the data cells, "
        "(e) U-Net map, and (f) the same slot equalized again with the map. Outlines mark the "
        "jammed cells; fainter points carry bits the decoder is told to trust less. It is the "
        f"first slot of a {crc['standard']['slots']}-slot batch that the map recovers: "
        f"{crc['standard']['failures']} of the {crc['standard']['slots']} slots fail the CRC "
        f"without the map and {crc['fused_soft']['failures']} with it."
    )
    return fig, caption


@paper_style()
def data_aided_passes():
    """The data-aided passes on one slot whose every pilot is jammed: the channel estimate and the
    equalized data symbols after the first attempt and after each of the first two passes (the
    slot of the site's passes animation)."""
    from matplotlib.patches import FancyArrowPatch

    from jamloc.site.cases import CASES

    d = _case_payload(CASES["aided_flip"])
    meta = d["metadata"]
    a0, n = meta["alloc"]
    ch = d["channel"]
    true = np.array(ch["true"], dtype=float)
    x = a0 - 0.5 + (np.arange(ch["num_sc"]) + 0.5) / 12
    pilot_x = a0 - 0.5 + (np.array(ch["pilot_sc"]) + 0.5) / 12
    ls = np.hypot(*np.array(ch["ls"], dtype=float).T)
    passes = d["passes"][:3]
    num_points = len(d["x_hat"][passes[0]["x_hat"]])
    errors = [
        float(np.mean(np.sum((np.array(p["h"], dtype=float) - true) ** 2, axis=1))) for p in passes
    ]
    ymax = 1.15 * float(np.max(np.hypot(*true.T))) * 1.6

    width, height = 7.16, 1.72
    fig = Figure(figsize=(width, height))
    names = ("First attempt", "Pass 1", "Pass 2")
    pair, chan_w, cons_w, gap, arrow = 2.08, 1.15, 0.85, 0.08, 0.2
    bottom, h = 0.5, 0.85
    lefts = [0.32 + k * (pair + arrow) for k in range(3)]
    for k, (step, left) in enumerate(zip(passes, lefts, strict=True)):
        c = fig.add_axes((left / width, bottom / height, chan_w / width, h / height))
        q = fig.add_axes(
            ((left + chan_w + gap) / width, bottom / height, cons_w / width, h / height)
        )
        c.plot(x, np.hypot(*true.T), color="0.6", lw=1.6)
        c.plot(pilot_x, ls, ls="none", marker="s", ms=1.3, mfc=JAM_COLOR, mec="none", alpha=0.35)
        c.plot(x, np.hypot(*np.array(step["h"], dtype=float).T), color=JAM_COLOR, lw=0.8)
        c.set(xlim=(a0 - 0.5, a0 + n - 0.5), ylim=(0, ymax), xticks=[a0, a0 + n - 1])
        c.set_xlabel("PRB", labelpad=1)
        c.tick_params(length=2, width=0.5, pad=1.5, labelleft=k == 0)
        c.grid(False)
        if k == 0:
            c.set_ylabel("Magnitude", labelpad=1)
        xy = np.array(d["x_hat"][step["x_hat"]], dtype=float)
        llr = np.array(d["abs_llr"][step["abs_llr"]], dtype=float)
        _constellation(q, xy, np.zeros(num_points, dtype=bool), "", reliability=llr)
        crc = "CRC pass" if step["crc"] else "CRC fail"
        box_c, box_q = c.get_position(), q.get_position()
        fig.text(
            (box_c.x0 + box_q.x1) / 2,
            (bottom + h + 0.08) / height,
            f"MSE {errors[k]:.3f}, {crc}",
            ha="center",
            va="bottom",
        )
        fig.text(
            (box_c.x0 + box_q.x1) / 2,
            0.02 / height,
            f"({'ace'[k]}), ({'bdf'[k]}) {names[k]}",
            ha="center",
            va="bottom",
        )
        if k:
            mid = (bottom + h / 2) / height
            fig.add_artist(
                FancyArrowPatch(
                    ((left - arrow + 0.03) / width, mid),
                    ((left - 0.03) / width, mid),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=6,
                    lw=0.7,
                    color="0.2",
                )
            )
    caption = (
        f"Data-aided passes on one slot whose every pilot is jammed ({meta['jsr_db']:+g} dB JSR, "
        f"TDL-C, {meta['snr_db']} dB SNR, the UE on PRBs {a0}-{a0 + n - 1}). (a), (c), (e) The channel "
        "magnitude (grey), the LS estimates of the jammed pilots (red squares) and the estimate of the "
        "soft receiver with the U-Net map (red line). (b), (d), (f) The equalized data symbols, fainter "
        f"where the decoder is told to trust them less. The first attempt rests on jammed pilots only, "
        f"its mean squared error is {errors[0]:.3f} and the CRC fails. After one pass the decoded data "
        f"cells act as pilots, the error falls to {errors[1]:.3f} and the CRC passes. A second pass, "
        f"which the receiver skips once the CRC has passed, would bring it to {errors[2]:.3f}."
    )
    return fig, caption


@paper_style()
def after_decoding(jsr_db: float = 3.0, seed: int = 11):
    """One slot of a hopping jammer, which the U-Net never saw in training: the true map, the
    network's map before decoding, the p-values of the exact tests after decoding and the map they
    give. The first slot of the batch that decodes and whose network map has at least as many
    errors as the slot's median."""
    import dataclasses

    from jamloc.experiments.decoded_map import _power_pvalues, _slot_pvalues
    from jamloc.experiments.recovery import detector_cells, val_threshold
    from jamloc.recovery import MapAwareReceiver

    _require_models(["fused"])
    cfg = dataclasses.replace(CFG, jammer_hopping=True)
    link = JammedPUSCHLink(cfg)
    a0, n = ALLOC
    seed_everything(seed)
    batch = 32
    sc = Scenario(
        channel="C",
        alloc_start=a0,
        alloc_size=n,
        attack=torch.full((batch,), ATTACKS.index("data")),
        jam_start=torch.full((batch,), JAM[0]),
        jam_size=torch.full((batch,), JAM[1]),
        jsr_db=torch.full((batch,), float(jsr_db)),
        snr_db=torch.full((batch,), 15.0),
    )
    out = link(sc, decode=False)
    model = load_model("fused", 0)
    unet = detector_cells(out, cfg, "fused", model, val_threshold("fused", model)).cpu()
    rx = MapAwareReceiver(out.tx, cfg)
    alloc = slice(a0, a0 + n)
    with torch.no_grad():
        res = rx(
            out.y_alloc,
            out.no,
            unet[:, alloc].to(out.y.device),
            mode="soft",
            iterations=3,
            cellwise="shrink",
            refine=True,
        )
    crc = res["tb_crc"].reshape(-1).bool().cpu()
    pv = _power_pvalues(out.y, out.no).cpu()
    pv[:, alloc] = _slot_pvalues(rx, out, rx.decoded_symbols(res["b_hat"])).cpu()
    after = pv < FA
    truth = out.labels.bool().cpu()
    errors = ((unet != truth).flatten(1).sum(1)).numpy()
    pick = int(np.flatnonzero(crc.numpy() & (errors >= np.median(errors)))[0])
    t, u, p, m = truth[pick].numpy(), unet[pick].numpy(), pv[pick].numpy(), after[pick].numpy()
    sched = np.zeros_like(t)
    sched[alloc] = True

    width, height = 3.5, 1.55
    fig = Figure(figsize=(width, height))
    names = ("(a) Jammer", "(b) U-Net", "(c) $-\\log_{10} p$", "(d) After decoding")
    lefts = (0.3, 1.1, 1.9, 2.7)
    axes = [fig.add_axes((x / width, 0.3 / height, 0.62 / width, 1.0 / height)) for x in lefts]
    codes = {
        0: np.where(t, 2, np.where(sched, 1, 0)),
        1: np.where(u, 2, np.where(sched, 1, 0)),
        3: np.where(m, 2, np.where(sched, 1, 0)),
    }
    for k, ax in enumerate(axes):
        if k == 2:
            ax.imshow(
                np.clip(-np.log10(np.maximum(p, 1e-12)), 0, 6),
                cmap=PARULA,
                vmin=0,
                vmax=6,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            ax.set(xticks=[0, CFG.dmrs_symbol, CFG.num_symbols - 1], yticks=[0, 6, 12, 18, 23])
            ax.tick_params(direction="in", length=2, width=0.5, pad=1.5)
            ax.grid(False)
        else:
            _cell_grid(ax, codes[k], [EMPTY_COLOR, ALLOC_COLOR, JAM_COLOR])
        if k:
            _outline(ax, t, color="black", lw=0.6)
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("PRB", labelpad=1)
        ax.set_xlabel("Symbol", labelpad=1)
        ax.set_title(names[k], fontsize=6.5, pad=2)
    fp_u, fn_u = int((u & ~t).sum()), int((~u & t).sum())
    fp_m, fn_m = int((m & ~t).sum()), int((~m & t).sum())
    caption = (
        f"One slot with data jamming at {jsr_db:+g} dB JSR by a jammer that moves its block of "
        f"{JAM[1]} PRBs on every OFDM symbol (TDL-C, 15 dB SNR, the UE on PRBs {a0}-{a0 + n - 1}). "
        "(a) Jammed cells. (b) The U-Net's map before decoding, trained only on jammers that fill "
        f"PRB rows, with {fp_u} false alarms and {fn_u} misses. (c) The p-values of the exact tests "
        "after the slot has decoded, the decoded-data residual inside the allocation and the "
        f"received power outside it. (d) The map of the tests at 1%, with {fp_m} false alarms and "
        f"{fn_m} misses. Outlines mark jammed cells. The slot is the first of a {batch}-slot batch "
        "that decodes and whose network map has at least the median number of errors, and (c) is "
        "clipped at 6."
    )
    return fig, caption


def _estimates_without_map(cfg, attack: str, jsr_db: float, seed: int, slot: int = 0) -> dict:
    """Jamming power that each estimate gives every cell of one simulated slot without a map.

    ``"prb"`` is one value per PRB row, ``"cell"`` each cell on its own and ``"shrink"`` each
    cell shrunk towards its row, every one ``[n, 14]`` over the allocation, in units of the
    jammer's power per resource element and nan on the DMRS symbols. ``"truth"`` marks the
    jammed cells. The jammer hits PRBs ``JAM`` of the allocation ``ALLOC`` (TDL-C, 15 dB SNR).
    """
    from jamloc.recovery import MapAwareReceiver, lmmse_interpolation

    a0, n = ALLOC
    seed_everything(seed)
    batch = 4
    sc = Scenario(
        channel="C",
        alloc_start=a0,
        alloc_size=n,
        attack=torch.full((batch,), ATTACKS.index(attack)),
        jam_start=torch.full((batch,), JAM[0]),
        jam_size=torch.full((batch,), JAM[1]),
        jsr_db=torch.full((batch,), float(jsr_db)),
        snr_db=torch.full((batch,), 15.0),
    )
    out = JammedPUSCHLink(cfg)(sc, decode=False)

    # Channel estimate with every pilot weighted by its residual, as the receiver without a map
    rx = MapAwareReceiver(out.tx, cfg)
    y, no = out.y_alloc, out.no
    h_ls, pilot_sc, pilot_power = dmrs_ls(y, out.tx)
    rx._pilot_weights = "all"
    clean, pilot_var, r, interfered = rx._pilots(h_ls, pilot_sc, pilot_power, no, None, "cell")
    h_sc, err_sc = lmmse_interpolation(
        h_ls, pilot_sc, clean, pilot_var, y.shape[-1], cfg.scs, cfg.tau_max
    )

    # The three estimates of every data cell, averaged over its 12 subcarriers
    power = 10 ** (jsr_db / 10)
    estimates = {}
    for key, option in (("prb", "prb"), ("cell", True), ("shrink", "shrink")):
        rx._cellwise = option
        with torch.no_grad():
            v = rx._jamming_variance(y, no, h_sc, err_sc, None, r, interfered, "cell")
        grid = torch.full((cfg.num_symbols, y.shape[-1]), float("nan"), device=v.device)
        grid[rx.data_sym, rx.data_sc] = v[slot]
        estimates[key] = (grid.reshape(cfg.num_symbols, n, 12).mean(-1).T / power).cpu().numpy()
    estimates["truth"] = out.labels[slot, a0 : a0 + n].bool().cpu().numpy()
    return estimates


def jamming_estimates(jsr_db: float = 9.0, seed: int = 7, slot: int = 0):
    """One slot of a hopping data jammer and three estimates of the jamming power of every cell,
    all without a map: one value per PRB row, each cell on its own, and shrinkage of each cell
    towards its row. The estimates are in units of the jammer's power per resource element."""
    import dataclasses

    cfg = dataclasses.replace(CFG, jammer_hopping=True)
    a0, n = ALLOC
    est = _estimates_without_map(cfg, "data", jsr_db, seed, slot)
    names = ("(b) Per PRB", "(c) Per cell", "(d) Shrinkage")
    estimates = [est["prb"], est["cell"], est["shrink"]]
    truth = est["truth"]
    data = np.ones_like(truth)
    data[:, list(cfg.dmrs_symbols)] = False

    width, height = 3.5, 1.62
    fig = Figure(figsize=(width, height))
    lefts = (0.3, 1.04, 1.78, 2.52)
    axes = [fig.add_axes((x / width, 0.3 / height, 0.66 / width, 1.05 / height)) for x in lefts]
    x_edges, y_edges = np.arange(cfg.num_symbols + 1) - 0.5, np.arange(n + 1) - 0.5
    codes = np.where(~data, 1, np.where(truth, 2, 0))
    axes[0].pcolormesh(
        x_edges,
        y_edges,
        codes,
        cmap=ListedColormap([ALLOC_COLOR, DMRS_COLOR, JAM_COLOR]),
        vmin=-0.5,
        vmax=2.5,
        edgecolors="#B3B3B3",
        linewidth=0.2,
    )
    axes[0].set_title("(a) Jammer", fontsize=6.5, pad=2)
    for ax, name, img in zip(axes[1:], names, estimates, strict=True):
        mesh = ax.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(img), cmap=PARULA, vmin=0, vmax=1.5
        )
        _outline(ax, truth, color="black", lw=0.6)
        ax.set_title(name, fontsize=6.5, pad=2)
    for k, ax in enumerate(axes):
        ax.set(
            xlim=(-0.5, cfg.num_symbols - 0.5),
            ylim=(-0.5, n - 0.5),
            xticks=[0, cfg.dmrs_symbol, cfg.num_symbols - 1],
            yticks=[0, n // 2, n - 1],
        )
        ax.grid(False)
        ax.tick_params(direction="in", length=2, width=0.5, pad=1.5, labelleft=k == 0)
        ax.set_xlabel("Symbol", labelpad=1)
    axes[0].set_ylabel("PRB", labelpad=1)
    cax = fig.add_axes(((lefts[-1] + 0.7) / width, 0.3 / height, 0.05 / width, 1.05 / height))
    fig.colorbar(mesh, cax=cax).ax.tick_params(labelsize=5, length=1.5, pad=1)

    hit, miss = truth & data, ~truth & data
    stats = [(np.nanmean(img[hit]), np.nanmean(img[miss])) for img in estimates]
    caption = (
        f"One slot of a data jammer that moves its block of {JAM[1]} PRBs on every OFDM symbol "
        f"({jsr_db:+g} dB JSR, TDL-C, 15 dB SNR, the UE on PRBs {a0}-{a0 + n - 1}, shown inside "
        "the allocation). (a) Jammed cells, with the DMRS symbol in blue. (b)-(d) The jamming "
        "power each estimate gives every data cell without a map, in units of the jammer's power "
        "per resource element, with the jammed cells outlined. The mean on jammed and on clean "
        "cells is "
        + ", ".join(
            f"{j:.2f} and {c:.2f} {label}"
            for label, (j, c) in zip(("per PRB", "per cell", "with shrinkage"), stats, strict=True)
        )
        + ". One value per PRB spreads the jammer over the clean cells of its rows, the cell's "
        "own estimate is noisy, and shrinkage keeps the jammed cells and leaves the clean ones "
        "near zero."
    )
    return fig, caption


def shrinkage_example(jsr_db: float = 9.0, seed: int = 7):
    """The data cells of one PRB row under a block jammer and under a hopping jammer: each cell's
    own estimate of the jamming power, the row mean, and the value after shrinkage."""
    import dataclasses

    # (a) A row of the jammed block, every data symbol jammed
    block = _estimates_without_map(CFG, "data", jsr_db, seed)
    rows = [(block, JAM[0] - ALLOC[0], "(a) Block jammer")]
    # (b) Of the rows that the hopping jammer hits on 2 to 5 of their 13 data symbols, the one
    # where it is strongest
    hop = _estimates_without_map(
        dataclasses.replace(CFG, jammer_hopping=True), "data", jsr_db, seed
    )
    hits = (hop["truth"] & ~np.isnan(hop["cell"])).sum(axis=1)
    candidates = np.flatnonzero((hits >= 2) & (hits <= 5))
    strength = [np.nanmean(hop["cell"][r][hop["truth"][r]]) for r in candidates]
    row = int(candidates[int(np.argmax(strength))])
    rows.append((hop, row, "(b) Hopping jammer"))

    with paper_style():
        fig, axes = new_figure(COLUMN_WIDTH, 1.75, ncols=2, sharey=True)
        for ax, (est, r, title) in zip(axes, rows, strict=True):
            sym = np.flatnonzero(~np.isnan(est["cell"][r]))  # the data symbols
            for s in sym[est["truth"][r, sym]]:
                ax.axvspan(s - 0.5, s + 0.5, color=JAM_COLOR, alpha=0.15, lw=0)
            ax.plot(
                sym,
                est["cell"][r, sym],
                "o",
                ms=3,
                mfc="none",
                mew=0.7,
                color="0.4",
                label="Own estimate",
            )
            ax.axhline(est["prb"][r, sym[0]], color="0.2", ls="--", lw=0.8, label="Row mean")
            ax.plot(sym, est["shrink"][r, sym], "s", ms=2.6, color=CLEAN_COLOR, label="Shrinkage")
            ax.set(title=title, xlabel="OFDM symbol", xticks=[0, 4, 8, 13])
            ax.grid(False)
        axes[0].set_ylabel("Jamming power")
        fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncols=3)

    caption = (
        "The estimated jamming power of the data cells of one PRB row, (a) under a block jammer "
        f"that hits every data symbol and (b) under a hopping jammer that hits {hits[row]} of "
        f"them ({jsr_db:+g} dB JSR, TDL-C, 15 dB SNR). The circles show each cell's own estimate, "
        "the dashed line the row mean and the squares the values after shrinkage, all in units of "
        "the jammer's power per resource element, and the jammed symbols are shaded."
    )
    return fig, caption


FIGURES = {
    "jamming_estimates": jamming_estimates,
    "shrinkage_example": shrinkage_example,
    "after_decoding": after_decoding,
    "system_views": system_views,
    "data_aided_passes": data_aided_passes,
    "attack_patterns": attack_patterns,
    "residual_explained": residual_explained,
    "constellations": constellations,
    "feature_distributions": feature_distributions,
    "detector_outputs": detector_outputs,
    "training_curves": training_curves,
    "constellations_with_map": constellations_with_map,
}

# Result figures written by the other scripts: (path from results/figures, title, caption,
# command that produces it). Shown in the gallery when they exist.
RESULT_FIGURES = [
    (
        "../ber_bler/ber_bler_A.png",
        "BER/BLER sweep, TDL-A",
        (
            "BLER of the unprotected link when the jammer covers the whole allocation, against JSR per "
            "jammed resource element (left) and against the jammer's total energy in the slot (right). "
            "Shaded bands are 95% confidence intervals; a hollow triangle marks a point without any "
            "block error, drawn at its 95% upper bound. Per unit of energy, pilot jamming is the most "
            "damaging attack."
        ),
        "python -m jamloc.experiments.ber_bler --channel A",
    ),
    (
        "../ber_bler/ber_bler_C.png",
        "BER/BLER sweep, TDL-C",
        "The same for TDL-C.",
        "python -m jamloc.experiments.ber_bler --channel C",
    ),
    (
        "../feature_maps.png",
        "Feature maps of one slot per attack",
        (
            "Label, power, DMRS residual and EVM maps of one slot per attack (grey: feature not "
            "defined). The jammer block partly leaves the allocation, where only power can see it."
        ),
        "python -m jamloc.experiments.feature_maps",
    ),
    (
        "../eval/test_pd_vs_jsr_scheduled.png",
        "Detection, test set",
        (
            "Probability of detecting a jammed cell inside the allocation at 1% per-cell false alarm, "
            "against JSR, mean and +-1 standard deviation over training seeds."
        ),
        "python -m jamloc.evaluate --split test",
    ),
    (
        "../eval/test_pd_vs_jsr_all.png",
        "Detection, all jammed cells",
        "The same, including jammed cells outside the allocation (easier: no UE signal there).",
        "python -m jamloc.evaluate --split test",
    ),
    (
        "../eval/sched_pd_vs_jsr_scheduled.png",
        "Detection, scheduling-aware jammer",
        "The jammer hits exactly the scheduled PRBs, an attacker no detector saw in training.",
        "python -m jamloc.evaluate --split sched",
    ),
    (
        "../eval/test_snr5-15_pd_vs_jsr_scheduled.png",
        "Detection at SNR 5-15 dB",
        "Test slots with low SNR only.",
        "python -m jamloc.evaluate --split test --snr-db 5 15",
    ),
    (
        "../eval/test_snr15-25_pd_vs_jsr_scheduled.png",
        "Detection at SNR 15-25 dB",
        "Test slots with high SNR only.",
        "python -m jamloc.evaluate --split test --snr-db 15 25",
    ),
    (
        "../recovery/recovery_A.png",
        "Map-aware recovery, TDL-A",
        (
            "BLER of five receivers decoding the same slots: no protection; DMRS-only "
            "interference estimation without a map (what a receiver can do today: it cannot see "
            "data jamming); the classic power-test map with dropped pilots and erased data (Li et "
            "al.); the fused U-Net's map used softly (the map says where, the receiver estimates "
            "how strong); and the true map used the same way. Green background: JSRs where the "
            "soft P+R+E receiver beats no protection. Bands: 95% confidence; hollow triangles: no "
            "block error (95% upper bound). The title gives the BLER without any jammer."
        ),
        "python -m jamloc.experiments.recovery --channel A",
    ),
    (
        "../recovery/recovery_A_ablation.png",
        "Recovery ablation, TDL-A",
        (
            "Hard against soft use of the map: dropping flagged pilots and erasing flagged data "
            "costs BLER at weak jamming, where false alarms hit clean cells; weighting them by the "
            "estimated jamming power does not. Also the map of the U-Net without the residual "
            "(P+E), and the true map used both ways."
        ),
        "python -m jamloc.experiments.recovery --channel A",
    ),
    (
        "../recovery/recovery_C.png",
        "Map-aware recovery, TDL-C",
        "The same for TDL-C.",
        "python -m jamloc.experiments.recovery --channel C",
    ),
    (
        "../recovery/recovery_C_ablation.png",
        "Recovery ablation, TDL-C",
        "The same for TDL-C.",
        "python -m jamloc.experiments.recovery --channel C",
    ),
    (
        "../theory/residual_theory.png",
        "The DMRS residual against its theory",
        (
            "(a) Without jamming, the residual of a PRB is a weighted sum of exponentials that "
            "depends on neither the channel nor the SNR, so its 1% threshold is computed from the "
            "allocation alone: the measured false-alarm rate stays at 1% from 5 to 25 dB on both "
            "channels, with no training data. With delays beyond tau_max (unseen 1000 ns channel) "
            "the assumption breaks and so does the rate. (b) Probability of exceeding the threshold "
            "for pilot jamming of 4 of 12 PRBs. The lines show the theory. Filled markers "
            "show TDL-A and open markers show TDL-C link simulations. A neighbouring PRB can "
            "exceed the threshold when jamming energy leaks through the projection. "
            "(c), when shown, gives detection for a jammed PRB after averaging the residual "
            "over K slots. The lines show the theory and the markers show pilot-level "
            "Monte Carlo results, all at 1% false alarm."
        ),
        "python -m jamloc.experiments.residual_theory",
    ),
    (
        "../generalization/generalization.png",
        "Conditions never seen in training",
        (
            "Detectors trained on 100/300 ns channels at up to 3 m/s with a Gaussian jammer, "
            "thresholds unchanged. A QPSK jammer changes nothing. Long delays and high speed raise "
            "the false alarms of everything that uses receiver features (residual, EVM), because "
            "the channel no longer fits the delay subspace or changes within the slot; the power "
            "feature alone is robust."
        ),
        "python -m jamloc.experiments.generalization",
    ),
    (
        "../robust_training/robust_training.png",
        "Mixed training against unseen conditions",
        (
            "The same U-Nets retrained on batches cycling through nominal channels, long delay "
            "spreads and speeds up to 30 m/s, against the nominally trained ones."
        ),
        "python -m jamloc.experiments.robust_training",
    ),
    (
        "../size_sweep/size_sweep.png",
        "How small can the U-Net be?",
        (
            "F1 and Pd of the fused detector against the number of parameters (width of the first "
            "level 4, 6, 8, 12), and the CPU time per slot with feature extraction (one thread, "
            "plain PyTorch and frozen graph). A 30 kHz slot lasts 0.5 ms."
        ),
        "python -m jamloc.experiments.size_sweep",
    ),
    (
        "../margin/margin.png",
        "Detection before damage",
        (
            "The jammer covers exactly the user's PRBs at 15 dB SNR, as in the BLER experiment. "
            "Solid: share of jammed slots flagged at 1% slot false alarm; dotted: share of jammed "
            "cells flagged at 1% cell false alarm; black: BLER of the unprotected link, which is "
            "all a detector built on CRC failures or link KPIs can see. Every attack is flagged in "
            "90% of slots at least 8.2 dB of JSR before it pushes the BLER to 10%, so there is no "
            "range of jamming powers in which it hurts unseen. The residual matters most for pilot "
            "jamming, the attacker's most efficient option (margin 14.0 dB with it, 10.6 dB "
            "without, on TDL-A)."
        ),
        "python -m jamloc.experiments.detection_margin",
    ),
    (
        "../calibration/calibration.png",
        "False alarms across SNR and under noise-estimation error",
        (
            "Thresholds from benign validation slots, false alarms measured on benign test slots. "
            "Left: with one threshold the U-Nets flag almost 4% of the cells at low SNR and 0.1% "
            "at high SNR; the block test and the residual, whose nulls are exact, stay at 1%. "
            "Middle: one threshold per 4 dB SNR bin (the gNB knows its SNR) flattens the U-Nets. "
            "Right: every feature is divided by the estimated noise power, so an underestimate "
            "inflates them all. The learned detectors and the block test rely on the absolute "
            "noise floor and are very sensitive (15% false alarms at -0.5 dB for the U-Net); the "
            "residual degrades slowly and exactly as its theory predicts (grey). Overestimating "
            "the noise by 0.5 dB costs nothing. The fused U-Net trained with a noise estimate "
            "jittered by up to 1 dB (dotted) stays near 1% within +-1 dB at a cost of 0.003 F1."
        ),
        "python -m jamloc.experiments.calibration",
    ),
    (
        "../theory/multislot.png",
        "Averaging the residual over several slots",
        (
            "For a jammer that stays on the same PRBs, averaging each PRB's residual over K slots "
            "keeps the null exact (every weight divided by K and repeated K times), so the "
            "threshold is still computed, not learned. Lines: Pd from the formula; circles: "
            "pilot-level Monte Carlo with the jammer's fading held over the K slots. Eight slots "
            "(4 ms) lower the JNR needed for 90% detection from 10.7 to 4.4 dB; on the benign "
            "link-level slots the K-slot thresholds give 0.92-1.15% false alarms."
        ),
        "python -m jamloc.experiments.multislot",
    ),
    (
        "../attack_training/attack_training.png",
        "Smarter jammers and attack-mixed training",
        (
            "Two jammers that use their knowledge of the DMRS configuration: one jams only the "
            "pilot subcarriers of the DMRS symbol, the other moves its PRB block on every OFDM "
            "symbol. Trained on whole-PRB attacks, the U-Net paints whole PRB rows and fails on "
            "the hopping jammer; retrained on batches that mix the three jammers, it recovers "
            "without losing anything on the original attacks."
        ),
        "python -m jamloc.experiments.attack_training",
    ),
]


def _missing(title: str, why: str) -> str:
    return f'<div class="missing"><b>{html.escape(title)}</b>: Not produced yet - {why}</div>'


def _figure(src: str, title: str, caption: str) -> str:
    return (
        f'<figure><img src="{src}" alt="{html.escape(title)}"><figcaption><b>'
        f"{html.escape(title)}</b>: {html.escape(caption)}</figcaption></figure>"
    )


def write_gallery(captions: dict, skipped: dict) -> None:
    """``results/figures/index.html``: every figure with its caption, readable offline.

    Missing figures are listed with the command that produces them.
    """
    out = results_dir() / "figures"
    parts = ["<h2>Explanatory figures</h2>"]
    commands = {"system_diagram": "python -m jamloc.diagrams"}
    commands.update(
        {name: f"python -m jamloc.experiments.figures --only {name}" for name in FIGURES}
    )
    for name, command in commands.items():
        if name in skipped:
            parts.append(_missing(name, html.escape(skipped[name])))
        elif (out / f"{name}.png").is_file():
            parts.append(_figure(f"{name}.png", name, captions.get(name, "")))
        else:
            parts.append(_missing(name, f"run <code>{html.escape(command)}</code>"))
    parts.append("<h2>Results</h2>")
    for rel, title, caption, command in RESULT_FIGURES:
        if (out / rel).resolve().is_file():
            parts.append(_figure(rel, title, caption))
        else:
            parts.append(_missing(title, f"run <code>{html.escape(command)}</code>"))
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Figure gallery</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;"
        "background:#fff;color:#222;line-height:1.45}"
        "figure{margin:0 0 32px}img{max-width:100%;border:1px solid #ddd}"
        "figcaption{font-size:14px;margin-top:6px}"
        ".missing{background:#fff4e5;border:1px solid #f0c27b;padding:8px;margin-bottom:12px}"
        "</style></head><body><h1>Jamming localization: figures</h1>"
        "<p>Generated by <code>python -m jamloc.experiments.figures</code>; PDFs for the paper "
        "are next to each PNG.</p>" + "".join(parts) + "</body></html>"
    )
    (out / "index.html").write_text(page, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description="Explanatory figures with captions, and a gallery.")
    p.add_argument("--only", nargs="+", choices=list(FIGURES), help="only these figures")
    p.add_argument(
        "--gallery-only",
        action="store_true",
        help="only rebuild index.html from the figures already there (e.g. after "
        "unzipping a Colab run, so its reference figures are kept)",
    )
    args = p.parse_args(argv)

    out = results_dir() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    captions_path = out / "captions.json"
    captions = {}
    if captions_path.is_file():
        captions = json.loads(captions_path.read_text(encoding="utf-8"))

    captions["system_diagram"] = SYSTEM_DIAGRAM_CAPTION
    skipped = {}
    for name in [] if args.gallery_only else (args.only or FIGURES):
        try:
            with paper_style():
                fig, caption = FIGURES[name]()
        except MissingInput as missing:
            skipped[name] = str(missing)
            print(f"Skipped {name}: {missing}")
            continue
        save(fig, out / name)
        captions[name] = caption
        print(f"Wrote {out / name}.pdf/.png")
    captions_path.write_text(json.dumps(captions, indent=2), encoding="utf-8")
    write_gallery(captions, skipped)
    print(f"Gallery: {out / 'index.html'}")


if __name__ == "__main__":
    main()
