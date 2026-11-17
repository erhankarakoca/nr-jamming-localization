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
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from sionna.phy import config

from jamloc.baselines import evm_scores, residual_scores
from jamloc.config import SimConfig
from jamloc.dataset import load
from jamloc.evaluate import FA
from jamloc.features import FEATURE_NAMES, delay_subspace, dmrs_ls, power_map, residual_map
from jamloc.jammer import ATTACKS, cell_mask
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.metrics import threshold_at_fa
from jamloc.model import VARIANTS, select_inputs
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.receiver import data_positions
from jamloc.train import checkpoint_path, load_model

CFG = SimConfig()
ALLOC = (6, 12)  # example allocation used by the explanatory figures: PRBs 6-17
JAM = (10, 4)  # example jammer block: PRBs 10-13 (inside the allocation)

# Colours (Okabe-Ito) shared by the explanatory figures
ALLOC_COLOR, DMRS_COLOR, JAM_COLOR = "#CFE3F3", "#0072B2", "#D55E00"
CLEAN_COLOR = "#0072B2"


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


def attack_patterns():
    fig, axes = new_figure(7.16, 2.4, ncols=4, sharey=True)
    titles = {
        "none": "No jammer",
        "pilot": "Pilot jamming",
        "data": "Data jamming",
        "partial": "Partial-band jamming",
    }
    for ax, attack in zip(axes, ATTACKS, strict=True):
        cells = cell_mask(
            torch.tensor([ATTACKS.index(attack)]),
            torch.tensor([JAM[0]]),
            torch.tensor([JAM[1]]),
            CFG,
        )[0].numpy()
        # Scheduled PRBs, and the DMRS symbol inside them
        ax.add_patch(
            Rectangle(
                (-0.5, ALLOC[0] - 0.5),
                CFG.num_symbols,
                ALLOC[1],
                facecolor=ALLOC_COLOR,
                edgecolor="none",
            )
        )
        ax.add_patch(
            Rectangle(
                (CFG.dmrs_symbol - 0.5, ALLOC[0] - 0.5),
                1,
                ALLOC[1],
                facecolor=DMRS_COLOR,
                edgecolor="none",
            )
        )
        # Jammed cells, hatched so they stay visible on top of either colour
        for prb, sym in zip(*np.nonzero(cells), strict=True):
            ax.add_patch(
                Rectangle(
                    (sym - 0.5, prb - 0.5),
                    1,
                    1,
                    facecolor="none",
                    edgecolor=JAM_COLOR,
                    hatch="////",
                    linewidth=0.6,
                )
            )
        ax.set(
            xlim=(-0.5, CFG.num_symbols - 0.5),
            ylim=(-0.5, CFG.num_prb - 0.5),
            title=titles[attack],
            xlabel="OFDM symbol",
            xticks=[0, 2, 6, 13],
        )
        ax.grid(False)
    axes[0].set(ylabel="PRB", yticks=[0, 6, 10, 13, 17, 23])
    legend = [
        Patch(facecolor=ALLOC_COLOR, label="Scheduled PRBs (UE data)"),
        Patch(facecolor=DMRS_COLOR, label="DMRS pilots (symbol 2)"),
        Patch(facecolor="white", edgecolor=JAM_COLOR, hatch="////", label="Jammed cells = labels"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncols=3, frameon=False)
    caption = (
        "The slot as the gNB sees it: 24 PRBs (rows) by 14 OFDM symbols (columns). The UE is "
        "scheduled on PRBs 6-17 (light blue) and sends its DMRS pilots on symbol 2 (dark blue); "
        "the hatched cells are where the jammer transmits, here on PRBs 10-13. Pilot jamming "
        "hits only the DMRS symbol, data jamming every other symbol, and partial-band jamming "
        "all of them. The hatched cells are exactly the labels the detector has to find; the "
        "jammer does not know the allocation, so in general its block can also fall outside it."
    )
    return fig, caption


# --- 2. The whole system ---------------------------------------------------------------------


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.3",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.6,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5)


def _arrow(ax, start, end, dashed=False, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.8,
            "color": "#333333",
            "linestyle": "--" if dashed else "-",
            "connectionstyle": f"arc3,rad={rad}",
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def system_diagram():
    fig, ax = new_figure(7.16, 2.7)
    ax.set(xlim=(0, 100), ylim=(0, 50))
    ax.axis("off")
    tx, rx, feat, det = "#EEEEEE", "#D6E9F8", "#FBE3D2", "#D5F0E6"

    # Transmitters and channels
    _box(ax, 1, 38, 12, 8, "Jammer\n(Gaussian noise)", tx)
    _box(ax, 17, 38, 10, 8, "TDL\nchannel", tx)
    _box(ax, 1, 24, 12, 8, "UE\nPUSCH slot", tx)
    _box(ax, 17, 24, 10, 8, "TDL\nchannel", tx)
    ax.text(33, 28, "+", ha="center", va="center", fontsize=12)
    _box(ax, 37, 24, 8, 8, "thermal\nnoise", tx)
    # Standard receiver
    _box(ax, 50, 24, 10, 8, "LS channel\nestimate", rx)
    _box(ax, 64, 24, 10, 8, "LMMSE\nequalizer", rx)
    _box(ax, 78, 24, 8, 8, "demapper\n(LLRs)", rx)
    _box(ax, 90, 24, 9, 8, "LDPC\ndecoder", rx)
    # Detector
    features_text = "features per PRB x symbol:\npower P   |   DMRS residual R   |   EVM E"
    _box(ax, 40, 4, 34, 9, features_text, feat)
    _box(ax, 77, 4, 10, 9, "U-Net\n(66 k param.)", det)
    _box(ax, 90, 4, 9, 9, "jamming\nmap", det)

    for a, b in [
        ((13, 42), (17, 42)),
        ((13, 28), (17, 28)),
        ((27, 28), (31.5, 28)),
        ((34.5, 28), (37, 28)),
        ((45, 28), (50, 28)),
        ((60, 28), (64, 28)),
        ((74, 28), (78, 28)),
        ((86, 28), (90, 28)),
        ((74, 8.5), (77, 8.5)),
        ((87, 8.5), (90, 8.5)),
    ]:
        _arrow(ax, a, b)
    _arrow(ax, (27, 42), (32.5, 30), rad=-0.2)
    # The detector reads three intermediate results of the receiver
    for x_from, x_to, label in [
        (41, 46, "received grid"),
        (55, 57, "LS pilots"),
        (69, 69, "equalized symbols"),
    ]:
        _arrow(ax, (x_from, 24), (x_to, 13.5))
        ax.text(x_to + 0.8, 15.5, label, fontsize=5.5, color="#555555")
    # Map-aware receiver: the map goes back into the receiver along one dashed line
    ax.plot([94.5, 94.5, 53], [13.5, 20.5, 20.5], ls="--", lw=0.8, color="#333333")
    _arrow(ax, (53, 20.5), (53, 23.5), dashed=True)
    _arrow(ax, (82, 20.5), (82, 23.5), dashed=True)
    ax.text(
        99,
        44,
        "dashed: map-aware receiver\ndrops flagged pilots, erases flagged LLRs",
        fontsize=6,
        ha="right",
        va="center",
        color="#333333",
    )
    caption = (
        "Simulation and detector at a glance. The UE's PUSCH slot and the jammer's noise pass "
        "through independent TDL fading channels and add up at the gNB together with thermal "
        "noise. The standard receiver (top row) estimates the channel from the DMRS, equalizes, "
        "demaps and decodes. The detector reads three results the receiver computes anyway - the "
        "received grid, the LS pilot estimates and the equalized symbols - turns them into one "
        "value per PRB x symbol (power, DMRS residual, EVM), and a small U-Net outputs the "
        "jamming map. Dashed: the map-aware receiver feeds the map back to drop jammed pilots "
        "and erase jammed data."
    )
    return fig, caption


# --- 3. How the DMRS residual works ----------------------------------------------------------


def residual_explained():
    fig, axes = new_figure(7.16, 3.8, nrows=2, ncols=2, sharex="col")
    alloc_prbs = np.arange(ALLOC[0], ALLOC[0] + ALLOC[1])
    jammed = (alloc_prbs >= JAM[0]) & (alloc_prbs < JAM[0] + JAM[1])
    r_db = {}
    seed = _example_seed("C")
    for col, (attack, title) in enumerate([("none", "No jammer"), ("pilot", "Pilot jamming")]):
        batch = _example_batch(attack, 0.0, 20.0, "C", seed=seed)
        h_ls, pilot_sc, _ = dmrs_ls(batch.y_alloc, batch.tx)
        u = delay_subspace(pilot_sc, CFG.scs, CFG.tau_max)
        fit = (h_ls @ u.conj()) @ u.T  # projection onto the physically possible channels
        prb_of_pilot = ALLOC[0] + pilot_sc.cpu().numpy() / 12

        ax = axes[0, col]
        for prb in alloc_prbs[jammed] if attack == "pilot" else []:
            ax.axvspan(prb, prb + 1, color=JAM_COLOR, alpha=0.12, lw=0)
        ax.plot(
            prb_of_pilot,
            h_ls[0].abs().cpu(),
            "o",
            ms=2.5,
            color="#555555",
            label="LS estimate at each pilot",
        )
        ax.plot(
            prb_of_pilot,
            fit[0].abs().cpu(),
            "-",
            color=CLEAN_COLOR,
            label="closest physically possible channel",
        )
        ax.set(title=title, ylabel="|channel estimate|" if col == 0 else None)

        ax = axes[1, col]
        r = 10 * np.log10(residual_map(batch, CFG)[0, alloc_prbs, CFG.dmrs_symbol].cpu().numpy())
        r_db[attack] = r
        colors = [JAM_COLOR if (j and attack == "pilot") else CLEAN_COLOR for j in jammed]
        ax.bar(alloc_prbs + 0.5, r, width=0.8, color=colors)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set(
            xlabel="PRB",
            ylabel="residual R [dB]" if col == 0 else None,
            ylim=(min(-3, r.min() - 1), max(25, r.max() + 2)),
        )
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor=JAM_COLOR, alpha=0.25, label="jammed PRBs 10-13"))
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)
    distance = np.maximum(JAM[0] - alloc_prbs, alloc_prbs - (JAM[0] + JAM[1] - 1))  # PRBs away
    near = r_db["pilot"][(distance >= 1) & (distance <= 2)]
    far = r_db["pilot"][distance > 2]
    caption = (
        "How the DMRS residual tells jamming apart from the user's own signal (TDL-C, SNR 20 dB, "
        "same slot in both columns). Top: magnitude of the raw LS channel estimate at every "
        "pilot (dots) and its projection onto the channels that are physically possible with "
        "delays up to 3 us (line). A real channel varies smoothly across frequency, so without "
        "jamming the line follows the dots; on the jammed PRBs (shaded) the dots jump around in "
        "a way no real channel can. Bottom: what is left after the projection, per PRB, relative "
        f"to noise. Without jamming it stays near 0 dB (mean {_fmt_db(r_db['none'].mean())}; "
        f"single PRBs range from {_fmt_db(r_db['none'].min())} to {_fmt_db(r_db['none'].max())} "
        "because each PRB has only 6 pilots). With pilot jamming at a JSR of only 0 dB it rises "
        f"to {_fmt_db(r_db['pilot'][jammed].mean())} on the jammed PRBs. The projection also "
        f"spreads part of the jamming to nearby PRBs: up to {_fmt_db(near.max())} within two "
        f"PRBs of the block, {_fmt_db(far.mean())} on average farther away. The detector learns "
        "this spread, but it is one reason its maps sometimes reach a little beyond the block."
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


# (attack, JSR) per row: weak pilot jamming shows what the residual adds, strong pilot jamming
# shows the wrong-symbol effect of EVM, and the last two rows show data and partial-band jamming
EXAMPLE_ROWS = (("pilot", -12.0), ("pilot", 3.0), ("data", -9.0), ("partial", -9.0))


def _candidates(d: dict) -> list[np.ndarray]:
    """Test slots for each example row: jammer block fully inside a wide allocation."""
    alloc = d["features"][:, FEATURE_NAMES.index("alloc_mask")].max(axis=2) > 0
    jammed = d["labels"].any(axis=2)
    inside = (jammed & alloc).any(axis=1) & ~(jammed & ~alloc).any(axis=1)
    out = []
    for attack, jsr in EXAMPLE_ROWS:
        rows = (d["attack"] == ATTACKS.index(attack)) & (d["jsr_db"] == jsr)
        cand = np.flatnonzero(rows & inside & (d["alloc_size"] >= 10))
        if cand.size == 0:  # tiny datasets (tests): relax the conditions
            cand = np.flatnonzero(rows) if rows.any() else np.flatnonzero(d["attack"] > 0)
        out.append(cand)
    return out


def _median_slot(cand: np.ndarray, values: np.ndarray) -> int:
    return int(cand[np.argsort(values, kind="stable")[len(values) // 2]])


def detector_outputs(split: str = "test", seed: int | None = None):
    from jamloc.evaluate import available_seeds
    from jamloc.experiments.recovery import val_threshold

    if seed is None:  # the first training seed that has every model
        seed = (available_seeds() or [0])[0]
    _require_models(("fused", "no_residual", "power"), seed)
    d, val = _load_split(split), _load_split("val")
    unets = (("fused", "U-Net P+R+E"), ("no_residual", "U-Net P+E"), ("power", "U-Net power only"))
    models = {v: load_model(v, seed) for v, _ in unets}
    thresholds = {v: val_threshold(v, models[v]) for v, _ in unets}

    def predict(variant, idx):
        x = select_inputs(torch.from_numpy(d["features"][idx]).to(config.device), variant)
        with torch.no_grad():
            return models[variant](x).cpu().numpy() > thresholds[variant]

    evm_thr = threshold_at_fa(evm_scores(val["features"])[val["labels"] == 0], FA)
    res_thr = threshold_at_fa(residual_scores(val["features"])[val["labels"] == 0], FA)

    def hit_rate(p, idx):
        lab = d["labels"][idx].astype(bool)
        return (p & lab).sum(axis=(1, 2)) / np.maximum(lab.sum(axis=(1, 2)), 1)

    # Example slots, each chosen by a stated rule rather than by hand. Row 1: the median
    # advantage of the fused U-Net over the one without the residual. Row 2 (the wrong-symbol
    # effect, which only some slots show): the median slot among those where the EVM threshold
    # flags unjammed cells. Rows 3-4: the median detection rate of the fused U-Net.
    cand = _candidates(d)
    stats = {}
    picks = []
    for row, idx in enumerate(cand):
        fused_pred = predict("fused", idx)
        fused_rate = hit_rate(fused_pred, idx)
        if row == 0:
            pe_pred = predict("no_residual", idx)
            lab = d["labels"][idx].astype(bool)
            stats["fused"] = (fused_pred & lab).sum() / max(lab.sum(), 1)  # pooled over cells
            stats["pe"] = (pe_pred & lab).sum() / max(lab.sum(), 1)
            picks.append(_median_slot(idx, fused_rate - hit_rate(pe_pred, idx)))
            continue
        if row == 1:
            lab = d["labels"][idx].astype(bool)
            evm_fa = ((evm_scores(d["features"][idx]) > evm_thr) & ~lab).sum(axis=(1, 2))
            stats["evm_fa"], stats["evm_share"] = evm_fa.mean(), (evm_fa > 0).mean()
            if (evm_fa > 0).any():
                picks.append(_median_slot(idx[evm_fa > 0], evm_fa[evm_fa > 0]))
                continue
        picks.append(_median_slot(idx, fused_rate))
    stats["n"] = [len(c) for c in cand]

    preds = {title: predict(v, picks) for v, title in unets}
    preds["EVM threshold"] = evm_scores(d["features"][picks]) > evm_thr
    preds["Residual threshold"] = residual_scores(d["features"][picks]) > res_thr
    columns = ["true map", *preds]

    # 0 outside the allocation, 1 inside and correctly not flagged, 2 hit, 3 miss, 4 false alarm
    cmap = ListedColormap(["#FFFFFF", "#E3EEF8", "#009E73", "#E69F00", "#CC79A7"])
    fig, axes = new_figure(
        7.16, 5.4, nrows=len(picks), ncols=len(columns), sharex=True, sharey=True
    )
    counts = {}
    for row, i in enumerate(picks):
        truth = d["labels"][i].astype(bool)
        inside = np.zeros_like(truth)
        a0, n = int(d["alloc_start"][i]), int(d["alloc_size"][i])
        inside[a0 : a0 + n] = True
        for col, title in enumerate(columns):
            p = truth if title == "true map" else preds[title][row]
            img = np.select([p & truth, ~p & truth, p & ~truth, inside], [2, 3, 4, 1], 0)
            counts[(row, title)] = (
                int((p & truth).sum()),
                int(truth.sum()),
                int((p & ~truth).sum()),
            )
            ax = axes[row, col]
            ax.imshow(
                img,
                cmap=cmap,
                vmin=0,
                vmax=4,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            ax.grid(False)
            ax.set_xticks([0, 2, 13])
            if row == 0:
                ax.set_title(title, fontsize=7)
            if col == 0:
                attack, jsr = EXAMPLE_ROWS[row]
                ax.set_ylabel(f"{attack}, {jsr:+.0f} dB\nPRB", fontsize=7)
            if row == len(picks) - 1:
                ax.set_xlabel("symbol")
    legend = [
        Patch(facecolor="#009E73", label="hit (jammed, flagged)"),
        Patch(facecolor="#E69F00", label="miss (jammed, not flagged)"),
        Patch(facecolor="#CC79A7", label="false alarm"),
        Patch(facecolor="#E3EEF8", edgecolor="#999999", label="UE allocation"),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncols=4, frameon=False)

    caption = (
        "What each detector reports on four test slots (jammer block inside the allocation), all "
        "thresholded at 1% false alarm on the validation set. Green: jammed cells found; orange: "
        "missed; pink: flagged although not jammed; light blue: the UE's allocation. Each row "
        "is picked by a stated rule, not by hand: row 1 is the slot with the median advantage of "
        "the fused U-Net over the U-Net without the residual, row 2 the median slot among those "
        "where the EVM threshold flags unjammed cells, rows 3-4 the slot with the median "
        "detection rate of the fused U-Net. Weak pilot jamming (row 1): over all "
        f"{stats['n'][0]} such test slots the fused U-Net finds {stats['fused']:.0%} of the jammed "
        f"DMRS cells and the U-Net without the residual {stats['pe']:.0%}. Strong pilot jamming "
        f"(row 2): the EVM threshold never finds the jammed DMRS cells (it has no value there), "
        f"and in {stats['evm_share']:.0%} of the {stats['n'][1]} slots it flags data cells that "
        f"were never jammed ({stats['evm_fa']:.1f} per slot on average) - the wrong-symbol effect. "
        "The residual threshold cannot see data jamming at all."
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


FIGURES = {
    "attack_patterns": attack_patterns,
    "system_diagram": system_diagram,
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
            "for pilot jamming of 4 of 12 PRBs: lines are the formula, markers the link simulation; "
            "the PRB next to the block is a false alarm caused by leakage of the projection."
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
    for name in FIGURES:
        if name in skipped:
            parts.append(_missing(name, html.escape(skipped[name])))
        elif (out / f"{name}.png").is_file():
            parts.append(_figure(f"{name}.png", name, captions.get(name, "")))
        else:
            parts.append(
                _missing(
                    name, f"run <code>python -m jamloc.experiments.figures --only {name}</code>"
                )
            )
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
