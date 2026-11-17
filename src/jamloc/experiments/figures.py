"""Figures that explain *why* things behave as they do, each with a caption.

Every figure function returns ``(figure, caption)``. The caption says what the figure shows, how
to read it and why it matters; numbers quoted in a caption are computed from the same data as
the figure, so they cannot drift apart.

    python -m jamloc.experiments.figures                  # all figures + gallery
    python -m jamloc.experiments.figures --only constellations

Output: ``results/figures/{name}.pdf`` and ``.png``, ``captions.json`` and ``index.html``.
"""

import argparse
import json

import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from jamloc.config import SimConfig
from jamloc.features import delay_subspace, dmrs_ls, residual_map
from jamloc.jammer import ATTACKS, cell_mask
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.paths import results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.receiver import data_positions

CFG = SimConfig()
ALLOC = (6, 12)  # example allocation used by the explanatory figures: PRBs 6-17
JAM = (10, 4)  # example jammer block: PRBs 10-13 (inside the allocation)

# Colours (Okabe-Ito) shared by the explanatory figures
ALLOC_COLOR, DMRS_COLOR, JAM_COLOR = "#CFE3F3", "#0072B2", "#D55E00"
CLEAN_COLOR = "#0072B2"


def _example_batch(attack: str, jsr_db: float, snr_db: float, channel: str, seed: int = 7):
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
    return JammedPUSCHLink(CFG)(sc)


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
    for col, (attack, title) in enumerate([("none", "No jammer"), ("pilot", "Pilot jamming")]):
        batch = _example_batch(attack, 0.0, 20.0, "C")
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
    neighbours = r_db["pilot"][[JAM[0] - 1 - ALLOC[0], JAM[0] + JAM[1] - ALLOC[0]]]
    caption = (
        "How the DMRS residual tells jamming apart from the user's own signal (TDL-C, SNR 20 dB, "
        "same slot in both columns). Top: magnitude of the raw LS channel estimate at every "
        "pilot (dots) and its projection onto the channels that are physically possible with "
        "delays up to 3 us (line). A real channel varies smoothly across frequency, so without "
        "jamming the line follows the dots; on the jammed PRBs (shaded) the dots jump around in "
        "a way no real channel can. Bottom: what is left after the projection, per PRB, relative "
        f"to noise. Without jamming it stays near 0 dB (mean {r_db['none'].mean():.1f} dB); on "
        f"the jammed PRBs it rises to {r_db['pilot'][jammed].mean():.0f} dB on average, at a "
        "JSR of only 0 dB. Two details: with only 6 pilots per PRB the clean values swing by a few "
        f"dB ({r_db['none'].min():.0f} to {r_db['none'].max():.0f} dB here), and the PRBs right "
        f"next to the jammed block rise a little (about {neighbours.mean():.0f} dB) "
        "because the projection spreads part of the jamming into its neighbours. The detector "
        "learns both effects."
    )
    return fig, caption


# --- 4. Why pilot jamming shows up in the data symbols -----------------------------------------


def constellations():
    fig, axes = new_figure(7.16, 2.6, ncols=3, sharex=True, sharey=True)
    titles = {"none": "No jammer", "pilot": "Pilot jamming", "data": "Data jamming"}
    spread = {}
    for ax, attack in zip(axes, ("none", "pilot", "data"), strict=True):
        # Seed 12: a channel draw without a deep fade on PRBs 10-13, so the only difference
        # between the panels is the jammer (a fade alone would also scatter those symbols)
        batch = _example_batch(attack, 0.0, 20.0, "A", seed=12)
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


FIGURES = {
    "attack_patterns": attack_patterns,
    "system_diagram": system_diagram,
    "residual_explained": residual_explained,
    "constellations": constellations,
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Explanatory figures with captions.")
    p.add_argument("--only", nargs="+", choices=list(FIGURES), help="only these figures")
    args = p.parse_args(argv)

    out = results_dir() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    captions_path = out / "captions.json"
    captions = {}
    if captions_path.is_file():
        captions = json.loads(captions_path.read_text(encoding="utf-8"))

    for name in args.only or FIGURES:
        with paper_style():
            fig, caption = FIGURES[name]()
        save(fig, out / name)
        captions[name] = caption
        print(f"Wrote {out / name}.pdf/.png")
    captions_path.write_text(json.dumps(captions, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
