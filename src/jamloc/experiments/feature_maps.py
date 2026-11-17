"""Feature maps of one slot per attack type (benign, pilot, data, partial-band).

Rows are attacks; columns are the label (jammed cells) and the three features in dB. The UE is
scheduled on PRBs 4-15 and the jammer hits PRBs 10-17, so part of the jammed block lies outside
the allocation, where only the power feature can see it. Gray cells: feature not defined.

    python -m jamloc.experiments.feature_maps --snr-db 15 --jsr-db 0
"""

import argparse

import matplotlib as mpl
import numpy as np
import torch

from jamloc.config import SimConfig
from jamloc.features import extract, masks
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.paths import results_dir
from jamloc.plotting import new_figure, paper_style, save

ALLOC = (4, 12)  # first PRB, number of PRBs
JAM = (10, 8)
COLUMNS = ("label", "power_db", "residual_db", "evm_db")
TITLES = {
    "label": "Jammed cells",
    "power_db": "Power P [dB]",
    "residual_db": "DMRS residual R [dB]",
    "evm_db": "EVM E [dB]",
}


def simulate(snr_db: float, jsr_db: float, channel: str, seed: int):
    cfg = SimConfig()
    seed_everything(seed)
    n = len(ATTACKS)
    sc = Scenario(
        channel=channel,
        alloc_start=ALLOC[0],
        alloc_size=ALLOC[1],
        attack=torch.arange(n),
        jam_start=torch.full((n,), JAM[0]),
        jam_size=torch.full((n,), JAM[1]),
        jsr_db=torch.full((n,), jsr_db),
        snr_db=torch.full((n,), snr_db),
    )
    batch = JammedPUSCHLink(cfg)(sc)
    f = extract(batch, cfg).cpu().numpy()
    _, dmrs, data = (m.cpu().numpy() for m in masks(batch, cfg))
    maps = {
        "label": batch.labels.cpu().numpy().astype(float),
        "power_db": f[:, 0],
        "residual_db": np.where(dmrs > 0, f[:, 1], np.nan),  # NaN = not defined
        "evm_db": np.where(data > 0, f[:, 2], np.nan),
    }
    return maps


def plot(maps, snr_db, jsr_db, channel):
    fig, axes = new_figure(
        7.16, 6.0, nrows=len(ATTACKS), ncols=len(COLUMNS), sharex=True, sharey=True
    )
    for j, col in enumerate(COLUMNS):
        values = maps[col]
        if col == "label":
            cmap, vmin, vmax = mpl.colormaps["Greys"], 0, 1
        else:
            cmap = mpl.colormaps["viridis"].with_extremes(bad="#BBBBBB")
            finite = values[np.isfinite(values)]
            vmin, vmax = float(finite.min()), float(finite.max())  # same scale in a column
        for i, attack in enumerate(ATTACKS):
            ax = axes[i, j]
            im = ax.imshow(
                values[i],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            ax.grid(False)
            if i == 0:
                ax.set_title(TITLES[col])
            if j == 0:
                ax.set_ylabel(f"{attack}\nPRB")
            if i == len(ATTACKS) - 1:
                ax.set_xlabel("OFDM symbol")
        if col != "label":
            fig.colorbar(im, ax=axes[:, j], shrink=0.6)
    fig.suptitle(
        f"TDL-{channel}, SNR {snr_db:g} dB, JSR {jsr_db:g} dB per jammed RE; "
        f"UE on PRBs {ALLOC[0]}-{sum(ALLOC) - 1}, jammer on PRBs {JAM[0]}-{sum(JAM) - 1}",
        fontsize=8,
    )
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--snr-db", type=float, default=15.0)
    p.add_argument("--jsr-db", type=float, default=0.0)
    p.add_argument("--channel", choices=("A", "C"), default="A")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    maps = simulate(args.snr_db, args.jsr_db, args.channel, args.seed)
    stem = results_dir() / "feature_maps"
    with paper_style():
        fig = plot(maps, args.snr_db, args.jsr_db, args.channel)
    save(fig, stem)
    print(f"Wrote {stem}.pdf/.png")


if __name__ == "__main__":
    main()
