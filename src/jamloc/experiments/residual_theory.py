"""Does the DMRS residual behave as the theory says? (``jamloc.theory``)

(a) **False alarm without jamming, against SNR.** Every scheduled DMRS cell of the benign test
    slots is compared with its analytic 1% threshold (from the allocation size and the PRB's
    position, no training data). The theory says the rate is 1% at every SNR and on every channel
    as long as the channel's delays stay below ``tau_max``; the ``longdelay`` split (TDL-C with
    delays up to 8.7 us) shows what happens when they do not.
(b) **Detection against the jammer-to-noise ratio.** UE on 12 PRBs, pilot jamming of 4 PRBs in the
    middle of the allocation, SNR 10 dB. Simulated detection rates of the jammed PRBs and the
    false-alarm rate of the unjammed PRB next to the block (leakage) against the formula
    (jammer channel: block Rayleigh fading).

    python -m jamloc.experiments.residual_theory
"""

import argparse
import csv
import itertools

import numpy as np
import torch

from jamloc import theory
from jamloc.config import SimConfig
from jamloc.dataset import CHANNELS, load
from jamloc.features import FEATURE_NAMES, residual_map
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.metrics import wilson_interval
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import new_figure, paper_style, save

FA = 0.01
ALLOC = (6, 12)  # UE on PRBs 6-17
JAM = (4, 4)  # jammed PRBs 4-7 of the allocation (PRBs 10-13 of the carrier)
GROUPS = {"inside the block": (5, 6), "block edge": (4, 7), "next to the block": (3, 8)}
SNR_BINS = np.arange(5.0, 25.1, 2.5)


def false_alarm_vs_snr(split: str) -> list[dict]:
    """Analytic-threshold false alarms on the scheduled DMRS cells of a split's benign slots."""
    d = load(split)
    benign = d["attack"] == 0
    dmrs = SimConfig().dmrs_symbol
    r_all = 10 ** (d["features"][:, FEATURE_NAMES.index("residual_db"), :, dmrs] / 10)
    rows = []
    for ch in np.unique(d["channel"][benign]):
        flags, snrs = [], []
        for i in np.flatnonzero(benign & (d["channel"] == ch)):
            n, start = int(d["alloc_size"][i]), int(d["alloc_start"][i])
            thr = np.array([theory.threshold(n, pos, FA) for pos in range(n)])
            flags.append(r_all[i, start : start + n] > thr)
            snrs.append(np.full(n, d["snr_db"][i]))
        flags, snrs = np.concatenate(flags), np.concatenate(snrs)
        for lo, hi in itertools.pairwise(SNR_BINS):
            sel = (snrs >= lo) & (snrs < hi)
            rows.append(
                {
                    "split": split,
                    "channel": CHANNELS[ch],
                    "snr_db": (lo + hi) / 2,
                    "false_alarms": int(flags[sel].sum()),
                    "cells": int(sel.sum()),
                }
            )
    return rows


def simulate_detection(jnr_dbs, snr_db: float, slots: int, batch_size: int) -> list[dict]:
    cfg = SimConfig()
    link = JammedPUSCHLink(cfg)
    seed_everything(8_000_000)
    rows = []
    for channel in CHANNELS:
        for jnr in jnr_dbs:
            exceed = np.zeros(ALLOC[1])
            count = 0
            for _ in range(int(np.ceil(slots / batch_size))):
                n = batch_size
                sc = Scenario(
                    channel=channel,
                    alloc_start=ALLOC[0],
                    alloc_size=ALLOC[1],
                    attack=torch.full((n,), ATTACKS.index("pilot")),
                    jam_start=torch.full((n,), ALLOC[0] + JAM[0]),
                    jam_size=torch.full((n,), JAM[1]),
                    jsr_db=torch.full((n,), float(jnr - snr_db)),
                    snr_db=torch.full((n,), float(snr_db)),
                )
                batch = link(sc)
                r = (
                    residual_map(batch, cfg)[:, ALLOC[0] : sum(ALLOC), cfg.dmrs_symbol]
                    .cpu()
                    .numpy()
                )
                thr = np.array([theory.threshold(ALLOC[1], pos, FA) for pos in range(ALLOC[1])])
                exceed += (r > thr).sum(0)
                count += n
            for group, positions in GROUPS.items():
                rows.append(
                    {
                        "channel": channel,
                        "jnr_db": float(jnr),
                        "group": group,
                        "exceed": int(exceed[list(positions)].sum()),
                        "cells": count * len(positions),
                    }
                )
            print(
                f"TDL-{channel} JNR {jnr:+5.1f} dB: "
                + "  ".join(f"{r['group']} {r['exceed'] / r['cells']:.3f}" for r in rows[-3:]),
                flush=True,
            )
    return rows


def theory_curves(jnr_grid) -> dict:
    jammed = range(JAM[0], sum(JAM))
    n = ALLOC[1]
    return {
        group: np.mean(
            [
                theory.detection_probability(jnr_grid, n, pos, jammed, FA, draws=4000)
                for pos in positions
            ],
            axis=0,
        )
        for group, positions in GROUPS.items()
    }


CHANNEL_STYLE = {
    ("test", "A"): {"color": "#0072B2", "marker": "o", "label": "TDL-A (100 ns)"},
    ("test", "C"): {"color": "#D55E00", "marker": "s", "label": "TDL-C (300 ns)"},
    ("longdelay", "C"): {"color": "#CC79A7", "marker": "D", "label": "TDL-C, 1000 ns (unseen)"},
}
GROUP_STYLE = {
    "inside the block": {"color": "#0072B2", "marker": "o"},
    "block edge": {"color": "#56B4E9", "marker": "^"},
    "next to the block": {"color": "#D55E00", "marker": "s"},
}


def plot(fa_rows, pd_rows, curves, jnr_grid):
    fig, (ax_fa, ax_pd) = new_figure(7.16, 2.6, ncols=2)
    for (split, ch), style in CHANNEL_STYLE.items():
        sel = [r for r in fa_rows if r["split"] == split and r["channel"] == ch]
        if not sel:
            continue
        x = np.array([r["snr_db"] for r in sel])
        k = np.array([r["false_alarms"] for r in sel])
        n = np.array([r["cells"] for r in sel])
        lo, hi = wilson_interval(k, n)
        ax_fa.errorbar(x, k / n, yerr=[k / n - lo, hi - k / n], capsize=2, ms=3, **style)
    ax_fa.axhline(FA, color="k", lw=0.8, ls=":", label="design value (1%)")
    ax_fa.set(
        yscale="log",
        xlabel="SNR [dB]",
        ylabel="False-alarm rate, benign slots",
        title="(a) Analytic threshold, no training data",
    )
    ax_fa.legend(loc="upper left", fontsize=6)

    for group, style in GROUP_STYLE.items():
        ax_pd.plot(jnr_grid, curves[group], color=style["color"], lw=1.0)
        for ch, mfc in (("A", style["color"]), ("C", "none")):
            sel = [r for r in pd_rows if r["group"] == group and r["channel"] == ch]
            k = np.array([r["exceed"] for r in sel])
            n = np.array([r["cells"] for r in sel])
            ax_pd.plot(
                [r["jnr_db"] for r in sel],
                k / n,
                ls="none",
                marker=style["marker"],
                mfc=mfc,
                mec=style["color"],
                label=f"{group}, TDL-{ch}",
            )
    ax_pd.plot([], [], color="k", lw=1.0, label="theory (lines)")
    ax_pd.set(
        xlabel="Jammer-to-noise ratio per RE [dB]",
        ylabel="P(residual above 1% threshold)",
        title="(b) Pilot jamming of 4 of 12 PRBs",
        ylim=(0, 1.02),
    )
    ax_pd.legend(loc="upper left", fontsize=5.5, ncols=1)
    return fig


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [
            {k: (v if k in ("split", "channel", "group") else float(v)) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jnr-db", type=float, nargs="+", default=list(np.arange(-6.0, 19.0, 3.0)))
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--slots", type=int, default=512, help="slots per JNR point and channel")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args(argv)

    out = results_dir() / "theory"
    out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        fa_rows = []
        for split in ("test", "longdelay"):
            if (data_dir() / f"{split}.npz").is_file():
                fa_rows += false_alarm_vs_snr(split)
        _write(fa_rows, out / "residual_false_alarm.csv")
        pd_rows = simulate_detection(args.jnr_db, args.snr_db, args.slots, args.batch_size)
        _write(pd_rows, out / "residual_detection.csv")
    fa_rows = _read(out / "residual_false_alarm.csv")
    pd_rows = _read(out / "residual_detection.csv")
    jnr_grid = np.linspace(min(r["jnr_db"] for r in pd_rows), max(r["jnr_db"] for r in pd_rows), 41)
    with paper_style():
        fig = plot(fa_rows, pd_rows, theory_curves(jnr_grid), jnr_grid)
    save(fig, out / "residual_theory")
    total = {}
    for r in fa_rows:
        key = (r["split"], r["channel"])
        k, n = total.get(key, (0, 0))
        total[key] = (k + r["false_alarms"], n + r["cells"])
    for (split, ch), (k, n) in total.items():
        print(f"{split} TDL-{ch}: false-alarm rate {k / n:.4f} over {int(n)} benign DMRS cells")
    print(f"Wrote {out / 'residual_theory'}.pdf/.png and the CSVs")


if __name__ == "__main__":
    main()
