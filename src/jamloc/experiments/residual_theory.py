"""Do the known-symbol residuals behave as the theory says? (``jamloc.theory``)

(a) **Calibration of the null laws.** The fraction of benign cells whose p-value under the analytic
    law falls below a level, against the level: the DMRS residual of the scheduled DMRS cells of
    the benign test slots (and of the ``longdelay`` split, whose channels leave the model), and the
    decoded-data residual of the data and DMRS cells of benign slots that pass their CRC
    (``MapAwareReceiver.decoded_cell_test``). An exact law lies on the diagonal.
(b) **False alarm without jamming, against SNR.** Every scheduled DMRS cell of the benign test
    slots is compared with its analytic 1% threshold (from the allocation size and the PRB's
    position, no training data). The theory says the rate is 1% at every SNR and on every channel
    as long as the channel's delays stay below ``tau_max``; the ``longdelay`` split (TDL-C with
    delays up to 8.7 us) shows what happens when they do not.
    The decoded-data residual of the data cells is shown as well.
(c) **Detection against the jammer-to-noise ratio.** UE on 12 PRBs, pilot jamming of 4 PRBs in the
    middle of the allocation, SNR 10 dB. Simulated detection rates of the jammed PRBs and the
    false-alarm rate of the unjammed PRB next to the block (leakage) against the formula
    (jammer channel: block Rayleigh fading).
(d) **K-slot detection.** When available, the saved results from ``multislot`` compare theory
    and pilot-level Monte Carlo for the average residual of a jammed PRB over K slots.

    python -m jamloc.experiments.residual_theory
    python -m jamloc.experiments.residual_theory --plot-only
"""

import argparse
import csv
import itertools

import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

from jamloc import theory
from jamloc.config import SimConfig
from jamloc.dataset import CHANNELS, draw_scenario, load
from jamloc.experiments.multislot import plot_on_axis, read_results
from jamloc.features import FEATURE_NAMES, residual_map
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.metrics import wilson_interval
from jamloc.paths import data_dir, results_dir
from jamloc.plotting import BLACK, BLUE, RED, new_figure, paper_style, save
from jamloc.recovery import MapAwareReceiver

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


LEVELS = np.logspace(-3, 0, 25)


def _calibration(kind: str, channel: str, log10_p: np.ndarray) -> list[dict]:
    return [
        {
            "kind": kind,
            "channel": channel,
            "level": float(a),
            "below": int((log10_p < np.log10(a)).sum()),
            "cells": int(log10_p.size),
        }
        for a in LEVELS
    ]


def dmrs_calibration(split: str) -> list[dict]:
    """Calibration of the DMRS residual law on the scheduled DMRS cells of benign slots."""
    d = load(split)
    benign = d["attack"] == 0
    dmrs = SimConfig().dmrs_symbol
    r_all = 10 ** (d["features"][:, FEATURE_NAMES.index("residual_db"), :, dmrs] / 10)
    rows = []
    for ch in np.unique(d["channel"][benign]):
        logp = []
        for i in np.flatnonzero(benign & (d["channel"] == ch)):
            n, start = int(d["alloc_size"][i]), int(d["alloc_start"][i])
            r = r_all[i, start : start + n]
            logp += [theory.log10_pvalue(r[pos], n, pos) for pos in range(n)]
        kind = "dmrs" if split == "test" else f"dmrs_{split}"
        rows += _calibration(kind, CHANNELS[ch], np.asarray(logp, float))
    return rows


def decoded_null(slots: int, batch_size: int) -> tuple[list[dict], list[dict]]:
    """Calibration and false alarms against SNR of the decoded-data residual on benign slots
    (TDL-A and TDL-C, SNR uniform in 5-25 dB), decoded by the standard receiver; only slots that
    pass their CRC are kept, as a receiver would."""
    cfg = SimConfig()
    link = JammedPUSCHLink(cfg)
    cal, fa = [], []
    data_syms = [s for s in range(cfg.num_symbols) if s not in cfg.dmrs_symbols]
    for channel in CHANNELS:
        logp = {"decoded_data": [], "decoded_dmrs": []}
        flags, snrs = [], []
        for b in range(int(np.ceil(slots / batch_size))):
            rng = seed_everything(9_000_000 + 1000 * CHANNELS.index(channel) + b)
            sc = draw_scenario(
                rng,
                batch_size,
                cfg,
                channel,
                attacks=(0,),
                jsr_db=0.0,
                snr_db=(5.0, 25.0),
                scheduling_aware=False,
            )
            out = link(sc, decode=True)
            ok = out.rx["tb_crc"].reshape(-1).bool()
            rx = MapAwareReceiver(out.tx, cfg)
            with torch.no_grad():
                flag, r, count = rx.decoded_cell_test(
                    out.y_alloc, out.no, rx.decoded_symbols(out.rx["b_hat"]), FA
                )
            m = count.double()
            logp_all = torch.log10(
                torch.special.gammaincc(m.clamp_min(1), m * r.double()).clamp_min(1e-300)
            )
            for kind, syms in (
                ("decoded_data", data_syms),
                ("decoded_dmrs", list(cfg.dmrs_symbols)),
            ):
                logp[kind].append(logp_all[ok][:, :, syms].flatten().cpu().numpy())
            f = flag[ok][:, :, data_syms].flatten(1).cpu().numpy()
            flags.append(f.reshape(-1))
            snrs.append(np.repeat(sc.snr_db[ok.cpu()].numpy(), f.shape[1]))
        for kind, values in logp.items():
            cal += _calibration(kind, channel, np.concatenate(values))
        flags, snrs = np.concatenate(flags), np.concatenate(snrs)
        for lo, hi in itertools.pairwise(SNR_BINS):
            sel = (snrs >= lo) & (snrs < hi)
            fa.append(
                {
                    "split": "decoded",
                    "channel": channel,
                    "snr_db": (lo + hi) / 2,
                    "false_alarms": int(flags[sel].sum()),
                    "cells": int(sel.sum()),
                }
            )
        print(f"decoded residual, TDL-{channel}: done", flush=True)
    return cal, fa


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
    ("test", "A"): (BLUE, "o", "-", "TDL-A (100 ns)"),
    ("test", "C"): (RED, "s", "--", "TDL-C (300 ns)"),
    ("longdelay", "C"): (BLACK, "D", "-.", "TDL-C (1000 ns)"),
    ("decoded", "C"): ("#2CA02C", "^", ":", "Decoded data"),
}
CALIBRATION_STYLE = {
    ("dmrs", "A"): (BLUE, "o", "-", "DMRS, TDL-A"),
    ("dmrs", "C"): (RED, "s", "--", "DMRS, TDL-C"),
    ("decoded_data", "C"): ("#2CA02C", "^", ":", "Decoded, data"),
    ("decoded_dmrs", "C"): ("#9467BD", "v", ":", "Decoded, DMRS"),
    ("dmrs_longdelay", "C"): (BLACK, "D", "-.", "DMRS, 1000 ns"),
}
GROUP_STYLE = {
    "next to the block": (BLACK, "s", "-.", "Next to the block"),
    "inside the block": (RED, "o", "-", "Inside the block"),
    "block edge": (BLUE, "^", "--", "Block edge"),
}


def plot_calibration(ax, cal_rows):
    for (kind, ch), (color, marker, line, label) in CALIBRATION_STYLE.items():
        sel = [r for r in cal_rows if r["kind"] == kind and r["channel"] == ch and r["cells"]]
        if not sel:
            continue
        x = np.array([r["level"] for r in sel])
        y = np.array([r["below"] / r["cells"] for r in sel])
        ax.plot(
            x,
            y,
            color=color,
            ls=line,
            lw=0.9,
            marker=marker,
            ms=2.5,
            mfc="none",
            markevery=4,
            label=label,
        )
    ax.plot([1e-3, 1], [1e-3, 1], color="black", lw=0.6, ls=(0, (1, 2)))
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(1e-3, 1),
        ylim=(5e-4, 1.2),
        xlabel="Level of the analytic law",
        ylabel="False-alarm rate",
    )
    ax.set_title("(a)", loc="left")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.02, -0.02),
        frameon=False,
        labelspacing=0.25,
        handlelength=1.5,
        handletextpad=0.4,
        fontsize=6,
    )


def plot(fa_rows, pd_rows, curves, jnr_grid, multislot_results=None, cal_rows=None):
    ncols = 2 + (multislot_results is not None) + (cal_rows is not None)
    fig, axes = new_figure(7.16, 2.1, ncols=ncols)
    if cal_rows is not None:
        plot_calibration(axes[0], cal_rows)
        axes = axes[1:]
    ax_fa, ax_pd = axes[:2]
    for (split, ch), (color, marker, line, label) in CHANNEL_STYLE.items():
        sel = [r for r in fa_rows if r["split"] == split and r["channel"] == ch and r["cells"]]
        if not sel:
            continue
        x = np.array([r["snr_db"] for r in sel])
        k = np.array([r["false_alarms"] for r in sel])
        n = np.array([r["cells"] for r in sel])
        lo, hi = wilson_interval(k, n)
        ax_fa.errorbar(
            x,
            k / n,
            yerr=[k / n - lo, hi - k / n],
            capsize=2,
            ms=3,
            mfc="none",
            lw=0.8,
            elinewidth=0.6,
            capthick=0.6,
            color=color,
            marker=marker,
            ls=line,
            label=label,
        )
    ax_fa.axhline(FA, color="black", lw=0.8, ls=":", label="1% target")
    ax_fa.set(
        yscale="log",
        yticks=[0.01, 0.1, 1],
        xlabel="SNR (dB)",
        ylabel="False-alarm rate",
    )
    ax_fa.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_fa.set_title("(b)" if cal_rows is not None else "(a)", loc="left")
    ax_fa.legend(loc="center left", bbox_to_anchor=(0, 0.46), frameon=False)

    for group, (color, marker, line, label) in GROUP_STYLE.items():
        ax_pd.plot(jnr_grid, curves[group], color=color, ls=line, lw=0.9, label=label)
        # Marker shape identifies the PRB group; fill distinguishes the two channels.
        for ch, mfc in (("A", color), ("C", "none")):
            sel = [r for r in pd_rows if r["group"] == group and r["channel"] == ch]
            k = np.array([r["exceed"] for r in sel])
            n = np.array([r["cells"] for r in sel])
            ax_pd.plot(
                [r["jnr_db"] for r in sel],
                k / n,
                ls="none",
                marker=marker,
                mfc=mfc,
                mec=color,
                ms=3,
                mew=0.65,
            )
    ax_pd.set(
        xlabel="JNR (dB)",
        ylabel="Exceedance probability",
        ylim=(-0.03, 1.08),
    )
    ax_pd.set_title("(c)" if cal_rows is not None else "(b)", loc="left")
    ax_pd.legend(
        loc="upper left", frameon=False, labelspacing=0.35, handlelength=1.6, handletextpad=0.5
    )
    if multislot_results is not None:
        plot_on_axis(axes[2], *multislot_results)
        axes[2].set_title("(d)" if cal_rows is not None else "(c)", loc="left")
    for ax in fig.axes:
        ax.tick_params(length=2.5, width=0.6, pad=2)
    return fig


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [
            {
                k: (v if k in ("split", "channel", "group", "kind") else float(v))
                for k, v in r.items()
            }
            for r in csv.DictReader(f)
        ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jnr-db", type=float, nargs="+", default=list(np.arange(-6.0, 19.0, 3.0)))
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--slots", type=int, default=512, help="slots per JNR point and channel")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--decoded-slots", type=int, default=2048, help="benign slots per channel")
    p.add_argument("--only-decoded", action="store_true", help="redo only the new panels' data")
    args = p.parse_args(argv)

    out = results_dir() / "theory"
    out.mkdir(parents=True, exist_ok=True)
    if args.only_decoded:
        cal = []
        for split in ("test", "longdelay"):
            if (data_dir() / f"{split}.npz").is_file():
                cal += dmrs_calibration(split)
        dcal, dfa = decoded_null(args.decoded_slots, args.batch_size)
        _write(cal + dcal, out / "residual_calibration.csv")
        _write(dfa, out / "decoded_false_alarm.csv")
    if not args.plot_only and not args.only_decoded:
        fa_rows = []
        for split in ("test", "longdelay"):
            if (data_dir() / f"{split}.npz").is_file():
                fa_rows += false_alarm_vs_snr(split)
        _write(fa_rows, out / "residual_false_alarm.csv")
        pd_rows = simulate_detection(args.jnr_db, args.snr_db, args.slots, args.batch_size)
        _write(pd_rows, out / "residual_detection.csv")
    fa_rows = _read(out / "residual_false_alarm.csv")
    if (out / "decoded_false_alarm.csv").is_file():
        fa_rows += _read(out / "decoded_false_alarm.csv")
    cal_rows = None
    if (out / "residual_calibration.csv").is_file():
        cal_rows = _read(out / "residual_calibration.csv")
    pd_rows = _read(out / "residual_detection.csv")
    jnr_grid = np.linspace(min(r["jnr_db"] for r in pd_rows), max(r["jnr_db"] for r in pd_rows), 41)
    try:
        multislot_results = read_results(out)
    except FileNotFoundError as missing:
        print(f"Skipping the K-slot panel: {missing.filename} is missing")
        multislot_results = None
    with paper_style():
        fig = plot(fa_rows, pd_rows, theory_curves(jnr_grid), jnr_grid, multislot_results, cal_rows)
    save(fig, out / "residual_theory")
    total = {}
    for r in fa_rows:
        key = (r["split"], r["channel"])
        k, n = total.get(key, (0, 0))
        total[key] = (k + r["false_alarms"], n + r["cells"])
    for (split, ch), (k, n) in total.items():
        print(f"{split} TDL-{ch}: false-alarm rate {k / n:.4f} over {int(n)} benign DMRS cells")
    print(f"Wrote {out / 'residual_theory'}.pdf/.png")


if __name__ == "__main__":
    main()
