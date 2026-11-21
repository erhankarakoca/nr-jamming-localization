"""How wide can the jammer be? Detection and decoding against the number of jammed PRBs.

Both parts look at the jammer's width, and so at how much of the allocation's pilots (pilot and
partial-band attacks) or data cells (data and partial-band attacks) it covers:

* ``detection`` - Pd of the fused U-Net at 1% per-cell false alarm (thresholds from ``val``, as in
  ``jamloc.evaluate``) on the ``test`` split, per attack, JSR and jammer width, over the jammed
  cells inside the allocation; mean and sample standard deviation over the training seeds. The
  split's jammer blocks are 2-12 PRBs wide at random places on the 24-PRB carrier (training draws
  them the same way), so a block can cover none, part or all of the allocation: Pd is also given
  against the fraction of the allocation's PRBs the jammer covers.
* ``decoding`` - BLER of the recovery receivers with the jammer on 2 ... 12 of the allocation's 12
  PRBs, from the runs of ``recovery --jam-prb N`` (TDL-C, 15 dB SNR), run with
  ``JAMLOC_RESULTS_DIR`` set to ``results/width_sweep`` (so they land in its ``recovery`` folder).
  Pooled over the detector seeds, with 95% Wilson intervals.

    python -m jamloc.experiments.width_sweep                    # both parts
    python -m jamloc.experiments.width_sweep --part detection   # no BLER runs needed
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from jamloc.dataset import load
from jamloc.features import FEATURE_NAMES
from jamloc.jammer import ATTACKS
from jamloc.metrics import detection_rate, threshold_at_fa, wilson_interval
from jamloc.paths import results_dir
from jamloc.plotting import BLACK, BLUE, GRAY, GREEN, RED, new_figure, paper_style, save

FA = 0.01
NOMINAL_ATTACKS = ("pilot", "data", "partial")
# Fraction of the allocation's PRBs the jammer covers, in bins (0: misses the allocation)
COVER_BINS = ((0.0, 0.0), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
WIDTHS = (2, 4, 6, 8, 10, 12)  # jammed PRBs of the BLER runs (allocation: 12 PRBs)
BLER_RECEIVERS = ("standard", "lmmse", "p_r_e_soft", "p_r_e_soft_da", "true_soft", "true_soft_da")
# Legend names of this figure (the recovery experiment's styles otherwise)
BLER_LABELS = {
    "standard": "No protection",
    "lmmse": "LMMSE, no map",
    "p_r_e_soft": "U-Net map",
    "p_r_e_soft_da": "U-Net map, data-aided",
    "true_soft": "True map",
    "true_soft_da": "True map, data-aided",
}
ATTACK_TITLES = {
    "pilot": "Pilot jamming",
    "data": "Data jamming",
    "partial": "Partial-band jamming",
}
JSR_COLORS = {-15.0: GRAY, -9.0: GREEN, -3.0: BLUE, 3.0: RED, 9.0: BLACK}


def out_dir():
    path = results_dir() / "width_sweep"
    path.mkdir(parents=True, exist_ok=True)
    return path


def allocation_cover(d: dict) -> np.ndarray:
    """Per slot: fraction of the allocation's PRBs inside the jammer's block."""
    j0, jn = d["jam_start"].astype(int), d["jam_size"].astype(int)
    a0, an = d["alloc_start"].astype(int), d["alloc_size"].astype(int)
    overlap = np.clip(np.minimum(j0 + jn, a0 + an) - np.maximum(j0, a0), 0, None)
    return overlap / an


def detection_rows() -> list[dict]:
    """Pd of the fused U-Net per seed, attack, JSR and width (and allocation cover bin)."""
    from jamloc.evaluate import _logits, available_seeds
    from jamloc.train import load_model

    val, test = load("val"), load("test")
    labels = test["labels"].astype(bool)
    in_alloc = test["features"][:, FEATURE_NAMES.index("alloc_mask")] > 0
    jammed = labels & in_alloc  # jammed cells where the UE's signal is present
    cover = allocation_cover(test)
    rows = []
    for seed in available_seeds():
        model = load_model("fused", seed)
        s_val = _logits(model, "fused", torch.from_numpy(val["features"]))
        thr = threshold_at_fa(s_val[val["labels"] == 0], FA)
        s_test = _logits(model, "fused", torch.from_numpy(test["features"]))
        for a in (1, 2, 3):
            for jsr in np.unique(test["jsr_db"][test["attack"] == a]):
                point = (test["attack"] == a) & (test["jsr_db"] == jsr)
                groups = [
                    ("width", float(w), point & (test["jam_size"] == w)) for w in range(2, 13)
                ]
                groups += [
                    ("cover", hi, point & (cover >= lo) & (cover <= hi)) for lo, hi in COVER_BINS
                ]
                for kind, value, sel in groups:
                    pos = s_test[sel][jammed[sel]]
                    if pos.size == 0:
                        continue
                    rows.append(
                        {
                            "seed": seed,
                            "attack": ATTACKS[a],
                            "jsr_db": float(jsr),
                            "group": kind,
                            "value": value,
                            "pd": detection_rate(pos, thr),
                            "num_cells": int(pos.size),
                            "num_slots": int(sel.sum()),
                        }
                    )
    return rows


def coverage_rows() -> list[dict]:
    """How much of the allocation (and so of its pilots) the test split's jammers cover."""
    test = load("test")
    cover = allocation_cover(test)
    rows = []
    for a in (1, 2, 3):
        c = cover[test["attack"] == a]
        hit = c[c > 0]
        rows.append(
            {
                "attack": ATTACKS[a],
                "slots": int(c.size),
                "share_hitting_allocation": float((c > 0).mean()),
                "median_cover_when_hit": float(np.median(hit)),
                "share_cover_over_half": float((c > 0.5).mean()),
                "share_cover_all": float((c >= 1).mean()),
            }
        )
    return rows


def mean_over_seeds(rows: list[dict]) -> list[dict]:
    keys = {}
    for r in rows:
        keys.setdefault((r["attack"], r["jsr_db"], r["group"], r["value"]), []).append(r)
    out = []
    for (attack, jsr, group, value), rs in keys.items():
        pd = np.array([r["pd"] for r in rs])
        out.append(
            {
                "attack": attack,
                "jsr_db": jsr,
                "group": group,
                "value": value,
                "pd": float(pd.mean()),
                "pd_std": float(pd.std(ddof=1)) if len(pd) > 1 else 0.0,
                "seeds": len(pd),
                "num_cells": int(np.mean([r["num_cells"] for r in rs])),
                "num_slots": int(np.mean([r["num_slots"] for r in rs])),
            }
        )
    return sorted(out, key=lambda r: (r["attack"], r["group"], r["jsr_db"], r["value"]))


def plot_detection(rows: list[dict]):
    fig, axes = new_figure(7.16, 2.2, ncols=3, sharey=True)
    for ax, attack in zip(axes, NOMINAL_ATTACKS, strict=True):
        for jsr, color in JSR_COLORS.items():
            sel = [
                r
                for r in rows
                if r["attack"] == attack and r["group"] == "width" and r["jsr_db"] == jsr
            ]
            if sel:
                ax.plot(
                    [r["value"] for r in sel],
                    [r["pd"] for r in sel],
                    color=color,
                    marker="o",
                    ms=3,
                    label=f"{jsr:+g} dB",
                )
        ax.set(title=f"{attack} jamming", xlabel="Jammed PRBs (of 24)", xticks=range(2, 13, 2))
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Pd (jammed cells in the allocation)")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=5,
        frameon=False,
        title="JSR",
    )
    return fig


def bler_rows(runs, channel: str = "C") -> list[dict]:
    """Pooled BLER per width, attack, JSR and receiver from the ``recovery --jam-prb`` runs in
    the folder ``runs``."""
    from jamloc.experiments.recovery import _read_rows, pool_seeds

    rows = []
    for w in WIDTHS:
        path = runs / f"recovery_{channel}_jam{w}.csv"
        if not path.is_file():
            continue
        for r in pool_seeds(_read_rows(path)):
            lo, hi = wilson_interval(np.array([r["block_errors"]]), np.array([r["blocks"]]))
            rows.append(
                {
                    "jam_prb": w,
                    "cover": w / 12,
                    "attack": r["attack"],
                    "jsr_db": r["jsr_db"],
                    "receiver": r["receiver"],
                    "block_errors": r["block_errors"],
                    "blocks": r["blocks"],
                    "bler": r["bler"],
                    "bler_lo": float(lo[0]),
                    "bler_hi": float(hi[0]),
                }
            )
    return sorted(rows, key=lambda r: (r["attack"], r["jsr_db"], r["receiver"], r["jam_prb"]))


def plot_bler_width(rows: list[dict], jsrs=(3.0, 9.0)):
    from jamloc.experiments.recovery import STYLE
    from jamloc.plotting import plot_bler, zero_error_handle

    fig, axes = new_figure(7.16, 1.9 * len(jsrs) + 0.6, nrows=len(jsrs), ncols=3, sharex=True)
    axes = np.atleast_2d(axes)
    floor = 1.0 / max(r["blocks"] for r in rows)
    for i, jsr in enumerate(jsrs):
        for j, attack in enumerate(NOMINAL_ATTACKS):
            ax = axes[i, j]
            for rx in BLER_RECEIVERS:
                sel = [
                    r
                    for r in rows
                    if r["attack"] == attack and r["jsr_db"] == jsr and r["receiver"] == rx
                ]
                if not sel:
                    continue
                style = {k: v for k, v in STYLE[rx].items() if k != "label"}
                plot_bler(
                    ax,
                    [r["jam_prb"] for r in sel],
                    [r["block_errors"] for r in sel],
                    [r["blocks"] for r in sel],
                    style,
                    BLER_LABELS[rx],
                )
            # Each tick also gives the share of the allocation (pilots and/or data cells) jammed
            ax.set(yscale="log", ylim=(floor / 2, 1.5), xticks=WIDTHS)
            ax.set_xticklabels([f"{w}\n{round(100 * w / 12)}%" for w in WIDTHS])
            if i == 0:
                ax.set_title(ATTACK_TITLES[attack])
            if i == len(jsrs) - 1 and j == 1:  # one label for the three columns
                ax.set_xlabel("Jammed PRBs of the UE's 12, and their share of its allocation")
            if j == 0:
                ax.set_ylabel(f"BLER, JSR {jsr:+g} dB")
    handles = axes[0, 0].get_legend_handles_labels()[0] + [zero_error_handle()]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False)
    return fig


def _write(rows, path):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--part", choices=("detection", "decoding", "both"), default="both")
    p.add_argument("--channel", choices=("A", "C"), default="C", help="channel of the BLER runs")
    p.add_argument(
        "--runs",
        type=Path,
        help="folder of the recovery_<channel>_jam<N>.csv runs (default: width_sweep/recovery)",
    )
    args = p.parse_args(argv)
    out = out_dir()

    if args.part in ("detection", "both"):
        per_seed = detection_rows()
        _write(per_seed, out / "width_detection_per_seed.csv")
        rows = mean_over_seeds(per_seed)
        _write(rows, out / "width_detection.csv")
        _write(coverage_rows(), out / "allocation_cover.csv")
        with paper_style():
            save(plot_detection(rows), out / "width_detection")
        print(f"Wrote {out / 'width_detection.csv'}, allocation_cover.csv and the figure")

    if args.part in ("decoding", "both"):
        rows = bler_rows(args.runs or out / "recovery", args.channel)
        if not rows:
            print("No recovery_*_jam<N>.csv runs found: run results/width_sweep/run.sh first")
            return
        _write(rows, out / f"width_bler_{args.channel}.csv")
        with paper_style():
            save(plot_bler_width(rows), out / f"width_bler_{args.channel}")
        print(f"Wrote {out / f'width_bler_{args.channel}.csv'} and the figure")


if __name__ == "__main__":
    main()
