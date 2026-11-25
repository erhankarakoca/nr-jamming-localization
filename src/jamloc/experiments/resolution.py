"""How finely to estimate the jamming power, and with which map: tables and figures.

The resolution study crosses the map a soft receiver gets (none, the EVM threshold, the U-Net,
the true map) with how it estimates the jamming power of the data cells (one value per PRB, the
same scaled by 2 or 4, each cell on its own, or shrinkage of each cell towards its PRB). Its runs
are ordinary ``jamloc.experiments.recovery`` runs with these receivers (``docs/reproducing.md``).

``merge`` joins run folders that decoded the same slots (same seed, cases and JSR grid) into one
folder of per-seed files. A receiver found in more than one folder must have the same counts in
all of them, which checks that the slots are the same. ``--plot-only`` of the recovery script
then pools the seeds and writes the paired differences.

    python -m jamloc.experiments.resolution merge OUT DIR [DIR ...]
    python -m jamloc.experiments.resolution table DIR
    python -m jamloc.experiments.resolution sweep DIR [DIR]   # BLER against JSR, one figure
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from jamloc.plotting import new_figure, paper_style, plot_bler, save, zero_error_handle

# map: {estimate: receiver}
ARMS = {
    "none": {
        "per PRB": "nomap_prb",
        "per PRB x2": "nomap_prb2",
        "per PRB x4": "nomap_prb4",
        "per cell": "nomap_cell",
        "shrinkage": "nomap_shrink",
    },
    "EVM": {"per PRB": "evm_prb", "per cell": "evm_cell", "shrinkage": "evm_shrink"},
    "U-Net": {
        "per PRB": "p_r_e_soft",
        "per PRB x2": "p_r_e_prb2",
        "per PRB x4": "p_r_e_prb4",
        "per cell": "p_r_e_cell",
        "shrinkage": "p_r_e_shrink",
    },
    "true, per PRB": {"per PRB": "true_prb_soft", "shrinkage": "true_prb_shrink"},
    "true": {"per PRB": "true_soft", "per cell": "true_cell", "shrinkage": "true_shrink"},
}
# (file stem, attack, JSR as written in the CSV, column title)
CASES = [
    ("recovery_C", "none", "nan", "benign"),
    ("recovery_C", "pilot", "9.0", "pilot"),
    ("recovery_C", "data", "9.0", "data"),
    ("recovery_C", "partial", "9.0", "partial"),
    ("recovery_C_pulsed", "data", "9.0", "pulsed data"),
    ("recovery_C_pulsed", "partial", "9.0", "pulsed partial"),
    ("recovery_C_hopping", "data", "9.0", "hopping data"),
    ("recovery_C_hopping", "partial", "9.0", "hopping partial"),
    ("recovery_C", "pilot_whole_allocation", "3.0", "all pilots +3"),
    ("recovery_C", "pilot_whole_allocation", "9.0", "all pilots +9"),
]
QAM_CASES = [
    ("recovery_C_snr18_mcs13", a, j, f"{a} {float(j):+g}")
    for a in ("pilot", "data", "partial")
    for j in ("3.0", "9.0")
]
# Names as in the paper: (label, color, marker, line style)
SWEEP_RECEIVERS = {
    "standard": ("Baseline receiver", "#000000", "x", ":"),
    "nomap_prb": ("No map, PRB average", "#999999", "s", "--"),
    "evm_shrink": ("EVM map, shrinkage", "#E69F00", "D", "-."),
    "p_r_e_shrink": ("U-Net map, shrinkage", "#0072B2", "o", "-"),
    "nomap_shrink": ("No map, shrinkage", "#009E73", "h", "-"),
    "true_shrink": ("True map, shrinkage", "#CC0000", "^", ":"),
}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def merge(out: Path, folders: list[Path]) -> list[str]:
    """Merge the per-seed CSVs and per-slot outcomes of ``folders`` into ``out``. The first folder
    that has a receiver at a point provides it, and every later folder with the same receiver and
    point must agree on its block errors and blocks. Returns the disagreements."""
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    names = sorted({p.name for d in folders for p in d.glob("recovery_*_seed*.csv")})
    for name in names:
        rows, slots = {}, {}
        for d in folders:
            path = d / name
            if not path.is_file():
                continue
            for r in _read(path):
                key = (r["receiver"], r["attack"], r["jsr_db"])
                if key in rows:
                    old = rows[key]
                    if (old["block_errors"], old["blocks"]) != (r["block_errors"], r["blocks"]):
                        problems.append(
                            f"{name} {key}: {old['block_errors']}/{old['blocks']} in an earlier "
                            f"folder, {r['block_errors']}/{r['blocks']} in {d}"
                        )
                    continue
                rows[key] = r
            npz = path.with_name(path.stem + "_slots.npz")
            if npz.is_file():
                for k, v in np.load(npz).items():
                    slots.setdefault(k, v)
        fields = list(next(iter(rows.values())))
        with (out / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows.values())
        np.savez_compressed(out / (Path(name).stem + "_slots.npz"), **slots)
    return problems


def pooled(folder: Path, stem: str) -> dict:
    """{(receiver, attack, jsr): [block errors, blocks]} summed over the seed files of a stem."""
    out = defaultdict(lambda: [0, 0])
    for path in sorted(folder.glob(f"{stem}_seed*.csv")):
        if "_slots" in path.name:
            continue
        for r in _read(path):
            key = (r["receiver"], r["attack"], r["jsr_db"])
            out[key][0] += int(r["block_errors"])
            out[key][1] += int(r["blocks"])
    return out


def table(folder: Path, cases=CASES) -> list[dict]:
    """BLER (%) of every arm at every case. The scaled per-PRB estimates appear as one row that
    takes, per case, the best of the scales 1, 2 and 4, the most favorable choice for them."""
    data = {stem: pooled(folder, stem) for stem in {c[0] for c in cases}}

    def bler(rx, stem, attack, jsr):
        e, n = data[stem].get((rx, attack, jsr), (0, 0))
        return (100 * e / n, n) if n else (None, 0)

    rows = []
    for mask, arms in ARMS.items():
        estimates = {name: [rx] for name, rx in arms.items() if " x" not in name}
        if "per PRB x2" in arms:
            estimates["per PRB, best scale"] = [
                arms["per PRB"],
                arms["per PRB x2"],
                arms["per PRB x4"],
            ]
        for name, rxs in estimates.items():
            row = {"map": mask, "estimate": name}
            for stem, attack, jsr, title in cases:
                # a tie keeps the unscaled estimate, the first in ``rxs``
                values = [(bler(rx, stem, attack, jsr), i, rx) for i, rx in enumerate(rxs)]
                values = [(b, i, n, rx) for (b, n), i, rx in values if b is not None]
                row[title] = min(values)[0:1] + min(values)[2:] if values else None
            rows.append(row)
    return rows


def markdown(rows, cases=CASES) -> str:
    titles = [c[3] for c in cases]
    lines = ["| map | estimate | " + " | ".join(titles) + " |", "|" + " --- |" * (len(titles) + 2)]
    for r in rows:
        cells = []
        for t in titles:
            v = r[t]
            if v is None:
                cells.append("-")
            else:
                scale = f" (x{v[2][-1]})" if v[2][-1] in "24" else ""
                cells.append(f"{v[0]:.2f}{scale}")
        lines.append(f"| {r['map']} | {r['estimate']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Panels of the sweep figure: (file stem, attack, title)
SWEEP_PANELS = [
    ("recovery_C", "pilot", "(a) Pilot only"),
    ("recovery_C", "partial", "(b) Pilot + data"),
    ("recovery_C_hopping", "data", "(c) Hopping, data only"),
    ("recovery_C_hopping", "partial", "(d) Hopping, pilot + data"),
]
# With 16-QAM (paper v10): three QPSK panels and two 16-QAM panels, with a label over each group
QAM = "_snr18_mcs13"
SWEEP_PANELS_QAM = [
    ("recovery_C", "pilot", "(a) Block, pilot only"),
    ("recovery_C_hopping", "data", "(b) Hopping, data only"),
    ("recovery_C_hopping", "partial", "(c) Hopping, pilot + data"),
    (f"recovery_C{QAM}", "partial", "(d) Block, pilot + data"),
    (f"recovery_C_hopping{QAM}", "partial", "(e) Hopping, pilot + data"),
]
SWEEP_GROUPS_QAM = [(0, 2, "QPSK at 15 dB SNR"), (3, 4, "16-QAM at 18 dB SNR")]


def bler_floor(values) -> float:
    """Bottom of a log BLER axis, the decade below the lowest point drawn and at most 1e-4."""
    return min(1e-4, 10.0 ** np.floor(np.log10(min(values, default=1e-4))))


def sweep(folders: list[Path], out: Path, panels=SWEEP_PANELS, groups=()) -> None:
    """BLER against JSR, one panel per case of ``panels``, for ``SWEEP_RECEIVERS``. ``groups``
    puts a label over panels ``first`` to ``last``: ``[(first, last, label), ...]``."""
    data = {}
    for folder in folders:
        for stem in {p[0] for p in panels}:
            for key, v in pooled(folder, stem).items():
                data[(stem, *key)] = v
    with paper_style():
        if groups:  # one subfigure per group, with its label on top
            fig = Figure(figsize=(7.16, 1.9), layout="constrained")
            widths = [last - first + 1 for first, last, _ in groups]
            subfigs = fig.subfigures(1, len(groups), width_ratios=widths, wspace=0.01)
            axes = []
            for sub, width, (_, _, label) in zip(subfigs, widths, groups, strict=True):
                sub.suptitle(label, fontsize=7)
                axes += list(np.atleast_1d(sub.subplots(1, width, sharey=True)))
        else:
            fig, axes = new_figure(width=7.16, height=1.95, ncols=len(panels), sharey=True)
        for ax, (stem, attack, title) in zip(axes, panels, strict=True):
            for rx, (label, color, marker, ls) in SWEEP_RECEIVERS.items():
                points = sorted(
                    (float(k[3]), v)
                    for k, v in data.items()
                    if k[0] == stem and k[1] == rx and k[2] == attack and k[3] != "nan"
                )
                if not points:
                    continue
                x = [p[0] for p in points]
                errors = [p[1][0] for p in points]
                blocks = [p[1][1] for p in points]
                style = {"color": color, "marker": marker, "ls": ls, "ms": 3, "lw": 1}
                plot_bler(ax, x, errors, blocks, style, label)
            ax.set_yscale("log")
            ax.set_xlabel("JSR per jammed RE [dB]")
            ax.set_title(title, fontsize=7)
        axes[0].set_ylabel("BLER")
        drawn = [y for ax in axes for line in ax.get_lines() for y in line.get_ydata() if y > 0]
        for ax in axes:  # every panel on the same y axis, also across groups
            ax.set_ylim(bler_floor(drawn), 1.2)
        legend = {}  # one entry per receiver, from whichever panel has its curve
        for ax in axes:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                legend.setdefault(label, handle)
        labels = [s[0] for s in SWEEP_RECEIVERS.values() if s[0] in legend]
        handles = [legend[label] for label in labels]
        handles.append(zero_error_handle())
        labels.append(handles[-1].get_label())
        fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=6)
    save(fig, out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge", help="merge run folders that decoded the same slots")
    m.add_argument("out", type=Path)
    m.add_argument("folders", type=Path, nargs="+")
    t = sub.add_parser("table", help="BLER of every arm at every case")
    t.add_argument("folder", type=Path)
    t.add_argument("--qam", action="store_true", help="the 16-QAM cases")
    s = sub.add_parser("sweep", help="BLER against JSR")
    s.add_argument("folders", type=Path, nargs="+")
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--with-qam", action="store_true", help="the panels of paper v10, with 16-QAM")
    args = p.parse_args(argv)
    if args.command == "merge":
        problems = merge(args.out, args.folders)
        print("\n".join(problems) or "every receiver found twice has the same counts")
    elif args.command == "table":
        cases = QAM_CASES if args.qam else CASES
        print(markdown(table(args.folder, cases), cases))
    else:
        out = args.out or args.folders[0] / "resolution_sweep"
        if args.with_qam:
            sweep(args.folders, out, SWEEP_PANELS_QAM, SWEEP_GROUPS_QAM)
        else:
            sweep(args.folders, out)
        print(f"Wrote {out}.pdf/.png")


if __name__ == "__main__":
    main()
