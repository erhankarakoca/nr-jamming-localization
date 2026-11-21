"""Averaging the DMRS residual over several slots: exact threshold and sensitivity gain.

A jammer that stays on the same PRBs for several slots can be found at lower power by averaging
each PRB's residual over K slots. With the same allocation and independent noise in every slot,
the average is again a weighted sum of exponentials (``jamloc.theory.slot_average``), so its 1%
threshold and its Pd are exact, like those of one slot. The price is K slots of detection delay.

Checks:

* **False alarm, link level.** Benign ``test`` and ``val`` slots with the same allocation are
  grouped K at a time, their stored residuals averaged per PRB and compared with the K-slot
  threshold. (The slots of a group have independent channels; without jamming that does not
  matter, since the residual does not depend on the channel.)
* **Pd, pilot level.** Monte Carlo of noise and a Gaussian jammer on the pilots, the jammer's
  Rayleigh gain per PRB drawn once per K slots, projected with the same subspace as the receiver.

    python -m jamloc.experiments.multislot
    python -m jamloc.experiments.multislot --plot-only  # reuse the two saved CSVs
"""

import argparse
import csv

import numpy as np

from jamloc.config import SimConfig
from jamloc.dataset import load
from jamloc.features import FEATURE_NAMES
from jamloc.paths import results_dir
from jamloc.plotting import new_figure, paper_style, save
from jamloc.theory import PILOTS_PER_PRB, detection_probability, projector, threshold

SLOTS = (1, 2, 4, 8)
ALLOC, PRB, JAMMED = 12, 5, range(4, 8)  # PRB 5 of a 12-PRB allocation, PRBs 4-7 jammed
JNR_DB = np.arange(-9.0, 15.1, 1.0)
MC_JNR_DB = np.arange(-9.0, 15.1, 3.0)
FA = 0.01


def link_false_alarm(slots: int) -> tuple[int, int]:
    """Exceedances and tests of the K-slot threshold on benign link-level residuals."""
    cfg = SimConfig()
    exceed, tests = 0, 0
    for split in ("test", "val"):
        d = load(split)
        benign = np.flatnonzero(d["attack"] == 0)
        r = 10 ** (d["features"][:, FEATURE_NAMES.index("residual_db"), :, cfg.dmrs_symbol] / 10)
        allocs = np.stack([d["alloc_start"][benign], d["alloc_size"][benign]], 1)
        for start, size in np.unique(allocs, axis=0):
            idx = benign[(allocs[:, 0] == start) & (allocs[:, 1] == size)]
            groups = len(idx) // slots
            if not groups:
                continue
            mean = r[idx[: groups * slots], start : start + size].reshape(groups, slots, size)
            mean = mean.mean(1)
            for pos in range(int(size)):
                exceed += int((mean[:, pos] > threshold(int(size), pos, FA, cfg, slots)).sum())
                tests += groups
    return exceed, tests


def monte_carlo_pd(slots: int, jnr_db, draws: int = 4000, seed: int = 0) -> np.ndarray:
    """Pilot-level simulation of the K-slot residual of PRB ``PRB`` under jamming."""
    cfg = SimConfig()
    rng = np.random.default_rng(seed)
    p = projector(ALLOC, cfg)
    block = slice(PILOTS_PER_PRB * PRB, PILOTS_PER_PRB * (PRB + 1))
    norm = np.trace(p[block, block]).real
    t = threshold(ALLOC, PRB, FA, cfg, slots)
    jammed = np.zeros(ALLOC)
    jammed[list(JAMMED)] = 1
    n_pilots = PILOTS_PER_PRB * ALLOC

    def cn(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)

    out = []
    for rho in 10 ** (np.asarray(jnr_db) / 10):
        gain = np.repeat(rng.exponential(1.0, (draws, ALLOC)) * jammed, PILOTS_PER_PRB, 1)
        r = np.zeros(draws)
        for _ in range(slots):  # the same jammer gain, new noise and jammer samples each slot
            y = cn(draws, n_pilots) + np.sqrt(rho * gain) * cn(draws, n_pilots)
            residual = y @ p.T  # rows: P_perp y
            r += (np.abs(residual[:, block]) ** 2).sum(1) / norm
        out.append(float((r / slots > t).mean()))
    return np.array(out)


def run() -> tuple[list[dict], list[dict]]:
    curves, checks = [], []
    for k in SLOTS:
        pd = detection_probability(JNR_DB, ALLOC, PRB, JAMMED, FA, slots=k)
        curves += [{"slots": k, "jnr_db": j, "pd_theory": v} for j, v in zip(JNR_DB, pd)]
        mc = monte_carlo_pd(k, MC_JNR_DB)
        exceed, tests = link_false_alarm(k)
        checks.append(
            {
                "slots": k,
                "threshold_mid": threshold(ALLOC, PRB, FA, slots=k),
                "threshold_edge": threshold(ALLOC, 0, FA, slots=k),
                "fa_link": exceed / tests,
                "fa_tests": tests,
                "jnr_pd50_db": float(np.interp(0.5, pd, JNR_DB)),
                "jnr_pd90_db": float(np.interp(0.9, pd, JNR_DB)),
                **{f"pd_mc_{j:+.0f}dB": v for j, v in zip(MC_JNR_DB, mc)},
            }
        )
        print(
            f"K={k}: threshold {checks[-1]['threshold_mid']:.3f}, link false alarm "
            f"{exceed}/{tests} = {exceed / tests:.4f}, JNR for Pd 0.5 / 0.9: "
            f"{checks[-1]['jnr_pd50_db']:.1f} / {checks[-1]['jnr_pd90_db']:.1f} dB",
            flush=True,
        )
    return curves, checks


def plot_on_axis(ax, curves, checks, *, show_mc_key=False):
    """Draw the same K-slot comparison in the standalone figure and the theory panel."""
    styles = (
        ("#000000", "-", "o"),
        ("#0033CC", "--", "s"),
        ("#CC0000", "-.", "^"),
        ("#008000", ":", "D"),
    )
    for k, (color, line, marker) in zip(SLOTS, styles, strict=True):
        rows = sorted((r for r in curves if r["slots"] == k), key=lambda r: r["jnr_db"])
        ax.plot(
            [r["jnr_db"] for r in rows],
            [r["pd_theory"] for r in rows],
            color=color,
            ls=line,
            lw=0.9,
            label=f"K = {k}",
        )
        (check,) = [r for r in checks if r["slots"] == k]
        # The Monte Carlo grid is encoded in the column names of the saved run.
        points = sorted(
            (float(key.removeprefix("pd_mc_").removesuffix("dB")), value)
            for key, value in check.items()
            if key.startswith("pd_mc_") and key.endswith("dB")
        )
        jnr, pd = zip(*points, strict=True)
        ax.plot(
            jnr,
            pd,
            ls="none",
            marker=marker,
            mfc="none",
            ms=3,
            mew=0.65,
            color=color,
        )
    ax.set(
        xlabel="JNR (dB)",
        ylabel=r"$P_\mathrm{d}$",
        ylim=(0, 1.02),
    )
    if show_mc_key:
        ax.plot([], [], ls="none", marker="o", mfc="none", color="#555555", label="Monte Carlo")
    ax.legend(loc="lower right", ncols=2, frameon=False, columnspacing=0.9)


def plot(curves, checks):
    fig, ax = new_figure(3.5, 2.4)
    plot_on_axis(ax, curves, checks, show_mc_key=True)
    return fig


def _write(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_results(directory):
    """Load the theory curves and Monte Carlo checks without running the simulation."""
    rows = []
    for name in ("multislot_curves.csv", "multislot.csv"):
        with (directory / name).open(encoding="utf-8", newline="") as f:
            rows.append([{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)])
    return tuple(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plot-only", action="store_true", help="plot the saved CSVs")
    args = p.parse_args(argv)
    out = results_dir() / "theory"
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        curves, checks = read_results(out)
    else:
        curves, checks = run()
        _write(curves, out / "multislot_curves.csv")
        _write(checks, out / "multislot.csv")
    with paper_style():
        fig = plot(curves, checks)
    save(fig, out / "multislot")


if __name__ == "__main__":
    main()
