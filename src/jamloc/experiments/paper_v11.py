"""The decoding runs of paper v11, one folder per case, and the numbers the paper quotes.

Every case is one ``jamloc.experiments.recovery`` run per seed of the v11 U-Net (``--unet all``,
``experiments.one_unet``), under the Monte Carlo rule of v9 (70 block errors or 160 batches of 64
slots). The sweeps decode the receivers of Fig. 5 from -9 to +9 dB JSR. The ``*9`` cases decode
every receiver of the text at +9 dB (and +3 dB with every pilot jammed). Each modulation runs
at the SNR where the baseline receiver fails on about 0.8% of the blocks without a jammer (random
allocations, ``docs/plan-v11.md``): QPSK (MCS 5) at 8 dB, 16-QAM (MCS 13) at 15 dB and 64-QAM
(MCS 17) at 18 dB. At the 15 dB of paper v10, the receivers that estimate the jamming power made
almost no errors with QPSK, so their curves followed chance.

    python -m jamloc.experiments.paper_v11 run ROOT [--cases ...] [--parallel 8]
    python -m jamloc.experiments.paper_v11 pool ROOT        # pooled CSVs and paired differences
    python -m jamloc.experiments.paper_v11 numbers ROOT     # BLER and differences at each point
    python -m jamloc.experiments.resolution sweep-v11 ROOT  # Fig. 5
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jamloc.paths import results_dir

SEEDS = (0, 1, 2)
COMMON = ("--channel", "C", "--target-block-errors", "70", "--max-mc-iter", "160", "--unet", "all")
JSR = ("--jsr-db", "-9", "-6", "-3", "0", "3", "6", "9")
FIG = ("standard", "nomap_prb", "nomap_shrink", "evm_shrink", "p_r_e_shrink", "true_soft")
FIG += ("true_shrink",)
ALL = (*FIG, "nomap_cell", "true_prb_soft")
QPSK = ("--snr-db", "8")
QAM16 = ("--snr-db", "15", "--mcs", "13")
QAM64 = ("--snr-db", "18", "--mcs", "17")
HOP = ("--jammer", "hopping")
WHOLE = ("--attacks", "pilot_whole_allocation")
BLOCK = ("--attacks", "none", "pilot", "data", "partial")
# case: (options after COMMON, receivers)
CASES = {
    "q_nom": ((*QPSK, "--attacks", "pilot", *JSR), FIG),
    "q_hop": ((*QPSK, *HOP, *JSR), FIG),
    "q_whole": ((*QPSK, *WHOLE, *JSR), FIG),
    "m16_nom": ((*QAM16, "--attacks", "partial", *JSR), FIG),
    "m16_hop": ((*QAM16, *HOP, "--attacks", "partial", *JSR), FIG),
    "m64_nom": ((*QAM64, "--attacks", "partial", *JSR), FIG),
    "m64_hop": ((*QAM64, *HOP, "--attacks", "partial", *JSR), FIG),
    "d2_nom": ((*QPSK, "--dmrs2", "--attacks", "pilot", *JSR), FIG),
    "d2_hop": ((*QPSK, "--dmrs2", *HOP, "--attacks", "partial", *JSR), FIG),
    "d2_whole": ((*QPSK, "--dmrs2", *WHOLE, *JSR), FIG),
    "q9": ((*QPSK, *BLOCK, "--jsr-db", "9"), ALL),
    "q9_hop": ((*QPSK, *HOP, "--jsr-db", "9"), ALL),
    "q9_pulse": ((*QPSK, "--jammer", "pulsed", "--jsr-db", "9"), ALL),
    "q9_whole": ((*QPSK, *WHOLE, "--jsr-db", "3", "9"), ALL),
    "m16_9": ((*QAM16, *BLOCK, "--jsr-db", "9"), ALL),
    "m16_9hop": ((*QAM16, *HOP, "--jsr-db", "9"), ALL),
    "m64_9": ((*QAM64, *BLOCK, "--jsr-db", "9"), ALL),
    "m64_9hop": ((*QAM64, *HOP, "--jsr-db", "9"), ALL),
    "d2_9": ((*QPSK, "--dmrs2", *BLOCK, "--jsr-db", "9"), ALL),
}
# Options that name the run's files (``recovery.main``): what --plot-only needs to find them
NAMING = ("--snr-db", "--mcs", "--jammer", "--dmrs2")


def _naming_options(options) -> list[str]:
    out, i = [], 0
    while i < len(options):
        if options[i] in NAMING:
            takes_value = options[i] != "--dmrs2"
            out += options[i : i + 1 + takes_value]
            i += 1 + takes_value
        else:
            i += 1
    return out


def _link_models(folder: Path, models: Path) -> None:
    """``folder/models`` points to the trained models (a copy where links are not allowed)."""
    link = folder / "models"
    if link.exists() or link.is_symlink():
        return
    try:
        link.symlink_to(models.resolve(), target_is_directory=True)
    except OSError:  # e.g. Windows without the right to create links
        shutil.copytree(models, link)


def commands(root: Path, cases, seeds=SEEDS) -> list[tuple[Path, list[str], Path]]:
    """(results folder, command, log file) of every case and seed."""
    out = []
    for case in cases:
        options, receivers = CASES[case]
        for seed in seeds:
            cmd = [sys.executable, "-m", "jamloc.experiments.recovery", *COMMON, *options]
            cmd += ["--seeds", str(seed), "--receivers", *receivers]
            out.append((root / case, cmd, root / "logs" / f"{case}_{seed}.log"))
    return out


def run(root: Path, cases, parallel: int, models: Path, seeds=SEEDS) -> list[int]:
    """Run every case and seed, ``parallel`` processes at a time with one thread each. Returns
    the exit codes."""
    (root / "logs").mkdir(parents=True, exist_ok=True)
    for case in cases:
        (root / case).mkdir(parents=True, exist_ok=True)
        _link_models(root / case, models)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def one(job):
        folder, cmd, log = job
        with log.open("w", encoding="utf-8") as f:
            code = subprocess.call(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**env, "JAMLOC_RESULTS_DIR": str(folder)},
            )
        with (root / "logs" / "runs.log").open("a", encoding="utf-8") as f:
            f.write(f"{log.stem} exit {code}\n")
        return code

    with ThreadPoolExecutor(parallel) as pool:
        return list(pool.map(one, commands(root, cases, seeds)))


def pool(root: Path, cases) -> None:
    """Pool the seeds of every case found under ``root`` and write its paired differences."""
    for case in cases:
        if not (root / case / "recovery").is_dir():
            continue
        options, _ = CASES[case]
        cmd = [sys.executable, "-m", "jamloc.experiments.recovery", "--channel", "C"]
        cmd += [*_naming_options(options), "--plot-only"]
        env = {**os.environ, "JAMLOC_RESULTS_DIR": str(root / case)}
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
        print(f"pooled {case}")


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def counts(folder: Path) -> dict:
    """{(attack, jsr, receiver): [block errors, blocks]} summed over the seed files."""
    out = {}
    for path in sorted(folder.glob("recovery_*_seed*.csv")):
        for r in _rows(path):
            c = out.setdefault((r["attack"], r["jsr_db"], r["receiver"]), [0, 0])
            c[0] += int(r["block_errors"])
            c[1] += int(r["blocks"])
    return out


def paired(folder: Path) -> dict:
    """{(attack, jsr, first, second): row} of the paired differences."""
    out = {}
    for path in folder.glob("recovery_*_paired.csv"):
        for r in _rows(path):
            out[(r["attack"], r["jsr_db"], r["first"], r["second"])] = r
    return out


def difference(pairs: dict, attack: str, jsr: str, rx: str, ref: str):
    """BLER(rx) - BLER(ref) in percentage points with its 95% interval, or None."""
    r, sign = pairs.get((attack, jsr, ref, rx)), -1
    if r is None:
        r, sign = pairs.get((attack, jsr, rx, ref)), 1
    if r is None:
        return None
    d, lo, hi = (sign * 100 * float(r[k]) for k in ("delta", "delta_lo", "delta_hi"))
    return d, min(lo, hi), max(lo, hi)


# Differences the text quotes: (receiver, reference)
PAIRS = (
    ("p_r_e_shrink", "nomap_shrink"),
    ("true_shrink", "nomap_shrink"),
    ("p_r_e_shrink", "true_shrink"),
    ("nomap_shrink", "nomap_cell"),
    ("nomap_prb", "standard"),
    ("evm_shrink", "nomap_shrink"),
    ("true_soft", "true_shrink"),
)


def numbers(root: Path, cases) -> str:
    """Markdown: per case and point, the BLER (%) of every receiver, then the quoted paired
    differences."""
    lines = []
    for case in cases:
        folder = root / case / "recovery"
        if not folder.is_dir():
            continue
        c, pairs = counts(folder), paired(folder)
        receivers = [rx for rx in CASES[case][1] if any(k[2] == rx for k in c)]
        lines += [f"### {case}", "", "| point | " + " | ".join(receivers) + " |"]
        lines.append("| --- |" + " --- |" * len(receivers))
        points = sorted({k[:2] for k in c}, key=lambda p: (p[0], float(p[1])))
        for attack, jsr in points:
            cells = []
            for rx in receivers:
                e, n = c.get((attack, jsr, rx), (0, 0))
                cells.append("-" if n == 0 else f"0/{n}" if e == 0 else f"{100 * e / n:.3g}")
            lines.append(f"| {attack} {jsr} | " + " | ".join(cells) + " |")
        lines.append("")
        for attack, jsr in points:
            parts = []
            for rx, ref in PAIRS:
                d = difference(pairs, attack, jsr, rx, ref)
                if d is not None:
                    parts.append(f"{rx} - {ref} {d[0]:+.2f} ({d[1]:+.2f} to {d[2]:+.2f})")
            if parts:
                lines.append(f"- {attack} {jsr}: " + "; ".join(parts))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "pool", "numbers"):
        s = sub.add_parser(name)
        s.add_argument("root", type=Path)
        s.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
        if name == "run":
            s.add_argument("--parallel", type=int, default=os.cpu_count() or 1)
            s.add_argument("--models", type=Path, default=results_dir() / "models")
            s.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = p.parse_args(argv)
    if args.command == "run":
        codes = run(args.root, args.cases, args.parallel, args.models, args.seeds)
        print(f"{sum(c != 0 for c in codes)} of {len(codes)} runs failed")
    elif args.command == "pool":
        pool(args.root, args.cases)
    else:
        print(numbers(args.root, args.cases))


if __name__ == "__main__":
    main()
