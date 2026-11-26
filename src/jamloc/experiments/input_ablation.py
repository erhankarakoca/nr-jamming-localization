"""Which inputs and which network: the general U-Net against two ablations of it.

Three detectors are trained on ``train_gen`` with the schedule of ``experiments.general_training``
(learning rate halved on a plateau, early stopping, thresholds from ``val_gen``):

* the U-Net on power, DMRS residual and EVM, the detector of the paper;
* the same U-Net without the EVM, which asks whether the EVM still helps once the residual is
  there (the EVM rises on the data cells of a PRB whose pilots are jammed, and on fast or
  long-delay channels without any jammer);
* a residual CNN of the same size on the same inputs that keeps the full 24 x 14 resolution
  (``model.SlotResNet``), which asks whether the U-Net's pooling path is needed.

They are compared on the eight test grids of the general training: F1 per attack type, false
alarms on scored cells and the Pd of weak pilot jamming, three training seeds each. Decoding runs
of ``experiments.recovery`` then give each map to the shrinkage receiver (``p_r_e_shrink``,
``pr_shrink``, ``resnet_shrink``) on the same slots, one folder per case
(``docs/reproducing.md``), and ``--decoding`` prints their BLER and paired differences.

    python -m jamloc.experiments.input_ablation --seeds 0 1 2
    python -m jamloc.experiments.input_ablation --decoding results/input_ablation
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from jamloc.dataset import generate, load
from jamloc.evaluate import _logits, aggregate, evaluate_scores
from jamloc.experiments.general_training import CONDITIONS, pilot_pd, pilot_pd_matched
from jamloc.paths import data_dir, results_dir
from jamloc.train import checkpoint_path, load_model, train

# Table name: (input variant, architecture)
DETECTORS = {
    "U-Net, P+R+E": ("fused", "unet"),
    "U-Net, P+R": ("no_evm", "unet"),
    "ResNet, P+R+E": ("fused", "resnet"),
}
DATA, VAL_SPLIT = "gen", "val_gen"
VALUES = ("f1", "f1_pilot", "f1_data", "f1_partial", "fa", "pd_pilot", "pd_pilot_matched")


def prepare(seeds, epochs: int, patience: int):
    """Generate the missing splits and train the missing models."""
    for split in (f"train_{DATA}", VAL_SPLIT, *CONDITIONS):
        if not (data_dir() / f"{split}.npz").is_file():
            generate(split, batch_size=128)
    for seed in seeds:
        for variant, arch in DETECTORS.values():
            if checkpoint_path(variant, seed, data=DATA, arch=arch).is_file():
                continue
            train(
                variant,
                epochs=epochs,
                seed=seed,
                data=DATA,
                arch=arch,
                schedule="plateau",
                patience=patience,
            )


def run(seeds) -> list[dict]:
    """F1 per attack, false alarms on scored cells and weak-pilot Pd, per grid, detector and
    seed."""
    val = load(VAL_SPLIT)
    val_features = torch.from_numpy(val["features"])
    models, val_scores = {}, {}
    for name, (variant, arch) in DETECTORS.items():
        for seed in seeds:
            model = load_model(variant, seed, data=DATA, arch=arch)
            models[name, seed] = model
            val_scores[name, seed] = _logits(model, variant, val_features)
    rows = []
    for split, condition in CONDITIONS.items():
        test = load(split)
        features = torch.from_numpy(test["features"])
        for name, (variant, _) in DETECTORS.items():
            for seed in seeds:
                s_test = _logits(models[name, seed], variant, features)
                pd_rows, f1_rows = evaluate_scores(
                    {variant: (val_scores[name, seed], s_test)}, val, test
                )
                by = {r["attack"]: r for r in f1_rows}
                rows.append(
                    {
                        "condition": condition,
                        "detector": name,
                        "seed": seed,
                        "f1": by["all"]["f1"],
                        # attack types missing from a grid (pulsed: no pilot attack) are nan
                        "f1_pilot": by.get("pilot", {}).get("f1", np.nan),
                        "f1_data": by.get("data", {}).get("f1", np.nan),
                        "f1_partial": by.get("partial", {}).get("f1", np.nan),
                        "fa": by["all"]["fa_scored"],
                        "pd_pilot": pilot_pd(pd_rows),
                        "pd_pilot_matched": pilot_pd_matched(s_test, test),
                    }
                )
        print(f"{condition}: done", flush=True)
    return rows


def _cell(r: dict, key: str) -> str:
    if np.isnan(r[f"{key}_mean"]):
        return "-"
    return f"{r[f'{key}_mean']:.3f} +- {r[f'{key}_std']:.3f}"


def table(rows: list[dict]) -> str:
    """Markdown tables: F1 / false alarms per grid and detector, and F1 per attack type with the
    weak-pilot Pd on the matched grid."""
    by = {(r["condition"], r["detector"]): r for r in rows}
    names = list(DETECTORS)
    lines = [
        "| Test grid | " + " | ".join(f"{n}: F1 / FA" for n in names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for condition in CONDITIONS.values():
        cells = [by[condition, n] for n in names if (condition, n) in by]
        if len(cells) < len(names):
            continue
        lines.append(
            f"| {condition} | "
            + " | ".join(f"{_cell(r, 'f1')} / {100 * r['fa_mean']:.1f}%" for r in cells)
            + " |"
        )
    lines += [
        "",
        "| Detector | Grid | F1 pilot | F1 data | F1 partial | Pilot Pd at -15 dB | Same at 1% FA |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for condition in ("matched", "16-QAM"):
        for n in names:
            r = by.get((condition, n))
            if r is None:
                continue
            keys = ("f1_pilot", "f1_data", "f1_partial", "pd_pilot", "pd_pilot_matched")
            lines.append(f"| {n} | {condition} | " + " | ".join(_cell(r, k) for k in keys) + " |")
    return "\n".join(lines)


# Decoding runs: folder name and title, receivers of the table, and the pairs it differences
DECODING_CASES = {
    "nom": "QPSK block",
    "hop": "QPSK hopping",
    "whole": "QPSK every pilot",
    "qam16": "16-QAM block",
    "qamhop": "16-QAM hopping",
}
DECODING_RECEIVERS = {
    "nomap_shrink": "No map",
    "p_r_e_shrink": "U-Net P+R+E",
    "pr_shrink": "U-Net P+R",
    "resnet_shrink": "ResNet P+R+E",
    "true_shrink": "True map",
}
# (receiver, reference): the table gives BLER(receiver) - BLER(reference) on the same slots
DECODING_PAIRS = (("pr_shrink", "p_r_e_shrink"), ("resnet_shrink", "p_r_e_shrink"))


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def decoding_table(root: Path) -> str:
    """BLER (%) of each map with shrinkage, pooled over the seeds, and the paired differences
    with their 95% intervals (``recovery.paired_rows``), one row per point of every case."""
    head = " | ".join(DECODING_RECEIVERS.values())
    diffs = " | ".join(
        f"{DECODING_RECEIVERS[a]} - {DECODING_RECEIVERS[b]}" for a, b in DECODING_PAIRS
    )
    lines = [f"| Case | Point | {head} | {diffs} |", "| --- | --- |" + " --- |" * 7]
    for case, title in DECODING_CASES.items():
        folder = root / case / "recovery"
        for paired_path in sorted(folder.glob("recovery_*_paired.csv")):
            stem = paired_path.name.removesuffix("_paired.csv")
            counts = {}  # (attack, jsr, receiver): [block errors, blocks]
            for path in sorted(folder.glob(f"{stem}_seed*.csv")):
                for r in _rows(path):
                    c = counts.setdefault((r["attack"], r["jsr_db"], r["receiver"]), [0, 0])
                    c[0] += int(r["block_errors"])
                    c[1] += int(r["blocks"])
            paired = {
                (r["attack"], r["jsr_db"], r["first"], r["second"]): r for r in _rows(paired_path)
            }
            for attack, jsr in sorted({k[:2] for k in counts}):
                cells = []
                for rx in DECODING_RECEIVERS:
                    e, n = counts.get((attack, jsr, rx), (0, 0))
                    cells.append("-" if n == 0 else f"0/{n}" if e == 0 else f"{100 * e / n:.2f}")
                for rx, ref in DECODING_PAIRS:
                    r = paired.get((attack, jsr, ref, rx))
                    sign = 1
                    if r is None:  # stored the other way round
                        r, sign = paired.get((attack, jsr, rx, ref)), -1
                    if r is None:
                        cells.append("-")
                        continue
                    # delta = BLER(first) - BLER(second); report BLER(rx) - BLER(ref)
                    d, lo, hi = (
                        -sign * 100 * float(r[k]) for k in ("delta", "delta_lo", "delta_hi")
                    )
                    lo, hi = min(lo, hi), max(lo, hi)
                    cells.append(f"{d:+.2f} ({lo:+.2f} to {hi:+.2f})")
                name = "every pilot" if attack == "pilot_whole_allocation" else attack
                point = "no jammer" if jsr == "nan" else f"{name} {float(jsr):+g} dB"
                lines.append(f"| {title} | {point} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=80, help="most epochs (as the general training)")
    p.add_argument("--patience", type=int, default=10, help="early stopping (as the general one)")
    p.add_argument("--prepare-only", action="store_true", help="generate and train, no evaluation")
    p.add_argument("--table-only", action="store_true", help="print the table of the saved CSV")
    p.add_argument(
        "--decoding", type=Path, help="print the BLER of the decoding runs in this folder"
    )
    args = p.parse_args(argv)
    if args.decoding:
        print(decoding_table(args.decoding))
        return

    out = results_dir() / "input_ablation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "input_ablation.csv"
    if not args.table_only:
        prepare(args.seeds, args.epochs, args.patience)
        if args.prepare_only:
            return
        rows = aggregate(run(args.seeds), VALUES)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(path, encoding="utf-8", newline="") as f:
        text = ("condition", "detector")
        rows = [{k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    print(table(rows))


if __name__ == "__main__":
    main()
