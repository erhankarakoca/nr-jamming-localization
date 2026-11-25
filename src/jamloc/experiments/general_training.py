"""General training: one detector for every jammer, channel and modulation of its training mix.

The fused U-Net is trained on ``train_gen``, whose batches cycle through 18 combinations of the
block, hopping and pilot-comb jammers, the nominal, long-delay and fast channels, and QPSK and
16-QAM (thresholds from ``val_gen``). It is compared with the nominally trained U-Net (QPSK, block
jammers, nominal channels, thresholds from ``val``) on the matched test grid, on grids of each
condition of the mix, and on the pulsed jammer, which is in neither mix. A third U-Net is trained
on the same mix without the long delay spreads (``train_gen_short``), to see whether those
channels make the network trust the DMRS residual less.

The nominal model is trained as in ``experiments.paper_run`` (15 epochs, cosine schedule). The
general ones run up to ``--epochs`` with a learning rate that halves on a plateau and stop after
``--patience`` epochs without improvement.

    python -m jamloc.experiments.general_training --seeds 0 1 2
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from jamloc.dataset import generate, load
from jamloc.evaluate import _logits, aggregate, evaluate_scores
from jamloc.features import FEATURE_NAMES
from jamloc.paths import data_dir, results_dir
from jamloc.train import checkpoint_path, load_model, train

# Test grids and how the tables name them
CONDITIONS = {
    "test": "matched",
    "hopjam": "hopping",
    "combjam": "pilot comb",
    "longdelay": "1000 ns delay",
    "fast": "30 m/s",
    "test_qam16": "16-QAM",
    "hopjam_qam16": "16-QAM hopping",
    "pulsejam": "pulsed (unseen)",
}
# Training data tag: (name, validation split for the thresholds)
TRAININGS = {
    "": ("nominal", "val"),
    "gen": ("general", "val_gen"),
    "gen_short": ("general, no long delays", "val_gen_short"),
}
VARIANT = "fused"
VALUES = ("f1", "f1_pilot", "f1_data", "f1_partial", "fa", "pd_pilot", "pd_pilot_matched")
WEAK_JSR_DB = -15.0


def pilot_pd(pd_rows: list[dict], jsr_db: float = WEAK_JSR_DB) -> float:
    """Pd of the jammed scheduled pilot cells at ``jsr_db`` over both channels (nan if absent)."""
    for r in pd_rows:
        if (r["attack"], r["channel"], r["cells"], r["jsr_db"]) == (
            "pilot",
            "both",
            "scheduled",
            jsr_db,
        ):
            return r["pd"]
    return np.nan


def pilot_pd_matched(s_test: np.ndarray, test: dict, fa: float = 0.01) -> float:
    """The same Pd with the threshold that flags ``fa`` of the grid's own clean scored cells, so
    that detectors with different operating points compare at one false-alarm rate."""
    labels = test["labels"].astype(bool)
    clean = s_test[~labels & (s_test > -np.inf)]
    threshold = np.quantile(clean, 1 - fa)
    weak = (test["attack"] == 1) & (test["jsr_db"] == WEAK_JSR_DB)
    if not weak.any():
        return np.nan
    scheduled = labels & (test["features"][:, FEATURE_NAMES.index("alloc_mask")] > 0)
    return float((s_test[weak][scheduled[weak]] > threshold).mean())


def prepare(seeds, epochs: int, patience: int):
    """Generate the missing splits and train the missing models."""
    # "train" / "val" for the nominal data, "train_gen" / "val_gen" for data "gen", ...
    splits = [f"{kind}_{data}" if data else kind for data in TRAININGS for kind in ("train", "val")]
    for split in splits + list(CONDITIONS):
        if not (data_dir() / f"{split}.npz").is_file():
            generate(split, batch_size=128)
    for seed in seeds:
        for data in TRAININGS:
            if checkpoint_path(VARIANT, seed, data=data).is_file():
                continue
            if data:
                train(
                    VARIANT,
                    epochs=epochs,
                    seed=seed,
                    data=data,
                    schedule="plateau",
                    patience=patience,
                )
            else:
                train(VARIANT, epochs=15, seed=seed)


def run(seeds) -> list[dict]:
    """F1 per attack, false-alarm rate on scored cells and weak-pilot Pd, per grid, training
    and seed."""
    rows = []
    vals = {val_split: load(val_split) for _, val_split in TRAININGS.values()}
    for split, condition in CONDITIONS.items():
        test = load(split)
        for data, (training, val_split) in TRAININGS.items():
            val = vals[val_split]
            for seed in seeds:
                model = load_model(VARIANT, seed, data=data)
                s_val = _logits(model, VARIANT, torch.from_numpy(val["features"]))
                s_test = _logits(model, VARIANT, torch.from_numpy(test["features"]))
                pd_rows, f1_rows = evaluate_scores({VARIANT: (s_val, s_test)}, val, test)
                by = {r["attack"]: r for r in f1_rows}
                rows.append(
                    {
                        "condition": condition,
                        "training": training,
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
            print(f"{condition}, {training}: done", flush=True)
    return rows


def _joined(cells: list[dict], key: str) -> str:
    """The trainings' means of ``key`` as "a / b / c", or "-" where the grid has no value."""
    if np.isnan(cells[0][f"{key}_mean"]):
        return "-"
    return " / ".join(f"{r[f'{key}_mean']:.2f}" for r in cells)


def table(rows: list[dict]) -> str:
    """Markdown table per grid: F1 / false alarms of every training, and weak-pilot Pd at the
    validation thresholds and at a matched 1% false-alarm rate."""
    names = [name for name, _ in TRAININGS.values()]
    head = " | ".join(f"{n}: F1 / FA" for n in names)
    lines = [
        f"| Test grid | {head} | Pilot Pd at -15 dB | Same at 1% FA |",
        "| --- | " + "--- | " * (len(names) + 2),
    ]
    by = {(r["condition"], r["training"]): r for r in rows}
    for condition in CONDITIONS.values():
        cells = [by[(condition, n)] for n in names]
        f1_fa = " | ".join(
            f"{r['f1_mean']:.3f} +- {r['f1_std']:.3f} / {100 * r['fa_mean']:.1f}%" for r in cells
        )
        pd = _joined(cells, "pd_pilot")
        pd_matched = _joined(cells, "pd_pilot_matched")
        lines.append(f"| {condition} | {f1_fa} | {pd} | {pd_matched} |")
    return "\n".join(lines)


# Decoding runs of the general U-Net's map (one folder per case, see docs/reproducing.md)
DECODING_CASES = {
    "hop": "QPSK hopping",
    "whole": "QPSK every pilot",
    "nom": "QPSK block",
    "qam16": "16-QAM block",
    "qamhop": "16-QAM hopping",
}
DECODING_RECEIVERS = {
    "standard": "No protection",
    "nomap_shrink": "No map, shrinkage",
    "p_r_e_soft": "Nominal U-Net, per PRB",
    "p_r_e_shrink": "Nominal U-Net, shrinkage",
    "gen_soft": "General U-Net, per PRB",
    "gen_shrink": "General U-Net, shrinkage",
    "true_shrink": "True map, shrinkage",
}


def decoding_counts(root) -> tuple[dict, dict]:
    """Block errors and blocks per (case, attack, JSR, receiver), summed over the seed files of
    ``root/<case>/recovery``. The JSR of slots without a jammer is nan, stored as None."""
    errors, blocks = {}, {}
    for case in DECODING_CASES:
        for path in sorted((root / case / "recovery").glob("*_seed*.csv")):
            with open(path, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    jsr = float(r["jsr_db"])
                    key = (case, r["attack"], None if np.isnan(jsr) else jsr, r["receiver"])
                    errors[key] = errors.get(key, 0) + int(r["block_errors"])
                    blocks[key] = blocks.get(key, 0) + int(r["blocks"])
    return errors, blocks


def _column(case: str, attack: str, jsr_db) -> str:
    """Column title of one point, e.g. "hopping partial +9" or "block, no jammer"."""
    name = DECODING_CASES[case].split(" ", 1)[1]
    if jsr_db is None:
        return f"{name}, no jammer"
    if attack == "pilot_whole_allocation":  # the case name already says it
        return f"{name} {jsr_db:+.0f}"
    return f"{name} {attack} {jsr_db:+.0f}"


def decoding_table(root) -> str:
    """Markdown tables of the pooled BLER (%), one for QPSK and one for 16-QAM."""
    errors, blocks = decoding_counts(root)
    points = sorted(
        {k[:3] for k in errors},
        key=lambda p: (list(DECODING_CASES).index(p[0]), p[1], -99 if p[2] is None else p[2]),
    )
    out = []
    for modulation in ("QPSK", "16-QAM"):
        cols = [p for p in points if DECODING_CASES[p[0]].startswith(modulation)]
        if not cols:
            continue
        head = " | ".join(_column(c, a, j) for c, a, j in cols)
        out += [f"{modulation}, BLER (%)", "", f"| Receiver | {head} |"]
        out.append("| --- |" + " --- |" * len(cols))
        for rx, name in DECODING_RECEIVERS.items():
            cells = []
            for c, a, j in cols:
                n, e = blocks.get((c, a, j, rx), 0), errors.get((c, a, j, rx), 0)
                cells.append("-" if n == 0 else f"0/{n}" if e == 0 else f"{100 * e / n:.2f}")
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=80, help="most epochs of the general training")
    p.add_argument("--patience", type=int, default=10, help="early stopping of the general one")
    p.add_argument("--table-only", action="store_true")
    p.add_argument(
        "--decoding", type=Path, help="print the pooled BLER of the decoding runs in this folder"
    )
    args = p.parse_args(argv)
    if args.decoding:
        print(decoding_table(args.decoding))
        return

    out = results_dir() / "general_training"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "general_training.csv"
    if not args.table_only:
        prepare(args.seeds, args.epochs, args.patience)
        rows = aggregate(run(args.seeds), VALUES)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with open(path, encoding="utf-8", newline="") as f:
        text = ("condition", "training")
        rows = [{k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    print(table(rows))


if __name__ == "__main__":
    main()
