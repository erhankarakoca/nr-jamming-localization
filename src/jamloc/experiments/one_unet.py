"""One U-Net for one and two DMRS symbols and three modulations (paper v11).

The fused U-Net is trained on ``train_all``, whose batches cycle through 54 conditions: the
jammers, channels and modulations of the general training (``experiments.general_training``) plus
64-QAM, each with one and with two DMRS symbols. It is trained as the general U-Net (learning
rate halved on a plateau, early stopping) and takes one threshold from ``val_all`` for every
configuration.

It is compared with the general U-Net of paper v10 (``train_gen``, one DMRS symbol, QPSK and
16-QAM, threshold from ``val_gen``) on test grids with one DMRS symbol and on the same grids with
two (``dataset.DMRS2_GRIDS``): F1 per attack type, false alarms on scored cells and the Pd of weak
pilot jamming, three training seeds each. The pulsed jammer stays out of both trainings.

    python -m jamloc.experiments.one_unet --seeds 0 1 2
    python -m jamloc.experiments.one_unet --table-only
"""

import argparse
import csv

import numpy as np
import torch

from jamloc.dataset import dmrs2_split, generate, load
from jamloc.evaluate import _logits, aggregate, evaluate_scores
from jamloc.experiments.general_training import pilot_pd, pilot_pd_matched
from jamloc.paths import data_dir, results_dir
from jamloc.train import checkpoint_path, load_model, train

VARIANT = "fused"
# Table name: (training data tag, validation split of the threshold)
DETECTORS = {"v10 U-Net": ("gen", "val_gen"), "v11 U-Net": ("all", "val_all")}
NEW = "all"  # the detector this experiment trains
# Test grids with one DMRS symbol and how the tables name them. Each also exists with two DMRS
# symbols (``test`` as the nominal ``test_dmrs2``, the others as ``dmrs2_split``).
GRIDS = {
    "test": "QPSK block",
    "hopjam": "QPSK hopping",
    "combjam": "QPSK pilot comb",
    "longdelay": "QPSK 1000 ns",
    "fast": "QPSK 30 m/s",
    "test_qam16": "16-QAM block",
    "hopjam_qam16": "16-QAM hopping",
    "test_qam64": "64-QAM block",
    "hopjam_qam64": "64-QAM hopping",
    "pulsejam": "QPSK pulsed (unseen)",
}
VALUES = ("f1", "f1_pilot", "f1_data", "f1_partial", "fa", "pd_pilot", "pd_pilot_matched")


def splits(dmrs: int) -> dict[str, str]:
    """{split: grid name} of the test grids with ``dmrs`` DMRS symbols (1 or 2)."""
    if dmrs == 1:
        return dict(GRIDS)
    return {dmrs2_split(s): name for s, name in GRIDS.items()}


def prepare(seeds, epochs: int, patience: int):
    """Generate the missing splits and train the missing models (the v10 U-Net is not trained
    again, it comes from ``experiments.general_training``)."""
    needed = [f"train_{NEW}", f"val_{NEW}", *(v for _, v in DETECTORS.values())]
    needed += [*splits(1), *splits(2)]
    for split in needed:
        if not (data_dir() / f"{split}.npz").is_file():
            generate(split, batch_size=128)
    for seed in seeds:
        if checkpoint_path(VARIANT, seed, data=NEW).is_file():
            continue
        train(VARIANT, epochs=epochs, seed=seed, data=NEW, schedule="plateau", patience=patience)


def run(seeds) -> list[dict]:
    """F1 per attack, false alarms on scored cells and weak-pilot Pd, per grid, number of DMRS
    symbols, detector and seed. A detector without its checkpoint or validation split is left
    out."""
    models, vals = {}, {}
    for name, (data, val_split) in DETECTORS.items():
        if not (data_dir() / f"{val_split}.npz").is_file():
            continue
        val = load(val_split)
        features = torch.from_numpy(val["features"])
        for seed in seeds:
            if not checkpoint_path(VARIANT, seed, data=data).is_file():
                continue
            model = load_model(VARIANT, seed, data=data)
            models[name, seed] = model
            vals[name, seed] = (val, _logits(model, VARIANT, features))
    rows = []
    for dmrs in (1, 2):
        for split, grid in splits(dmrs).items():
            test = load(split)
            features = torch.from_numpy(test["features"])
            for (name, seed), model in models.items():
                val, s_val = vals[name, seed]
                s_test = _logits(model, VARIANT, features)
                pd_rows, f1_rows = evaluate_scores({VARIANT: (s_val, s_test)}, val, test)
                by = {r["attack"]: r for r in f1_rows}
                rows.append(
                    {
                        "grid": grid,
                        "dmrs": dmrs,
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
            print(f"{grid}, {dmrs} DMRS: done", flush=True)
    return rows


def table(rows: list[dict]) -> str:
    """Markdown table: F1 / false alarms (% of scored clean cells) per grid, number of DMRS
    symbols and detector, then the pilot F1 and weak-pilot Pd of the block grids."""
    by = {(r["grid"], int(r["dmrs"]), r["detector"]): r for r in rows}
    cols = [(d, n) for d in (1, 2) for n in DETECTORS]
    head = " | ".join(f"{n}, {d} DMRS" for d, n in cols)
    lines = [f"| Grid | {head} |", "| --- |" + " --- |" * len(cols)]
    for grid in GRIDS.values():
        cells = []
        for d, n in cols:
            r = by.get((grid, d, n))
            cells.append(
                "-" if r is None else f"{r['f1_mean']:.3f} / {100 * r['fa_mean']:.1f}%"
            )
        lines.append(f"| {grid} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| Grid | Detector | DMRS | F1 pilot | Pilot Pd at -15 dB | Same at 1% FA |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for grid in ("QPSK block", "16-QAM block", "64-QAM block"):
        for d, n in cols:
            r = by.get((grid, d, n))
            if r is None:
                continue
            values = [r[f"{k}_mean"] for k in ("f1_pilot", "pd_pilot", "pd_pilot_matched")]
            cells = " | ".join("-" if np.isnan(v) else f"{v:.3f}" for v in values)
            lines.append(f"| {grid} | {n} | {d} | {cells} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=80, help="most epochs (as the general training)")
    p.add_argument("--patience", type=int, default=10, help="early stopping (as the general one)")
    p.add_argument("--prepare-only", action="store_true", help="generate and train, no evaluation")
    p.add_argument("--table-only", action="store_true", help="print the table of the saved CSV")
    args = p.parse_args(argv)

    out = results_dir() / "one_unet"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "one_unet.csv"
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
        text = ("grid", "detector")
        rows = [{k: v if k in text else float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    print(table(rows))


if __name__ == "__main__":
    main()
