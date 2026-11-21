"""Compare jamming detection and BLER with one or two PUSCH DMRS symbols per slot.

The nominal and two-DMRS detectors are evaluated on their respective test splits. Thresholds
target 1% false alarms among scored, non-jammed cells on the matching validation split. Detection
uses both TDL models. The F1 and false-alarm columns count all carrier cells of the test grid; the
Pd columns count the jammed cells inside the allocation at -15 dB JSR, as in the paper.

Recovery uses TDL-C at 15 dB SNR, with 12 scheduled PRBs and a block of 4 jammed PRBs inside
that allocation. Each receiver decodes the same batches at a point. The additional first-only
pilot attack tests what happens when the second DMRS symbol is left clean.

    python -m jamloc.experiments.dmrs_count --seeds 0 1 2 --batch-size 64 --batches 100
    python -m jamloc.experiments.dmrs_count --small

The full run reuses the nominal validation and test splits and nominal checkpoints. It builds
missing two-DMRS splits and checkpoints. The smoke test trains for one epoch on tiny splits in
temporary directories, keeping the same attack and JSR grid for the BLER comparison.
"""

import argparse
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from itertools import product
from tempfile import TemporaryDirectory

import numpy as np
import torch

from jamloc.baselines import block_test_scores, evm_scores
from jamloc.config import SimConfig
from jamloc.dataset import generate, load
from jamloc.evaluate import FA, _logits, _write_csv
from jamloc.experiments.recovery import ALLOC, JAM_SIZE, _scenario, simulate_point
from jamloc.features import FEATURE_NAMES
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, seed_everything
from jamloc.metrics import detection_rate, f1, threshold_at_fa
from jamloc.paths import data_dir, results_dir
from jamloc.train import checkpoint_path, load_model, train

VARIANTS = ("fused", "no_residual")
RECEIVERS = ("standard", "lmmse", "p_r_e_soft", "true_soft")
JSR_DB = (-15, -3, 3, 9)
SMALL_SPLITS = {"train": 64, "val": 64, "test": 60}
TABLE_COLUMNS = (
    ("f1_pilot", "Pilot F1"),
    ("f1_data", "Data F1"),
    ("f1_partial", "Partial F1"),
    ("f1_all", "Overall F1"),
    ("fa", "False alarm"),
    ("pd_pilot_m15", "Pilot Pd at -15 dB"),
    ("pd_data_m15", "Data Pd at -15 dB"),
)
DETECTOR_LABELS = {
    "fused": "U-Net P+R+E",
    "no_residual": "U-Net P+E",
    "block_test": "Block test",
    "evm_threshold": "EVM threshold",
}


@contextmanager
def _small_artifacts(out):
    """Give the smoke test its own datasets and checkpoints."""
    with TemporaryDirectory(dir=data_dir()) as data, TemporaryDirectory(dir=out) as results:
        roots = {"JAMLOC_DATA_DIR": data, "JAMLOC_RESULTS_DIR": results}
        previous = {key: os.environ.get(key) for key in roots}
        os.environ.update(roots)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key)
                else:
                    os.environ[key] = value


def _prepare(data: str, seeds, epochs: int, small: bool):
    """Build missing datasets and checkpoints for a new training condition."""
    suffix = f"_{data}" if data else ""
    missing = [
        (variant, seed)
        for variant, seed in product(VARIANTS, seeds)
        if not checkpoint_path(variant, seed, data=data).is_file()
    ]
    for name in ("train", "val", "test") if missing else ("val", "test"):
        split = f"{name}{suffix}"
        if not (data_dir() / f"{split}.npz").is_file():
            generate(split, SMALL_SPLITS[name] if small else None)
    for variant, seed in missing:
        train(variant, epochs=epochs, seed=seed, data=data)


def _detection_row(count, detector, seed, threshold, scores, test):
    labels = test["labels"].astype(bool)
    inside = test["features"][:, FEATURE_NAMES.index("alloc_mask")] > 0
    pred = scores > threshold
    row = {
        "dmrs_count": count,
        "detector": detector,
        "seed": seed,
        "threshold": threshold,
    }
    for attack in ("pilot", "data", "partial"):
        rows = test["attack"] == ATTACKS.index(attack)
        row[f"f1_{attack}"] = f1(pred[rows], labels[rows])
        if attack in ("pilot", "data"):
            weak = rows & (test["jsr_db"] == -15)
            cells = labels[weak] & inside[weak]  # jammed cells inside the allocation
            row[f"pd_{attack}_m15"] = detection_rate(scores[weak][cells], threshold)
    row["f1_all"] = f1(pred, labels)
    row["fa"] = float(pred[~labels].mean())
    return row


def _detection(data: str, count: int, seeds):
    """Detection rows and the fitted fused detector and threshold for each recovery run."""
    suffix = f"_{data}" if data else ""
    val, test = load(f"val{suffix}"), load(f"test{suffix}")
    x_val, x_test = (torch.from_numpy(d["features"]) for d in (val, test))
    clean = val["labels"] == 0
    rows, detectors = [], {}
    for variant, seed in product(VARIANTS, seeds):
        model = load_model(variant, seed, data=data)
        threshold = threshold_at_fa(_logits(model, variant, x_val)[clean], FA)
        scores = _logits(model, variant, x_test)
        rows.append(_detection_row(count, variant, seed, threshold, scores, test))
        if variant == "fused":
            detectors[seed] = {"fused": (model, threshold)}

    for name, score in (
        ("block_test", lambda d: block_test_scores(d, val)),
        ("evm_threshold", lambda d: evm_scores(d["features"])),
    ):
        threshold = threshold_at_fa(score(val)[clean], FA)
        rows.append(_detection_row(count, name, "", threshold, score(test), test))
    return rows, detectors


def _recovery(count: int, detectors, batch_size: int, batches: int):
    """Run every receiver for the same number of batches at every recovery point."""
    rows = []
    cases = {"all": ("pilot", "data", "partial")}
    if count == 2:
        cases["first"] = ("pilot",)
    for seed, models in detectors.items():
        for jammer_dmrs, attacks in cases.items():
            cfg = SimConfig(dmrs_additional_position=count - 1, jammer_dmrs=jammer_dmrs)
            link = JammedPUSCHLink(cfg)
            # Replay the same pilot-attack batches for both jammer DMRS scopes.
            rng = seed_everything(1000 + seed)
            for attack, jsr in product(attacks, JSR_DB):
                scenario_fn = partial(_scenario, rng, batch_size, "C", attack, jsr, 15.0)
                counts, _ = simulate_point(
                    link, models, scenario_fn, RECEIVERS, batches, target_block_errors=float("inf")
                )
                for receiver, (_, _, errors, blocks) in counts.items():
                    rows.append(
                        {
                            "dmrs_count": count,
                            "jammer_dmrs": jammer_dmrs,
                            "seed": seed,
                            "channel": "C",
                            "snr_db": 15.0,
                            "ue_prbs": ALLOC[1],
                            "jammed_prbs": JAM_SIZE,
                            "attack": attack,
                            "jsr_db": jsr,
                            "receiver": receiver,
                            "block_errors": int(errors),
                            "blocks": int(blocks),
                            "bler": errors / blocks,
                        }
                    )
                print(
                    f"[{count} DMRS, {jammer_dmrs}, seed {seed}] {attack} at {jsr:+g} dB",
                    flush=True,
                )
    return rows


def _write_table(rows, path, small: bool):
    """Write mean detection metrics over the requested training seeds."""
    header = ["DMRS count", "Detector", *(label for _, label in TABLE_COLUMNS)]
    lines = [
        "# Detection with one or two DMRS symbols",
        "",
        "Thresholds target 1% false alarms among scored, non-jammed validation cells.",
        "F1 and false alarms use the whole carrier of the matching test split; Pd uses the jammed",
        "cells inside the allocation.",
        "U-Net results are averaged over the requested training seeds.",
    ]
    if small:
        lines.append("Smoke-test results use tiny splits and one training epoch.")
    lines += ["", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for count, detector in product((1, 2), DETECTOR_LABELS):
        group = [r for r in rows if r["dmrs_count"] == count and r["detector"] == detector]
        means = [np.mean([r[key] for r in group]) for key, _ in TABLE_COLUMNS]
        fields = [str(count), DETECTOR_LABELS[detector], *(f"{x:.1%}" for x in means)]
        lines.append("| " + " | ".join(fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="detector training seeds")
    p.add_argument("--epochs", type=int, default=15, help="epochs for missing two-DMRS models")
    p.add_argument("--batch-size", type=int, help="slots per recovery batch (default 64)")
    p.add_argument("--batches", "--max-mc-iter", type=int, help="batches per point (default 100)")
    p.add_argument(
        "--small", action="store_true", help="tiny splits, one epoch and four slots per point"
    )
    p.add_argument(
        "--no-recovery",
        action="store_true",
        help="only the detection tables (recovery.csv of an earlier run is kept)",
    )
    args = p.parse_args(argv)
    if (
        args.epochs < 1
        or (args.batch_size is not None and args.batch_size < 1)
        or (args.batches is not None and args.batches < 1)
    ):
        p.error("--epochs, --batch-size and --batches must be positive")
    batch_size = args.batch_size or (4 if args.small else 64)
    batches = args.batches or (1 if args.small else 100)

    if not args.small:
        needed = [data_dir() / f"{split}.npz" for split in ("val", "test")]
        needed += [checkpoint_path(v, s) for v, s in product(VARIANTS, args.seeds)]
        missing = [str(path) for path in needed if not path.is_file()]
        if missing:
            p.error(
                "generate the nominal splits and train the nominal models first. Missing:\n"
                + "\n".join(missing)
            )

    out = results_dir() / "dmrs_count"
    out.mkdir(parents=True, exist_ok=True)
    with _small_artifacts(out) if args.small else nullcontext():
        if args.small:
            _prepare("", args.seeds, 1, small=True)
        _prepare("dmrs2", args.seeds, 1 if args.small else args.epochs, args.small)
        detection, detectors = [], {}
        for count, data in ((1, ""), (2, "dmrs2")):
            rows, detectors[count] = _detection(data, count, args.seeds)
            detection.extend(rows)
        _write_csv(detection, out / "detection.csv")
        _write_table(detection, out / "detection.md", args.small)

        if args.no_recovery:
            print(f"Wrote detection.csv and detection.md in {out}")
            return
        recovery = [
            row
            for count, models in detectors.items()
            for row in _recovery(count, models, batch_size, batches)
        ]
        _write_csv(recovery, out / "recovery.csv")
    print(f"Wrote detection.csv, detection.md and recovery.csv in {out}")


if __name__ == "__main__":
    main()
