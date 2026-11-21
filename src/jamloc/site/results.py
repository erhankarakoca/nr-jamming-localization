"""Package the paper's result tables for the site without recomputing experiments."""

import csv
import json
import math
import shutil
from pathlib import Path

PD_CSV = "eval/test_pd_vs_jsr.csv"
MARGIN_CSV = "margin/margin_vs_snr_ci.csv"
MATCHED_CSV = "intervals/test_matched_by_attack_f1_ci.csv"
MATCHED_PAIRED_CSV = "intervals/test_matched_by_attack_paired.csv"
DMRS_DETECTION_CSV = "dmrs_count/detection.csv"
DMRS_RECOVERY_CSV = "dmrs_count/recovery.csv"
RECOVERY_RUNS = {
    "nominal": "recovery_C",
    "pulsed": "recovery_C_pulsed",
    "hopping": "recovery_C_hopping",
}

POPULATIONS = {
    "pd": (
        "Detection probability is the fraction of jammed cells inside the scheduled allocation "
        "that are flagged on the test split, pooling TDL-A and TDL-C. Thresholds target 1% "
        "per-cell false alarms on validation. Lines are means over training seeds 0, 1, and 2. "
        "The plotted bands span one standard deviation on either side, clipped to [0, 1]."
    ),
    "margin": (
        "The fused U-Net's smallest margin over pilot, data, and partial-band jamming on the "
        "scheduled PRBs is the JSR at 10% standard-receiver BLER minus the JSR at 90% slot "
        "detection. Slot detection averages over training seeds 0, 1, and 2. Its JSR grid runs "
        "from −30 to 0 dB at 15 dB SNR and from −36 to 0 dB at 5, 8, and 11 dB SNR, in 3 dB "
        "steps. The 1% and 0.1% per-slot false-alarm thresholds are calibrated on every other "
        "benign batch in each margin split, starting with the first. The remaining benign "
        "batches measure the false-alarm rate. The 95% intervals use a stratified batch bootstrap "
        "with binomial resampling of block errors. Open markers give upper bounds where slot "
        "detection did not reach 90% on the tested JSR grid."
    ),
    "recovery": (
        "Block error rate (BLER) pools block errors and decoded blocks across detector training "
        "seeds 0, 1, and 2 at +9 dB JSR on TDL-C at 15 dB SNR with known noise power. The "
        "allocation spans 12 PRBs and the jammer occupies 4 PRBs when active. Receivers share the same draws "
        "but can stop at different block counts. The paired 95% intervals use only slots "
        "decoded by both receivers, so their BLER differences can differ from the pooled table."
    ),
    "matched": (
        "On the nominal test split, thresholds are fitted by attack and detector to flag 1% "
        "of scored clean cells, rounded down to a whole cell. F1 uses all cell decisions. "
        "Detection probability uses jammed scheduled cells at the weakest tested JSR. "
        "The values average over training seeds 0, 1, and 2. The 95% batch-bootstrap intervals hold "
        "the matched decisions fixed."
    ),
    "dmrs": (
        "One DMRS symbol per slot against two (symbols 2 and 11), each with detectors trained "
        "on its own data. F1 and false alarms cover the whole carrier of the test split; Pd counts "
        "the jammed cells inside the allocation at −15 dB JSR. U-Net values average training "
        "seeds 0, 1, and 2. BLER pools the same seeds at +9 dB JSR on TDL-C at 15 dB SNR with the "
        "UE on 12 PRBs and the jammer on 4. In the last row the jammer assumes one front-loaded "
        "DMRS symbol and jams only the first."
    ),
}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _number(value: str) -> float | None:
    number = float(value) if value else math.nan
    return number if math.isfinite(number) else None


def _fields(row, *, labels, numbers="", counts=""):
    return {
        **{key: row[key] for key in labels.split()},
        **{key: _number(row[key]) for key in numbers.split()},
        **{key: int(row[key]) for key in counts.split()},
    }


def _pd(rows):
    return [
        _fields(
            r, labels="detector attack", numbers="jsr_db pd_mean pd_std", counts="num_cells n_seeds"
        )
        for r in rows
        if r["cells"] == "scheduled" and r["channel"] == "both"
    ]


def _margin(rows):
    return [
        {
            **_fields(
                r,
                labels="channel",
                numbers="snr_db slot_fa_target margin_db margin_lo_db margin_hi_db",
                counts="boot_draws",
            ),
            "point_any_attack_censored": r["point_any_attack_censored"].lower() == "true",
        }
        for r in rows
        if r["detector"] == "fused"
    ]


def _recovery(path):
    from jamloc.experiments.recovery import _read_rows, pool_seeds

    rows = pool_seeds(r for r in _read_rows(path) if r["jsr_db"] == 9)
    keys = ["receiver", "jammer", "attack", "jsr_db", "snr_db", "bler", "block_errors", "blocks"]
    return [{key: row[key] for key in keys} for row in rows]


def _recovery_paired(rows, jammer):
    return [
        {
            "jammer": jammer,
            **_fields(
                r,
                labels="first second attack",
                numbers="jsr_db bler_first bler_second delta delta_lo delta_hi",
                counts="slots only_first only_second",
            ),
        }
        for r in rows
        if _number(r["jsr_db"]) == 9
    ]


def _matched(rows):
    return [
        _fields(
            r,
            labels="detector attack",
            numbers="f1 f1_lo f1_hi fa_scored fa_scored_lo fa_scored_hi jsr_db pd pd_lo pd_hi",
            counts="n_seeds",
        )
        for r in rows
        if r["seed"] == "mean"
    ]


def _matched_paired(rows):
    return [
        _fields(
            r,
            labels="first second metric attack",
            numbers="jsr_db delta delta_lo delta_hi",
            counts="n_seeds",
        )
        for r in rows
        if r["seed"] == "mean"
    ]


def _dmrs_detection(rows):
    # U-Nets have one row per training seed, the model-free tests one row: average per detector.
    groups = {}
    for r in rows:
        groups.setdefault((int(r["dmrs_count"]), r["detector"]), []).append(r)
    fields = ("f1_pilot", "f1_all", "fa", "pd_pilot_m15", "pd_data_m15")
    return [
        {
            "dmrs_count": count,
            "detector": detector,
            "n_seeds": len(group),
            **{f: sum(float(r[f]) for r in group) / len(group) for f in fields},
        }
        for (count, detector), group in groups.items()
    ]


def _dmrs_recovery(rows):
    pooled = {}
    for r in rows:
        if float(r["jsr_db"]) != 9:
            continue
        key = (int(r["dmrs_count"]), r["jammer_dmrs"], r["attack"], r["receiver"])
        p = pooled.setdefault(key, [0, 0])
        p[0] += int(r["block_errors"])
        p[1] += int(r["blocks"])
    return [
        {
            "dmrs_count": count,
            "jammer_dmrs": jammer,
            "attack": attack,
            "receiver": receiver,
            "block_errors": errors,
            "blocks": blocks,
            "bler": errors / blocks if blocks else None,
        }
        for (count, jammer, attack, receiver), (errors, blocks) in pooled.items()
    ]


def _recovery_files(directory: Path, stem: str) -> tuple[Path, Path]:
    for name in (f"{stem}_replay", stem):
        points = directory / f"{name}.csv"
        intervals = directory / f"{name}_paired.csv"
        if points.is_file() and intervals.is_file():
            return points, intervals
    raise FileNotFoundError(
        f"Missing a complete recovery pair for {stem} in {directory}. "
        f"Expected {stem}_replay.csv with {stem}_replay_paired.csv "
        f"or {stem}.csv with {stem}_paired.csv."
    )


def write_results(source: Path, build: Path) -> None:
    """Write data/results.json and the original CSVs, including paper replay filenames."""
    downloads = build / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    paths = {kind: [] for kind in POPULATIONS}

    def copy(kind, path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing paper result {path}")
        target = downloads / path.name
        shutil.copyfile(path, target)
        paths[kind].append(target.relative_to(build).as_posix())
        return path

    pd = copy("pd", source / PD_CSV)
    margin = copy("margin", source / MARGIN_CSV)
    matched = copy("matched", source / MATCHED_CSV)
    matched_paired = copy("matched", source / MATCHED_PAIRED_CSV)
    recovery, paired = [], []
    recovery_dir = source / "recovery"
    for jammer, stem in RECOVERY_RUNS.items():
        points, intervals = _recovery_files(recovery_dir, stem)
        recovery.extend(_recovery(copy("recovery", points)))
        paired.extend(_recovery_paired(_read(copy("recovery", intervals)), jammer))
    for path in sorted(recovery_dir.glob("recovery_C*_replay_paired.csv")):
        if path.stem.removesuffix("_replay_paired") not in RECOVERY_RUNS.values():
            copy("recovery", path)

    dmrs_detection = copy("dmrs", source / DMRS_DETECTION_CSV)
    dmrs_recovery = copy("dmrs", source / DMRS_RECOVERY_CSV)

    payload = {
        "pd": {"rows": _pd(_read(pd))},
        "margin": {"rows": _margin(_read(margin))},
        "recovery": {"rows": recovery, "paired": paired},
        "matched": {
            "rows": _matched(_read(matched)),
            "paired": _matched_paired(_read(matched_paired)),
        },
        "dmrs": {
            "rows": _dmrs_detection(_read(dmrs_detection)),
            "recovery": _dmrs_recovery(_read(dmrs_recovery)),
        },
    }
    for kind, section in payload.items():
        if not section["rows"]:
            raise ValueError(f"No eligible {kind} rows in {', '.join(paths[kind])}")
        section.update(population=POPULATIONS[kind], downloads=paths[kind])
    (build / "data" / "results.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
