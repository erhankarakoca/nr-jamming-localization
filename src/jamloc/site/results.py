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
RECOVERY_CURVES_CSV = "recovery_pr9/recovery_C.csv"
RECOVERY_CURVES_RECEIVERS = ("standard", "lmmse", "block_soft", "p_r_e_soft", "true_soft")
# The data-aided receiver study: its own results folder, one recovery run per SNR
DATA_AIDED_RUNS = {
    15.0: "receiver_study/recovery/recovery_C",
    8.0: "receiver_study/recovery/recovery_C_snr8",
}
DATA_AIDED_RECEIVERS = (
    "standard",
    "lmmse",
    "lmmse_da",
    "p_r_e_soft",
    "p_r_e_soft_da",
    "true_soft",
    "true_soft_da",
)
RECOVERY_RUNS = {
    "nominal": "recovery_C",
    "pulsed": "recovery_C_pulsed",
    "hopping": "recovery_C_hopping",
}

POPULATIONS = {
    "pd": (
        "Detection probability is the fraction of jammed cells inside the scheduled allocation "
        "that are flagged on the test split, pooling TDL-A and TDL-C. Thresholds target 1% "
        "per-cell false alarms on validation. Curves are means over training seeds 0, 1, and 2; the "
        "hover text and the table give the standard deviation over the seeds."
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
    "recovery_curves": (
        "Block error rate (BLER) pools block errors and decoded blocks over detector training "
        "seeds 0, 1, and 2 on TDL-C at 15 dB SNR with known noise power. The nominal jammer "
        "sweep spans −15 to +9 dB JSR and includes a benign reference. The UE occupies 12 PRBs. "
        "The jammer occupies 4 PRBs for pilot, data, and partial-band jamming and all 12 PRBs "
        "for whole-allocation pilot jamming."
    ),
    "matched": (
        "On the nominal test split, thresholds are fitted by attack and detector to flag 1% "
        "of scored clean cells, rounded down to a whole cell. F1 uses all cell decisions. "
        "Detection probability uses jammed scheduled cells at the weakest tested JSR. "
        "The values average over training seeds 0, 1, and 2. The 95% batch-bootstrap intervals hold "
        "the matched decisions fixed."
    ),
    "data_aided": (
        "Pilot jamming of every DMRS cell of the 12-PRB allocation (PRBs 6-17), TDL-C, known noise "
        "power; at 8 dB SNR also data and partial-band jamming of 4 PRBs. Errors pool detector "
        "training seeds 0, 1, and 2, up to 1,600 slots each; a receiver stops after 100 block "
        "errors, so denominators differ. Data-aided: up to three passes after a failed decoding, "
        "in which the decoded symbols of the data cells the map calls clean act as extra pilots; a "
        "block keeps the first decoding whose CRC passes. The paired CSVs give each difference "
        "with a 95% interval on the slots both receivers decoded."
    ),
    "dmrs": (
        "One DMRS symbol per slot against two (symbols 2 and 11). The U-Nets are trained on their own "
        "DMRS configuration; the block test and the EVM threshold are calibrated on the matching "
        "validation data. F1 and false alarms cover the whole carrier of the test split; Pd counts "
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


def _recovery_curves(rows):
    from jamloc.experiments.recovery import pool_seeds

    selected = []
    for row in rows:
        if (
            row["receiver"] not in RECOVERY_CURVES_RECEIVERS
            or row["jammer"] != "nominal"
            or row["seed"] not in ("0", "1", "2")
            or _number(row["snr_db"]) != 15
        ):
            continue
        jsr = _number(row["jsr_db"])
        if (jsr is None and row["attack"] == "none") or (jsr is not None and -15 <= jsr <= 9):
            selected.append({**row, "jsr_db": jsr, "snr_db": 15.0})
    keys = ("attack", "jsr_db", "receiver", "block_errors", "blocks", "bler")
    return [{key: row[key] for key in keys} for row in pool_seeds(selected)]


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


def _data_aided(paths):
    """BLER per SNR, attack, JSR and receiver of the data-aided study, pooled over seeds."""
    pooled = {}
    for snr, path in paths.items():
        for r in _read(path):
            if r["receiver"] not in DATA_AIDED_RECEIVERS or r["jsr_db"] in ("", "nan"):
                continue
            key = (snr, r["attack"], float(r["jsr_db"]), r["receiver"])
            p = pooled.setdefault(key, [0, 0])
            p[0] += int(r["block_errors"])
            p[1] += int(r["blocks"])
    return [
        {
            "snr_db": snr,
            "attack": attack,
            "jsr_db": jsr,
            "receiver": receiver,
            "block_errors": errors,
            "blocks": blocks,
            "bler": errors / blocks,
        }
        for (snr, attack, jsr, receiver), (errors, blocks) in sorted(pooled.items())
    ]


def _data_aided_summary(rows):
    """One sentence from the pooled 15 dB whole-allocation counts, or "" if a value is missing."""
    found = {
        (r["jsr_db"], r["receiver"]): f"{100 * r['bler']:.1f}%"
        for r in rows
        if r["snr_db"] == 15.0 and r["attack"] == "pilot_whole_allocation"
    }
    needed = [(j, rx) for j in (9.0, 3.0) for rx in ("p_r_e_soft", "p_r_e_soft_da", "lmmse_da")]
    if any(key not in found for key in needed):
        return ""
    text = (
        "With every pilot of the allocation jammed there is no clean pilot to estimate the "
        "channel from. At 15 dB SNR, data-aided passes lower the BLER with the U-Net map from "
        f"{found[9.0, 'p_r_e_soft']} to {found[9.0, 'p_r_e_soft_da']} at +9 dB JSR and from "
        f"{found[3.0, 'p_r_e_soft']} to {found[3.0, 'p_r_e_soft_da']} at +3 dB; the same passes "
        f"without a map reach {found[9.0, 'lmmse_da']} and {found[3.0, 'lmmse_da']}."
    )
    # At 8 dB SNR: data and partial-band jamming of 4 PRBs, where the map decides which data cells
    # may serve as pilots
    low = {
        (r["receiver"], r["attack"], r["jsr_db"]): 100 * r["bler"]
        for r in rows
        if r["snr_db"] == 8.0 and r["attack"] in ("data", "partial")
    }
    points = {(a, j) for _, a, j in low}
    if points and all(("p_r_e_soft", *k) in low and ("p_r_e_soft_da", *k) in low for k in points):
        before = [low["p_r_e_soft", *k] for k in points]
        after = [low["p_r_e_soft_da", *k] for k in points]
        text += (
            " At 8 dB SNR, data and partial-band jamming of 4 PRBs still leave "
            f"{min(before):.1f}–{max(before):.1f}% BLER with the map; the passes lower it to "
            f"{min(after):.1f}–{max(after):.1f}%. Without a map they change little there, because "
            "they then also treat jammed data cells as pilots."
        )
    # State any whole-allocation point where the passes without a map end lower than with it
    whole = {
        (r["snr_db"], r["jsr_db"], r["receiver"]): 100 * r["bler"]
        for r in rows
        if r["attack"] == "pilot_whole_allocation"
    }
    for snr, jsr in sorted({(k[0], k[1]) for k in whole}):
        mapped, blind = whole.get((snr, jsr, "p_r_e_soft_da")), whole.get((snr, jsr, "lmmse_da"))
        if mapped is not None and blind is not None and blind < mapped:
            text += (
                f" With every pilot jammed at {snr:g} dB SNR and {jsr:+g} dB JSR, the passes "
                f"without a map end lower than with it ({blind:.1f}% against {mapped:.1f}%)."
            )
    return text


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

    def copy(kind, path, prefix=""):
        if not path.is_file():
            raise FileNotFoundError(f"Missing paper result {path}")
        target = downloads / f"{prefix}{path.name}"  # a prefix keeps same-named files apart
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
    curves = copy("recovery_curves", source / RECOVERY_CURVES_CSV, "recovery_curves_")

    aided = {}
    for snr, stem in DATA_AIDED_RUNS.items():
        if snr != 15.0 and not (source / f"{stem}.csv").is_file():
            continue  # the lower-SNR run is optional; the 15 dB one is required
        aided[snr] = copy("data_aided", source / f"{stem}.csv", "data_aided_")
        copy("data_aided", source / f"{stem}_paired.csv", "data_aided_")
    dmrs_detection = copy("dmrs", source / DMRS_DETECTION_CSV)
    dmrs_recovery = copy("dmrs", source / DMRS_RECOVERY_CSV)

    payload = {
        "pd": {"rows": _pd(_read(pd))},
        "margin": {"rows": _margin(_read(margin))},
        "recovery": {"rows": recovery, "paired": paired},
        "recovery_curves": {"rows": _recovery_curves(_read(curves))},
        "matched": {
            "rows": _matched(_read(matched)),
            "paired": _matched_paired(_read(matched_paired)),
        },
        "data_aided": {"rows": _data_aided(aided)},
        "dmrs": {
            "rows": _dmrs_detection(_read(dmrs_detection)),
            "recovery": _dmrs_recovery(_read(dmrs_recovery)),
        },
    }
    for kind, section in payload.items():
        if not section["rows"]:
            raise ValueError(f"No eligible {kind} rows in {', '.join(paths[kind])}")
        section.update(population=POPULATIONS[kind], downloads=paths[kind])
    payload["data_aided"]["summary"] = _data_aided_summary(payload["data_aided"]["rows"])
    (build / "data" / "results.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
