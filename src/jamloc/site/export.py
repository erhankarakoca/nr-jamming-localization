"""Export one buffered uplink slot per case. JSON schema, version 2.

manifest.json has schema_version, git_commit, created_at (UTC ISO 8601), and cases in
display order. Each entry has name, title, file (relative to the manifest) and size_bytes.
Each invocation lists the cases requested in that invocation.

Each case file has schema_version and the following fields. All indices start at zero.
Numbers have about four significant digits. Undefined values are JSON null.

* metadata: the Case fields (alloc and jam are [carrier start PRB, count], jam gives the
  nominal block before hopping or pulsing), config (SimConfig), device, git_commit,
  noise_variance and true_noise_variance for the slot. thresholds maps each detector to
  its validation split, model training data, target false-alarm rate and threshold value.
  batch_crc maps each receiver name to {"slots": n, "failures": k} for the full case
  batch, including the selected slot. A false receiver CRC counts as one failure.
  selection has rule, slot, hit_rate (fused U-Net over all jammed carrier cells) and
  batch_median. For median_hit_rate choose the hit rate nearest the arithmetic median,
  first index on a tie. If there are no jammed cells, both rates are null and slot zero
  is shown. first_crc_flip selects the first standard failure that fused_soft recovers,
  with flip_count and flip_frequency. first_true_soft_failure selects the first true_soft
  CRC failure, with failure_count and failure_frequency. Both frequencies are fractions
  of the full case batch.
  For paper_realization, block_min_relative_db is the weakest block PRB's benign power
  relative to the allocation mean. The original figure uses the first seed at or above −3 dB.
* grids: power_db, residual_db, evm_db from the detector inputs, allocation and truth.
  All grids are [carrier PRB][OFDM symbol], 24 by 14. Masks and decisions are 0/1.
* detectors: one entry per available detector with scores, decisions and unscored grids.
  Scores are raw logits for networks, linear ratios for feature tests and negative log10
  p-values for the block test. Unscored cells have null scores and decision 0. Decisions
  are made at full precision before rounding.
* prb: r_b and threshold (linear residual and analytic 1% threshold) in allocation order.
  pilots: subcarrier (relative to allocation start), ls, projection and residual_power
  (squared magnitude, linear). Each complex value is a [real, imaginary] pair.
* receivers: entries named standard or <map>_<mode>, each with crc (boolean) and keys
  x_hat and abs_llr into the two like-named tables at the top level. Each table stores
  a shared numerical array once. x_hat is a list of float16-rounded complex points.
  abs_llr gives the mean absolute LLR of the two QPSK bits per point. Both arrays run
  in symbol order, skipping the DMRS symbol, then in increasing allocation subcarrier
  order (12 per PRB). For point i, divmod(i, 12 * alloc_size) gives data-symbol ordinal
  and allocation subcarrier.
* fig3 and jsr_sweep also have benign (jsr_db null for zero power, pilots, r_b), from the
  same batch seed and slot index. jsr_sweep adds sweep, nine such frames with finite
  jsr_db values from -15 to +9. The selected index comes from its 0 dB batch.
"""

import json
import math
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from sionna.phy import config

from jamloc import theory
from jamloc.dataset import load
from jamloc.evaluate import FA
from jamloc.experiments.recovery import (
    NEST_DETECTORS,
    TRAINED_ON,
    coarsen_to_prb,
    detector_cells,
    val_threshold,
)
from jamloc.features import (
    FEATURE_NAMES,
    delay_subspace,
    dmrs_ls,
    extract,
    power_map,
    residual_per_prb,
)
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, SlotBatch, seed_everything
from jamloc.paths import REPO_ROOT, results_dir
from jamloc.recovery import MapAwareReceiver
from jamloc.site.cases import Case
from jamloc.train import load_model

SCHEMA_VERSION = 2


@contextmanager
def _cpu():
    """Choose the device before constructing any Sionna blocks or loading checkpoints."""
    previous = config.device
    config.device = "cpu"
    try:
        with torch.no_grad():
            yield
    finally:
        config.device = previous


def _load_detector(name, seed, noise_estimate):
    """Load the checkpoint and calibrate it on the split that uses the same inputs."""
    variant = name
    options, split = {}, "val"
    if noise_estimate == "comb":
        split = NEST_DETECTORS[name]
        options = {"data": "nest"}
    elif name in TRAINED_ON:
        variant = "fused"
        options, split = TRAINED_ON[name]
    elif name == "fused_atk2":
        variant, options, split = "fused", {"data": "atk2"}, "val_atk2"

    training_data = None
    if name == "block_test":
        model = load(split)
    elif name.endswith("_threshold"):
        model = None
    else:
        model = load_model(variant, seed, **options)
        training_data = "train" + ("_" + options["data"] if options else "")
    calibration = {
        "split": split,
        "training_data": training_data,
        "false_alarm_rate": FA,
        "value": val_threshold(variant, model, split),
    }
    return variant, model, calibration


def _batch(case, jsr_db, *, receive):
    """Reset before constructing the link so all power levels use the same random draws."""
    seed_everything(case.seed)
    n = case.batch_size
    sc = Scenario(
        channel=case.channel,
        alloc_start=case.alloc[0],
        alloc_size=case.alloc[1],
        attack=torch.full((n,), ATTACKS.index(case.attack)),
        jam_start=torch.full((n,), case.jam[0]),
        jam_size=torch.full((n,), case.jam[1]),
        jsr_db=torch.full((n,), float(jsr_db) if jsr_db is not None else -torch.inf),
        snr_db=torch.full((n,), float(case.snr_db)),
    )
    return JammedPUSCHLink(case.config())(sc, decode=receive, receive=receive)


def _select(case, truth, flags, crc):
    """Choose the representative, paper, or conditioned slot and keep its denominator."""
    counts = truth.sum((1, 2))
    rates = (truth & flags).sum((1, 2)).double() / counts
    ordered = rates[counts > 0].sort().values
    selection = {"rule": case.selection_rule, "batch_median": None}
    slot = 0
    if len(ordered):
        # Both central observations are equally close to the arithmetic median. Comparing
        # to their observed values keeps a rounding error from deciding a tie.
        lower, upper = ordered[(len(ordered) - 1) // 2], ordered[len(ordered) // 2]
        selection["batch_median"] = float((lower + upper) / 2)
        if case.selection_rule == "median_hit_rate":
            slot = int(((rates == lower) | (rates == upper)).nonzero()[0].item())
    if case.selection_rule in ("first_crc_flip", "first_true_soft_failure"):
        if case.selection_rule == "first_crc_flip":
            matches = ~crc["standard"] & crc["fused_soft"]
            field, outcome = "flip", "CRC flip"
        else:
            matches = ~crc["true_soft"]
            field, outcome = "failure", "true_soft CRC failure"
        indices = matches.nonzero().flatten()
        if not len(indices):
            raise ValueError(f"{case.name}: no {outcome} among {len(matches)} slots")
        slot = int(indices[0])
        selection[f"{field}_count"] = len(indices)
        selection[f"{field}_frequency"] = len(indices) / len(matches)
    selection.update(slot=slot, hit_rate=float(rates[slot]) if counts[slot] else None)
    return selection


@dataclass
class CaseRun:
    """Full precision results retained until the selected slot is serialized."""

    case: Case
    batch: SlotBatch
    maps: dict
    thresholds: dict
    receivers: dict
    selection: dict


def simulate_case(case: Case, *, load_detector: Callable = _load_detector) -> CaseRun:
    """Decode every map on the same received allocation and then select one slot."""
    with _cpu():
        batch = _batch(case, case.jsr_db, receive=True)
        cfg = case.config()
        maps, thresholds = {}, {}
        for name in case.detectors:
            variant, model, calibration = load_detector(
                name, case.detector_seed, case.noise_estimate
            )
            thresholds[name] = calibration
            maps[name] = detector_cells(
                batch, cfg, variant, model, calibration["value"], return_scores=True
            )

        receiver = MapAwareReceiver(batch.tx, cfg)
        alloc = slice(case.alloc[0], sum(case.alloc))
        truth = batch.labels[:, alloc]
        sources = {name: maps[name][1][:, alloc] for name in case.decode_maps}
        sources.update(true=truth, true_prb=coarsen_to_prb(truth, cfg.dmrs_symbol))
        receivers = {"standard": batch.rx}
        for name, cells in sources.items():
            for mode in ("soft", "hard") if name == "fused" else ("soft",):
                receivers[f"{name}_{mode}"] = receiver(batch.y_alloc, batch.no, cells, mode=mode)
        crc = {name: rx["tb_crc"].bool().flatten() for name, rx in receivers.items()}
        selection = _select(case, batch.labels, maps["fused"][1], crc)
        return CaseRun(case, batch, maps, thresholds, receivers, selection)


def _numbers(value):
    """Convert tensors to JSON numbers, preserving masks and mapping nonfinite values to null."""
    if isinstance(value, torch.Tensor):
        if value.is_complex():
            value = torch.view_as_real(value)
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_numbers(x) for x in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = float(f"{value:.4g}")
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, dict):
        return {k: _numbers(v) for k, v in value.items()}
    return value


def _pilot_frame(batch, cfg, slot, jsr_db):
    """Keep unclipped residuals for the figure and animation, including clean neighbors."""
    h, sc, power = dmrs_ls(batch.y_alloc[slot : slot + 1], batch.tx)
    u = delay_subspace(sc, cfg.scs, cfg.tau_max).to(h.dtype)
    projection = (h @ u.conj()) @ u.T
    r_b = residual_per_prb(h, sc, batch.no[slot : slot + 1], power, cfg)
    return _numbers(
        {
            "jsr_db": jsr_db,
            "pilots": {
                "subcarrier": sc,
                "ls": h[0],
                "projection": projection[0],
                "residual_power": (h - projection)[0].abs().square(),
            },
            "r_b": r_b[0],
        }
    )


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _share(table, name, values):
    """Exactly equal displayed arrays can be referenced by several receivers."""
    for key, previous in table.items():
        if previous == values:
            return key
    table[name] = values
    return name


def payload(run: CaseRun, *, git_commit: str | None = None) -> dict:
    """Build the versioned JSON document for the selected slot."""
    case, batch = run.case, run.batch
    cfg, i = case.config(), run.selection["slot"]
    features = dict(zip(FEATURE_NAMES, extract(batch, cfg)[i], strict=True))
    grids = {
        "power_db": features["power_db"],
        "residual_db": features["residual_db"].masked_fill(features["dmrs_mask"] == 0, torch.nan),
        "evm_db": features["evm_db"].masked_fill(features["data_mask"] == 0, torch.nan),
        "allocation": features["alloc_mask"].int(),
        "truth": batch.labels[i].int(),
    }
    detectors = {
        name: {
            "scores": scores[i],
            "decisions": flags[i].int(),
            "unscored": (~scores[i].isfinite()).int(),
        }
        for name, (scores, flags) in run.maps.items()
    }
    receivers, points, reliability, batch_crc = {}, {}, {}, {}
    for name, rx in run.receivers.items():
        crc = rx["tb_crc"].bool().flatten()
        batch_crc[name] = {"slots": len(crc), "failures": int((~crc).sum())}
        xy = torch.view_as_real(rx["x_hat"][i]).half()
        llr = rx["llr"][i].reshape(len(xy), 2).abs().mean(-1)
        receivers[name] = {
            "crc": bool(crc[i]),
            "x_hat": _share(points, name, _numbers(xy)),
            "abs_llr": _share(reliability, name, _numbers(llr)),
        }
    anchor = _pilot_frame(batch, cfg, i, case.jsr_db)
    result = {
        "schema_version": SCHEMA_VERSION,
        "metadata": _numbers(
            {
                **asdict(case),
                "config": asdict(cfg),
                "device": str(batch.y.device),
                "git_commit": git_commit or _git_commit(),
                "selection": run.selection,
                "batch_crc": batch_crc,
                "thresholds": run.thresholds,
                "noise_variance": batch.no[i],
                "true_noise_variance": batch.no_true[i],
            }
        ),
        "grids": _numbers(grids),
        "detectors": _numbers(detectors),
        "prb": {
            "r_b": anchor["r_b"],
            "threshold": _numbers(
                [theory.threshold(case.alloc[1], pos, FA, cfg) for pos in range(case.alloc[1])]
            ),
        },
        "pilots": anchor["pilots"],
        "receivers": receivers,
        "x_hat": points,
        "abs_llr": reliability,
    }
    if case.name in ("fig3", "jsr_sweep"):
        with _cpu():
            benign = _batch(case, None, receive=False)
            result["benign"] = _pilot_frame(benign, cfg, i, None)
            if case.selection_rule == "paper_realization":
                level = power_map(benign.y, benign.no, cfg)[i].mean(-1)
                weakest = level[case.jam[0] : sum(case.jam)].min()
                reference = level[case.alloc[0] : sum(case.alloc)].mean()
                result["metadata"]["selection"]["block_min_relative_db"] = _numbers(
                    10 * torch.log10(weakest / reference)
                )
            if case.name == "jsr_sweep":
                result["sweep"] = [
                    anchor
                    if db == case.jsr_db
                    else _pilot_frame(_batch(case, db, receive=False), cfg, i, db)
                    for db in range(-15, 10, 3)
                ]
    return result


def _write(path: Path, document: dict):
    """Publish complete UTF-8 documents so an interrupted write cannot truncate a case."""
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export(cases):
    """Calibrate each detector once per export and write the manifest after its cases."""
    directory = results_dir() / "site" / "data"
    directory.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    load_detector = cache(_load_detector)
    entries = []
    for case in cases:
        path = directory / f"{case.name}.json"
        document = payload(simulate_case(case, load_detector=load_detector), git_commit=commit)
        _write(path, document)
        size = path.stat().st_size
        entries.append(
            {"name": case.name, "title": case.title, "file": path.name, "size_bytes": size}
        )
        print(
            f"{case.name}: slot {document['metadata']['selection']['slot']}, {size} bytes",
            flush=True,
        )
    _write(
        directory / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "git_commit": commit,
            "created_at": datetime.now(UTC).isoformat(),
            "cases": entries,
        },
    )
