"""How much sampling uncertainty is there in the detector comparison?

Slots sharing an allocation are resampled together. The bootstrap draws simulation batches
with replacement within each (channel, attack, JSR) stratum. The 2.5th and 97.5th percentiles
give 95% intervals conditional on the validation thresholds and trained checkpoints. Every
detector and training seed uses the same draws, so differences are paired and seed means are
computed within each draw. Scheduled Pd pools channels after resampling within strata.

Older splits recover batches from consecutive runs of identical channel, attack, JSR,
allocation start and size, and scheduling-aware flag. This can merge adjacent simulation
batches that happen to draw the same allocation. PRB F1 gives credit for flagging the right
PRB anywhere in a slot. Its threshold is calibrated on scored negative validation PRBs, and
the reported false-alarm rate uses all negative PRBs in the evaluated attack group.

    python -m jamloc.experiments.intervals --split test --boot 1000
    python -m jamloc.experiments.intervals --split sched --pairs fused:no_residual fused:block_test
"""

import argparse
import csv

import numpy as np

from jamloc import evaluate
from jamloc.dataset import SPLITS, load
from jamloc.features import FEATURE_NAMES
from jamloc.jammer import ATTACKS
from jamloc.metrics import f1, threshold_at_fa
from jamloc.paths import results_dir

STRATUM_KEYS = ("channel", "attack", "jsr_db")
BATCH_KEYS = (*STRATUM_KEYS, "alloc_start", "alloc_size", "scheduling_aware")
DEFAULT_PAIRS = (
    ("fused", "no_residual"),
    ("fused", "block_test"),
    ("fused", "fused_pixel"),
    ("fused", "fused_pixel_wide"),  # spatial context at equal capacity
    ("fused_comb", "fused"),  # what the empty-comb input adds
    ("block_test_comb", "block_test"),
    ("no_residual", "power"),
)


def batch_ids(split: dict) -> np.ndarray:
    """Read the true batch IDs, or number consecutive runs of matching metadata."""
    if "batch" in split:
        return split["batch"]
    starts = np.zeros(len(split["attack"]), dtype=bool)
    starts[:1] = True
    for key in BATCH_KEYS:
        values = split[key]
        starts[1:] |= values[1:] != values[:-1]
    return np.cumsum(starts) - 1


class BatchBootstrap:
    """One set of stratified batch multiplicities shared by every metric and model."""

    def __init__(self, split: dict, n_boot: int):
        _, first_slot, self.slot_batch = np.unique(
            batch_ids(split), return_index=True, return_inverse=True
        )
        slot_strata = np.column_stack([split[key] for key in STRATUM_KEYS])
        batch_strata = slot_strata[first_slot]
        if not np.array_equal(slot_strata, batch_strata[self.slot_batch]):
            raise ValueError("Each simulation batch must belong to one channel/attack/JSR stratum.")
        self.strata, stratum = np.unique(batch_strata, axis=0, return_inverse=True)
        self.draws = []
        self.n_batches = len(first_slot)
        rng = np.random.default_rng(0)
        for i in range(len(self.strata)):
            batches = np.flatnonzero(stratum == i)
            n = len(batches)
            # The multinomial gives the counts of n uniform draws with replacement.
            weights = np.ones((n_boot + 1, n), dtype=np.int64)
            weights[1:] = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
            self.draws.append((batches, weights))

    def totals(self, slot_counts: np.ndarray) -> np.ndarray:
        """Sum counts into batches, then return [stratum, draw, count] totals.

        Draw zero uses every original batch once, giving the full-split point estimate.
        """
        batch_counts = np.zeros((self.n_batches, slot_counts.shape[1]), dtype=np.int64)
        np.add.at(batch_counts, self.slot_batch, slot_counts)
        return np.stack([weights @ batch_counts[batches] for batches, weights in self.draws])


def confusion_counts(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """TP, FP and FN per slot, for either cells or PRBs."""
    axes = tuple(range(1, labels.ndim))
    tp = (pred & labels).sum(axis=axes)
    return np.column_stack((tp, pred.sum(axis=axes) - tp, labels.sum(axis=axes) - tp))


def ratio(numerator, denominator, empty=np.nan):
    return np.divide(
        numerator, denominator, out=np.full(np.shape(denominator), empty), where=denominator != 0
    )


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """Vectorized F1 for bootstrap totals, with the same empty case as metrics.f1."""
    tp, fp, fn = counts.T
    return ratio(2 * tp, 2 * tp + fp + fn, empty=1.0)


def prb_scores(scores: np.ndarray) -> np.ndarray:
    """Max over scored cells in each (slot, PRB), or -inf if none are finite."""
    return np.max(scores, axis=-1, where=np.isfinite(scores), initial=-np.inf)


def validation_thresholds(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Calibrate the cell and PRB predictions separately on validation negatives."""
    cell = threshold_at_fa(scores[labels == 0], evaluate.FA)
    prb = threshold_at_fa(prb_scores(scores)[~labels.any(axis=-1)], evaluate.FA)
    return cell, prb


def detector_samples(scores, thresholds, labels, scheduled, attack, bootstrap):
    """Point estimates followed by shared bootstrap replicates for one detector and seed."""
    cell_threshold, prb_threshold = thresholds
    pred = scores > cell_threshold
    prb_pred = prb_scores(scores) > prb_threshold
    prb_labels = labels.any(axis=-1)
    counts = np.column_stack(
        (
            confusion_counts(pred, labels),
            (pred & scheduled).sum(axis=(1, 2)),
            scheduled.sum(axis=(1, 2)),
            confusion_counts(prb_pred, prb_labels),
            (~prb_labels).sum(axis=1),
        )
    )
    totals = bootstrap.totals(counts)
    _, stratum_attack, jsr = bootstrap.strata.T
    samples = {}
    for code, group in [(a, ATTACKS[a]) for a in (1, 2, 3)] + [(None, "all")]:
        all_slots = code is None
        use = slice(None) if all_slots else stratum_attack == code
        if not all_slots and not use.any():
            continue
        slots = slice(None) if all_slots else attack == code
        cell, _, prb, neg = np.split(totals[use].sum(axis=0), (3, 5, 8), axis=1)
        cell_f1, prb_f1 = f1_from_counts(cell), f1_from_counts(prb)
        cell_f1[0] = f1(pred[slots], labels[slots])
        prb_f1[0] = f1(prb_pred[slots], prb_labels[slots])
        samples["f1", group, None] = cell_f1
        samples["prb_f1", group, None] = prb_f1
        samples["prb_fa", group, None] = ratio(prb[:, 1], neg[:, 0])
        if not all_slots:
            for db in np.unique(jsr[use]):
                hit, jammed = totals[use & (jsr == db), :, 3:5].sum(axis=0).T
                samples["pd", group, float(db)] = ratio(hit, jammed)
    return samples


def collect_samples(val, split, seeds, bootstrap):
    """Reduce one seed's scores before evaluating the next seed."""
    val_labels = val["labels"].astype(bool)
    labels = split["labels"].astype(bool)
    alloc = split["features"][:, FEATURE_NAMES.index("alloc_mask")] > 0
    scheduled = labels & alloc
    results = {}
    for seed in seeds:
        val_scores = evaluate.score_split(val, seed, null=val)
        thresholds = {k: validation_thresholds(v, val_labels) for k, v in val_scores.items()}
        del val_scores
        scores = evaluate.score_split(split, seed, null=val)
        for detector, limits in thresholds.items():
            samples = detector_samples(
                scores.pop(detector), limits, labels, scheduled, split["attack"], bootstrap
            )
            results.setdefault(detector, {})[seed] = samples
    return results


def interval(name, samples):
    """Summarize the point estimate and the percentiles of defined bootstrap draws."""
    draws = samples[1:][np.isfinite(samples[1:])]
    lo, hi = np.percentile(draws, (2.5, 97.5)) if draws.size else (np.nan, np.nan)
    return {name: samples[0], f"{name}_lo": lo, f"{name}_hi": hi}


def with_seed_mean(by_seed):
    """Yield each seed and the mean, averaging statistics within each replicate."""
    for seed, samples in by_seed.items():
        yield seed, 1, samples
    mean = {
        key: np.mean([samples[key] for samples in by_seed.values()], axis=0)
        for key in next(iter(by_seed.values()))
    }
    yield "mean", len(by_seed), mean


def metric_rows(results):
    f1_rows, pd_rows, prb_rows = [], [], []
    for detector, by_seed in results.items():
        for seed, n_seeds, samples in with_seed_mean(by_seed):
            for (metric, attack, jsr_db), draws in samples.items():
                row = {"detector": detector, "attack": attack, "seed": seed, "n_seeds": n_seeds}
                if metric == "f1":
                    f1_rows.append({**row, **interval("f1", draws)})
                elif metric == "pd":
                    pd_rows.append({**row, "jsr_db": jsr_db, **interval("pd", draws)})
                elif metric == "prb_f1":
                    fa_draws = samples["prb_fa", attack, None]
                    prb_rows.append({**row, **interval("f1", draws), **interval("fa", fa_draws)})
    return f1_rows, pd_rows, prb_rows


def paired_rows(results, pairs):
    """First detector minus second, using only seeds with both checkpoints."""
    rows = []
    for first, second in pairs:
        a, b = results.get(first, {}), results.get(second, {})
        common = sorted(a.keys() & b.keys())
        if not common:
            continue
        keys = a[common[0]]
        lowest = {}
        for metric, attack, jsr in keys:
            if metric == "pd":
                lowest[attack] = min(jsr, lowest.get(attack, jsr))
        compare = [k for k in keys if k[0] == "f1" or (k[0] == "pd" and k[2] == lowest[k[1]])]
        deltas = {seed: {k: a[seed][k] - b[seed][k] for k in compare} for seed in common}
        for seed, n_seeds, samples in with_seed_mean(deltas):
            for (metric, attack, jsr_db), draws in samples.items():
                rows.append(
                    {
                        "first": first,
                        "second": second,
                        "metric": metric,
                        "attack": attack,
                        "jsr_db": jsr_db,
                        "seed": seed,
                        "n_seeds": n_seeds,
                        **interval("delta", draws),
                    }
                )
    return rows


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pair(value):
    parts = value.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("use FIRST:SECOND for each detector pair")
    return tuple(parts)


def _print_summary(split, seeds, n_boot, bootstrap, f1_rows, prb_rows):
    print(f"{split} | seeds {seeds} | {bootstrap.n_batches} batches | {n_boot} bootstrap draws")
    print(f"{'Detector':20s} {'Cell F1 [95% CI]':25s} {'PRB F1 [95% CI]':25s}")
    for cell, prb in zip(f1_rows, prb_rows, strict=True):
        if cell["seed"] == "mean" and cell["attack"] == "all":
            print(
                f"{cell['detector']:20s} "
                f"{cell['f1']:.4f} [{cell['f1_lo']:.4f}, {cell['f1_hi']:.4f}]   "
                f"{prb['f1']:.4f} [{prb['f1_lo']:.4f}, {prb['f1_hi']:.4f}]"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    splits = [s for s in SPLITS if not s.startswith(("train", "val"))]
    parser.add_argument("--split", default="test", choices=splits)
    parser.add_argument("--boot", type=int, default=1000, help="number of bootstrap draws")
    parser.add_argument(
        "--pairs",
        type=_pair,
        nargs="+",
        metavar="FIRST:SECOND",
        help="detector pairs to compare, with the second subtracted from the first",
    )
    args = parser.parse_args(argv)
    if args.boot < 1:
        parser.error("--boot must be positive")

    seeds = evaluate.available_seeds()
    if not seeds:
        raise SystemExit("No complete training seeds found in results/models.")
    val, split = load("val"), load(args.split)
    bootstrap = BatchBootstrap(split, args.boot)
    results = collect_samples(val, split, seeds, bootstrap)
    for first, second in args.pairs or ():
        if not (results.get(first, {}).keys() & results.get(second, {}).keys()):
            parser.error(f"no shared training seed for {first}:{second}")
    f1_rows, pd_rows, prb_rows = metric_rows(results)
    paired = paired_rows(results, args.pairs or DEFAULT_PAIRS)

    out = results_dir() / "intervals"
    out.mkdir(parents=True, exist_ok=True)
    common = ("detector", "attack", "seed", "n_seeds")
    f1_fields = ("f1", "f1_lo", "f1_hi")
    _write_csv(out / f"{args.split}_f1_ci.csv", f1_rows, (*common, *f1_fields))
    _write_csv(
        out / f"{args.split}_pd_ci.csv",
        pd_rows,
        (*common, "jsr_db", "pd", "pd_lo", "pd_hi"),
    )
    _write_csv(
        out / f"{args.split}_paired.csv",
        paired,
        (
            "first",
            "second",
            "metric",
            "attack",
            "jsr_db",
            "seed",
            "n_seeds",
            "delta",
            "delta_lo",
            "delta_hi",
        ),
    )
    _write_csv(
        out / f"{args.split}_prb_f1.csv",
        prb_rows,
        (*common, *f1_fields, "fa", "fa_lo", "fa_hi"),
    )
    _print_summary(args.split, seeds, args.boot, bootstrap, f1_rows, prb_rows)


if __name__ == "__main__":
    main()
