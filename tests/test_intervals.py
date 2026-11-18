import numpy as np
import pytest

from jamloc.experiments import intervals
from jamloc.metrics import f1


def _grid(attack):
    n = len(attack)
    return {
        "channel": np.zeros(n, dtype=int),
        "attack": np.array(attack),
        "jsr_db": np.full(n, -15.0),
        "alloc_start": np.zeros(n, dtype=int),
        "alloc_size": np.ones(n, dtype=int),
        "scheduling_aware": np.zeros(n, dtype=bool),
    }


@pytest.mark.parametrize(
    "key", ["channel", "attack", "jsr_db", "alloc_start", "alloc_size", "scheduling_aware"]
)
def test_batch_fallback_starts_new_run_when_metadata_changes(key):
    split = _grid([1] * 5)
    split[key][2:4] = split[key][2:4] + 1
    np.testing.assert_array_equal(intervals.batch_ids(split), [0, 0, 1, 1, 2])


def test_explicit_batch_ids_take_precedence():
    split = {**_grid([1, 1, 1]), "batch": np.array([42, 42, 8])}
    np.testing.assert_array_equal(intervals.batch_ids(split), [42, 42, 8])


def test_per_batch_counts_reproduce_f1():
    pred = np.array([[1, 1], [0, 1], [0, 0], [1, 0]], dtype=bool)
    label = np.array([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=bool)
    split = {**_grid([1] * 4), "batch": np.array([12, 12, 24, 24])}
    bootstrap = intervals.BatchBootstrap(split, 4)
    totals = bootstrap.totals(intervals.confusion_counts(pred, label))
    np.testing.assert_array_equal(totals[0, 0], [3, 1, 2])
    assert intervals.f1_from_counts(totals[0, 0]) == f1(pred, label)


def test_bootstrap_of_identical_batches_has_zero_width():
    split = {**_grid([1, 1, 1, 1]), "batch": np.array([0, 0, 1, 1])}
    bootstrap = intervals.BatchBootstrap(split, 50)
    counts = np.array([[2, 1, 0], [1, 0, 2], [2, 1, 0], [1, 0, 2]])
    samples = intervals.f1_from_counts(bootstrap.totals(counts)[0])
    assert intervals.interval("f1", samples) == {"f1": 2 / 3, "f1_lo": 2 / 3, "f1_hi": 2 / 3}


def test_bootstrap_keeps_the_number_of_batches_in_each_stratum():
    split = {**_grid([1, 1, 2, 2]), "batch": np.arange(4)}
    bootstrap = intervals.BatchBootstrap(split, 50)
    totals = bootstrap.totals(np.array([[10], [11], [100], [110]]))
    assert set(totals[0, 1:, 0]) == {20, 21, 22}
    assert set(totals[1, 1:, 0]) == {200, 210, 220}


def test_prb_max_pooling_and_threshold():
    # 101 scored negatives make the higher 99th percentile the second-highest score.
    scores = np.full((1, 103, 3), -np.inf)
    scores[0, :101, 1] = np.arange(101)
    scores[0, 0, 2] = 10
    scores[0, 101] = [np.nan, -np.inf, np.inf]
    scores[0, 102, 2] = 200
    labels = np.zeros_like(scores, dtype=bool)
    labels[0, 102, 0] = True

    pooled = intervals.prb_scores(scores)
    assert pooled.shape == (1, 103)
    np.testing.assert_array_equal(pooled[0, [0, 101, 102]], [10, -np.inf, 200])
    threshold = intervals.validation_thresholds(scores, labels)[1]
    assert threshold == 99
    np.testing.assert_array_equal(np.flatnonzero(pooled > threshold), [100, 102])


def test_detector_samples_include_benign_slots_and_use_scheduled_cells_for_pd():
    split = _grid([0, 1, 1])
    split["channel"][2] = 1
    scores = np.array([[[1, 0], [0, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 0]]])
    labels = np.array([[[0, 0], [0, 0]], [[1, 0], [1, 0]], [[1, 0], [0, 0]]], dtype=bool)
    scheduled = labels.copy()
    scheduled[:, 1] = False
    bootstrap = intervals.BatchBootstrap(split, 10)
    samples = intervals.detector_samples(
        scores.astype(float), (0.5, 0.5), labels, scheduled, split["attack"], bootstrap
    )

    assert {group for _, group, _ in samples} == {"pilot", "all"}
    assert samples["f1", "pilot", None][0] == 1 / 3
    assert samples["f1", "all", None][0] == 2 / 7
    assert samples["prb_f1", "pilot", None][0] == 1.0
    assert samples["prb_f1", "all", None][0] == 6 / 7
    assert samples["prb_fa", "all", None][0] == 1 / 3
    np.testing.assert_array_equal(samples["pd", "pilot", -15.0], np.full(11, 0.5))


def test_seed_mean_averages_within_each_paired_replicate():
    key = ("f1", "all", None)
    results = {
        "first": {0: {key: np.array([0.5, 0.75, 0.25])}, 1: {key: np.array([0.5, 0.25, 0.75])}},
        "second": {0: {key: np.array([0.25, 0.0, 0.5])}, 1: {key: np.array([0.25, 0.5, 0.0])}},
    }
    rows = intervals.paired_rows(results, [("first", "second")])
    mean = next(row for row in rows if row["seed"] == "mean")
    assert mean["delta"] == mean["delta_lo"] == mean["delta_hi"] == 0.25


def test_reports_pair_with_itself_on_each_seed_and_the_seed_mean():
    results = {
        "detector": {
            0: {
                ("f1", "pilot", None): np.array([0.5, 0.3, 0.7]),
                ("pd", "pilot", -15.0): np.array([0.4, 0.5, 0.6]),
                ("pd", "pilot", -12.0): np.array([0.6, 0.7, 0.8]),
            },
            1: {
                ("f1", "pilot", None): np.array([0.2, 0.1, 0.3]),
                ("pd", "pilot", -15.0): np.array([0.1, 0.0, 0.2]),
                ("pd", "pilot", -12.0): np.array([0.3, 0.2, 0.5]),
            },
        },
    }
    rows = intervals.paired_rows(results, [("detector", "detector")])
    assert {(row["metric"], row["jsr_db"], row["seed"]) for row in rows} == {
        (metric, jsr_db, seed)
        for metric, jsr_db in (("f1", None), ("pd", -15.0))
        for seed in (0, 1, "mean")
    }
    assert all(row["delta"] == row["delta_lo"] == row["delta_hi"] == 0 for row in rows)
