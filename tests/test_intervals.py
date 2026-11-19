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


def test_matched_predictions_spend_whole_cell_budget_through_ties():
    scores = np.array([9, 8, 8, 8, *([0] * 197), *([-np.inf] * 100), 10, 8, -np.inf])
    scored_neg = scores > -np.inf
    scored_neg[-3:] = False

    pred = intervals.matched_predictions(scores, scored_neg)

    assert scored_neg.sum() == 201
    assert (pred & scored_neg).sum() == 2
    assert pred[0] and pred[-3]
    assert not pred[scores == -np.inf].any()
    np.testing.assert_array_equal(pred, intervals.matched_predictions(scores, scored_neg))


def test_matched_variants_hold_predictions_fixed_and_pool_weakest_scheduled_pd():
    split = _grid([0, 1, 1, 1, 2, 3])
    split["jsr_db"] = np.array([0, -15, -15, -9, -12, -9])
    split["batch"] = np.arange(6)
    scores = np.empty((6, 1, 102))
    scores[:, 0, :100] = np.arange(100) + np.array([0, 100, 105, 101, 200, 300])[:, None]
    scores[:, 0, 100] = [-np.inf, 205, 198, 450, 300, 390]
    scores[:, 0, 101] = [-np.inf, 450, 450, 450, 450, 450]
    labels = np.zeros_like(scores, dtype=bool)
    labels[1:, 0, 100:] = True
    scheduled = labels.copy()
    scheduled[:, 0, 101] = False

    bootstrap = intervals.BatchBootstrap(split, 50)
    pooled, by_attack = intervals.matched_detector_samples(
        scores, labels, scheduled, split["attack"], bootstrap
    )
    assert pooled["fa_scored", "all", None][0] == 0.01
    assert pooled["fa_scored", "pilot", None][0] == 0
    assert pooled["fa_scored", "partial", None][0] == 0.06
    for group in ("all", "pilot", "data", "partial"):
        assert by_attack["fa_scored", group, None][0] == 0.01
    assert set(by_attack["fa_scored", "pilot", None][1:]) == {0, 0.01, 0.02}
    assert set(by_attack["pd", "pilot", -15.0]) == {0, 0.5, 1.0}
    assert by_attack["pd", "all", None][0] == 0.5
    assert ("pd", "pilot", -9.0) not in by_attack
    assert by_attack["f1", "all", None][0] == 16 / 24
    paired = intervals.paired_rows({"a": {0: by_attack}}, [("a", "a")])
    assert any(row["metric"] == "pd" and row["attack"] == "all" for row in paired)


def test_matched_groups_without_scored_negatives_use_split_wide_predictions():
    split = _grid([1, 2])
    scores = np.full((2, 1, 101), -np.inf)
    scores[0, 0, -1] = 100
    scores[1, 0, :100] = np.arange(100)
    labels = np.zeros_like(scores, dtype=bool)
    labels[:, 0, -1] = True

    bootstrap = intervals.BatchBootstrap(split, 5)
    pooled, by_attack = intervals.matched_detector_samples(
        scores, labels, labels, split["attack"], bootstrap
    )
    for samples in (pooled, by_attack):
        assert np.isnan(samples["fa_scored", "pilot", None]).all()
        assert (samples["pd", "pilot", -15.0] == 1).all()


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
