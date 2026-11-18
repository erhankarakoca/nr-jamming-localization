import csv

import numpy as np

from jamloc.dataset import generate, load
from jamloc.evaluate import aggregate, evaluate_scores, main
from jamloc.model import VARIANTS
from jamloc.train import train


def test_oracle_detector(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    generate("val", num_slots=32, batch_size=32)
    generate("test", num_slots=60, batch_size=32)
    val, test = load("val"), load("test")
    oracle = {"oracle": (val["labels"].astype(float), test["labels"].astype(float))}

    pd_rows, f1_rows = evaluate_scores(oracle, val, test)
    assert {"detector", "attack", "channel", "jsr_db", "cells", "pd"} <= set(pd_rows[0])
    assert all(r["pd"] == 1.0 for r in pd_rows if not np.isnan(r["pd"]))
    assert {r["cells"] for r in pd_rows} == {"all", "scheduled"}
    assert all(r["f1"] == 1.0 for r in f1_rows)
    assert all(r["fa"] == 0.0 and r["fa_scored"] == 0.0 for r in f1_rows)


def test_aggregate_over_seeds():
    rows = [
        {"detector": "fused", "attack": "pilot", "seed": 0, "pd": 0.4},
        {"detector": "fused", "attack": "pilot", "seed": 1, "pd": 0.6},
        {"detector": "power", "attack": "pilot", "seed": 0, "pd": 0.3},
    ]
    out = {r["detector"]: r for r in aggregate(rows, "pd")}
    assert np.isclose(out["fused"]["pd_mean"], 0.5)
    assert np.isclose(out["fused"]["pd_std"], np.std([0.4, 0.6], ddof=1))
    assert out["fused"]["n_seeds"] == 2 and "seed" not in out["fused"]
    assert out["power"]["pd_std"] == 0.0 and out["power"]["n_seeds"] == 1


def test_main_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)
    generate("test", num_slots=60, batch_size=32)
    for variant in VARIANTS:
        for seed in (0, 1):
            train(variant, epochs=1, batch_size=32, seed=seed)
    for seed in (0, 1):
        train("fused", epochs=1, batch_size=32, seed=seed, arch="pixel")

    main(["--split", "test"])
    out = tmp_path / "results" / "eval"
    with open(out / "test_pd_vs_jsr.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["detector"] for r in rows} == {
        *VARIANTS,
        "evm_threshold",
        "residual_threshold",
        "residual_cfar",
        "power_threshold",
        "block_test",
        "fused_pixel",
    }
    assert {"pd_mean", "pd_std", "n_seeds"} <= set(rows[0])
    assert {r["n_seeds"] for r in rows} == {"2"}
    with open(out / "test_f1.csv", encoding="utf-8", newline="") as f:
        assert {"f1_mean", "f1_std", "fa_mean"} <= set(next(csv.DictReader(f)))
    assert (out / "test_pd_vs_jsr_per_seed.csv").is_file()
    assert (out / "test_pd_vs_jsr_all.png").is_file()
    assert (out / "test_pd_vs_jsr_scheduled.png").is_file()

    main(["--split", "test", "--snr-db", "5", "15"])
    assert (out / "test_snr5-15_pd_vs_jsr.csv").is_file()
