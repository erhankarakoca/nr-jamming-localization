import csv

import numpy as np

from jamloc.dataset import generate, load
from jamloc.evaluate import evaluate_scores, main
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
    assert all(r["fa"] == 0.0 for r in f1_rows)


def test_main_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)
    generate("test", num_slots=60, batch_size=32)
    for variant in VARIANTS:
        train(variant, epochs=1, batch_size=32)

    main(["--split", "test"])
    out = tmp_path / "results" / "eval"
    with open(out / "test_pd_vs_jsr.csv", encoding="utf-8", newline="") as f:
        detectors = {r["detector"] for r in csv.DictReader(f)}
    assert detectors == {*VARIANTS, "evm_threshold", "residual_threshold"}
    assert (out / "test_f1.csv").is_file()
    assert (out / "test_pd_vs_jsr_all.png").is_file()
    assert (out / "test_pd_vs_jsr_scheduled.png").is_file()
