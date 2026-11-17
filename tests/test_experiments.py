import csv

from jamloc.experiments import ber_bler


def test_ber_bler_script_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    ber_bler.main(
        ["--channel", "A", "--batch-size", "8", "--max-mc-iter", "1", "--jsr-db", "-10", "5"]
    )
    out = tmp_path / "ber_bler"
    assert (out / "ber_bler_A.png").is_file()
    assert (out / "ber_bler_A.pdf").is_file()
    with open(out / "ber_bler_A.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    attacks = {r["attack"] for r in rows}
    assert attacks == {"none", "pilot", "data", "partial"}
    assert len(rows) == 1 + 3 * 2  # benign reference + 3 attacks x 2 JSR points
    assert {"jsr_db", "energy_db", "ber", "bler"} <= set(rows[0])

    # Redraw from the saved CSV without simulating again
    (out / "ber_bler_A.png").unlink()
    ber_bler.main(["--channel", "A", "--plot-only"])
    assert (out / "ber_bler_A.png").is_file()


def test_feature_maps_script_runs(tmp_path, monkeypatch):
    from jamloc.experiments import feature_maps

    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    feature_maps.main(["--snr-db", "15", "--jsr-db", "0"])
    assert (tmp_path / "feature_maps.png").is_file()
    assert (tmp_path / "feature_maps.pdf").is_file()


def test_latency_script_runs(tmp_path, monkeypatch):
    from jamloc.experiments import latency
    from jamloc.model import VARIANTS

    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    latency.main(["--reps", "5"])
    with open(tmp_path / "latency.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["variant"] for r in rows] == list(VARIANTS)
    for r in rows:
        assert float(r["feature_ms"]) > 0 and float(r["model_ms"]) > 0
        assert abs(float(r["total_ms"]) - float(r["feature_ms"]) - float(r["model_ms"])) < 1e-6
        assert 10_000 <= int(r["params"]) <= 100_000
        assert int(r["threads"]) == 1  # one CPU core by default


def test_ber_bler_title_uses_snr_from_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    args = ["--channel", "A", "--snr-db", "12", "--batch-size", "8", "--max-mc-iter", "1"]
    ber_bler.main([*args, "--jsr-db", "0"])
    rows = ber_bler.read_rows(tmp_path / "ber_bler" / "ber_bler_A.csv")
    assert all(r["snr_db"] == 12.0 for r in rows)
    assert "SNR 12 dB" in ber_bler.plot(rows, "A").axes[0].get_title()


def test_full_run_small(tmp_path, monkeypatch):
    import zipfile

    from jamloc.experiments import full_run

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    full_run.main(["--small", "--seeds", "0", "1"])
    archive = tmp_path / "results" / "full_run.zip"
    names = zipfile.ZipFile(archive).namelist()
    assert {"eval/test_f1.csv", "eval/sched_f1.csv", "eval/test_pd_vs_jsr_scheduled.pdf"} <= set(
        names
    )
    assert {"models/fused_seed0.pt", "models/fused_seed1.pt"} <= set(names)
    assert {
        "models/fused_seed0_history.csv",
        "recovery/recovery_A.csv",
        "recovery/recovery_C.csv",
        "figures/index.html",
        "figures/detector_outputs.png",
    } <= set(names)


def test_recovery_script_runs(tmp_path, monkeypatch):
    from jamloc.dataset import generate
    from jamloc.experiments import recovery
    from jamloc.train import train

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)
    for variant in ("fused", "no_residual"):
        train(variant, epochs=1, batch_size=32)

    args = ["--channel", "A", "--batch-size", "8", "--max-mc-iter", "1", "--jsr-db", "-3", "3"]
    recovery.main(args)
    out = tmp_path / "results" / "recovery"
    with open(out / "recovery_A.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    main_rows = {
        (r["receiver"], r["attack"], r["jsr_db"])
        for r in rows
        if r["attack"] in ("pilot", "data", "partial")
    }
    assert main_rows == {
        (rx, a, j)
        for rx in recovery.RECEIVERS
        for a in ("pilot", "data", "partial")
        for j in ("-3.0", "3.0")
    }
    limit = [r for r in rows if r["attack"] == "pilot_whole_allocation"]
    assert {r["receiver"] for r in limit} == {"standard", "true_map"}
    assert (out / "recovery_A.png").is_file() and (out / "recovery_A.pdf").is_file()
