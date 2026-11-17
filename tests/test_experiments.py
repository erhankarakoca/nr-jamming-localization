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
