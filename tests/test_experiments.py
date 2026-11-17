import csv

from jamloc.experiments import ber_bler


def test_ber_bler_script_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    ber_bler.main(["--channel", "A", "--batch-size", "8", "--max-mc-iter", "1",
                   "--jsr-db", "-10", "5"])
    out = tmp_path / "ber_bler"
    assert (out / "ber_bler_A.png").is_file()
    assert (out / "ber_bler_A.pdf").is_file()
    with open(out / "ber_bler_A.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    attacks = {r["attack"] for r in rows}
    assert attacks == {"none", "pilot", "data", "partial"}
    assert len(rows) == 1 + 3 * 2  # benign reference + 3 attacks x 2 JSR points
    assert {"jsr_db", "energy_db", "ber", "bler"} <= set(rows[0])
