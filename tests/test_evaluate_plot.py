import csv

from jamloc import evaluate


def test_plot_only_redraws_the_pd_figures_from_the_saved_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    out = tmp_path / "eval"
    out.mkdir()
    fields = ("detector", "attack", "channel", "jsr_db", "cells", "num_cells", "pd_mean", "pd_std")
    with open(out / "test_pd_vs_jsr.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for cells in ("all", "scheduled"):
            for attack in ("pilot", "data", "partial"):
                for jsr, pd in ((-15, 0.5), (0, 0.9)):
                    writer.writerow(("fused", attack, "both", jsr, cells, 100, pd, 0.01))
    evaluate.main(["--split", "test", "--plot-only"])
    for cells in ("all", "scheduled"):
        assert (out / f"test_pd_vs_jsr_{cells}.pdf").read_bytes().startswith(b"%PDF-")
