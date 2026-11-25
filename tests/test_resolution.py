import csv

import numpy as np

from jamloc.experiments import resolution

FIELDS = ["receiver", "attack", "jsr_db", "block_errors", "blocks"]


def _run(folder, seed, rows, slots):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"recovery_C_hopping_seed{seed}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rx, attack, jsr, errors, blocks in rows:
            writer.writerow(dict(zip(FIELDS, (rx, attack, jsr, errors, blocks), strict=True)))
    np.savez_compressed(folder / f"recovery_C_hopping_seed{seed}_slots.npz", **slots)


def test_merge_joins_runs_and_checks_shared_receivers(tmp_path):
    fail = np.array([True, False, False, False])
    _run(
        tmp_path / "old",
        0,
        [("p_r_e_soft", "data", "9.0", 2, 4), ("lmmse", "data", "9.0", 1, 4)],
        {"data;9;p_r_e_soft": fail | ~fail, "data;9;lmmse": fail},
    )
    _run(
        tmp_path / "new",
        0,
        [("nomap_shrink", "data", "9.0", 0, 4), ("lmmse", "data", "9.0", 1, 4)],
        {"data;9;nomap_shrink": ~(fail | ~fail), "data;9;lmmse": fail},
    )
    out = tmp_path / "merged"
    assert resolution.merge(out, [tmp_path / "old", tmp_path / "new"]) == []
    counts = resolution.pooled(out, "recovery_C_hopping")
    assert counts[("nomap_shrink", "data", "9.0")] == [0, 4]
    assert counts[("p_r_e_soft", "data", "9.0")] == [2, 4]
    assert set(np.load(out / "recovery_C_hopping_seed0_slots.npz")) == {
        "data;9;p_r_e_soft",
        "data;9;lmmse",
        "data;9;nomap_shrink",
    }

    # A shared receiver with other counts means the runs decoded other slots
    _run(tmp_path / "other", 0, [("lmmse", "data", "9.0", 3, 4)], {"data;9;lmmse": ~fail})
    problems = resolution.merge(tmp_path / "merged2", [tmp_path / "old", tmp_path / "other"])
    assert len(problems) == 1 and "lmmse" in problems[0]


def test_table_takes_the_best_scale_and_keeps_ties_unscaled(tmp_path):
    _run(
        tmp_path,
        0,
        [
            ("nomap_prb", "data", "9.0", 50, 100),
            ("nomap_prb2", "data", "9.0", 40, 100),
            ("nomap_prb4", "data", "9.0", 45, 100),
            ("p_r_e_soft", "data", "9.0", 0, 100),
            ("p_r_e_prb2", "data", "9.0", 0, 100),
            ("p_r_e_prb4", "data", "9.0", 1, 100),
        ],
        {},
    )
    cases = [("recovery_C_hopping", "data", "9.0", "hopping data")]
    rows = {(r["map"], r["estimate"]): r["hopping data"] for r in resolution.table(tmp_path, cases)}
    assert rows[("none", "per PRB")][0] == 50
    assert rows[("none", "per PRB, best scale")][0] == 40
    assert rows[("none", "per PRB, best scale")][-1] == "nomap_prb2"
    assert rows[("U-Net", "per PRB, best scale")][-1] == "p_r_e_soft"  # a tie stays unscaled
    assert rows[("true", "shrinkage")] is None  # not run
    text = resolution.markdown(resolution.table(tmp_path, cases), cases)
    assert "40.00 (x2)" in text and "| U-Net | per PRB, best scale | 0.00 |" in text


def test_sweep_uses_the_names_of_the_paper():
    from jamloc.experiments.resolution import SWEEP_PANELS, SWEEP_RECEIVERS

    labels = [style[0] for style in SWEEP_RECEIVERS.values()]
    assert labels[0] == "Baseline receiver" and "No map, PRB average" in labels
    assert [panel[2] for panel in SWEEP_PANELS] == [
        "(a) Pilot only",
        "(b) Pilot + data",
        "(c) Hopping, data only",
        "(d) Hopping, pilot + data",
    ]


def test_sweep_axis_shows_the_lowest_measured_point(tmp_path, monkeypatch):
    # One curve reaches 3 errors in 100,000 blocks, below the old fixed floor of 1e-4
    rows = [
        ("nomap_shrink", "data", "3.0", 50, 1000),
        ("nomap_shrink", "data", "9.0", 3, 100_000),
    ]
    _run(tmp_path / "run" / "recovery", 0, rows, {})
    saved = {}
    monkeypatch.setattr(resolution, "save", lambda fig, out: saved.setdefault("fig", fig))
    resolution.sweep([tmp_path / "run" / "recovery"], tmp_path / "sweep.pdf")
    low, high = saved["fig"].axes[0].get_ylim()
    assert low <= 3e-5 and high >= 0.05


def test_sweep_with_16qam_groups_its_panels_on_one_axis(tmp_path, monkeypatch):
    # QPSK hopping and 16-QAM hopping runs: five panels in two labelled groups, one y axis
    rows = [("nomap_shrink", "partial", "9.0", 30, 1000), ("standard", "data", "9.0", 500, 1000)]
    _run(tmp_path / "qpsk" / "recovery", 0, rows, {})
    qam = tmp_path / "qam" / "recovery"
    _run(qam, 0, [("p_r_e_shrink", "partial", "3.0", 2, 10_000)], {})
    (qam / "recovery_C_hopping_seed0.csv").rename(qam / "recovery_C_hopping_snr18_mcs13_seed0.csv")
    saved = {}
    monkeypatch.setattr(resolution, "save", lambda fig, out: saved.setdefault("fig", fig))
    resolution.sweep(
        [tmp_path / "qpsk" / "recovery", qam],
        tmp_path / "sweep",
        resolution.SWEEP_PANELS_QAM,
        resolution.SWEEP_GROUPS_QAM,
    )
    fig = saved["fig"]
    axes = [ax for sub in fig.subfigs for ax in sub.axes]
    assert len(axes) == 5
    assert [sub._suptitle.get_text() for sub in fig.subfigs] == [
        "QPSK at 15 dB SNR",
        "16-QAM at 18 dB SNR",
    ]
    assert len({ax.get_ylim() for ax in axes}) == 1 and axes[0].get_ylim()[0] <= 2e-4
    assert axes[4].get_lines()  # the 16-QAM hopping curve is drawn in the last panel
