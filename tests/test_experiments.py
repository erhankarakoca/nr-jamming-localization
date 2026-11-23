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
    assert {"jsr_db", "energy_db", "ber", "bler", "block_errors", "blocks"} <= set(rows[0])
    assert all(int(r["blocks"]) % 8 == 0 and int(r["blocks"]) > 0 for r in rows)

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
    rows = ber_bler.read_rows(tmp_path / "ber_bler" / "ber_bler_A_snr12.csv")
    assert all(r["snr_db"] == 12.0 for r in rows)
    assert "SNR 12 dB" in ber_bler.plot(rows, "A").axes[0].get_title()


def test_full_run_small(tmp_path, monkeypatch):
    import zipfile

    from jamloc.experiments import full_run

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    full_run.main(["--small", "--seeds", "1", "2"])
    archive = tmp_path / "results" / "full_run.zip"
    names = zipfile.ZipFile(archive).namelist()
    assert {"eval/test_f1.csv", "eval/sched_f1.csv", "eval/test_pd_vs_jsr_scheduled.pdf"} <= set(
        names
    )
    assert {"models/fused_seed1.pt", "models/fused_seed2.pt"} <= set(names)
    assert {
        "models/fused_seed1_history.csv",
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
        if rx != "p_r_e_atk_soft"  # its model (trained on mixed attacks) is not trained here
        for a in ("pilot", "data", "partial")
        for j in ("-3.0", "3.0")
    }
    benign = {r["receiver"] for r in rows if r["attack"] == "none"}
    assert benign == set(recovery.RECEIVERS) - {"p_r_e_atk_soft"}
    limit = [r for r in rows if r["attack"] == "pilot_whole_allocation"]
    assert {r["receiver"] for r in limit} == {
        "standard",
        "lmmse",
        "p_r_e_soft",
        "true_hard",
        "true_soft",
        "cell_soft",
        "p_r_e_cell",
        *recovery.DATA_AIDED,
    }
    assert (out / "recovery_A.png").is_file() and (out / "recovery_A.pdf").is_file()
    assert (out / "recovery_A_ablation.png").is_file()
    assert (out / "recovery_A_sources.png").is_file()  # receiver fixed, map source varied

    # Redraw from the saved CSV without simulating again
    (out / "recovery_A.png").unlink()
    recovery.main(["--channel", "A", "--plot-only"])
    assert (out / "recovery_A.png").is_file()
    assert all(int(r["block_errors"]) <= int(r["blocks"]) for r in rows)

    # Re-run one receiver: its rows are replaced, all others kept
    recovery.main([*args, "--receivers", "power_hard"])
    with open(out / "recovery_A.csv", encoding="utf-8", newline="") as f:
        merged = list(csv.DictReader(f))
    assert len(merged) == len(rows)
    assert {r["receiver"] for r in merged} == {r["receiver"] for r in rows}

    # A pulsed jammer, two detector seeds: only data and partial-band attacks, pooled figures
    train("fused", epochs=1, batch_size=32, seed=1)
    train("no_residual", epochs=1, batch_size=32, seed=1)
    recovery.main([*args[:-2], "3", "--jammer", "pulsed", "--seeds", "0", "1"])
    with open(out / "recovery_A_pulsed.csv", encoding="utf-8", newline="") as f:
        pulsed = list(csv.DictReader(f))
    assert {r["attack"] for r in pulsed} == {"data", "partial"}
    assert {r["seed"] for r in pulsed} == {"0", "1"} and {r["jammer"] for r in pulsed} == {"pulsed"}
    pooled = recovery.pool_seeds(recovery._read_rows(out / "recovery_A_pulsed.csv"))
    assert len(pooled) == len(pulsed) // 2
    assert (out / "recovery_A_pulsed_sources.png").is_file()

    # Per-slot outcomes: one array per receiver and point, consistent with the counts; paired
    # differences for every pair of receivers decoding the same point
    import numpy as np

    slots = dict(np.load(out / "recovery_A_pulsed_seed1_slots.npz"))
    for r in (r for r in pulsed if r["seed"] == "1"):
        failed = slots[recovery.slot_key(r["attack"], float(r["jsr_db"]), r["receiver"])]
        assert len(failed) == int(r["blocks"]) and failed.sum() == int(r["block_errors"])
    with open(out / "recovery_A_pulsed_paired.csv", encoding="utf-8", newline="") as f:
        paired = list(csv.DictReader(f))
    n_rx = len({r["receiver"] for r in pulsed})
    assert len(paired) == 2 * n_rx * (n_rx - 1) // 2  # two points (data, partial at 3 dB)
    for r in paired:
        assert float(r["delta_lo"]) <= float(r["delta"]) <= float(r["delta_hi"])
        assert int(r["slots"]) == 2 * 8  # both seeds, one batch of 8 slots each

    # A replay of the nominal run decodes only the quoted points, on the same slots
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (out / "recovery_A_seed0.csv").rename(earlier / "recovery_A_seed0.csv")
    recovery.main([*args, "--replay", str(earlier)])
    replayed = recovery._read_rows(out / "recovery_A_replay_seed0.csv")
    assert {(r["attack"], r["jsr_db"]) for r in replayed} == {
        ("none", r["jsr_db"]) for r in replayed if r["attack"] == "none"
    } | {("partial", 3.0)}
    assert not recovery.check_replay(replayed, earlier / "recovery_A_seed0.csv")

    # The noise power estimated per slot: only the detectors trained or calibrated on it
    generate("train_nest", num_slots=64, batch_size=32)
    generate("val_nest", num_slots=32, batch_size=32)
    train("fused", epochs=1, batch_size=32, data="nest")
    recovery.main([*args[:-2], "3", "--noise-estimate", "comb"])
    nest = recovery._read_rows(out / "recovery_A_nest.csv")
    assert {r["noise_estimate"] for r in nest} == {"comb"}
    assert {"p_r_e_soft", "block_soft", "true_soft"} <= {r["receiver"] for r in nest}
    assert not {"p_e_soft", "p_r_e_atk_soft"} & {r["receiver"] for r in nest}


def test_paper_run_plans_every_split_and_model(tmp_path, monkeypatch):
    from jamloc.experiments import paper_run

    splits = paper_run.all_splits()
    for name in ("train", "test", "unnorm", "train_atk2", "test_nest", "margin_snr8", "noise_m1"):
        assert name in splits
    assert len(splits) == len(set(splits))

    # Training: every checkpoint the later stages load, and nothing twice
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    trained = []
    monkeypatch.setattr(paper_run, "train", lambda v, **kw: trained.append((v, kw)))
    paper_run.stage_train([0, 1], epochs=1, small=True)
    names = {
        paper_run.checkpoint_path(
            v,
            kw["seed"],
            **{k: x for k, x in kw.items() if k in ("width", "data", "arch", "jitter", "hidden")},
        ).name
        for v, kw in trained
    }
    assert len(names) == len(trained)
    for name in (
        "fused_seed1.pt",
        "fused_comb_seed0.pt",
        "fused_pixel_h252_seed1.pt",
        "fused_atk2_seed0.pt",
        "fused_nest_seed1.pt",
        "evm_mix_seed0.pt",
        "fused_w4_seed1.pt",
        "fused_nj1_seed0.pt",
        "fused_pixel_seed0.pt",
    ):
        assert name in names
