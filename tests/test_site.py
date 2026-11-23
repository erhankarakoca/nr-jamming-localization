"""Contracts for the cases exported to the showcase."""

import pytest


def test_case_definitions():
    from jamloc.config import SimConfig
    from jamloc.experiments.recovery import NEST_DETECTORS
    from jamloc.jammer import ATTACKS
    from jamloc.site.cases import CASES, DETECTORS

    assert len(CASES) == 13
    for name, case in CASES.items():
        assert name == case.name and case.title
        assert case.attack in ATTACKS
        assert isinstance(case.config(), SimConfig)
        assert set(case.decode_maps) <= set(case.detectors) <= set(DETECTORS)
    assert CASES["hopping"].config().jammer_hopping
    assert CASES["pulsed"].config().jammer_duty == 0.5
    assert CASES["spoof"].config().jammer_waveform == "dmrs"
    assert CASES["pilot_whole_comb"].config().noise_estimate == "comb"
    assert set(CASES["pilot_whole_comb"].detectors) == set(NEST_DETECTORS)
    aided = CASES["aided_flip"]
    assert aided.attack == "pilot" and aided.jam == aided.alloc == (6, 12)
    assert (aided.channel, aided.snr_db, aided.jsr_db, aided.noise_estimate) == ("C", 15, 9, "true")


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    from jamloc.dataset import generate
    from jamloc.train import train

    root = tmp_path_factory.mktemp("site")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("JAMLOC_DATA_DIR", str(root / "data"))
        patch.setenv("JAMLOC_RESULTS_DIR", str(root / "results"))
        generate("train", num_slots=32, batch_size=16)
        generate("val", num_slots=32, batch_size=16)
        for variant in ("fused", "no_residual", "power"):
            train(variant, epochs=1, batch_size=16)
        yield root


def test_same_received_grid_and_selection(tiny_models, monkeypatch):
    from dataclasses import replace

    import torch

    from jamloc.receiver import PUSCHReceiverChain
    from jamloc.recovery import MapAwareReceiver
    from jamloc.site.cases import CASES
    from jamloc.site.export import payload, simulate_case

    grids = []

    def record(method):
        def call(self, y, *args, **kwargs):
            grids.append(y.data_ptr())
            return method(self, y, *args, **kwargs)

        return call

    for cls in (PUSCHReceiverChain, MapAwareReceiver):
        monkeypatch.setattr(cls, "call", record(cls.call))
    run = simulate_case(replace(CASES["pilot"], batch_size=4, levels=()))
    assert len(grids) == 5 and set(grids) == {run.batch.y_alloc.data_ptr()}
    assert run.batch.y_alloc.device.type == "cpu"
    labels = run.batch.labels
    scores, flags = run.maps["fused"]
    assert torch.equal(flags, scores > run.thresholds["fused"]["value"])
    rates = (labels & flags).sum((1, 2)) / labels.sum((1, 2))
    median = torch.quantile(rates, 0.5)
    assert run.selection["slot"] == int((rates - median).abs().argmin())
    assert run.selection["batch_median"] == float(median)
    document = payload(run)
    channel = document["channel"]
    assert channel["dmrs_symbol"] == 2 and channel["num_sc"] == 144
    assert channel["pilot_sc"] == list(range(0, 144, 2))
    assert channel["pilot_jammed"] == [48 <= sc < 96 for sc in channel["pilot_sc"]]
    assert channel["ls"] == document["pilots"]["ls"]
    assert channel["fit"] == document["pilots"]["projection"]
    assert channel["residual_prb"] == document["prb"]["r_b"]
    assert channel["threshold_prb"] == document["prb"]["threshold"]
    for source in ("true", "standard", "map"):
        assert len(channel[source]) == 144 and all(len(row) == 2 for row in channel[source])
    counts = document["metadata"]["batch_crc"]
    assert counts.keys() == document["receivers"].keys() == run.receivers.keys()
    for name, rx in run.receivers.items():
        outcomes = rx["tb_crc"].flatten().tolist()
        assert counts[name] == {"slots": run.case.batch_size, "failures": outcomes.count(False)}
        assert document["receivers"][name]["crc"] == bool(outcomes[run.selection["slot"]])


def test_flip_selection():
    import torch

    from jamloc.site.cases import CASES
    from jamloc.site.export import _select

    truth = torch.ones((4, 1, 1), dtype=torch.bool)
    standard = torch.tensor([True, False, False, False])
    protected = torch.tensor([True, False, True, True])
    crc = {"standard": standard, "fused_soft": protected}
    choice = _select(CASES["partial_flip"], truth, truth, crc)
    assert choice["slot"] == 2
    assert choice["flip_count"] == 2 and choice["flip_frequency"] == 0.5
    with pytest.raises(ValueError, match="no CRC flip"):
        _select(CASES["partial_flip"], truth, truth, {**crc, "fused_soft": standard})


def test_aided_flip_selects_first_soft_crc_recovered_by_passes():
    import torch

    from jamloc.site.cases import CASES
    from jamloc.site.export import _select

    truth = torch.ones((4, 1, 1), dtype=torch.bool)
    plain = torch.tensor([False, True, False, False])
    aided = torch.tensor([False, True, True, True])
    choice = _select(
        CASES["aided_flip"], truth, truth, {"fused_soft": plain, "fused_soft_da": aided}
    )
    assert choice["rule"] == "first_aided_crc_flip"
    assert choice["slot"] == 2
    assert choice["flip_count"] == 2 and choice["flip_frequency"] == 0.5
    assert choice["seed"] == CASES["aided_flip"].seed


def test_aided_payload_records_passes(tiny_models, monkeypatch):
    from dataclasses import replace

    from jamloc.site import export as exporter
    from jamloc.site.cases import CASES

    definition = replace(CASES["aided_flip"], batch_size=16)
    tried = []
    select = exporter._select

    def without_first_flip(case, truth, flags, crc):
        tried.append(case.seed)
        if case.seed == definition.seed:
            # Suppress qualifying slots in the initial batch to exercise the next-seed path.
            crc = {**crc, "fused_soft_da": crc["fused_soft"]}
        return select(case, truth, flags, crc)

    monkeypatch.setattr(exporter, "_select", without_first_flip)
    run = exporter.simulate_case(definition)
    case = exporter.payload(run)
    selection = case["metadata"]["selection"]
    assert len(tried) >= 2 and tried == list(range(definition.seed, tried[-1] + 1))
    assert selection["seed"] == case["metadata"]["seed"] == tried[-1]
    assert case["channel"]["pilot_jammed"] == [True] * 72
    assert 2 <= len(case["passes"]) <= 4
    assert case["passes"][0]["h"] == case["channel"]["map"]
    assert case["passes"][0]["x_hat"] == case["receivers"]["fused_soft"]["x_hat"]
    assert case["passes"][0]["crc"] is case["receivers"]["fused_soft"]["crc"] is False
    assert case["receivers"]["fused_soft_da"]["crc"] is True
    assert any(step["crc"] for step in case["passes"][1:])
    for step in case["passes"]:
        assert len(step["h"]) == 144 and all(len(row) == 2 for row in step["h"])
        assert len(case["x_hat"][step["x_hat"]]) == len(case["abs_llr"][step["abs_llr"]]) == 1872


def test_pilot_whole_comb_selects_first_true_soft_failure():
    import torch

    from jamloc.site.cases import CASES
    from jamloc.site.export import _select

    case = CASES["pilot_whole_comb"]
    truth = torch.ones((4, 1, 1), dtype=torch.bool)
    passes = torch.ones(4, dtype=torch.bool)
    crc = {"standard": ~passes, "fused_soft": passes, "true_soft": passes.clone()}
    crc["true_soft"][[1, 3]] = False
    selection = _select(case, truth, truth, crc)
    assert selection["rule"] == "first_true_soft_failure"
    assert selection["slot"] == 1
    assert selection["failure_count"] == 2 and selection["failure_frequency"] == 0.5
    with pytest.raises(ValueError, match="no true_soft CRC failure"):
        _select(case, truth, truth, {**crc, "true_soft": passes})


def test_partial_export_keeps_other_cases(tmp_path):
    import json

    from jamloc.site.export import _merged_entries

    (tmp_path / "pilot.json").write_text("{}", encoding="utf-8")
    old = [
        {"name": "pilot", "title": "old", "file": "pilot.json"},
        {"name": "benign", "title": "old", "file": "benign.json"},  # its file is gone
        {"name": "retired", "title": "old", "file": "pilot.json"},  # no longer registered
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"cases": old}), encoding="utf-8")
    fresh = [{"name": "data", "title": "new", "file": "data.json"}]
    assert [e["name"] for e in _merged_entries(tmp_path, fresh)] == ["pilot", "data"]


@pytest.mark.parametrize("dirty", [False, True])
def test_cli_payload_and_manifest(tiny_models, monkeypatch, dirty):
    import json
    from dataclasses import replace

    import numpy as np

    from jamloc.site import export as exporter
    from jamloc.site.__main__ import main
    from jamloc.site.cases import CASES

    def git_output(args, *, cwd, **kwargs):
        assert cwd == exporter.REPO_ROOT
        if "rev-parse" in args:
            return "example-commit\n"
        assert {"status", "--porcelain", "--untracked-files=no"} <= set(args)
        return " M site/index.html\n" if dirty else ""

    monkeypatch.setattr(exporter.subprocess, "check_output", git_output)
    monkeypatch.setitem(CASES, "benign", replace(CASES["benign"], batch_size=2))
    main(["export", "--cases", "benign"])
    folder = tiny_models / "results" / "site" / "data"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    (entry,) = manifest["cases"]
    path = folder / entry["file"]
    assert entry["name"] == "benign" and entry["title"] == CASES["benign"].title
    assert entry["size_bytes"] == path.stat().st_size < 250_000
    case = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == case["schema_version"] == 2
    meta = case["metadata"]
    assert manifest["git_commit"] == meta["git_commit"] and manifest["created_at"]
    assert manifest["git_commit"] == "example-commit"
    assert manifest["git_dirty"] is meta["git_dirty"] is dirty
    assert meta["selection"]["slot"] == 0 and meta["selection"]["hit_rate"] is None
    assert np.shape(case["grids"]["truth"]) == (24, 14)
    assert case["grids"]["residual_db"][6][0] is None
    assert case["grids"]["evm_db"][6][2] is None
    assert len(case["prb"]["r_b"]) == len(case["prb"]["threshold"]) == 12
    assert np.shape(case["pilots"]["ls"]) == np.shape(case["pilots"]["projection"]) == (72, 2)
    assert set(case["detectors"]) == set(CASES["benign"].detectors)
    for layer in case["detectors"].values():
        for key in ("scores", "decisions", "unscored"):
            assert np.shape(layer[key]) == (24, 14)
    evm = case["detectors"]["evm_threshold"]
    assert evm["scores"][6][2] is None and evm["unscored"][6][2] == 1
    assert evm["unscored"][6][0] == 0
    assert set(case["receivers"]) == {
        "standard",
        "fused_soft",
        "fused_hard",
        "true_soft",
        "true_prb_soft",
    }
    assert meta["batch_crc"].keys() == case["receivers"].keys()
    for name, rx in case["receivers"].items():
        assert np.shape(case["x_hat"][rx["x_hat"]]) == (1872, 2)
        assert len(case["abs_llr"][rx["abs_llr"]]) == 1872
        assert isinstance(rx["crc"], bool)
        counts = meta["batch_crc"][name]
        assert counts["slots"] == meta["batch_size"] == 2
        assert 0 <= counts["failures"] <= counts["slots"]
        if rx["crc"]:
            assert counts["failures"] < counts["slots"]
        else:
            assert counts["failures"] > 0


def test_sweep_repeats_channel_and_noise(tiny_models):
    from dataclasses import replace

    import numpy as np

    from jamloc.site.cases import CASES
    from jamloc.site.export import payload, simulate_case

    case = payload(simulate_case(replace(CASES["jsr_sweep"], batch_size=2)))
    benign = np.asarray(case["benign"]["pilots"]["ls"])
    clean = np.asarray(case["pilots"]["subcarrier"]) // 12 + 6
    clean = (clean < 10) | (clean > 13)
    frames = case["sweep"]
    assert [f["jsr_db"] for f in frames] == list(range(-15, 10, 3))
    for f in frames:
        ls = np.asarray(f["pilots"]["ls"])
        assert np.array_equal(ls[clean], benign[clean])
        assert len(f["pilots"]["residual_power"]) == 72 and len(f["r_b"]) == 12
    # With the same jammer draw the excess complex amplitude scales as sqrt(power).
    lo, hi = (np.asarray(f["pilots"]["ls"]) for f in (frames[0], frames[-1]))
    np.testing.assert_allclose(hi - benign, (lo - benign) * 10 ** (24 / 20), atol=0.02)


def test_levels_replay_the_slot_at_each_power(tiny_models):
    from dataclasses import replace

    import numpy as np

    from jamloc.site.cases import CASES
    from jamloc.site.export import payload, simulate_case

    case = replace(CASES["partial"], batch_size=2, levels=(None, -9, 3))
    document = payload(simulate_case(case))
    off, weak, own = document["levels"]
    assert own == {"jsr_db": 3} and off["jsr_db"] is None and weak["jsr_db"] == -9
    truth = np.asarray(document["grids"]["truth"], bool)
    assert truth.any() and not np.asarray(off["grids"]["truth"]).any()
    assert weak["grids"]["truth"] == document["grids"]["truth"]
    power = np.asarray(document["grids"]["power_db"], float)
    for level in (off, weak):
        # The same draws: cells the jammer leaves alone receive the same samples at every power
        level_power = np.asarray(level["grids"]["power_db"], float)
        np.testing.assert_array_equal(level_power[~truth], power[~truth])
        assert (level_power[truth] < power[truth]).all()
        assert level["detectors"].keys() == document["detectors"].keys()
        assert level["batch_crc"].keys() == level["receivers"].keys()
        assert level["receivers"].keys() == document["receivers"].keys()
        for name, rx in level["receivers"].items():
            assert np.shape(level["x_hat"][rx["x_hat"]]) == (1872, 2)
            assert len(level["abs_llr"][rx["abs_llr"]]) == 1872
            assert level["batch_crc"][name]["slots"] == 2


def test_median_tie_uses_first_slot():
    import torch

    from jamloc.site.cases import CASES
    from jamloc.site.export import _select

    truth = torch.ones((2, 1, 3), dtype=torch.bool)
    flags = torch.tensor([[[True, False, False]], [[True, True, False]]])
    choice = _select(CASES["pilot"], truth, flags, {})
    assert choice["batch_median"] == 0.5 and choice["slot"] == 0


def test_fig3_reports_original_benign_fade(tiny_models):
    from jamloc.site.cases import CASES
    from jamloc.site.export import payload, simulate_case

    case = payload(simulate_case(CASES["fig3"]))
    selection = case["metadata"]["selection"]
    assert selection["slot"] == 0 and selection["rule"] == "paper_realization"
    assert selection["block_min_relative_db"] >= -3
    assert case["benign"]["pilots"]["ls"][:24] == case["pilots"]["ls"][:24]
    assert case["benign"]["r_b"] != case["prb"]["r_b"]


def _write_case_data(data):
    import json
    from dataclasses import asdict

    from jamloc.site.cases import CASES

    data.mkdir(parents=True)
    entries = []
    for case in CASES.values():
        cfg = case.config()
        grid = [[0] * cfg.num_symbols] * cfg.num_prb
        width = case.alloc[1]
        num_points = width * 12 * (cfg.num_symbols - 1)
        receivers = {
            "standard",
            "fused_hard",
            "true_soft",
            "true_prb_soft",
            *(f"{name}_soft" for name in case.decode_maps),
        }
        if case.name == "aided_flip":
            receivers.add("fused_soft_da")
        pilots = {
            "subcarrier": list(range(0, width * 12, 2)),
            "residual_power": [0] * (width * 6),
            "ls": [[0, 0]] * (width * 6),
            "projection": [[0, 0]] * (width * 6),
        }
        document = {
            "schema_version": 2,
            "metadata": {
                **asdict(case),
                "config": asdict(cfg),
                "selection": {
                    "rule": case.selection_rule,
                    "hit_rate": 0,
                    "batch_median": 0,
                    "slot": 0,
                    "flip_count": 1,
                    "flip_frequency": 1 / 64,
                    "seed": case.seed,
                    "failure_count": 1,
                },
                "thresholds": {
                    name: {
                        "split": "val",
                        "training_data": "train",
                        "false_alarm_rate": 0.01,
                        "value": 0.5,
                    }
                    for name in case.detectors
                },
                "batch_crc": {name: {"slots": 64, "failures": 1} for name in receivers},
                "noise_variance": 0.04,
                "true_noise_variance": 0.03,
            },
            "grids": {
                key: grid for key in ("allocation", "truth", "power_db", "residual_db", "evm_db")
            },
            "detectors": {
                name: {key: grid for key in ("scores", "decisions", "unscored")}
                for name in case.detectors
            },
            "prb": {"r_b": [0] * width, "threshold": [1] * width},
            "pilots": pilots,
            "receivers": {
                name: {"crc": True, "x_hat": "shared", "abs_llr": "shared"} for name in receivers
            },
            "x_hat": {"shared": [[0, 0]] * num_points},
            "abs_llr": {"shared": [0] * num_points},
        }
        if case.levels:
            frame = {key: document[key] for key in ("grids", "detectors", "receivers")}
            frame.update(
                x_hat=document["x_hat"],
                abs_llr=document["abs_llr"],
                batch_crc=document["metadata"]["batch_crc"],
            )
            document["levels"] = [
                {"jsr_db": db} if db == case.jsr_db else {"jsr_db": db, **frame}
                for db in case.levels
            ]
        if case.name in ("fig3", "jsr_sweep"):
            document["benign"] = {"jsr_db": None, "pilots": pilots, "r_b": [0] * width}
        if case.name == "jsr_sweep":
            document["sweep"] = [
                {"jsr_db": db, "pilots": pilots, "r_b": [0] * width} for db in range(-15, 10, 3)
            ]
        if case.name in ("pilot", "partial_flip", "aided_flip"):
            document["channel"] = {
                "dmrs_symbol": cfg.dmrs_symbol,
                "num_sc": width * 12,
                "pilot_sc": pilots["subcarrier"],
                "pilot_jammed": [False] * (width * 6),
                "ls": pilots["ls"],
                "fit": pilots["projection"],
                **{name: [[0, 0]] * (width * 12) for name in ("true", "standard", "map")},
                "residual_prb": [0] * width,
                "threshold_prb": [1] * width,
            }
        if case.name == "aided_flip":
            document["passes"] = [
                {
                    "h": document["channel"]["map"],
                    "x_hat": "shared",
                    "abs_llr": "shared",
                    "crc": crc,
                }
                for crc in (False, True)
            ]
        path = data / f"{case.name}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        entries.append({"name": case.name, "title": case.title, "file": path.name})
    (data / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "cases": entries}), encoding="utf-8"
    )


def test_check_build_requires_page_cases(tmp_path):
    import json

    from jamloc.site.build import check_build

    data = tmp_path / "data"
    _write_case_data(data)
    (data / "results.json").write_text("{}", encoding="utf-8")
    path = data / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["cases"] = [entry for entry in manifest["cases"] if entry["name"] == "pilot"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ValueError, match=r"manifest\.json.*aided_flip.*hopping.*jsr_sweep.*partial_flip"
    ):
        check_build(tmp_path)


@pytest.mark.parametrize(
    "name, keys",
    [
        ("pilot", ("schema_version",)),
        ("pilot", ("grids", "residual_db")),
        ("partial_flip", ("metadata", "selection", "flip_count")),
        ("pilot_whole_comb", ("metadata", "thresholds", "fused", "value")),
        ("hopping", ("detectors", "fused_atk", "decisions")),
        ("hopping", ("metadata", "batch_crc", "fused_atk_soft")),
        ("pilot", ("x_hat", "shared")),
        ("jsr_sweep", ("sweep", 0, "pilots", "residual_power")),
        ("pilot", ("channel", "fit", 0)),
        ("partial_flip", ("channel", "standard", 0, 1)),
        ("aided_flip", ("passes", 0, "x_hat")),
        ("aided_flip", ("passes", 0, "h", 0)),
        ("data", ("levels",)),
        ("partial", ("levels", 1, "batch_crc")),
        ("hopping", ("levels", 2, "detectors", "fused_atk")),
        ("spoof", ("levels", 0, "receivers", "fused_atk2_soft")),
    ],
)
def test_check_build_requires_case_fields(tmp_path, name, keys):
    import json
    import re

    from jamloc.site.build import check_build

    data = tmp_path / "data"
    _write_case_data(data)
    (data / "results.json").write_text("{}", encoding="utf-8")
    path = data / f"{name}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    parent = document
    for key in keys[:-1]:
        parent = parent[key]
    del parent[keys[-1]]
    path.write_text(json.dumps(document), encoding="utf-8")
    field = next(key for key in reversed(keys) if isinstance(key, str))
    with pytest.raises(ValueError, match=rf"{name}\.json.*{re.escape(field)}"):
        check_build(tmp_path)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda levels: levels.pop(3), "expected the jammer powers"),
        (lambda levels: levels[0]["grids"].update(truth=[[1] * 14] * 24), "switched-off jammer"),
        (lambda levels: levels[7].update(grids={}), "own power refers"),
    ],
)
def test_check_build_checks_levels(tmp_path, change, message):
    import json

    from jamloc.site.build import check_build

    data = tmp_path / "data"
    _write_case_data(data)
    (data / "results.json").write_text("{}", encoding="utf-8")
    path = data / "partial.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["levels"][7] == {"jsr_db": 3}
    change(document["levels"])
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"partial\.json.*{message}"):
        check_build(tmp_path)


def test_check_build_rejects_nested_channel_residual(tmp_path):
    import json

    from jamloc.site.build import check_build

    data = tmp_path / "data"
    _write_case_data(data)
    (data / "results.json").write_text("{}", encoding="utf-8")
    path = data / "pilot.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["channel"]["residual_prb"][0] = [0]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"pilot\.json.*channel.residual_prb"):
        check_build(tmp_path)


def test_build_showcase(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from jamloc.site.__main__ import main
    from jamloc.site.build import PLOTLY_URL, build_site, check_build

    results = tmp_path / "results"
    papers = tmp_path / "paper-results"
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(results))
    data = results / "site" / "data"
    _write_case_data(data)

    inputs = {
        "eval/test_pd_vs_jsr.csv": (
            "detector,attack,channel,jsr_db,cells,num_cells,pd_mean,pd_std,n_seeds\n"
            "fused,pilot,both,-15,scheduled,18,0.5,0.1,3\n"
            "fused,pilot,both,-15,all,20,0.6,0.1,3\n"
            "fused,pilot,C,-15,scheduled,8,0.4,0.1,3\n"
        ),
        "margin/margin_vs_snr_ci.csv": (
            "snr_db,channel,detector,slot_fa_target,margin_db,boot_draws,margin_lo_db,"
            "margin_hi_db,point_any_attack_censored\n"
            "5,C,fused,0.001,-7.9,1000,-inf,-5.7,True\n"
        ),
        "intervals/test_matched_by_attack_f1_ci.csv": (
            "detector,attack,seed,n_seeds,f1,f1_lo,f1_hi,fa_scored,fa_scored_lo,fa_scored_hi,"
            "jsr_db,pd,pd_lo,pd_hi\n"
            "fused,pilot,mean,3,0.77,0.75,0.79,0.01,0.009,0.011,-15,0.65,0.62,0.67\n"
        ),
        "intervals/test_matched_by_attack_paired.csv": (
            "first,second,metric,attack,jsr_db,seed,n_seeds,delta,delta_lo,delta_hi\n"
            "fused,no_residual,f1,all,,mean,3,0.005,0.004,0.007\n"
        ),
        "dmrs_count/detection.csv": (
            "dmrs_count,detector,seed,threshold,f1_pilot,pd_pilot_m15,f1_data,pd_data_m15,"
            "f1_partial,f1_all,fa\n"
            "2,fused,0,0.7,0.8,0.6,0.9,0.7,0.9,0.9,0.01\n"
            "2,fused,1,0.7,0.9,0.8,0.9,0.7,0.9,0.9,0.01\n"
            "2,block_test,,30,0.6,0.3,0.9,0.4,0.8,0.8,0.01\n"
        ),
        "receiver_study/recovery/recovery_C.csv": (
            "receiver,jammer,noise_estimate,seed,attack,jsr_db,snr_db,ber,bler,block_errors,blocks\n"
            + "".join(
                f"{rx},nominal,true,{seed},pilot_whole_allocation,{jsr},15.0,0,0,{e},10\n"
                for rx, e in (("p_r_e_soft", 4), ("p_r_e_soft_da", 2), ("lmmse_da", 3))
                for jsr in (3.0, 9.0)
                for seed in (0, 1)
            )
        ),
        "receiver_study/recovery/recovery_C_paired.csv": (
            "first,second,attack,jsr_db,slots,only_first,only_second,"
            "bler_first,bler_second,delta,delta_lo,delta_hi\n"
            "p_r_e_soft,p_r_e_soft_da,pilot_whole_allocation,9,20,4,0,0.4,0.2,0.2,0.1,0.3\n"
        ),
        "width_sweep/width_bler_C.csv": (
            "jam_prb,cover,attack,jsr_db,receiver,block_errors,blocks,bler,bler_lo,bler_hi\n"
            "4,0.333,partial,3.0,standard,50,100,0.5,0.4,0.6\n"
            "4,0.333,partial,3.0,p_r_e_soft,0,4800,0,0,0.001\n"
            "8,0.667,partial,3.0,p_r_e_soft,90,100,0.9,0.8,0.95\n"
            "8,0.667,partial,3.0,fused_hard,1,10,0.1,0,0.4\n"
        ),
        "width_sweep/width_detection.csv": (
            "attack,jsr_db,group,value,pd,pd_std,seeds,num_cells,num_slots\n"
            "pilot,-9.0,width,2.0,0.76,0.02,3,113,81\n"
            "pilot,-9.0,cover,0.25,0.82,0.01,3,737,274\n"
        ),
        "recovery_pr9/recovery_C.csv": (
            "receiver,jammer,seed,attack,jsr_db,snr_db,ber,bler,block_errors,blocks\n"
            "standard,nominal,0,pilot,-15,15,0,0.2,2,10\n"
            "standard,nominal,1,pilot,-15,15,0,0.05,1,20\n"
            "standard,nominal,2,pilot,-15,15,0,0.1,3,30\n"
            "true_soft,nominal,0,pilot,9,15,0,0.1,1,10\n"
            "lmmse,nominal,0,none,nan,15,0,0,0,10\n"
            "lmmse,nominal,1,none,nan,15,0,0,0,20\n"
            "standard,nominal,3,pilot,-15,15,0,0.1,1,10\n"
            "standard,nominal,0,pilot,12,15,0,0.1,1,10\n"
            "standard,nominal,0,pilot,-15,8,0,0.1,1,10\n"
            "standard,hopping,0,pilot,-15,15,0,0.1,1,10\n"
            "fused_hard,nominal,0,pilot,9,15,0,0.1,1,10\n"
        ),
        "dmrs_count/recovery.csv": (
            "dmrs_count,jammer_dmrs,seed,channel,snr_db,ue_prbs,jammed_prbs,attack,jsr_db,"
            "receiver,block_errors,blocks,bler\n"
            "2,first,0,C,15.0,12,4,pilot,9,standard,3,10,0.3\n"
            "2,first,1,C,15.0,12,4,pilot,9,standard,1,10,0.1\n"
            "2,first,1,C,15.0,12,4,pilot,3,standard,9,10,0.9\n"
        ),
    }
    unselected = {}
    for suffix, jammer in (("", "nominal"), ("_pulsed", "pulsed"), ("_hopping", "hopping")):
        base = f"recovery/recovery_C{suffix}"
        stem = base if jammer == "nominal" else f"{base}_replay"
        inputs[f"{stem}.csv"] = (
            "receiver,jammer,noise_estimate,seed,attack,jsr_db,snr_db,ber,bler,block_errors,blocks\n"
            f"standard,{jammer},true,0,data,9,15,0,0.8,8,10\n"
            f"standard,{jammer},true,1,data,9,15,0,0.1,1,20\n"
            f"standard,{jammer},true,1,data,6,15,0,0,0,10\n"
        )
        inputs[f"{stem}_paired.csv"] = (
            "first,second,attack,jsr_db,slots,only_first,only_second,"
            "bler_first,bler_second,delta,delta_lo,delta_hi\n"
            "standard,p_r_e_soft,data,9,10,8,1,0.8,0.1,0.7,0.5,0.8\n"
        )
        if jammer == "pulsed":  # Both complete runs exist, with different estimates.
            unselected[f"{base}.csv"] = inputs[f"{stem}.csv"].replace("0.8,8,10", "0.2,2,10")
            unselected[f"{base}_paired.csv"] = inputs[f"{stem}_paired.csv"].replace(
                ",9,10,", ",9,99,"
            )
        if jammer == "nominal":  # An incomplete replay cannot override a complete original.
            unselected[f"{base}_replay_paired.csv"] = inputs[f"{stem}_paired.csv"].replace(
                ",9,10,", ",9,99,"
            )
    for name, contents in (inputs | unselected).items():
        path = papers / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write(contents)

    main(["build", "--results-from", str(papers)])
    build = results / "site" / "build"
    assert {p.name for p in build.glob("*.html")} == {
        "index.html",
        "walkthrough.html",
        "explore.html",
        "constellations.html",
        "animations.html",
        "results.html",
    }
    assert (build / ".nojekyll").is_file()
    content = json.loads((build / "data" / "results.json").read_text(encoding="utf-8"))
    pd = content["pd"]["rows"]
    assert len(pd) == 1 and pd[0]["pd_mean"] == 0.5 and pd[0]["num_cells"] == 18
    (margin,) = content["margin"]["rows"]
    assert margin["margin_lo_db"] is None and margin["point_any_attack_censored"] is True
    recovery = content["recovery"]
    assert {r["jammer"] for r in recovery["rows"]} == {"nominal", "pulsed", "hopping"}
    assert all(r["bler"] == 9 / 30 and r["blocks"] == 30 for r in recovery["rows"])
    assert len(recovery["paired"]) == 3 and all(r["slots"] == 10 for r in recovery["paired"])
    curves = {
        (r["attack"], r["jsr_db"], r["receiver"]): (r["block_errors"], r["blocks"], r["bler"])
        for r in content["recovery_curves"]["rows"]
    }
    assert curves == {
        ("pilot", -15, "standard"): (6, 60, 0.1),
        ("pilot", 9, "true_soft"): (1, 10, 0.1),
        ("none", None, "lmmse"): (0, 30, 0),
    }
    assert content["matched"]["rows"][0]["f1"] == 0.77
    assert content["matched"]["paired"][0]["jsr_db"] is None
    dmrs = {(r["dmrs_count"], r["detector"]): r for r in content["dmrs"]["rows"]}
    assert dmrs[2, "fused"]["n_seeds"] == 2
    assert dmrs[2, "fused"]["pd_pilot_m15"] == pytest.approx(0.7)
    assert dmrs[2, "block_test"]["n_seeds"] == 1
    aided = {(r["jsr_db"], r["receiver"]): r for r in content["data_aided"]["rows"]}
    row = aided[9.0, "p_r_e_soft_da"]
    assert row["blocks"] == 20 and row["bler"] == 0.2
    assert "from 40.0% to 20.0% at +9 dB" in content["data_aided"]["summary"]
    width = {(r["jam_prb"], r["receiver"]): r["bler"] for r in content["width_sweep"]["rows"]}
    assert width == {(4, "standard"): 0.5, (4, "p_r_e_soft"): 0, (8, "p_r_e_soft"): 0.9}
    (by_width,) = content["width_sweep"]["detection"]  # the width groups, not the cover bins
    assert by_width["jam_prb"] == 2 and by_width["pd"] == 0.76
    assert content["width_sweep"]["alloc_prb"] == 12
    (pooled,) = content["dmrs"]["recovery"]  # only the +9 dB rows, pooled over the seeds
    assert pooled["jammer_dmrs"] == "first" and pooled["blocks"] == 20 and pooled["bler"] == 0.2
    for section in content.values():
        assert section["population"]
        for download in section["downloads"]:
            assert (build / download).is_file()
    for name in inputs:
        folder = Path(name).parts[0]
        prefix = {
            "receiver_study": "data_aided_",
            "recovery_pr9": "recovery_curves_",
            "width_sweep": "width_",
        }.get(folder, "")
        target = build / "downloads" / f"{prefix}{Path(name).name}"
        assert target.read_bytes() == (papers / name).read_bytes()
    for name in unselected:
        assert not (build / "downloads" / Path(name).name).exists()

    check_build(build)
    (papers / "recovery" / "recovery_C_paired.csv").unlink()
    with pytest.raises(FileNotFoundError, match="complete recovery pair for recovery_C"):
        build_site(results_from=papers)
    assert json.loads((build / "data" / "results.json").read_text(encoding="utf-8")) == content

    page = build / "extra.html"
    for markup, error in (
        ('<a href="absent.csv">Missing</a>', "absent.csv"),
        ('<link rel="stylesheet" href="absent.css">', "absent.css"),
        ('<script src="absent.js"></script>', "absent.js"),
        ('<script src=""></script>', "file path"),
        ('<script src="//example.com/script.js"></script>', "pinned"),
        (f'<script src="{PLOTLY_URL}?v=2"></script>', "pinned"),
        (f'<script src="{PLOTLY_URL}"></script>', "results.html"),
    ):
        page.write_text(markup, encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            check_build(build)
    page.unlink()
    module = build / "extra.js"
    module.write_text('import "./missing.js";', encoding="utf-8")
    with pytest.raises(ValueError, match="missing.js"):
        check_build(build)
    module.unlink()
    (build / "data" / "pilot.json").unlink()
    with pytest.raises(ValueError, match="pilot.json"):
        check_build(build)


@pytest.mark.parametrize("dirty_at", ["before build", "during build", "before push", None])
def test_deploy_checks_source_worktree(tmp_path, monkeypatch, dirty_at):
    from subprocess import CompletedProcess

    from jamloc.site import deploy

    source = tmp_path / "source-worktree"
    main = tmp_path / "primary-checkout"
    output = tmp_path / "build"
    output.mkdir()
    calls = []

    def build(*, results_from):
        calls.append("build")
        return output

    def git(args, *, cwd, **kwargs):
        if "worktree" in args and "list" in args:
            return CompletedProcess(args, 0, f"worktree {main}\0\0", "")
        if "status" in args:
            assert {"--porcelain", "--untracked-files=normal", "--ignore-submodules=none"} <= set(
                args
            )
            dirty = {
                "before build": "?? new.html\n",
                "during build": " M site/index.html\n" if "build" in calls else "",
                "before push": "M  site/index.html\n" if "stage" in calls else "",
                None: "",
            }[dirty_at]
            return CompletedProcess(args, 0, dirty if cwd == source else "", "")
        if "ls-remote" in args:
            calls.append("remote")
            return CompletedProcess(args, 2, "", "")
        if args[1] == "add":
            calls.append("stage")
        if args[1] == "push":
            calls.append("push")
        return CompletedProcess(args, 0, "", "")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(deploy, "REPO_ROOT", source)
    monkeypatch.setattr(deploy, "build_site", build)
    monkeypatch.setattr(deploy, "check_build", lambda path: None)
    monkeypatch.setattr(deploy.subprocess, "run", git)
    if dirty_at is None:
        deploy.deploy_site()
        assert calls == ["build", "remote", "stage", "push"]
    else:
        with pytest.raises(ValueError, match=r"dirty.*source-worktree"):
            deploy.deploy_site()
        expected = {
            "before build": [],
            "during build": ["build"],
            "before push": ["build", "remote", "stage"],
        }
        assert calls == expected[dirty_at]
