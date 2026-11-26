"""One U-Net for one and two DMRS symbols and three modulations (paper v11)."""

import numpy as np
import pytest

from jamloc import dataset
from jamloc.config import SimConfig
from jamloc.dataset import ALL_CONDITIONS, DMRS2_GRIDS, SPLITS, dmrs2_split, generate
from jamloc.experiments import one_unet, recovery
from jamloc.features import FEATURE_NAMES
from jamloc.train import checkpoint_path, train


def test_all_conditions_cross_the_general_ones_with_64qam_and_two_dmrs():
    assert len(ALL_CONDITIONS) == 54
    assert len({repr(sorted(c.items())) for c in ALL_CONDITIONS}) == 54
    mcs = [c.get("mcs_index", SimConfig().mcs_index) for c in ALL_CONDITIONS]
    assert {m: mcs.count(m) for m in set(mcs)} == {5: 18, 13: 18, 17: 18}
    dmrs = [c.get("dmrs_additional_position", 0) for c in ALL_CONDITIONS]
    assert dmrs.count(0) == dmrs.count(1) == 27
    # every condition of the v10 training is still there, and the pulsed jammer is in none
    for c in dataset.GENERAL_CONDITIONS:
        assert c in ALL_CONDITIONS
    assert not any("jammer_duty" in c for c in ALL_CONDITIONS)
    # 64-QAM is MCS 17 of table 1, at the SNRs where link adaptation would choose it
    assert SimConfig(mcs_index=17).mcs_index == 17
    assert all(c["_snr_db"] == (16.0, 30.0) for c in ALL_CONDITIONS if c.get("mcs_index") == 17)


def test_two_dmrs_grids_differ_from_their_one_dmrs_grid_only_in_the_dmrs():
    for split in DMRS2_GRIDS:
        one, two = SPLITS[split], SPLITS[dmrs2_split(split)]
        assert two["cfg"] == {**one.get("cfg", {}), "dmrs_additional_position": 1}
        assert {k: v for k, v in two.items() if k not in ("cfg", "seed", "title")} == {
            k: v for k, v in one.items() if k not in ("cfg", "seed", "title")
        }
    seeds = [s["seed"] for s in SPLITS.values()]
    assert len(seeds) == len(set(seeds))
    assert SimConfig(dmrs_additional_position=1).dmrs_symbols == (2, 11)
    # the grids of the experiment exist with one and with two DMRS symbols
    for dmrs in (1, 2):
        assert all(s in SPLITS for s in one_unet.splits(dmrs))


def test_training_split_mixes_one_and_two_dmrs_symbols(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    # batches 0-1 use the first condition (one DMRS), batches 2-3 the second (two)
    generate("train_all", num_slots=32, batch_size=8)
    d = dataset.load("train_all")
    dmrs = d["features"][:, FEATURE_NAMES.index("dmrs_mask")]
    second = dmrs[:, :, 11].max(axis=1) > 0
    assert not second[:16].any() and second[16:].all()
    assert (dmrs[:, :, 2].max(axis=1) > 0).all()


def test_one_unet_trains_compares_and_tabulates(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    grids = {"test": "QPSK block", "test_qam64": "64-QAM block"}
    monkeypatch.setattr(one_unet, "GRIDS", grids)
    sizes = {"train_all": 64, "val_all": 32, "train_gen": 32, "val_gen": 32}

    def small(split, batch_size=64):
        return generate(split, num_slots=sizes.get(split, 60), batch_size=32)

    monkeypatch.setattr(one_unet, "generate", small)
    small("train_gen")
    small("val_gen")
    train("fused", epochs=1, batch_size=32, data="gen")  # stands in for the v10 U-Net
    one_unet.prepare([0], epochs=1, patience=0)
    assert checkpoint_path("fused", 0, data="all").name == "fused_all_seed0.pt"
    assert checkpoint_path("fused", 0, data="all").is_file()

    rows = one_unet.run([0])
    keys = {(r["grid"], r["dmrs"], r["detector"]) for r in rows}
    assert keys == {(g, d, n) for g in grids.values() for d in (1, 2) for n in one_unet.DETECTORS}
    assert all(0 <= r["f1"] <= 1 and 0 <= r["fa"] <= 1 for r in rows)
    text = one_unet.table(one_unet.aggregate(rows, one_unet.VALUES))
    assert "| 64-QAM block |" in text and "v11 U-Net, 2 DMRS" in text


def test_decoding_with_two_dmrs_symbols_and_the_v11_unet(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def link(cfg):
        seen["cfg"] = cfg
        raise Stop

    monkeypatch.setattr(recovery, "JammedPUSCHLink", link)
    with pytest.raises(Stop):
        recovery.simulate("C", 10.0, [9.0], 8, 1, 1, 0, ["standard"], dmrs2=True, mcs_index=17)
    assert seen["cfg"].dmrs_symbols == (2, 11) and seen["cfg"].mcs_index == 17
    with pytest.raises(Stop):
        recovery.simulate("C", 10.0, [9.0], 8, 1, 1, 0, ["standard"])
    assert seen["cfg"].dmrs_symbols == (2,)
    assert recovery.unet_source("all") == ({"data": "all"}, "val_all")
    assert "all" in recovery.TRAIN_COMMANDS
    assert np.isfinite(SimConfig().num_symbols)
