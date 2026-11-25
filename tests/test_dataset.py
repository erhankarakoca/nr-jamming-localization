import numpy as np

from jamloc.dataset import META_KEYS, TEST_JSR_DB, generate, load
from jamloc.features import FEATURE_NAMES


def test_generate_small(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    path = generate("train", num_slots=128, batch_size=32)
    assert path == tmp_path / "train.npz" and path.is_file()

    d = load("train")
    assert (
        d["features"].shape == (128, len(FEATURE_NAMES), 24, 14)
        and d["features"].dtype == np.float32
    )
    assert d["labels"].shape == (128, 24, 14) and d["labels"].dtype == np.uint8
    for key in META_KEYS:
        assert d[key].shape == (128,)
    assert set(np.unique(d["channel"])) == {0, 1}  # both TDL models

    benign = d["attack"] == 0
    assert benign.any() and not d["labels"][benign].any()
    assert d["labels"][~benign].any(axis=(1, 2)).all()

    generate("test", num_slots=60, batch_size=32)
    t = load("test")
    assert not np.array_equal(t["features"][:10], d["features"][:10])
    attacked = t["attack"] > 0
    assert set(np.unique(t["jsr_db"][attacked])) <= set(TEST_JSR_DB)
    assert set(np.unique(t["attack"])) == {0, 1, 2, 3}


def test_general_mix_covers_jammers_channels_and_modulations(tmp_path, monkeypatch):
    from jamloc.dataset import GENERAL_CONDITIONS, SPLITS

    # 3 jammers x 3 channel conditions x 2 modulations, the pulsed jammer held out
    assert len(GENERAL_CONDITIONS) == 18
    assert SPLITS["train_gen"]["mix"] is GENERAL_CONDITIONS
    qam = [c for c in GENERAL_CONDITIONS if c.get("mcs_index") == 13]
    assert len(qam) == 9 and all(c["_snr_db"] == (10.0, 25.0) for c in qam)
    assert not any("jammer_duty" in c for c in GENERAL_CONDITIONS)

    # Batches 0-1 use the first condition (QPSK), batches 2-3 the second (16-QAM)
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    generate("train_gen", num_slots=32, batch_size=8)
    d = load("train_gen")
    assert list(np.unique(d["mcs_index"])) == [5, 13]
    assert (d["snr_db"][d["mcs_index"] == 13] >= 10).all()
    assert np.isfinite(d["features"]).all()
