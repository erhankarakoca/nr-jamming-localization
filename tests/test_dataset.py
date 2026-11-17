import numpy as np

from jamloc.dataset import META_KEYS, TEST_JSR_DB, generate, load


def test_generate_small(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    path = generate("train", num_slots=128, batch_size=32)
    assert path == tmp_path / "train.npz" and path.is_file()

    d = load("train")
    assert d["features"].shape == (128, 6, 24, 14) and d["features"].dtype == np.float32
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
