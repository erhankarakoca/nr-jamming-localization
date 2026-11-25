import csv

from jamloc.dataset import generate
from jamloc.train import EarlyStopping, train


def test_early_stopping_waits_for_patience_epochs_without_improvement():
    stop = EarlyStopping(patience=2)
    assert not stop.step(1.0)
    assert not stop.step(0.9)  # improves
    assert not stop.step(0.95)  # one epoch without improvement
    assert stop.step(0.91)  # two epochs without improvement
    # Patience 0 turns early stopping off
    off = EarlyStopping(patience=0)
    assert not any(off.step(1.0) for _ in range(5))


def test_training_with_the_plateau_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    (tmp_path / "data").mkdir()
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)

    path = train("fused", epochs=3, schedule="plateau", patience=1)
    assert path.is_file()
    with open(path.with_name(f"{path.stem}_history.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert 1 <= len(rows) <= 3 and {"epoch", "train_loss", "val_loss", "lr"} <= set(rows[0])
