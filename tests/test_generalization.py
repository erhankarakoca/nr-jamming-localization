import csv
import dataclasses

import pytest
import torch

from jamloc.config import SimConfig
from jamloc.jammer import Jammer, cell_mask


def test_qpsk_jammer_has_constant_envelope_and_same_energy():
    cfg = dataclasses.replace(SimConfig(), jammer_waveform="qpsk")
    cells = cell_mask(torch.tensor([3]), torch.tensor([4]), torch.tensor([5]), cfg)
    z = Jammer(cfg)(cells, torch.tensor([6.0]))[0, 0, 0]
    jammed = z[cells[0].T.repeat_interleave(12, dim=1)]
    assert torch.allclose(jammed.abs() ** 2, torch.full_like(jammed.real, 10**0.6), rtol=1e-5)
    assert (z.abs()[~cells[0].T.repeat_interleave(12, dim=1)] == 0).all()


def test_unknown_waveform_is_rejected():
    cfg = dataclasses.replace(SimConfig(), jammer_waveform="chirp")
    cells = cell_mask(torch.tensor([1]), torch.tensor([0]), torch.tensor([2]), cfg)
    with pytest.raises(ValueError, match="waveform"):
        Jammer(cfg)(cells, torch.tensor([0.0]))


def test_size_sweep_and_generalization_small(tmp_path, monkeypatch):
    from jamloc.dataset import GENERALIZATION, generate, load
    from jamloc.experiments import generalization, size_sweep
    from jamloc.model import VARIANTS
    from jamloc.train import train

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)
    generate("test", num_slots=60, batch_size=32)
    for variant in VARIANTS:
        train(variant, epochs=1, batch_size=32)

    generalization.main(["--num-slots", "60"])
    longdelay = load("longdelay")
    assert set(longdelay["channel"].tolist()) == {1}  # TDL-C only
    with open(
        tmp_path / "results" / "generalization" / "generalization.csv", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    assert {r["condition"] for r in rows} == {"test", *GENERALIZATION}
    assert {"combjam", "hopjam"} <= set(GENERALIZATION)
    assert (tmp_path / "results" / "generalization" / "generalization.png").is_file()

    size_sweep.main(["--widths", "4", "12", "--seeds", "0", "--epochs", "1"])
    size_sweep.main(["--widths", "4", "12", "--latency", "--reps", "3"])
    out = tmp_path / "results" / "size_sweep"
    with open(out / "size_sweep.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["width"]) for r in rows] == [4, 12]
    assert int(rows[0]["params"]) < int(rows[1]["params"]) == 66_553
    assert (out / "size_sweep.png").is_file() and (out / "size_sweep_latency.csv").is_file()


def test_robust_training_small(tmp_path, monkeypatch):
    from jamloc.dataset import generate, load
    from jamloc.experiments import robust_training
    from jamloc.model import VARIANTS
    from jamloc.train import train

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    sizes = {"train": 64, "val": 32, "train_mix": 96, "val_mix": 32}
    sizes.update(test=60, longdelay=60, fast=60)
    for split, n in sizes.items():
        generate(split, num_slots=n, batch_size=16)
    for variant in VARIANTS:
        train(variant, epochs=1, batch_size=32)

    robust_training.main(["--seeds", "0", "--epochs", "1"])
    out = tmp_path / "results" / "robust_training"
    with open(out / "robust_training.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 2 * len(VARIANTS)  # conditions x trainings x detectors
    assert (out / "robust_training.png").is_file()
    assert (tmp_path / "results" / "models" / "fused_mix_seed0.pt").is_file()
    assert len(load("train_mix")["labels"]) == 96


def test_attack_training_small(tmp_path, monkeypatch):
    from jamloc.dataset import generate
    from jamloc.experiments import attack_training
    from jamloc.train import train

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    sizes = {"train": 64, "val": 32, "train_atk": 96, "val_atk": 32}
    sizes.update(test=60, combjam=60, hopjam=60)
    for split, n in sizes.items():
        generate(split, num_slots=n, batch_size=16)
    train("fused", epochs=1, batch_size=32)

    attack_training.main(["--seeds", "0", "--epochs", "1"])
    out = tmp_path / "results" / "attack_training"
    with open(out / "attack_training.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(attack_training.CONDITIONS) * 2  # conditions x trainings
    assert {r["condition"] for r in rows} == set(attack_training.CONDITIONS.values())
    assert (out / "attack_training.png").is_file()
    assert (tmp_path / "results" / "models" / "fused_atk_seed0.pt").is_file()
