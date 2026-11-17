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
    cfg = dataclasses.replace(SimConfig(), jammer_waveform="tone")
    cells = cell_mask(torch.tensor([1]), torch.tensor([0]), torch.tensor([2]), cfg)
    with pytest.raises(ValueError, match="waveform"):
        Jammer(cfg)(cells, torch.tensor([0.0]))


def test_size_sweep_and_generalization_small(tmp_path, monkeypatch):
    from jamloc.dataset import generate, load
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
    assert {r["condition"] for r in rows} == {"test", "longdelay", "fast", "qpskjam"}
    assert (tmp_path / "results" / "generalization" / "generalization.png").is_file()

    size_sweep.main(["--widths", "4", "12", "--seeds", "0", "--epochs", "1"])
    size_sweep.main(["--widths", "4", "12", "--latency", "--reps", "3"])
    out = tmp_path / "results" / "size_sweep"
    with open(out / "size_sweep.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["width"]) for r in rows] == [4, 12]
    assert int(rows[0]["params"]) < int(rows[1]["params"]) == 66_553
    assert (out / "size_sweep.png").is_file() and (out / "size_sweep_latency.csv").is_file()
