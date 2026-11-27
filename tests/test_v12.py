"""Paper v12: the pictures of the method and the cost of the receiver's steps."""

import numpy as np
import pytest
import torch

from jamloc.config import SimConfig
from jamloc.experiments import complexity
from jamloc.experiments.figures import CFG, _estimates_without_map, _pilot_weight_example
from jamloc.model import TinyUNet
from jamloc.recovery import lmmse_interpolation, lmmse_lowrank


@pytest.mark.parametrize("num_dmrs", [1, 2])
def test_lowrank_lmmse_matches_the_full_solve(num_dmrs):
    cfg = SimConfig()
    num_sc = 144
    gen = torch.Generator().manual_seed(0)
    pilot_sc = torch.arange(0, num_sc, 2).repeat(num_dmrs)  # comb-2, repeated per DMRS symbol
    shape = (3, pilot_sc.numel())
    h_ls = torch.complex(torch.randn(shape, generator=gen), torch.randn(shape, generator=gen))
    s2 = 0.05 * (1 + 30 * torch.rand(shape, generator=gen))  # pilot weights of every size
    clean = torch.ones(shape, dtype=torch.bool)
    clean[1, 20:30] = False  # a map that drops some pilots
    args = (h_ls, pilot_sc, clean, s2, num_sc, cfg.scs, cfg.tau_max)
    full, full_err = lmmse_interpolation(*args)
    low, low_err = lmmse_lowrank(*args)
    assert torch.allclose(low, full, rtol=0, atol=1e-5)
    assert torch.allclose(low_err, full_err, rtol=0, atol=1e-5)


def test_operation_counts():
    counts = complexity.operation_counts(72, 18, 144, 1872, 21)
    assert counts["pilots"] == 4 * 2 * 72 * 18
    assert counts["lmmse_fixed"] == 4 * 144 * 72
    assert counts["lmmse"] == round(4 * (72**3 / 3 + 72**2 * 145 + 2 * 144 * 72))
    # The delay subspace makes the per-slot estimate more than eight times cheaper
    assert counts["lmmse_lowrank"] * 8 < counts["lmmse"]
    assert counts["shrinkage"] == 10 * 1872


def test_unet_macs_of_the_paper_unet():
    assert complexity.unet_macs(TinyUNet(6), torch.zeros(1, 6, 24, 14)) == 5_451_264


def test_complexity_benchmark_gives_every_step():
    rows = complexity.benchmark(reps=1, dmrs2=False)
    assert [r["step"] for r in rows] == [key for key, _, _ in complexity.STEPS]
    assert all(r["ms"] > 0 for r in rows)
    by = {r["step"]: r for r in rows}
    assert by["unet"]["real_macs"] == 5_451_264
    assert by["decoding"]["edge_messages"] > 0
    assert by["lmmse_lowrank"]["real_macs"] < by["lmmse"]["real_macs"]
    assert "LMMSE" in complexity.table(rows)


def test_pilot_weights_keep_the_channel_estimate_on_jammed_prbs():
    ex = _pilot_weight_example(6.0, 15.0)
    jammed = ex["jammed_prb"]
    assert ex["r_db"][jammed].min() > 10  # far above what noise alone leaves
    assert ex["r_db"][~jammed].max() < 10
    nmse = ex["nmse"]
    assert nmse["weighted", "jammed"] < nmse["alike", "jammed"] - 3
    assert nmse["weighted", "clean"] < nmse["alike", "clean"]


def test_shrinkage_weights_from_the_receiver():
    est = _estimates_without_map(CFG, "data", 9.0, seed=7)
    data = ~np.isnan(est["cell"])
    w = est["weight"]
    assert np.all((w[data] >= 0) & (w[data] <= 1 + 1e-6))
    # A row the block jammer fills: its cells differ only by their noise, so none keeps any of
    # its own estimate
    row = 10 - 6  # PRB 10 of the allocation 6-17
    assert w[row][data[row]].max() < 0.05
    # Where neither estimate was clipped at zero, shrinkage is the row mean plus the weighted
    # deviation of the cell
    ok = data & (est["cell"] > 0) & (est["prb"] > 0)
    expected = np.maximum(est["prb"] + w * (est["cell"] - est["prb"]), 0)
    assert np.allclose(est["shrink"][ok], expected[ok], atol=1e-4)
    assert np.all(est["se"][data] >= 0)


def test_detection_curves_average_the_seeds_and_draw():
    from jamloc.experiments import detection_curves as dc

    rows = [
        {"detector": "unet", "grid": "test", "attack": "pilot", "jsr_db": -15.0, "pd": p}
        for p in (0.4, 0.6)
    ] + [{"detector": "evm", "grid": "test", "attack": "data", "jsr_db": 3.0, "pd": 0.5}]
    mean = dc.mean_rows(rows)
    assert mean[1] == {
        "detector": "unet",
        "grid": "test",
        "attack": "pilot",
        "jsr_db": -15.0,
        "pd": 0.5,
        "runs": 2,
    }
    fig = dc.figure(mean)
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert labels == ["U-Net, block", "EVM, block"]
