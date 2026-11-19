import numpy as np
import torch
from sionna.phy import config

from jamloc.config import SimConfig
from jamloc.dataset import GENERALIZATION, TEST_JSR_DB, generate, load
from jamloc.features import FEATURE_NAMES
from jamloc.jammer import cell_mask
from jamloc.ue import UEPool


def test_neighbor_waveform_matches_the_ue_pilot_comb():
    from jamloc.jammer import Jammer

    cfg = SimConfig(jammer_waveform="ue")
    device = config.device
    start, size = 3, 6
    cells = cell_mask(
        torch.tensor([3], device=device),
        torch.tensor([start], device=device),
        torch.tensor([size], device=device),
        cfg,
    )
    grid = Jammer(cfg)(cells, torch.zeros(1, device=device))[0, 0, 0]
    band = slice(12 * start, 12 * (start + size))
    block = grid[:, band]
    assert torch.count_nonzero(grid[:, : band.start]) == 0
    assert torch.count_nonzero(grid[:, band.stop :]) == 0

    # Sionna reserves both combs in the mask, but only the pilot comb has nonzero values.
    pattern = UEPool(cfg).transmitter(size).resource_grid.pilot_pattern
    ue_pilots = torch.zeros_like(block)
    ue_pilots[pattern.mask[0, 0].bool()] = pattern.pilots[0, 0]
    pilots = ue_pilots[cfg.dmrs_symbol] != 0
    even = torch.arange(block.shape[1], device=device) % 2 == 0
    assert torch.equal(pilots, even)

    dmrs = block[cfg.dmrs_symbol]
    data = block[torch.arange(cfg.num_symbols, device=device) != cfg.dmrs_symbol]
    torch.testing.assert_close(data.abs().square(), torch.ones_like(data.real))
    torch.testing.assert_close(dmrs[pilots].abs().square(), torch.full_like(dmrs[pilots].real, 2))
    assert torch.count_nonzero(dmrs[~pilots]) == 0
    for symbols, component_power in ((data, 0.5), (dmrs[pilots], 1.0)):
        torch.testing.assert_close(
            symbols.real.square(), torch.full_like(symbols.real, component_power)
        )
        torch.testing.assert_close(
            symbols.imag.square(), torch.full_like(symbols.imag, component_power)
        )
    stronger = Jammer(cfg)(cells, torch.full((1,), 10.0, device=device))[0, 0, 0]
    torch.testing.assert_close(stronger.abs().square(), 10 * grid.abs().square())


def test_generate_tiny_neighbor_split(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    spec = GENERALIZATION["neighbor"]
    assert spec["seed"] == 19_000_000
    assert spec["cfg"] == {"jammer_waveform": "ue"}
    assert spec["attacks"] == (3,)
    assert spec["title"]

    assert generate("neighbor", num_slots=1) == tmp_path / "neighbor.npz"
    d = load("neighbor")
    attacked = d["attack"] == 3
    assert set(d["attack"]) == {0, 3}
    assert set(d["jsr_db"][attacked]) == set(TEST_JSR_DB)
    assert not d["labels"][~attacked].any()
    for labels, start, size in zip(
        d["labels"][attacked], d["jam_start"][attacked], d["jam_size"][attacked], strict=True
    ):
        assert labels[start : start + size].all()
        assert labels.sum() == size * SimConfig().num_symbols


def test_cell_shares_pool_cells_and_count_unscored_interference_as_unflagged():
    from jamloc.experiments.neighbor import share_rows

    labels = np.array([[[1, 1], [1, 0]], [[1, 0], [1, 1]]], dtype=np.uint8)
    scores = np.array([[[2, 1], [-np.inf, 3]], [[2, -np.inf], [2, 2]]])
    features = np.zeros((2, len(FEATURE_NAMES), 2, 2))
    features[:, FEATURE_NAMES.index("alloc_mask"), 0, :] = 1
    rows = share_rows(scores, {"labels": labels, "features": features, "jsr_db": [-3, 0]}, 1)

    first, second, pooled = rows
    assert first["isr_db"] == -3 and second["isr_db"] == 0 and pooled["isr_db"] == "all"
    assert [r["scheduled"] for r in rows] == [0.5, 1, 2 / 3]
    assert [r["unscheduled"] for r in rows] == [0, 1, 2 / 3]
    assert [r["fa"] for r in rows] == [1, 0, 0.5]
    assert first["fa_scored"] == pooled["fa_scored"] == 1
    assert np.isnan(second["fa_scored"])
