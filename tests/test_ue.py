import pytest
import torch

from jamloc.config import SimConfig
from jamloc.ue import UEPool, make_pusch_config, place_on_carrier, take_allocation

cfg = SimConfig()


@pytest.mark.parametrize("num_prb", [6, 24])
def test_pusch_config_matches_table1(num_prb):
    pc = make_pusch_config(cfg, num_prb)
    assert pc.dmrs_symbol_indices == [cfg.dmrs_symbol] == [2]
    assert pc.num_subcarriers == 12 * num_prb
    assert pc.tb.num_bits_per_symbol == 2  # QPSK


@pytest.mark.parametrize("start, num_prb", [(0, 6), (5, 8), (18, 6), (0, 24)])
def test_place_and_take_roundtrip(start, num_prb):
    x = torch.randn(3, 1, 1, 14, 12 * num_prb, dtype=torch.complex64) + 1
    grid = place_on_carrier(x, start, cfg)
    assert grid.shape[-1] == cfg.num_subcarriers == 288
    assert torch.equal(take_allocation(grid, start, num_prb), x)
    outside = torch.ones(288, dtype=torch.bool)
    outside[12 * start : 12 * (start + num_prb)] = False
    assert torch.all(grid[..., outside] == 0)


def test_transmitter_cached():
    pool = UEPool(cfg)
    assert pool.transmitter(8) is pool.transmitter(8)
    x, b = pool.transmitter(8)(2)
    assert x.shape == (2, 1, 1, 14, 96)
