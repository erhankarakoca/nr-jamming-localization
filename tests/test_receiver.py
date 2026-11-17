import torch
from sionna.phy import config
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import TDL

from jamloc.config import SimConfig
from jamloc.receiver import PUSCHReceiverChain, data_positions
from jamloc.ue import UEPool

cfg = SimConfig()
pool = UEPool(cfg)


def test_noiseless_identity():
    tx = pool.transmitter(6)
    rx = PUSCHReceiverChain(tx)
    x, b = tx(4)
    out = rx(x, torch.full((4,), 1e-4, device=config.device))

    sym, sc = data_positions(tx)
    x_data = x[:, 0, 0, sym, sc]
    assert out["x_hat"].shape == x_data.shape
    assert torch.allclose(out["x_hat"], x_data, atol=1e-2)
    assert torch.equal(out["b_hat"], b)


def test_benign_tdl_high_snr_decodes():
    config.seed = 1
    tx = pool.transmitter(8)
    rx = PUSCHReceiverChain(tx)
    tdl = TDL("A", cfg.delay_spread["A"], cfg.carrier_frequency, min_speed=0.0,
              max_speed=cfg.max_speed)
    channel = OFDMChannel(tdl, tx.resource_grid, normalize_channel=True)
    no = torch.full((16,), 10 ** (-30 / 10), device=config.device)

    x, b = tx(16)
    y = channel(x, no)
    out = rx(y, no)
    assert out["tb_crc"].all()
    assert torch.equal(out["b_hat"], b)
