"""Environment smoke test: checks that Sionna (PyTorch backend) is installed and
can produce a 5G NR PUSCH slot with the project's simulation parameters."""

import torch
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter

import jamloc


def test_package_imports():
    assert jamloc.__version__


def test_pusch_slot_matches_project_numerology():
    cfg = PUSCHConfig()
    cfg.carrier.subcarrier_spacing = 30  # kHz
    cfg.carrier.n_size_grid = 24  # PRBs on the carrier
    cfg.n_size_bwp = 24
    cfg.dmrs.config_type = 1  # comb-2
    cfg.dmrs.length = 1  # one front-loaded DMRS symbol
    cfg.tb.mcs_index = 5  # QPSK, r~0.37 (Sionna LDPC rejects r<1/5, i.e. MCS 0-2)

    tx = PUSCHTransmitter(cfg)
    x, b = tx(2)  # batch of 2 slots

    assert isinstance(x, torch.Tensor)
    # [batch, tx, antennas, OFDM symbols, subcarriers]
    assert x.shape[0] == 2
    assert x.shape[-2] == 14
    assert x.shape[-1] == 24 * 12
    assert b.shape[0] == 2
