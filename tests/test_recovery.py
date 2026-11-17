import torch
from sionna.phy import config
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import TDL

from jamloc.config import SimConfig
from jamloc.features import dmrs_ls
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, seed_everything
from jamloc.receiver import PUSCHReceiverChain, data_positions
from jamloc.recovery import MapAwareReceiver, lmmse_interpolation
from jamloc.ue import UEPool

cfg = SimConfig()
link = JammedPUSCHLink(cfg)
ALLOC = (6, 12)  # UE on PRBs 6-17
NONE, PILOT, DATA = (ATTACKS.index(a) for a in ("none", "pilot", "data"))


def _scenario(attack, jsr_db, snr_db, n=64, jam=(10, 4), seed=0):
    seed_everything(seed)
    return Scenario(
        channel="A",
        alloc_start=ALLOC[0],
        alloc_size=ALLOC[1],
        attack=torch.full((n,), attack),
        jam_start=torch.full((n,), jam[0]),
        jam_size=torch.full((n,), jam[1]),
        jsr_db=torch.full((n,), float(jsr_db)),
        snr_db=torch.full((n,), float(snr_db)),
    )


def _cells(batch):
    """True map cut to the allocation: [B, n, 14]."""
    return batch.labels[:, ALLOC[0] : ALLOC[0] + ALLOC[1]]


def _bler(out):
    return (~out["tb_crc"]).float().mean().item()


def _clean_slot(snr_db=40.0, n=8):
    """LS pilot estimates of a benign slot and the true channel on the DMRS symbol."""
    config.seed = 1
    tx = UEPool(cfg).transmitter(ALLOC[1])
    tdl = TDL("A", cfg.delay_spread["A"], cfg.carrier_frequency, max_speed=cfg.max_speed)
    channel = OFDMChannel(tdl, tx.resource_grid, normalize_channel=True, return_channel=True)
    no = torch.full((n,), 10 ** (-snr_db / 10), device=config.device)
    x, _ = tx(n)
    y, h = channel(x, no)
    h_ls, pilot_sc, pilot_power = dmrs_ls(y, tx)
    return h_ls, pilot_sc, no / pilot_power, h[:, 0, 0, 0, 0, cfg.dmrs_symbol]


def _interpolate(h_ls, pilot_sc, s2, unclean_prbs=()):
    clean = torch.ones_like(h_ls, dtype=torch.bool)
    for prb in unclean_prbs:
        clean[:, pilot_sc // 12 == prb] = False
    return lmmse_interpolation(h_ls, pilot_sc, clean, s2, 12 * ALLOC[1], cfg.scs, cfg.tau_max)


def test_interpolation_all_clean_recovers_channel():
    h_ls, pilot_sc, s2, h_true = _clean_slot()
    h_hat, err_var = _interpolate(h_ls, pilot_sc, s2)
    rel_err = (h_hat - h_true).abs() / h_true.abs().mean()
    assert rel_err.max() < 0.05
    assert err_var.max() < 0.05


def test_interpolation_gap_has_high_error_variance():
    h_ls, pilot_sc, s2, _ = _clean_slot()
    h_hat, err_var = _interpolate(h_ls, pilot_sc, s2, unclean_prbs=(4, 5, 6, 7))
    gap = torch.zeros(12 * ALLOC[1], dtype=torch.bool)
    gap[48:96] = True
    assert torch.isfinite(h_hat).all() and torch.isfinite(err_var).all()
    assert err_var[:, gap].mean() > 5 * err_var[:, ~gap].mean()


def test_interpolation_gap_at_edge_is_finite():
    h_ls, pilot_sc, s2, _ = _clean_slot()
    h_hat, err_var = _interpolate(h_ls, pilot_sc, s2, unclean_prbs=(0, 1, 2, 3))
    assert torch.isfinite(h_hat).all() and torch.isfinite(err_var).all()
    assert err_var.max() <= 1 + 1e-4 and err_var.min() >= 0


def test_empty_map_decodes_benign():
    out = link(_scenario(NONE, 0.0, 25.0, n=32))
    cells = torch.zeros_like(_cells(out))
    assert _bler(MapAwareReceiver(out.tx)(out.y_alloc, out.no, cells)) == 0.0


def test_few_false_alarms_still_decode():
    out = link(_scenario(NONE, 0.0, 25.0, n=32))
    cells = torch.zeros_like(_cells(out))
    gen = torch.Generator().manual_seed(0)
    for b in range(cells.shape[0]):
        prbs = torch.randint(0, ALLOC[1], (4,), generator=gen)
        syms = torch.tensor([3, 7, 11, cfg.dmrs_symbol])  # three data cells and one DMRS cell
        cells[b, prbs, syms] = True
    assert _bler(MapAwareReceiver(out.tx)(out.y_alloc, out.no, cells)) == 0.0


def test_erasure_zeroes_flagged_llrs():
    out = link(_scenario(DATA, 6.0, 15.0, n=8))
    cells = _cells(out)
    llr = MapAwareReceiver(out.tx)(out.y_alloc, out.no, cells, decode=False)["llr"][:, 0, 0]
    sym, sc = data_positions(out.tx)
    flagged = cells[:, sc // 12, sym].repeat_interleave(2, dim=1)  # two bits per QPSK symbol
    assert flagged.any()
    assert torch.all(llr[flagged] == 0)
    assert llr[~flagged].abs().sum() > 0


def test_true_map_beats_unprotected():
    for attack in (PILOT, DATA):
        out = link(_scenario(attack, 6.0, 15.0), decode=True)
        protected = MapAwareReceiver(out.tx)(out.y_alloc, out.no, _cells(out))
        assert _bler(protected) < _bler(out.rx), ATTACKS[attack]


def test_no_clean_pilot_falls_back():
    sc = _scenario(PILOT, -10.0, 15.0, n=16, jam=ALLOC)  # every DMRS cell of the allocation
    out = link(sc, decode=True)
    protected = MapAwareReceiver(out.tx)(out.y_alloc, out.no, _cells(out))
    standard = PUSCHReceiverChain(out.tx)(out.y_alloc, out.no)
    assert torch.isfinite(protected["llr"]).all()
    assert torch.equal(protected["b_hat"], standard["b_hat"])
