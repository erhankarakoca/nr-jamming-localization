import pytest
import torch

from jamloc.config import SimConfig
from jamloc.features import (
    FEATURE_NAMES,
    delay_subspace,
    dmrs_ls,
    evm_map,
    extract,
    masks,
    residual_map,
)
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, Scenario, draw_scenario, seed_everything

cfg = SimConfig()
link = JammedPUSCHLink(cfg)
NONE, PILOT, DATA = (ATTACKS.index(a) for a in ("none", "pilot", "data"))


def _batch(seed, channel, attack, snr_db, jsr_db=0.0, batch=64, **kw):
    rng = seed_everything(seed)
    sc = draw_scenario(
        rng, batch, cfg, channel, attacks=(attack,), jsr_db=jsr_db, snr_db=snr_db, **kw
    )
    return link(sc)


def test_delay_subspace():
    out = _batch(0, "C", NONE, snr_db=60.0, batch=16)
    h_ls, pilot_sc, _ = dmrs_ls(out.y_alloc, out.tx)
    u = delay_subspace(pilot_sc, cfg.scs, cfg.tau_max)
    n_p, n_basis = u.shape
    assert n_basis < n_p == pilot_sc.numel()
    assert torch.allclose(u.conj().T @ u, torch.eye(n_basis, dtype=u.dtype), atol=1e-4)

    # A noiseless TDL-C channel lies inside the subspace
    residual = h_ls - (h_ls @ u.conj()) @ u.T
    leak = (residual.abs() ** 2).sum() / (h_ls.abs() ** 2).sum()
    assert leak < 1e-4


def test_benign_residual_is_noise_level():
    out = _batch(1, "C", NONE, snr_db=30.0)
    _, dmrs, _ = masks(out, cfg)
    r = residual_map(out, cfg)
    assert 0.7 < r[dmrs.bool()].mean().item() < 1.4


def test_benign_evm_is_noise_level():
    out = _batch(2, "A", NONE, snr_db=15.0)
    _, _, data = masks(out, cfg)
    e = evm_map(out, cfg)
    assert 0.5 < e[data.bool()].mean().item() < 1.5


def test_pilot_attack_raises_residual():
    pilot = _batch(3, "A", PILOT, snr_db=20.0, scheduling_aware=True)
    _, dmrs, _ = masks(pilot, cfg)
    assert residual_map(pilot, cfg)[dmrs.bool()].mean().item() > 5.0


def test_data_attack_leaves_residual_at_noise_level():
    data_attack = _batch(4, "A", DATA, snr_db=20.0, scheduling_aware=True)
    _, dmrs, data = masks(data_attack, cfg)
    assert 0.8 < residual_map(data_attack, cfg)[dmrs.bool()].mean().item() < 1.25
    assert evm_map(data_attack, cfg)[data.bool()].mean().item() > 3.0


def test_residual_stays_low_away_from_jammed_prbs():
    # Full-band allocation, pilot jamming on PRBs 0-5 only
    seed_everything(6)
    n = 32
    sc = Scenario(
        channel="A",
        alloc_start=0,
        alloc_size=24,
        attack=torch.full((n,), PILOT),
        jam_start=torch.zeros(n, dtype=torch.long),
        jam_size=torch.full((n,), 6),
        jsr_db=torch.zeros(n),
        snr_db=torch.full((n,), 20.0),
    )
    r = residual_map(link(sc), cfg)[:, :, cfg.dmrs_symbol]  # [B, 24]
    assert r[:, :6].mean().item() > 5.0
    assert r[:, 8:].mean().item() < 1.5  # two or more PRBs away from the jammed block


@pytest.mark.parametrize("alloc", [(0, 6), (18, 6)])
def test_features_finite_and_masked(alloc):
    out = _batch(5, "C", 3, snr_db=0.0, jsr_db=20.0, batch=16, alloc=alloc)
    f = extract(out, cfg)
    assert f.shape == (16, len(FEATURE_NAMES), 24, 14) and f.dtype == torch.float32
    assert torch.isfinite(f).all()
    names = list(FEATURE_NAMES)
    r, e = f[:, names.index("residual_db")], f[:, names.index("evm_db")]
    dmrs, data = f[:, names.index("dmrs_mask")], f[:, names.index("data_mask")]
    assert torch.all(r[dmrs == 0] == 0) and torch.all(e[data == 0] == 0)
    start, size = alloc
    alloc_rows = f[:, names.index("alloc_mask")].amax(dim=2)  # [B, 24]
    assert alloc_rows[:, start : start + size].all() and alloc_rows.sum(1).eq(size).all()


def test_benign_residual_is_flat_across_the_allocation():
    # Edge PRBs keep less noise after the projection; the normalization must account for it
    rng = seed_everything(8)
    sc = draw_scenario(rng, 256, cfg, "A", attacks=(NONE,), snr_db=20.0, alloc=(6, 12))
    r = residual_map(link(sc), cfg)[:, 6:18, cfg.dmrs_symbol].mean(0)
    assert torch.all((r > 0.9) & (r < 1.12)), r


def test_evm_map_matches_per_cell_average():
    from jamloc.receiver import data_positions

    out = _batch(9, "A", DATA, snr_db=15.0, batch=4, alloc=(3, 6))
    e = evm_map(out, cfg)
    x_hat, no_eff = out.rx["x_hat"], out.rx["no_eff"]
    decision = (
        torch.where(x_hat.real >= 0, 1.0, -1.0) + 1j * torch.where(x_hat.imag >= 0, 1.0, -1.0)
    ) / 2**0.5
    error = (x_hat - decision).abs() ** 2
    sym, sc = data_positions(out.tx)
    for prb in range(6):
        for s in range(14):
            cell = (sc // 12 == prb) & (sym == s)
            if s == cfg.dmrs_symbol:
                assert torch.all(e[:, 3 + prb, s] == 0)
                continue
            ref = error[:, cell].sum(1) / no_eff[:, cell].sum(1)
            assert torch.allclose(e[:, 3 + prb, s], ref, rtol=1e-5)
