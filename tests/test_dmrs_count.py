import csv
from itertools import product

import numpy as np
import pytest
import torch
from scipy.stats import gamma
from sionna.phy import config

from jamloc.baselines import block_test_scores, comb_scores, residual_cfar_scores
from jamloc.config import SimConfig
from jamloc.features import FEATURE_NAMES, comb_map, dmrs_ls, evm_map, masks, residual_map
from jamloc.jammer import Jammer, cell_mask
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.recovery import MapAwareReceiver, lmmse_interpolation
from jamloc.ue import UEPool


@pytest.mark.parametrize("position", range(4))
def test_dmrs_symbols_match_sionna_pilot_pattern(position):
    cfg = SimConfig(dmrs_additional_position=position)
    tx = UEPool(cfg).transmitter(6)
    pattern = tx.resource_grid.pilot_pattern.mask[0, 0]
    symbols = tuple(pattern.any(dim=1).nonzero().flatten().tolist())
    assert cfg.dmrs_symbols == symbols
    assert cfg.dmrs_symbol == symbols[0]
    assert len(symbols) == position + 1
    if position == 1:
        assert symbols == (2, 11)


@pytest.mark.parametrize("jammer_dmrs", ("all", "first"))
@pytest.mark.parametrize(
    ("waveform", "pilot_comb"),
    (("gaussian", False), ("gaussian", True), ("ue", False), ("dmrs", False)),
)
def test_pilot_jammer_transmits_on_exactly_the_labelled_cells(jammer_dmrs, waveform, pilot_comb):
    cfg = SimConfig(
        dmrs_additional_position=1,
        jammer_dmrs=jammer_dmrs,
        jammer_waveform=waveform,
        jammer_pilot_comb=pilot_comb,
    )
    seed_everything(43)
    attacks = torch.tensor([1, 2, 3], device=config.device)
    cells = cell_mask(attacks, torch.full_like(attacks, 4), torch.full_like(attacks, 3), cfg)
    sent = torch.zeros_like(cells)
    dmrs = cfg.dmrs_symbols
    sent[0, 4:7, dmrs if jammer_dmrs == "all" else dmrs[:1]] = True
    sent[1, 4:7] = True
    sent[1, 4:7, dmrs] = False
    sent[2, 4:7] = True
    assert torch.equal(cells, sent)
    x, _ = UEPool(cfg).transmitter(cfg.num_prb)(len(attacks))
    z = Jammer(cfg)(cells, torch.zeros(len(attacks), device=config.device), x)[:, 0, 0]
    assert torch.equal(
        z.reshape(len(attacks), cfg.num_symbols, cfg.num_prb, 12).ne(0).any(-1).transpose(1, 2),
        cells,
    )
    if pilot_comb or waveform != "gaussian":
        assert torch.count_nonzero(z[:, dmrs, 1::2]) == 0
    else:
        assert torch.count_nonzero(z[2, dmrs, 49:84:2]) == len(dmrs) * 3 * 6
    if waveform == "dmrs":
        ratio = z[2, dmrs, 48:84:2] / x[2, 0, 0, dmrs, 48:84:2]
        torch.testing.assert_close(ratio, ratio[0, 0].expand_as(ratio))


def test_sparse_pilot_comb_labels_only_transmitted_cells():
    cfg = SimConfig(dmrs_additional_position=1, jammer_pilot_comb=True, jammer_tones=1)
    rng = seed_everything(59)
    sc = draw_scenario(rng, 16, cfg, "A", attacks=(1,), jam_size=8, alloc=(6, 12))
    link = JammedPUSCHLink(cfg)
    grids = []
    hook = link.jammer.register_forward_hook(lambda module, args, z: grids.append(z))
    try:
        batch = link(sc, receive=False)
    finally:
        hook.remove()
    z = grids[0][:, 0, 0].reshape(sc.batch_size, cfg.num_symbols, cfg.num_prb, 12)
    assert torch.equal(batch.labels, z.ne(0).any(dim=-1).transpose(1, 2))
    assert 0 < batch.labels.sum() < sc.batch_size * 8 * len(cfg.dmrs_symbols)


@pytest.mark.parametrize("position", (1, 2, 3))
def test_residual_and_empty_comb_are_calibrated_on_each_dmrs_symbol(position):
    cfg = SimConfig(dmrs_additional_position=position)
    rng = seed_everything(52)
    sc = draw_scenario(rng, 256, cfg, "C", attacks=(0,), snr_db=15, alloc=(6, 12))
    batch = JammedPUSCHLink(cfg)(sc, decode=True)
    dmrs, data = cfg.dmrs_symbols, cfg.data_symbols
    res = residual_map(batch, cfg)
    comb = comb_map(batch, cfg)
    assert torch.count_nonzero(res[:, :, data]) == torch.count_nonzero(comb[:, :, data]) == 0
    assert torch.all((res[:, 6:18, dmrs].mean(dim=(0, 1)) - 1).abs() < 0.05)
    assert torch.all((comb[:, :, dmrs].mean(dim=(0, 1)) - 1).abs() < 0.05)
    alloc_mask, dmrs_mask, data_mask = masks(batch, cfg)
    assert torch.equal(dmrs_mask[:, :, dmrs], alloc_mask[:, :, dmrs])
    assert torch.equal(data_mask[:, :, data], alloc_mask[:, :, data])
    assert torch.count_nonzero(evm_map(batch, cfg)[:, :, dmrs]) == 0
    assert batch.rx["tb_crc"].float().mean() > 0.99


def test_one_dmrs_lmmse_matches_original_interpolation():
    cfg = SimConfig()
    rng = seed_everything(53)
    sc = draw_scenario(rng, 8, cfg, "C", snr_db=15, alloc=(6, 12))
    batch = JammedPUSCHLink(cfg)(sc, receive=False)
    h_ls, pilot_sc, pilot_power = dmrs_ls(batch.y_alloc, batch.tx)
    h_sc, err_sc = lmmse_interpolation(
        h_ls,
        pilot_sc,
        torch.ones_like(h_ls, dtype=torch.bool),
        batch.no / pilot_power,
        batch.y_alloc.shape[-1],
        cfg.scs,
        cfg.tau_max,
    )
    actual = MapAwareReceiver(batch.tx, cfg)(
        batch.y_alloc, batch.no, None, decode=False, mode="lmmse"
    )
    assert torch.equal(actual["h_hat"][:, 0, 0, 0, 0, 0], h_sc)
    assert torch.equal(actual["err_var"][:, 0, 0, 0, 0, 0], err_sc)


def test_soft_channel_estimate_downweights_the_jammed_dmrs_symbol():
    cfg = SimConfig(dmrs_additional_position=1, jammer_dmrs="first")
    seed_everything(54)
    tx = UEPool(cfg).transmitter(12)
    x, _ = tx(16)
    device = config.device
    # A slot-constant delayed path plus AWGN and a jammer on four PRBs of the first DMRS.
    h = torch.exp(-2j * torch.pi * torch.arange(x.shape[-1], device=device) * cfg.scs * 1e-6)
    no = torch.full((len(x),), 10 ** (-15 / 10), device=device)
    cells = torch.zeros(len(x), 12, cfg.num_symbols, dtype=torch.bool, device=device)
    cells[:, 4:8, cfg.dmrs_symbol] = True
    noise = torch.randn_like(x) * no.sqrt()[:, None, None, None, None]
    jam = torch.randn_like(x[..., cfg.dmrs_symbol, 48:96]) * 10 ** (9 / 20)
    y = x * h + noise
    y[..., cfg.dmrs_symbol, 48:96] += jam
    rx = MapAwareReceiver(tx, cfg)
    errors = {}
    for mode in ("lmmse", "soft", "hard", "dmrs", "slot"):
        out = rx(y, no, cells, decode=False, mode=mode)
        h_hat = out["h_hat"][:, 0, 0, 0, 0]
        assert torch.equal(h_hat[:, :1].expand_as(h_hat), h_hat)
        assert torch.isfinite(out["llr"]).all()
        errors[mode] = (h_hat - h).abs().square().mean()
    assert errors["soft"] < 0.25 * errors["lmmse"]
    assert errors["hard"] < 0.25 * errors["lmmse"]
    cells[:, :, cfg.dmrs_symbols] = True
    fallback = rx(y, no, cells, decode=False, mode="hard")
    standard_h, standard_err = rx.standard_estimator(y, no)
    assert torch.equal(fallback["h_hat"], standard_h)
    assert torch.equal(fallback["err_var"], standard_err.expand_as(fallback["err_var"]))


def test_dmrs2_block_test_uses_individual_pilots_and_the_data_null(tmp_path, monkeypatch):
    from jamloc.dataset import generate, load

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    generate("val_dmrs2", num_slots=96, batch_size=32)
    val = load("val_dmrs2")
    cfg = SimConfig(dmrs_additional_position=1)
    # On clean scheduled data cells, the empirical EVM tail is close to its 1% nominal level.
    clean_data = (val["attack"][:, None, None] == 0) & (
        val["features"][:, FEATURE_NAMES.index("data_mask")] > 0
    )
    clean_scores = block_test_scores(val, val)[clean_data]
    assert 0 < (clean_scores > 2).mean() < 0.04
    first, second = cfg.dmrs_symbols
    slot, prb = np.argwhere(val["features"][:, FEATURE_NAMES.index("alloc_mask"), :, first] == 0)[0]
    on = val["alloc_start"][slot]
    d = {k: v.copy() for k, v in val.items()}
    d["features"][slot, FEATURE_NAMES.index("power_db"), prb, cfg.data_symbols] = 3.0
    for feature in ("residual_db", "comb_db"):
        d["features"][slot, FEATURE_NAMES.index(feature), on, [first, second]] = [0, 15]
    scores = block_test_scores(d, val)
    assert np.isfinite(scores).all()
    assert scores[slot, on, second] > scores[slot, on, first]
    assert residual_cfar_scores(d)[slot, on, second] > residual_cfar_scores(d)[slot, on, first]
    assert (
        comb_scores(d["features"])[slot, on, second] > comb_scores(d["features"])[slot, on, first]
    )
    m = 12 * len(cfg.data_symbols)
    expected = -gamma.logsf(10**0.3, a=m, scale=1 / m) / np.log(10)
    np.testing.assert_allclose(scores[slot, prb, cfg.data_symbols], expected, rtol=1e-5)
    np.testing.assert_allclose(
        scores[..., cfg.data_symbols],
        scores[..., cfg.data_symbols[:1]].repeat(len(cfg.data_symbols), axis=-1),
    )


def test_dmrs_count_small(tmp_path, monkeypatch):
    from jamloc.experiments import dmrs_count

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    dmrs_count.main(["--small", "--seeds", "0", "--batch-size", "2", "--batches", "1"])
    out = tmp_path / "results" / "dmrs_count"
    with (out / "detection.csv").open(encoding="utf-8") as f:
        detection = list(csv.DictReader(f))
    assert len(detection) == 8
    assert {(r["dmrs_count"], r["detector"]) for r in detection} == set(
        product(("1", "2"), ("fused", "no_residual", "block_test", "evm_threshold"))
    )
    assert (out / "detection.md").is_file()
    with (out / "recovery.csv").open(encoding="utf-8") as f:
        recovery = list(csv.DictReader(f))
    cases = {(str(n), "all", a) for n, a in product((1, 2), ("pilot", "data", "partial"))}
    cases.add(("2", "first", "pilot"))
    assert {(r["dmrs_count"], r["jammer_dmrs"], r["attack"]) for r in recovery} == cases
    assert {float(r["jsr_db"]) for r in recovery} == {-15, -3, 3, 9}
    assert {r["receiver"] for r in recovery} == {"standard", "lmmse", "p_r_e_soft", "true_soft"}
    assert len(recovery) == len(cases) * 4 * 4 and {int(r["blocks"]) for r in recovery} == {2}

    # Detection only: the tables are rewritten, the recovery results of the earlier run are kept
    before = (out / "recovery.csv").read_bytes()
    (out / "detection.md").unlink()
    dmrs_count.main(["--small", "--seeds", "0", "--no-recovery"])
    assert (out / "detection.md").is_file()
    assert (out / "recovery.csv").read_bytes() == before
