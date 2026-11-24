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
    protected = MapAwareReceiver(out.tx)(out.y_alloc, out.no, _cells(out), history=True)
    standard = PUSCHReceiverChain(out.tx)(out.y_alloc, out.no)
    assert torch.isfinite(protected["llr"]).all()
    assert torch.equal(protected["b_hat"], standard["b_hat"])
    (first,) = protected["history"]
    assert torch.equal(first["h_sc"], standard["h_hat"][:, 0, 0, 0, 0, cfg.dmrs_symbol])
    assert torch.equal(first["err_var"], standard["err_var"][:, 0, 0, 0, 0, cfg.dmrs_symbol])


def test_carrier_sized_map_is_rejected():
    import pytest

    out = link(_scenario(DATA, 0.0, 15.0, n=4))
    with pytest.raises(ValueError, match="allocation"):
        MapAwareReceiver(out.tx)(
            out.y_alloc, out.no, out.labels
        )  # [B, 24, 14] instead of [B, 12, 14]


def test_soft_mode_ignores_false_alarms():
    # Every cell flagged on clean slots: hard erasure would lose everything, the soft receiver
    # measures no jamming power and decodes as usual
    out = link(_scenario(NONE, 0.0, 15.0, n=32))
    everything = torch.ones_like(_cells(out))
    rx = MapAwareReceiver(out.tx)
    assert _bler(rx(out.y_alloc, out.no, everything, mode="soft")) == 0.0
    assert _bler(rx(out.y_alloc, out.no, everything, mode="hard")) == 1.0


def test_soft_mode_estimates_the_jamming_power():
    from jamloc.features import residual_per_prb

    out = link(_scenario(DATA, 6.0, 15.0, n=64))
    cells = _cells(out)
    h_ls, pilot_sc, pilot_power = dmrs_ls(out.y_alloc, out.tx)
    r = residual_per_prb(h_ls, pilot_sc, out.no, pilot_power, cfg)
    every_pilot = torch.ones_like(h_ls, dtype=torch.bool)
    num_sc = out.y_alloc.shape[-1]
    h_sc, err_sc = lmmse_interpolation(
        h_ls, pilot_sc, every_pilot, out.no / pilot_power, num_sc, cfg.scs, cfg.tau_max
    )
    rx = MapAwareReceiver(out.tx)
    var = rx._jamming_variance(out.y_alloc, out.no, h_sc, err_sc, cells, r, None, "soft")
    sym, sc = data_positions(out.tx)
    flagged = cells[:, sc // 12, sym]
    # Jammer energy per RE is 10^0.6 ~ 4, seen through a unit-power fading channel
    assert 2.5 < var[flagged].mean().item() < 6.0
    assert torch.all(var[~flagged] == 0)


def test_soft_and_dmrs_modes_beat_unprotected():
    for attack, modes in ((PILOT, ("soft", "dmrs")), (DATA, ("soft",))):
        out = link(_scenario(attack, 6.0, 15.0), decode=True)
        for mode in modes:
            protected = MapAwareReceiver(out.tx)(out.y_alloc, out.no, _cells(out), mode=mode)
            assert _bler(protected) < _bler(out.rx), (ATTACKS[attack], mode)


def test_dmrs_mode_decodes_clean_slots_and_needs_no_map():
    out = link(_scenario(NONE, 0.0, 15.0, n=32))
    assert _bler(MapAwareReceiver(out.tx)(out.y_alloc, out.no, None, mode="dmrs")) == 0.0


def test_unknown_mode_is_rejected():
    import pytest

    out = link(_scenario(NONE, 0.0, 15.0, n=2))
    with pytest.raises(ValueError, match="mode"):
        MapAwareReceiver(out.tx)(out.y_alloc, out.no, _cells(out), mode="gentle")


def test_soft_decoder_returns_the_transmitted_codeword():
    out = link(_scenario(NONE, -30.0, 30.0, n=8), decode=True)
    rx = MapAwareReceiver(out.tx)
    llr = rx(out.y_alloc, out.no, _cells(out), mode="soft")["llr"]
    post = rx.soft_decoder(rx.layer_demapper(llr))
    codeword = out.tx._tb_encoder(out.b)
    assert post.shape == codeword.shape
    assert torch.equal(post > 0, codeword > 0.5)  # LLR > 0 means bit 1


def test_soft_qpsk_matches_the_constellation():
    from sionna.phy.mapping import Constellation

    from jamloc.recovery import soft_qpsk

    points = Constellation("qam", 2).points  # on Sionna's device, so the LLRs go there too
    bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=config.device)
    mean, var = soft_qpsk(40.0 * (2 * bits - 1))  # confident LLRs of each point's label
    assert torch.allclose(mean, points, atol=1e-6)
    assert torch.allclose(var, torch.zeros_like(var), atol=1e-6)
    mean, var = soft_qpsk(torch.zeros(1, 2, device=config.device))  # nothing known
    assert mean.abs().item() == 0 and var.item() == 1  # mean 0, unit variance


def test_sum_over_symbols_matches_a_sum_per_subcarrier():
    from jamloc.receiver import data_positions
    from jamloc.recovery import sum_over_symbols

    # In the data cells of a 12-PRB slot, every subcarrier appears once per data symbol
    tx = UEPool(cfg).transmitter(ALLOC[1])
    sym, sc = data_positions(tx)
    num_sc = 12 * ALLOC[1]
    gen = torch.Generator(device=config.device).manual_seed(0)
    values = torch.randn(3, len(sc), generator=gen, device=config.device)
    out = sum_over_symbols(values, sym, sc, cfg.num_symbols, num_sc)
    assert out.shape == (3, num_sc)
    for k in range(num_sc):
        assert torch.allclose(out[:, k], values[:, sc == k].sum(1), rtol=1e-5, atol=1e-6)
    # Same inputs, same bits, also on a GPU
    assert torch.equal(out, sum_over_symbols(values, sym, sc, cfg.num_symbols, num_sc))


def test_data_aided_passes_rescue_blocks_without_clean_pilots():
    sc = _scenario(PILOT, 9.0, 15.0, n=32, jam=ALLOC)  # every DMRS cell of the allocation
    out = link(sc, decode=True)
    rx = MapAwareReceiver(out.tx)
    first = rx(out.y_alloc, out.no, _cells(out), mode="soft")
    aided = rx(out.y_alloc, out.no, _cells(out), mode="soft", iterations=3)
    assert (aided["tb_crc"] | ~first["tb_crc"]).all()  # a decoded block stays decoded
    assert aided["tb_crc"].sum() > first["tb_crc"].sum()  # and some failed ones are rescued
    passed = first["tb_crc"].reshape(-1)
    assert torch.equal(aided["b_hat"][passed], first["b_hat"][passed])


def test_history_records_each_data_aided_pass():
    sc = _scenario(PILOT, 9.0, 15.0, n=8, jam=ALLOC)
    batch = link(sc, decode=True)
    rx = MapAwareReceiver(batch.tx)
    args = batch.y_alloc, batch.no, _cells(batch)
    recorded = rx(*args, mode="soft", iterations=3, history=True)
    plain = rx(*args, mode="soft", iterations=3)
    assert recorded.keys() - plain.keys() == {"history"}
    assert all(torch.equal(recorded[k], plain[k]) for k in plain)

    history = recorded["history"]
    assert 1 < len(history) <= 4
    for index, step in enumerate(history):
        current = rx(*args, mode="soft", iterations=index)
        assert step["h_sc"].shape == step["err_var"].shape == (8, 144)
        assert step["x_hat"].shape == (8, 1872) and step["tb_crc"].shape == (8,)
        for key in ("x_hat", "llr"):
            assert torch.equal(step[key], current[key])
        for key, grid in (("h_sc", "h_hat"), ("err_var", "err_var")):
            assert torch.equal(step[key], current[grid][:, 0, 0, 0, 0, cfg.dmrs_symbol])
        cumulative = torch.stack([p["tb_crc"] for p in history[: index + 1]]).any(0)
        assert torch.equal(cumulative, current["tb_crc"].flatten())


def test_history_stops_when_all_slots_pass():
    out = link(_scenario(NONE, 0.0, 25.0, n=2), decode=True)
    rx = MapAwareReceiver(out.tx)
    decoded = rx(out.y_alloc, out.no, _cells(out), mode="soft", iterations=3, history=True)
    assert decoded["tb_crc"].all()
    assert len(decoded["history"]) == 1


def test_data_aided_passes_need_decoding():
    out = link(_scenario(PILOT, 9.0, 15.0, n=4, jam=ALLOC), decode=True)
    rx = MapAwareReceiver(out.tx)
    plain = rx(out.y_alloc, out.no, _cells(out), mode="soft", decode=False)
    passes = rx(out.y_alloc, out.no, _cells(out), mode="soft", decode=False, iterations=3)
    assert torch.equal(plain["llr"], passes["llr"]) and "b_hat" not in passes


def test_decoded_symbols_reproduce_the_transmitted_grid():
    out = link(_scenario(NONE, 0.0, 30.0, n=8), decode=False)
    rx = MapAwareReceiver(out.tx)
    x = rx.decoded_symbols(out.b)
    sym, sc = data_positions(out.tx)
    ratio = out.y_alloc[:, 0, 0, sym, sc] / x  # the channel, up to noise
    spread = (ratio - ratio.mean(dim=1, keepdim=True)).abs().mean()
    assert x.shape[1] == sym.numel()
    assert (ratio[:, 1:] - ratio[:, :-1]).abs().mean() < 0.2 * spread  # smooth across the grid


def test_slot_residual_has_unit_mean_without_jamming():
    out = link(_scenario(NONE, 0.0, 15.0, n=64), decode=False)
    rx = MapAwareReceiver(out.tx)
    r, count = rx.slot_residuals(out.y_alloc, out.no, rx.decoded_symbols(out.b))
    assert abs(r.mean().item() - 1) < 0.05
    dmrs = list(cfg.dmrs_symbols)
    assert (count[:, :, dmrs] == 6).all()  # the pilots of the DMRS comb
    data = [sym for sym in range(cfg.num_symbols) if sym not in dmrs]
    assert (count[:, :, data] == 12).all()  # every subcarrier of a data cell


def test_decoded_cell_test_finds_the_jammed_cells_of_any_support():
    out = link(_scenario(DATA, 9.0, 15.0, n=32), decode=False)  # all data symbols of 4 PRBs
    rx = MapAwareReceiver(out.tx)
    flags, _, _ = rx.decoded_cell_test(out.y_alloc, out.no, rx.decoded_symbols(out.b))
    jammed = _cells(out).bool()
    data = torch.ones_like(jammed)
    data[:, :, list(cfg.dmrs_symbols)] = False
    assert (flags & jammed & data).sum() > 0.95 * (jammed & data).sum()
    assert (flags & ~jammed & data).float().sum() < 0.03 * (~jammed & data).sum()


def test_refine_adds_the_jammed_cells_a_map_missed():
    out = link(_scenario(DATA, 9.0, 15.0, n=16), decode=False)
    rx = MapAwareReceiver(out.tx)
    x = rx.decoded_symbols(out.b)  # what a decoder that is sure of every symbol returns
    first = rx(out.y_alloc, out.no, torch.zeros_like(_cells(out)), mode="soft", decode=False)
    h_hat, err_var = first["h_hat"], first["err_var"]
    cells = rx._refine(
        out.y_alloc,
        out.no,
        h_hat,
        err_var,
        x,
        torch.zeros_like(x.real),
        torch.zeros_like(_cells(out)),
        0.01,
    )
    jammed = _cells(out).bool()
    data = torch.ones_like(jammed)
    data[:, :, list(cfg.dmrs_symbols)] = False
    assert (cells & jammed & data).sum() > 0.95 * (jammed & data).sum()
    assert (cells & ~jammed).float().sum() < 0.05 * (~jammed & data).sum()


def test_renoise_recovers_the_noise_power_from_decoded_data():
    out = link(_scenario(NONE, 0.0, 15.0, n=16), decode=False)
    rx = MapAwareReceiver(out.tx)
    x = rx.decoded_symbols(out.b)  # a decoder that is sure of every symbol
    sure = torch.zeros_like(x.real)
    no = out.no * 10  # e.g. an estimate from a jammed empty comb
    cells = torch.zeros_like(_cells(out))
    for _ in range(2):  # what two passes do
        first = rx(out.y_alloc, no, cells, mode="soft", decode=False)
        h_ls, pilot_sc, power = dmrs_ls(out.y_alloc, out.tx)
        clean, pilot_var, _, _ = rx._pilots(h_ls, pilot_sc, power, no, cells, "soft")
        h_sc, err_sc = rx._data_aided(
            out.y_alloc,
            no,
            h_ls,
            pilot_sc,
            clean,
            pilot_var,
            first["h_hat"],
            x,
            sure,
            torch.zeros_like(sure),
        )
        no = rx._renoise(
            out.y_alloc, no, *rx._on_grid(h_sc, err_sc, out.y_alloc), x, sure, cells, "soft"
        )
    error_db = (no / out.no).log10().abs() * 10
    assert error_db.max() < 0.5


def test_shrinkage_pools_a_filled_row_and_separates_a_moving_jammer():
    out = link(_scenario(NONE, 0.0, 15.0, n=32), decode=False)
    rx = MapAwareReceiver(out.tx)
    y = out.y_alloc.clone()
    data_symbols = list(cfg.data_symbols)
    moving = data_symbols[::2]  # PRB 8 is jammed on every other data symbol
    jam = torch.zeros(y.shape[-2:], dtype=torch.bool, device=y.device)
    jam[data_symbols, 2 * 12 : 6 * 12] = True  # PRBs 2-5 on every data symbol
    jam[moving, 8 * 12 : 9 * 12] = True
    torch.manual_seed(3)
    noise = (torch.randn_like(y.real) + 1j * torch.randn_like(y.real)) * 2**0.5  # power 4
    y = torch.where(jam, y + noise, y)
    cells = torch.zeros_like(_cells(out))
    cells[:, 2:6, data_symbols] = 1
    cells[:, 8, :] = 1  # the map flags the whole row of the moving jammer

    h_ls, pilot_sc, pilot_power = dmrs_ls(y, out.tx)
    h_sc, err_sc = lmmse_interpolation(
        h_ls,
        pilot_sc,
        torch.ones_like(h_ls, dtype=torch.bool),
        out.no / pilot_power,
        y.shape[-1],
        cfg.scs,
        cfg.tau_max,
    )
    estimates = {}
    for option in (False, True, "shrink"):
        rx._cellwise = option
        estimates[option] = rx._jamming_variance(y, out.no, h_sc, err_sc, cells, None, None, "soft")
    sym, sc = data_positions(out.tx)
    prb = sc // 12
    hit = jam[sym, sc]
    row8 = prb == 8
    # The moving jammer: shrinkage keeps the jammed and the clean cells of its row apart, the
    # row average gives both about half the power
    shrink = estimates["shrink"]
    assert shrink[:, row8 & hit].mean() > 3.0
    assert shrink[:, row8 & ~hit].mean() < 0.5
    assert estimates[False][:, row8 & ~hit].mean() > 1.0
    # The filled rows: shrinkage stays close to the row average, much closer than the per-cell
    # estimates are
    filled = (prb >= 2) & (prb < 6)
    to_row = (shrink[:, filled] - estimates[False][:, filled]).abs().mean()
    per_cell = (estimates[True][:, filled] - estimates[False][:, filled]).abs().mean()
    assert to_row < 0.3 * per_cell


def test_options_for_a_receiver_without_a_map():
    import pytest

    out = link(_scenario(DATA, 6.0, 15.0, n=16), decode=False)
    rx = MapAwareReceiver(out.tx)
    h_ls, pilot_sc, pilot_power = dmrs_ls(out.y_alloc, out.tx)
    h_sc, err_sc = lmmse_interpolation(
        h_ls,
        pilot_sc,
        torch.ones_like(h_ls, dtype=torch.bool),
        out.no / pilot_power,
        out.y_alloc.shape[-1],
        cfg.scs,
        cfg.tau_max,
    )
    args = (out.y_alloc, out.no, h_sc, err_sc)
    # Without a map, the per-PRB average treats every data cell as flagged, as a map that flags
    # everything does
    rx._cellwise = "prb"
    no_map = rx._jamming_variance(*args, None, None, None, "cell")
    rx._cellwise = False
    everything = torch.ones_like(_cells(out))
    assert torch.allclose(no_map, rx._jamming_variance(*args, everything, None, None, "soft"))

    # The scale multiplies the jamming power, so a larger one gives every LLR less weight
    def llr(scale):
        options = {"cellwise": "prb", "scale": scale}
        out_rx = rx(out.y_alloc, out.no, None, decode=False, mode="cell", history=True, **options)
        return out_rx["history"][0]["llr"].abs()

    one, two = llr(1.0), llr(2.0)
    assert torch.all(two <= one + 1e-4) and (two < one - 1e-3).any()

    # Every pilot weighted by its residual, or only those above the analytic threshold
    rx._pilot_weights = "all"
    _, pilot_var, _, interfered = rx._pilots(h_ls, pilot_sc, pilot_power, out.no, None, "cell")
    assert interfered.all()
    assert torch.all(pilot_var >= (out.no / pilot_power)[:, None] * (1 - 1e-6))
    rx._pilot_weights = "test"
    _, _, _, tested = rx._pilots(h_ls, pilot_sc, pilot_power, out.no, None, "cell")
    assert tested.float().mean() < interfered.float().mean()

    with pytest.raises(ValueError, match="cellwise"):
        rx(out.y_alloc, out.no, None, decode=False, mode="cell", cellwise="row")
    with pytest.raises(ValueError, match="pilot_weights"):
        rx(out.y_alloc, out.no, None, decode=False, mode="cell", pilot_weights="some")
