import torch
from sionna.phy import config

from jamloc.config import SimConfig
from jamloc.jammer import ATTACKS, Jammer, cell_mask

cfg = SimConfig()


def _mask(attack: str, start: int, size: int) -> torch.Tensor:
    code = torch.tensor([ATTACKS.index(attack)])
    return cell_mask(code, torch.tensor([start]), torch.tensor([size]), cfg)[0]


def test_attack_patterns():
    rows = torch.zeros(24, dtype=torch.bool)
    rows[4:9] = True
    not_dmrs = torch.ones(14, dtype=torch.bool)
    not_dmrs[2] = False

    pilot = _mask("pilot", 4, 5)
    assert pilot.shape == (24, 14)
    assert torch.equal(pilot[:, 2], rows)
    assert not pilot[:, not_dmrs].any()

    data = _mask("data", 4, 5)
    assert not data[:, 2].any()
    assert torch.equal(data[:, not_dmrs], rows[:, None].expand(24, 13))

    partial = _mask("partial", 4, 5)
    assert torch.equal(partial, rows[:, None].expand(24, 14))

    assert not _mask("none", 4, 5).any()

    edge = _mask("partial", 20, 4)
    assert edge[20:24].all() and not edge[:20].any()


def test_jammer_power():
    batch = 64
    cells = cell_mask(
        torch.full((batch,), ATTACKS.index("partial")),
        torch.zeros(batch, dtype=torch.long),
        torch.full((batch,), 24),
        cfg,
    )
    cells[:, :3] = False  # leave PRBs 0-2 unjammed
    z = Jammer(cfg)(cells, torch.full((batch,), 10.0))
    assert z.shape == (batch, 1, 1, 14, 288)
    power = z.abs() ** 2
    jammed = power[..., 36:]
    assert abs(jammed.mean().item() - 10.0) / 10.0 < 0.05
    assert torch.all(power[..., :36] == 0)


def test_pilot_comb_jammer_leaves_the_empty_comb_clean():
    import dataclasses

    comb = dataclasses.replace(cfg, jammer_pilot_comb=True)
    cells = cell_mask(torch.tensor([3, 1]), torch.tensor([2, 5]), torch.tensor([4, 3]), comb)
    z = Jammer(comb)(cells, torch.tensor([0.0, 0.0]))[:, 0, 0]  # [B, 14, 288]
    for b, (start, size) in enumerate(((2, 4), (5, 3))):
        band = slice(12 * start, 12 * (start + size))
        dmrs = z[b, comb.dmrs_symbol, band]
        assert (dmrs[1::2] == 0).all() and (dmrs[::2] != 0).all()  # pilots on the even comb
    assert (z[0, 5, 24:72] != 0).all()  # partial-band: data symbols fully jammed
    assert (z[1, 5] == 0).all()  # pilot attack: nothing on the data symbols


def test_hopping_jammer_moves_the_block_every_symbol():
    hop = torch.tensor([[0, 3, 7, 20, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    mask = cell_mask(torch.tensor([3]), torch.tensor([0]), torch.tensor([4]), cfg, hop)[0]
    for s in range(14):
        assert torch.equal(torch.nonzero(mask[:, s])[:, 0], torch.arange(hop[0, s], hop[0, s] + 4))
    # pilot attack: only the DMRS symbol, at that symbol's position
    pilot = cell_mask(torch.tensor([1]), torch.tensor([0]), torch.tensor([4]), cfg, hop)[0]
    assert pilot[:, cfg.dmrs_symbol].sum() == 4 and pilot.sum() == 4
    assert pilot[hop[0, cfg.dmrs_symbol], cfg.dmrs_symbol]


def test_hopping_jammer_in_the_link():
    import dataclasses

    from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything

    hop = dataclasses.replace(cfg, jammer_hopping=True)
    rng = seed_everything(3)
    sc = draw_scenario(rng, 4, hop, "A", attacks=(3,), jsr_db=0.0, jam_size=5)
    labels = JammedPUSCHLink(hop)(sc).labels.cpu()
    assert (labels.sum(1) == 5).all()  # 5 contiguous PRBs on every symbol
    starts = labels.float().argmax(1)  # first jammed PRB per symbol
    assert (starts.max(1).values > starts.min(1).values).all()  # the block moves


def test_pulsed_jammer_keeps_a_random_subset_of_symbols():
    from jamloc.jammer import pulse_mask

    batch = 200
    attack = torch.tensor([3] * (batch - 1) + [1])  # partial-band, and one pilot attack
    cells = cell_mask(attack, torch.full((batch,), 4), torch.full((batch,), 5), cfg)
    gen = torch.Generator().manual_seed(0)
    pulsed = pulse_mask(cells, 0.5, gen)
    assert not (pulsed & ~cells).any()  # only cells of the continuous attack
    per_slot = pulsed.any(1).sum(1)  # jammed symbols per slot
    assert (per_slot >= 1).all()  # a jammed slot stays jammed
    assert 5 < per_slot[:-1].float().mean() < 9  # about half of the 14 symbols
    assert torch.equal(pulsed[-1], cells[-1])  # a pilot attack (one symbol) is unchanged
    jammed = pulsed[:, 4:9]  # whole PRB block on the symbols that are on
    assert torch.equal(jammed.all(1), jammed.any(1))


def test_tone_jammer_has_one_constant_tone_per_prb():
    import dataclasses

    tone = dataclasses.replace(cfg, jammer_waveform="tone", jammer_tones=1)
    cells = cell_mask(torch.tensor([3]), torch.tensor([6]), torch.tensor([4]), tone)
    z = Jammer(tone)(cells, torch.tensor([3.0]))[0, 0, 0]  # [14, 288]
    on = z.abs() > 0
    for prb in range(24):
        used = on[:, 12 * prb : 12 * prb + 12]
        if 6 <= prb < 10:
            assert used.sum(1).eq(1).all()  # one subcarrier on every symbol ...
            assert used.any(0).sum() == 1  # ... and always the same one
        else:
            assert not used.any()
    values = z[:, on.any(0)]  # [14, 4]: a continuous tone keeps its value over the symbols
    assert torch.allclose(values, values[:1].expand_as(values))
    assert torch.allclose(values.abs() ** 2, torch.full_like(values.real, 10**0.3), rtol=1e-5)


def test_spoofing_jammer_replays_the_pilots():
    import dataclasses

    spoof = dataclasses.replace(cfg, jammer_waveform="dmrs")
    # The UE grid has QPSK pilots of energy 2 on the even subcarriers of PRBs 4-9, DMRS symbol
    # only. It is built on Sionna's device, where the jammer returns its output.
    dev = config.device
    x = torch.zeros(1, 1, 1, 14, 288, dtype=torch.complex64, device=dev)
    pilots = (torch.randint(0, 2, (36,), device=dev) * 2 - 1) + 1j * (
        torch.randint(0, 2, (36,), device=dev) * 2 - 1
    )
    x[0, 0, 0, spoof.dmrs_symbol, 48:120:2] = pilots.to(torch.complex64)
    cells = cell_mask(torch.tensor([3]), torch.tensor([2]), torch.tensor([10]), spoof)
    z = Jammer(spoof)(cells, torch.tensor([0.0]), x)[0, 0, 0]
    dmrs = z[spoof.dmrs_symbol]
    # Inside the allocation: the UE's pilots (unit energy) times one common phase
    ratio = dmrs[48:120:2] / (x[0, 0, 0, spoof.dmrs_symbol, 48:120:2] / 2**0.5)
    assert torch.allclose(ratio, ratio[:1].expand_as(ratio), atol=1e-5)
    assert torch.allclose(ratio.abs(), torch.ones_like(ratio.real), atol=1e-5)
    # Outside the allocation (PRBs 2-3, 10-11): unit-energy symbols; the empty comb stays clean
    outside = dmrs[24:48:2].abs()
    assert torch.allclose(outside, torch.ones_like(outside), atol=1e-5)
    assert (dmrs[25:144:2] == 0).all()
    assert (z[5, 24:144] != 0).all()  # partial-band: data symbols get Gaussian noise


def test_spoofing_is_invisible_to_the_residual():
    """A replayed DMRS adds a channel-like term to the LS estimates, which the projection removes
    (the residual's blind spot) - but only if it covers the whole allocation: a spoofed block
    inside it leaves a step at its edges, which the residual sees. Gaussian jamming always shows."""
    from jamloc.features import residual_per_prb

    n = 12
    pilot_sc = torch.arange(0, 12 * n, 2)
    f = pilot_sc.double() * cfg.scs
    ue = torch.exp(-2j * torch.pi * f * 0.4e-6).to(torch.complex64)
    jammer = (0.8 * torch.exp(-2j * torch.pi * f * 1.7e-6 + 0.3j)).to(torch.complex64)
    no = torch.tensor([0.01])  # no noise is added below, so an unjammed residual is about 0

    def residual(h_ls):
        return residual_per_prb(h_ls[None], pilot_sc, no, 2.0, cfg)[0]

    whole = residual(ue + jammer)  # spoofing on every scheduled PRB
    assert whole.max() < 0.05

    block = ue.clone()
    block[24:48] += jammer[24:48]  # spoofing only PRBs 4-7 of the allocation
    r_block = residual(block)
    assert r_block[[3, 4, 7, 8]].min() > 3  # the steps at the block edges show

    gen = torch.Generator().manual_seed(1)
    noise = torch.complex(torch.randn(24, generator=gen), torch.randn(24, generator=gen)) * 0.57
    gaussian = ue.clone()
    gaussian[24:48] += noise  # Gaussian jamming of PRBs 4-7 with the same energy
    assert residual(gaussian)[4:8].min() > 10
