import torch

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
