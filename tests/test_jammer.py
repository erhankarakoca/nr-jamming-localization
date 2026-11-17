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
