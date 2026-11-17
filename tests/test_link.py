import torch

from jamloc.config import SimConfig
from jamloc.jammer import ATTACKS, cell_mask
from jamloc.link import JammedPUSCHLink, Scenario, draw_scenario, seed_everything

cfg = SimConfig()
link = JammedPUSCHLink(cfg)
PILOT, PARTIAL = ATTACKS.index("pilot"), ATTACKS.index("partial")


def _bler(batch) -> float:
    return (~batch.rx["tb_crc"]).float().mean().item()


def test_labels_match_jammer():
    rng = seed_everything(0)
    sc = draw_scenario(rng, 8, cfg, "A", attacks=(PILOT,), jsr_db=0.0, snr_db=20.0)
    out = link(sc)
    expected = cell_mask(sc.attack, sc.jam_start, sc.jam_size, cfg)
    assert torch.equal(out.labels.cpu(), expected)
    assert out.y.shape == (8, 14, 288)
    assert out.y_alloc.shape == (8, 1, 1, 14, 12 * sc.alloc_size)


def test_same_seed_same_output():
    ys = []
    for _ in range(2):
        rng = seed_everything(5)
        ys.append(link(draw_scenario(rng, 4, cfg, "C")).y)
    assert torch.equal(ys[0], ys[1])


def test_benign_high_snr_decodes():
    rng = seed_everything(1)
    sc = draw_scenario(rng, 32, cfg, "A", attacks=(0,), snr_db=25.0)
    assert _bler(link(sc, decode=True)) == 0.0


def test_partial_jamming_breaks_link():
    rng = seed_everything(2)
    sc = draw_scenario(
        rng, 32, cfg, "A", attacks=(PARTIAL,), jsr_db=10.0, snr_db=25.0, scheduling_aware=True
    )
    assert _bler(link(sc, decode=True)) > 0.5


def test_jammer_outside_allocation_is_harmless():
    seed_everything(3)
    batch = 16
    sc = Scenario(
        channel="A",
        alloc_start=0,
        alloc_size=8,
        attack=torch.full((batch,), PARTIAL),
        jam_start=torch.full((batch,), 12),
        jam_size=torch.full((batch,), 9),
        jsr_db=torch.full((batch,), 10.0),
        snr_db=torch.full((batch,), 25.0),
    )
    out = link(sc, decode=True)
    assert _bler(out) == 0.0
    jammed_prbs = out.labels.any(dim=2)  # [B, 24]
    assert jammed_prbs[:, 12:21].all() and not jammed_prbs[:, :12].any()
    assert not jammed_prbs[:, 21:].any()


def test_default_config():
    rng = seed_everything(7)
    out = JammedPUSCHLink()(draw_scenario(rng, 2, cfg, "A"))
    assert out.y.shape == (2, 14, 288)
