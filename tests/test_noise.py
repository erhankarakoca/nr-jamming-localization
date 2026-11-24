"""Noise power the receiver assumes, the empty-comb feature, and the new dataset metadata."""

import dataclasses

import numpy as np
import pytest
import torch

from jamloc.baselines import comb_scores
from jamloc.config import SimConfig
from jamloc.dataset import SPLITS, generate, load, margin_split, noise_split
from jamloc.features import FEATURE_NAMES, comb_map
from jamloc.jammer import ATTACKS
from jamloc.link import JammedPUSCHLink, draw_scenario, seed_everything
from jamloc.noise import _median_of_exponentials, empty_comb, estimated_snr_db

cfg = SimConfig()
link = JammedPUSCHLink(cfg)
PILOT = ATTACKS.index("pilot")


def test_empty_comb_is_the_other_half_of_the_dmrs_symbol():
    out = link(draw_scenario(seed_everything(0), 2, cfg, "A", alloc=(4, 8)))
    sym, sc = empty_comb(out.tx)
    assert torch.all(sym == cfg.dmrs_symbol)
    assert len(sc) == 6 * 8  # six empty subcarriers per PRB
    assert torch.all(sc % 2 == 1)  # port 0 of type-1 DMRS uses the even subcarriers


def test_median_of_exponentials_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for m in (36, 37, 144):
        sim = np.median(rng.exponential(size=(20_000, m)), axis=1).mean()
        assert sim == pytest.approx(_median_of_exponentials(m), rel=0.01)


def test_set_noise_error_reaches_the_receiver():
    sc = draw_scenario(seed_everything(1), 4, cfg, "A", attacks=(0,), noise_error_db=1.0)
    out = link(sc)
    assert torch.allclose(out.no, out.no_true * 10**0.1)
    # without an error the receiver keeps the true noise power
    out = link(draw_scenario(seed_everything(1), 4, cfg, "A", attacks=(0,)))
    assert torch.equal(out.no, out.no_true)


def test_comb_estimate_is_unbiased_without_jamming():
    est = JammedPUSCHLink(dataclasses.replace(cfg, noise_estimate="comb"))
    ratios = []
    for i in range(6):
        sc = draw_scenario(seed_everything(10 + i), 64, cfg, "C", attacks=(0,), alloc=(0, 24))
        out = est(sc)
        ratios.append((out.no / out.no_true).cpu())
    ratios = torch.cat(ratios)
    assert ratios.mean().item() == pytest.approx(1.0, abs=0.03)
    # 144 empty REs: the median of 144 exponentials has a relative spread of about 12%
    assert 0.08 < ratios.std().item() < 0.17


def test_comb_estimate_is_biased_when_the_whole_dmrs_symbol_is_jammed():
    est = JammedPUSCHLink(dataclasses.replace(cfg, noise_estimate="comb"))
    sc = draw_scenario(
        seed_everything(3),
        32,
        cfg,
        "A",
        attacks=(PILOT,),
        jsr_db=0.0,
        snr_db=15.0,
        alloc=(0, 24),
        scheduling_aware=True,
    )
    out = est(sc)
    assert (out.no / out.no_true).median().item() > 10  # the jamming dominates the empty comb


def test_comb_feature_sees_full_pilot_jamming_but_not_the_comb_jammer():
    sc = draw_scenario(
        seed_everything(4), 32, cfg, "A", attacks=(PILOT,), jsr_db=0.0, snr_db=20.0, jam_size=6
    )
    full = comb_map(link(sc), cfg)[:, :, cfg.dmrs_symbol]
    comb_jammer = JammedPUSCHLink(dataclasses.replace(cfg, jammer_pilot_comb=True))
    sc = draw_scenario(
        seed_everything(4), 32, cfg, "A", attacks=(PILOT,), jsr_db=0.0, snr_db=20.0, jam_size=6
    )
    only_pilots = comb_map(comb_jammer(sc), cfg)[:, :, cfg.dmrs_symbol]
    jammed = torch.zeros_like(full, dtype=torch.bool)
    for b in range(len(sc.jam_start)):
        jammed[b, sc.jam_start[b] : sc.jam_start[b] + sc.jam_size[b]] = True
    assert full[jammed].mean().item() > 30  # about the 20 dB JNR
    assert full[~jammed].mean().item() == pytest.approx(1.0, abs=0.1)
    assert only_pilots[jammed].mean().item() == pytest.approx(1.0, abs=0.15)


def test_comb_scores_are_calibrated_without_jamming():
    # Synthetic benign features: the comb power is the mean of 6 unit exponentials
    rng = np.random.default_rng(0)
    n = 4000
    features = np.zeros((n, len(FEATURE_NAMES), 24, 14), dtype=np.float32)
    comb = rng.exponential(size=(n, 24, 6)).mean(-1)
    features[:, FEATURE_NAMES.index("comb_db"), :, cfg.dmrs_symbol] = 10 * np.log10(comb)
    features[:, FEATURE_NAMES.index("dmrs_mask"), :, cfg.dmrs_symbol] = 1
    scores = comb_scores(features)[:, :, cfg.dmrs_symbol]
    assert (scores > 2).mean() == pytest.approx(0.01, abs=0.002)


def test_unnormalized_channels_vary_from_slot_to_slot():
    def energy(normalize):
        lk = JammedPUSCHLink(dataclasses.replace(cfg, normalize_channel=normalize))
        sc = draw_scenario(seed_everything(5), 64, cfg, "A", attacks=(0,), snr_db=40.0)
        y = lk(sc).y  # at 40 dB SNR the received energy is the channel's
        data = [s for s in range(cfg.num_symbols) if s != cfg.dmrs_symbol]
        start, size = sc.alloc_start, sc.alloc_size
        return (y[:, data, 12 * start : 12 * (start + size)].abs() ** 2).mean(dim=(1, 2))

    # Normalization fixes the energy of the whole carrier, so the allocation's share still
    # varies a little. Over 64 slots its spread is about 0.14 in the median over 30 seeds, on
    # CPU and GPU. This seed gives 0.139 on CPU and 0.150 on CUDA, whose generator draws other
    # channels. Without normalization it is 0.5-1.2.
    assert energy(True).std().item() < 0.25
    assert energy(False).std().item() > 2 * energy(True).std().item()
    assert energy(False).mean().item() == pytest.approx(1.0, abs=0.3)


def test_snr_estimate_tracks_the_true_snr():
    # On the whole carrier the channel has unit energy in every slot, so the received SNR is the
    # nominal one; on part of it, the estimate follows the fading of those PRBs instead
    sc = draw_scenario(
        seed_everything(6), 64, cfg, "C", attacks=(0,), snr_db=(5.0, 25.0), alloc=(0, 24)
    )
    out = link(sc)
    est = estimated_snr_db(out.y_alloc, out.no, cfg).cpu()
    assert (est - sc.snr_db).abs().median().item() < 0.2


def test_new_splits_and_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    assert margin_split(15.0) == "margin" and margin_split(8.0) == "margin_snr8"
    assert noise_split(-0.5) == "noise_m0.5" and noise_split(1.0) in SPLITS
    for name in ("unnorm", "train_atk2", "val_atk2", "train_nest", "test_nest", "sched_nest"):
        assert name in SPLITS

    generate("test", num_slots=60)
    d = load("test")
    assert d["features"].shape[1] == len(FEATURE_NAMES)
    assert {"batch", "snr_db_est", "noise_error_db"} <= set(d)
    assert np.bincount(d["batch"]).max() <= 16  # grid splits use small batches
    assert np.all(d["noise_error_db"] == 0)

    # Every fourth pair of batches of train_atk2 is pilot spoofing on the scheduled PRBs
    generate("train_atk2", num_slots=8 * 8, batch_size=8)
    d = load("train_atk2")
    spoof = (d["batch"] // 2) % 4 == 3
    assert np.all(d["scheduling_aware"][spoof]) and not np.any(d["scheduling_aware"][~spoof])
    assert set(np.unique(d["attack"][spoof])) <= {0, 1}
