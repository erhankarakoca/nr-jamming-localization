"""Block test, per-cell MLP, multi-slot residual, detection margin and calibration."""

import csv

import numpy as np
import torch

from jamloc import theory
from jamloc.baselines import block_test_scores
from jamloc.dataset import generate, load
from jamloc.features import FEATURE_NAMES
from jamloc.model import PixelMLP
from jamloc.train import checkpoint_path, load_model, train

P, R, E, ALLOC = (
    FEATURE_NAMES.index(n) for n in ("power_db", "residual_db", "evm_db", "alloc_mask")
)
DMRS = 2
DATA = np.arange(14) != DMRS


def test_block_test_follows_the_attack_structure(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    generate("val", num_slots=96, batch_size=32)
    val = load("val")
    s = block_test_scores(val, val)
    assert s.shape == val["labels"].shape

    # The 13 data cells of a PRB share one score (a whole-PRB statistic)
    data = s[:, :, DATA]
    assert np.isfinite(s).all()
    assert np.isclose(data, data[:, :, :1]).all()

    benign = np.flatnonzero(val["attack"] == 0)
    sched = val["features"][:, ALLOC, :, DMRS] > 0
    slot = benign[0]
    off, on = np.flatnonzero(~sched[slot]), np.flatnonzero(sched[slot])

    # Data jamming outside the allocation: +10 dB on the data symbols of an unscheduled PRB
    d = {k: v.copy() for k, v in val.items()}
    if len(off):
        d["features"][slot, P, off[0], DATA] += 10
        assert block_test_scores(d, val)[slot, off[0], DATA].min() > 5

    # Pilot-only jamming of a scheduled PRB: residual and DMRS power up, data power unchanged.
    # The PRB's EVM rise is then not reported on the data cells ...
    b = on[len(on) // 2]
    d = {k: v.copy() for k, v in val.items()}
    d["features"][slot, R, b, DMRS] = 20.0
    d["features"][slot, P, b, DMRS] += 10
    d["features"][slot, E, b, DATA] += 10
    pilot_only = block_test_scores(d, val)
    assert pilot_only[slot, b, DMRS] > 5
    assert (pilot_only[slot, b, DATA] == 0).all()  # p = 1: scored but never flagged
    # ... but with the data symbols' power up as well (partial-band jamming) it is
    d["features"][slot, P, b, DATA] += 10
    assert block_test_scores(d, val)[slot, b, DATA].min() > 2


def test_per_cell_mlp_sees_no_neighbours(tmp_path, monkeypatch):
    model = PixelMLP(6).eval()
    assert sum(p.numel() for p in model.parameters()) == 4801
    x = torch.randn(2, 6, 24, 14)
    y = model(x)
    x2 = x.clone()
    x2[:, :, 5, 5] += 3.0  # change one cell
    changed = (model(x2) - y).abs() > 1e-6
    assert changed[:, 5, 5].all() and changed.sum() == 2

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)
    path = train("fused", epochs=1, batch_size=32, arch="pixel")
    assert path == checkpoint_path("fused", arch="pixel") and path.name == "fused_pixel_seed0.pt"
    assert isinstance(load_model("fused", arch="pixel"), PixelMLP)


def test_slot_average_keeps_the_mean_and_lowers_the_threshold():
    lam = theory.weights(12, 5)
    for k in (2, 4):
        avg = theory.slot_average(lam, k)
        assert avg.shape == (k * lam.size,) and np.isclose(avg.sum(), lam.sum())
        t = theory.threshold(12, 5, 0.01, slots=k)
        assert np.isclose(theory.ccdf(t, avg), 0.01, rtol=0.02)
    assert theory.threshold(12, 5, slots=4) < theory.threshold(12, 5, slots=2)
    assert theory.threshold(12, 5, slots=2) < theory.threshold(12, 5)


def test_multislot_pd_matches_monte_carlo():
    from jamloc.experiments.multislot import ALLOC as N
    from jamloc.experiments.multislot import JAMMED, PRB, monte_carlo_pd

    pd = theory.detection_probability([3.0], N, PRB, JAMMED, slots=4, draws=1000)
    assert abs(pd[0] - monte_carlo_pd(4, [3.0], draws=4000)[0]) < 0.04


def _small_models(tmp_path, monkeypatch, variants):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=64, batch_size=32)
    for variant in variants:
        train(variant, epochs=1, batch_size=32)


def test_detection_margin_small(tmp_path, monkeypatch):
    from jamloc.experiments import detection_margin

    _small_models(tmp_path, monkeypatch, ("fused", "no_residual"))
    generate("margin", num_slots=80, batch_size=32)
    bler_dir = tmp_path / "results" / "ber_bler"
    bler_dir.mkdir(parents=True)
    for channel in ("A", "C"):
        with open(bler_dir / f"ber_bler_{channel}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["attack", "snr_db", "jsr_db", "bler"])
            for attack in ("pilot", "data", "partial"):
                for jsr in range(-20, 11, 5):
                    writer.writerow([attack, 15.0, jsr, 1 / (1 + np.exp(-jsr))])
                    writer.writerow([attack, 5.0, jsr, 0.5])  # another SNR: must be ignored
    detection_margin.main(["--seeds", "0"])
    out = tmp_path / "results" / "margin"
    assert (out / "margin.png").is_file()
    with open(out / "margin_summary.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["detector"] for r in rows} == set(detection_margin.DETECTORS)
    # linear interpolation between the grid points -5 and 0 dB of the logistic BLER curve
    lo, hi = 1 / (1 + np.exp(5)), 0.5
    expected = -5 + 5 * (0.1 - lo) / (hi - lo)
    assert all(np.isclose(float(r["jsr_bler10_db"]), expected) for r in rows)


def test_crossing():
    from jamloc.experiments.detection_margin import crossing

    assert crossing([0, 1, 2], [0.0, 0.5, 1.0], 0.75) == 1.5
    assert crossing([0, 1], [0.95, 1.0], 0.9) == -np.inf
    assert np.isnan(crossing([0, 1], [0.1, 0.2], 0.9))


def test_calibration_small(tmp_path, monkeypatch):
    from jamloc.experiments import calibration

    _small_models(tmp_path, monkeypatch, ("fused", "no_residual", "power"))
    train("fused", epochs=1, batch_size=32, jitter=1.0)  # the U-Net trained with noise jitter
    generate("test", num_slots=60, batch_size=32)
    calibration.main(["--seeds", "0"])
    out = tmp_path / "results" / "calibration"
    assert (out / "calibration.png").is_file()
    with open(out / "snr_thresholds.csv", encoding="utf-8", newline="") as f:
        assert {r["detector"] for r in csv.DictReader(f)} == set(calibration.DETECTORS)

    # Re-running one detector replaces only its rows
    calibration.main(["--seeds", "0", "--detectors", "residual_cfar"])
    with open(out / "snr_thresholds.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["detector"] for r in rows} == set(calibration.DETECTORS)
    assert sum(r["detector"] == "residual_cfar" for r in rows) == 2  # two threshold modes
    with open(out / "noise_mismatch.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    theory_fa = {float(r["delta_db"]): float(r["residual_theory_fa"]) for r in rows}
    assert 0.001 < theory_fa[0.0] < 0.1  # the design rate of the calibrated residual cutoff
    assert theory_fa[-1.0] > theory_fa[0.0] > theory_fa[1.0]  # underestimated noise: more


def test_shift_noise_only_moves_defined_features():
    from jamloc.experiments.calibration import shift_noise

    f = np.zeros((1, 6, 24, 14), np.float32)
    f[0, FEATURE_NAMES.index("dmrs_mask"), 3, DMRS] = 1
    f[0, FEATURE_NAMES.index("data_mask"), 3, 5] = 1
    g = shift_noise({"features": f}, 1.0)["features"]
    assert (g[0, P] == -1).all()
    assert g[0, R, 3, DMRS] == -1 and (g[0, R] != 0).sum() == 1
    assert g[0, E, 3, 5] == -1 and (g[0, E] != 0).sum() == 1


def test_noise_jitter_checkpoint_and_shift(tmp_path, monkeypatch):
    from jamloc.experiments.calibration import shift_noise
    from jamloc.model import noise_shift, select_inputs

    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path))
    assert checkpoint_path("fused", 2, jitter=1.0).name == "fused_nj1_seed2.pt"
    rng = np.random.default_rng(0)
    f = rng.normal(size=(3, 6, 24, 14)).astype(np.float32)
    f[:, 3:] = rng.integers(0, 2, size=(3, 3, 24, 14))  # masks
    for variant in ("fused", "power", "evm", "no_residual"):
        a = noise_shift(select_inputs(torch.from_numpy(f), variant), variant, torch.full((3,), 0.7))
        b = select_inputs(torch.from_numpy(shift_noise({"features": f}, 0.7)["features"]), variant)
        assert torch.allclose(a, b, atol=1e-6)


def test_gamma_tail_stays_finite_and_ordered():
    from jamloc.baselines import _log10_gamma_sf

    x = np.array([1.0, 2.0, 8.0, 20.0, 100.0])
    p = _log10_gamma_sf(x, 156)
    assert np.isfinite(p).all() and (np.diff(p) < 0).all()
