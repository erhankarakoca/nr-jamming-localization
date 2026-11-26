"""The U-Net without the EVM input, the residual CNN of the same size and their comparison."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from jamloc.dataset import generate
from jamloc.experiments import input_ablation, recovery
from jamloc.features import FEATURE_NAMES
from jamloc.model import SlotResNet, TinyUNet, select_inputs
from jamloc.train import checkpoint_path, load_model, train


def test_no_evm_inputs_are_the_fused_ones_without_the_evm():
    features = torch.randn(2, len(FEATURE_NAMES), 24, 14)
    fused = select_inputs(features, "fused")
    no_evm = select_inputs(features, "no_evm")
    evm = 2  # position of the EVM among the fused inputs (power, residual, EVM, masks)
    keep = [i for i in range(fused.shape[1]) if i != evm]
    assert torch.equal(no_evm, fused[:, keep])


def test_residual_cnn_matches_the_unet_size_and_sees_19_cells():
    model = SlotResNet(6).eval()
    unet = TinyUNet(6)
    size = sum(p.numel() for p in model.parameters())
    assert size == 66991
    assert abs(size - sum(p.numel() for p in unet.parameters())) / size < 0.01

    x = torch.randn(2, 6, 24, 14)
    with torch.no_grad():
        y = model(x)
        x2 = x.clone()
        x2[:, :, 12, 0] += 3.0  # change one cell at the start of the slot
        changed = (model(x2) - y).abs() > 1e-6
    assert y.shape == (2, 24, 14)
    # Nine 3 x 3 convolutions: the change reaches 9 cells in each direction and no further
    rows, cols = torch.nonzero(changed.any(0), as_tuple=True)
    assert rows.min() >= 12 - 9 and rows.max() <= 12 + 9 and cols.max() <= 9
    assert changed[:, 12, 9].any()


def test_residual_cnn_trains_and_loads_under_its_own_name(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)
    path = train("fused", epochs=1, batch_size=32, arch="resnet")
    assert path == checkpoint_path("fused", arch="resnet")
    assert path.name == "fused_resnet_seed0.pt"
    assert (
        checkpoint_path("fused", 1, data="gen", arch="resnet").name == "fused_resnet_gen_seed1.pt"
    )
    assert isinstance(load_model("fused", arch="resnet"), SlotResNet)

    path = train("no_evm", epochs=1, batch_size=32)
    assert path.name == "no_evm_seed0.pt"
    assert load_model("no_evm").enc1[0].in_channels == 5


def test_ablation_maps_are_loaded_only_when_their_receivers_are_asked_for(tmp_path, monkeypatch):
    class Loaded(Exception):
        pass

    def stop(seed):
        raise Loaded

    assert recovery.input_variant("no_evm_gen") == "no_evm"
    assert recovery.input_variant("resnet_gen") == "fused"
    assert recovery.input_variant("fused_gen") == "fused"
    assert recovery.input_variant("no_residual") == "no_residual"

    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path))
    (tmp_path / "val_gen.npz").touch()
    monkeypatch.setattr(recovery, "load", lambda split: {})
    monkeypatch.setattr(recovery, "checkpoint_path", lambda *a, **k: Path(__file__))
    monkeypatch.setattr(recovery, "val_threshold", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(recovery, "seed_everything", stop)
    cases = (
        (None, []),
        (["p_r_e_shrink", "pr_shrink"], [("no_evm", None)]),
        (["resnet_shrink"], [("fused", "resnet")]),
    )
    for only, expected in cases:
        calls = []

        def record(variant, seed=0, calls=calls, **options):
            if options.get("data") == "gen" and (variant != "fused" or "arch" in options):
                calls.append((variant, options.get("arch")))

        monkeypatch.setattr(recovery, "load_model", record)
        with pytest.raises(Loaded):
            recovery.simulate("C", 15.0, [9.0], 8, 1, 10, seed=0, only=only, unet="general")
        assert calls == expected


def _write(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_decoding_table_pools_seeds_and_reports_paired_differences(tmp_path):
    folder = tmp_path / "hop" / "recovery"
    folder.mkdir(parents=True)
    counts = {"p_r_e_shrink": (3, 10), "pr_shrink": (5, 10), "true_shrink": (0, 10)}
    for seed in (0, 1):
        rows = [
            {"receiver": rx, "attack": "partial", "jsr_db": "9.0", "block_errors": e, "blocks": n}
            for rx, (e, n) in counts.items()
        ]
        _write(folder / f"recovery_C_hopping_seed{seed}.csv", rows)
    # recovery.paired_rows stores the receiver listed first in RECEIVERS as "first"
    paired = {
        "first": "p_r_e_shrink",
        "second": "pr_shrink",
        "attack": "partial",
        "jsr_db": "9.0",
        "delta": -0.1,
        "delta_lo": -0.15,
        "delta_hi": -0.05,
    }
    _write(folder / "recovery_C_hopping_paired.csv", [paired])

    lines = input_ablation.decoding_table(tmp_path).splitlines()
    assert len(lines) == 3
    cells = [c.strip() for c in lines[2].strip("|").split("|")]
    assert cells[:2] == ["QPSK hopping", "partial +9 dB"]
    # No map and ResNet were not run; the others are pooled over both seeds
    assert cells[2:7] == ["-", "30.00", "50.00", "-", "0/20"]
    # BLER(P+R) - BLER(P+R+E), with the interval turned round
    assert cells[7] == "+10.00 (+5.00 to +15.00)"
    assert cells[8] == "-"


def test_detection_table_lists_every_detector_per_grid():
    rows = []
    for condition in input_ablation.CONDITIONS.values():
        for i, name in enumerate(input_ablation.DETECTORS):
            row = {"condition": condition, "detector": name, "n_seeds": 3}
            for v in input_ablation.VALUES:
                row[f"{v}_mean"] = 0.9 - 0.1 * i if v != "fa" else 0.01
                row[f"{v}_std"] = 0.001
            row["f1_pilot_mean"] = np.nan if condition.startswith("pulsed") else 0.8
            rows.append(row)
    text = input_ablation.table(rows)
    lines = text.splitlines()
    assert lines[0].count("F1 / FA") == len(input_ablation.DETECTORS)
    assert "| matched | 0.900 +- 0.001 / 1.0% | 0.800 +- 0.001 / 1.0% |" in text
    assert sum(line.startswith("| U-Net, P+R |") for line in lines) == 2
