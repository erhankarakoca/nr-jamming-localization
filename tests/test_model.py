import torch

from jamloc.dataset import generate
from jamloc.model import VARIANTS, TinyUNet, select_inputs
from jamloc.train import load_model, train


def test_param_count():
    model = TinyUNet(len(VARIANTS["fused"]))
    n = sum(p.numel() for p in model.parameters())
    assert 10_000 <= n <= 100_000


def test_output_shape():
    assert TinyUNet(6)(torch.randn(8, 6, 24, 14)).shape == (8, 24, 14)


def test_select_inputs():
    f = torch.zeros(2, 6, 24, 14)
    f[:, 0] = 20.0  # power in dB
    f[:, 3] = 1.0  # alloc mask
    x = select_inputs(f, "power")
    assert x.shape == (2, len(VARIANTS["power"]), 24, 14)
    assert torch.allclose(x[:, 0], torch.full((2, 24, 14), 2.0))  # dB channels scaled by 1/10
    assert torch.allclose(x[:, 1], torch.ones(2, 24, 14))  # masks unchanged


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    x = torch.randn(16, 6, 24, 14)
    y = (x[:, 0] > 1).float()
    model = TinyUNet(6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    first = None
    for _ in range(100):
        loss = loss_fn(model(x), y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_train_writes_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMLOC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("JAMLOC_RESULTS_DIR", str(tmp_path / "results"))
    generate("train", num_slots=64, batch_size=32)
    generate("val", num_slots=32, batch_size=32)
    path = train("evm", epochs=1, batch_size=32, seed=1)
    assert path == tmp_path / "results" / "models" / "evm_seed1.pt"
    model = load_model("evm", seed=1)
    assert isinstance(model, TinyUNet) and not model.training
