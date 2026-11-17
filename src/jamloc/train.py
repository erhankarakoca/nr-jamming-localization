"""Train one detector variant on the ``train`` split, keeping the epoch with the best ``val`` loss.

python -m jamloc.train --variant fused
python -m jamloc.train --variant all
"""

import argparse
import time

import torch
from sionna.phy import config
from torch import nn

from jamloc.dataset import load
from jamloc.model import VARIANTS, TinyUNet, select_inputs
from jamloc.paths import results_dir


def _tensors(split: str, variant: str, device: str):
    d = load(split)
    x = select_inputs(torch.from_numpy(d["features"]), variant).to(device)
    y = torch.from_numpy(d["labels"]).float().to(device)
    return x, y


def _checkpoint_path(variant: str):
    return results_dir() / "models" / f"{variant}.pt"


def train(variant: str, epochs: int = 30, batch_size: int = 128, lr: float = 1e-3, seed: int = 0):
    """Train a variant and save its best weights. Returns the checkpoint path."""
    torch.manual_seed(seed)
    device = config.device
    x_tr, y_tr = _tensors("train", variant, device)
    x_va, y_va = _tensors("val", variant, device)

    model = TinyUNet(x_tr.shape[1]).to(device)
    # Jammed cells are rare, so weight them up in the loss
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum().clamp_min(1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    path = _checkpoint_path(variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        for idx in torch.randperm(len(x_tr), generator=gen).split(batch_size):
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                loss_fn(model(x_va[i]), y_va[i]).item() * len(i)
                for i in torch.arange(len(x_va)).split(1024)
            ) / len(x_va)
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "variant": variant,
                    "in_channels": x_tr.shape[1],
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "state_dict": model.state_dict(),
                },
                path,
            )
        print(
            f"[{variant}] epoch {epoch + 1}/{epochs}  train loss {loss.item():.4f}  "
            f"val loss {val_loss:.4f}  ({time.perf_counter() - start:.1f} s)",
            flush=True,
        )
    return path


def load_model(variant: str) -> TinyUNet:
    """Trained model in eval mode, on ``sionna.phy.config.device``."""
    ckpt = torch.load(_checkpoint_path(variant), map_location=config.device)
    model = TinyUNet(ckpt["in_channels"]).to(config.device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def main(argv=None):
    p = argparse.ArgumentParser(description="Train a jamming-map detector.")
    p.add_argument("--variant", choices=[*VARIANTS, "all"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        print("Saved", train(variant, args.epochs, args.batch_size, args.lr, args.seed))


if __name__ == "__main__":
    main()
