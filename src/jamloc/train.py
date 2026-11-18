"""Train one detector variant on the ``train`` split, keeping the epoch with the best ``val`` loss.

python -m jamloc.train --variant fused
python -m jamloc.train --variant all --seeds 0 1 2
"""

import argparse
import csv
import time

import torch
from sionna.phy import config
from torch import nn

from jamloc.dataset import load
from jamloc.model import VARIANTS, PixelMLP, TinyUNet, noise_shift, select_inputs
from jamloc.paths import results_dir


def _tensors(split: str, variant: str, device: str):
    d = load(split)
    x = select_inputs(torch.from_numpy(d["features"]), variant).to(device)
    y = torch.from_numpy(d["labels"]).float().to(device)
    return x, y


DEFAULT_WIDTH = 12  # channels of the first U-Net level (66 553 parameters for the fused variant)
ARCHS = ("unet", "pixel")  # the U-Net, or the per-cell MLP of the spatial-context ablation


def build_model(arch: str, in_channels: int, width: int = DEFAULT_WIDTH):
    if arch == "unet":
        return TinyUNet(in_channels, base=width)
    if arch == "pixel":
        return PixelMLP(in_channels)
    raise ValueError(f"unknown architecture {arch!r}")


def checkpoint_path(
    variant: str,
    seed: int = 0,
    width: int = DEFAULT_WIDTH,
    data: str = "",
    arch: str = "unet",
    jitter: float = 0.0,
):
    """``fused_seed0.pt`` for the default width, ``fused_w4_seed0.pt`` for a narrower U-Net,
    ``fused_mix_seed0.pt`` for one trained on ``train_mix`` (``data="mix"``),
    ``fused_pixel_seed0.pt`` for the per-cell MLP (``arch="pixel"``, no width),
    ``fused_nj1_seed0.pt`` for one trained with +-1 dB noise-estimation error (``jitter=1``)."""
    if arch == "pixel":
        tag = f"{variant}_pixel"
    else:
        tag = variant if width == DEFAULT_WIDTH else f"{variant}_w{width}"
    tag = f"{tag}_{data}" if data else tag
    tag = f"{tag}_nj{jitter:g}" if jitter else tag
    return results_dir() / "models" / f"{tag}_seed{seed}.pt"


def train(
    variant: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    width: int = DEFAULT_WIDTH,
    data: str = "",
    arch: str = "unet",
    jitter: float = 0.0,
):
    """Train a variant and save its best weights. Returns the checkpoint path.

    ``data="mix"`` trains on ``train_mix`` / ``val_mix`` instead of ``train`` / ``val``;
    ``arch="pixel"`` trains the per-cell MLP instead of the U-Net. ``jitter > 0`` shows every
    training slot as if the noise power had been estimated with an error drawn uniformly from
    +-``jitter`` dB (``model.noise_shift``); validation stays exact.
    """
    torch.manual_seed(seed)
    device = config.device
    suffix = f"_{data}" if data else ""
    x_tr, y_tr = _tensors(f"train{suffix}", variant, device)
    x_va, y_va = _tensors(f"val{suffix}", variant, device)

    model = build_model(arch, x_tr.shape[1], width).to(device)
    # Jammed cells are rare, so weight them up in the loss
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum().clamp_min(1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    path = checkpoint_path(variant, seed, width, data, arch, jitter)
    path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    gen = torch.Generator().manual_seed(seed)
    history = []  # one row per epoch, saved next to the checkpoint for the training-curve figure
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        batch_losses = []
        for idx in torch.randperm(len(x_tr), generator=gen).split(batch_size):
            x = x_tr[idx]
            if jitter:
                delta = (torch.rand(len(idx), generator=gen) * 2 - 1) * jitter
                x = noise_shift(x, variant, delta.to(device))
            loss = loss_fn(model(x), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        sched.step()
        train_loss = sum(batch_losses) / len(batch_losses)

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
                    "seed": seed,
                    "in_channels": x_tr.shape[1],
                    "width": width,
                    "arch": arch,
                    "noise_jitter_db": jitter,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "state_dict": model.state_dict(),
                },
                path,
            )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[{variant}, {arch} width {width}, seed {seed}] epoch {epoch + 1}/{epochs}  "
            f"train loss {train_loss:.4f}  "
            f"val loss {val_loss:.4f}  ({time.perf_counter() - start:.1f} s)",
            flush=True,
        )

    with open(path.with_name(f"{path.stem}_history.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return path


def load_model(
    variant: str,
    seed: int = 0,
    width: int = DEFAULT_WIDTH,
    data: str = "",
    arch: str = "unet",
    jitter: float = 0.0,
):
    """Trained model in eval mode, on ``sionna.phy.config.device``."""
    path = checkpoint_path(variant, seed, width, data, arch, jitter)
    ckpt = torch.load(path, map_location=config.device)
    model = build_model(
        ckpt.get("arch", "unet"), ckpt["in_channels"], ckpt.get("width", DEFAULT_WIDTH)
    ).to(config.device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def main(argv=None):
    p = argparse.ArgumentParser(description="Train a jamming-map detector.")
    p.add_argument("--variant", choices=[*VARIANTS, "all"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="channels of the first level")
    p.add_argument("--arch", choices=ARCHS, default="unet", help="pixel: per-cell MLP ablation")
    p.add_argument(
        "--noise-jitter-db", type=float, default=0.0, help="train with +-this noise-estimate error"
    )
    args = p.parse_args(argv)
    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        for seed in args.seeds:
            path = train(
                variant,
                args.epochs,
                args.batch_size,
                args.lr,
                seed,
                args.width,
                arch=args.arch,
                jitter=args.noise_jitter_db,
            )
            print("Saved", path)


if __name__ == "__main__":
    main()
