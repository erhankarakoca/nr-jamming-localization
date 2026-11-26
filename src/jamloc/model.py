"""Small U-Net that turns the feature maps of a slot into a per-cell jamming map.

Input: a subset of the feature channels ``[B, C, 24, 14]`` (PRB x OFDM symbol). Output: one logit
per cell ``[B, 24, 14]``; ``sigmoid(logit)`` is the jamming probability.

Two resolution levels are enough for a 24 x 14 grid. The symbol axis is padded from 14 to 16 so
that it can be halved twice (24 x 16 -> 12 x 8 -> 6 x 4) and cropped back at the end.
"""

import torch
import torch.nn.functional as F
from torch import nn

from jamloc.config import SimConfig
from jamloc.features import FEATURE_NAMES

# Which feature channels each detector variant sees (all variants get the masks)
_P, _R, _E, _ALLOC, _DMRS, _DATA, _COMB = range(len(FEATURE_NAMES))
VARIANTS = {
    "fused": [_P, _R, _E, _ALLOC, _DMRS, _DATA],
    "power": [_P, _ALLOC, _DMRS, _DATA],
    "evm": [_E, _ALLOC, _DMRS, _DATA],
    "no_residual": [_P, _E, _ALLOC, _DMRS, _DATA],
}
# Ablations trained on their own (not part of ``--variant all``): the fused inputs plus the power
# on the empty DMRS comb, and the fused inputs without the EVM
EXTRA_VARIANTS = {
    "fused_comb": [_P, _R, _E, _COMB, _ALLOC, _DMRS, _DATA],
    "no_evm": [_P, _R, _ALLOC, _DMRS, _DATA],
}
ALL_VARIANTS = {**VARIANTS, **EXTRA_VARIANTS}
_DB_CHANNELS = (_P, _R, _E, _COMB)


def select_inputs(features: torch.Tensor, variant: str) -> torch.Tensor:
    """Pick the variant's channels; dB features are divided by 10 to keep inputs near [-1, 4]."""
    idx = ALL_VARIANTS[variant]
    scale = torch.tensor([0.1 if i in _DB_CHANNELS else 1.0 for i in idx], device=features.device)
    return features[:, idx] * scale[None, :, None, None]


def noise_shift(
    x: torch.Tensor, variant: str, delta_db: torch.Tensor, *, cfg: SimConfig | None = None
) -> torch.Tensor:
    """Inputs (from ``select_inputs``) as seen with a noise estimate ``delta_db`` ``[B]`` dB above
    the true noise power.

    Every feature is divided by the estimated noise variance, so P, R and E move by ``-delta_db``
    (``-delta_db / 10`` after the input scaling) where they are defined: P everywhere, R on the
    DMRS cells, E on the data cells, C on the DMRS symbols. Used to train detectors that tolerate
    noise-estimation error.
    """
    idx = ALL_VARIANTS[variant]
    shift = -delta_db.to(x.dtype)[:, None, None] / 10
    out = x.clone()
    for pos, ch in enumerate(idx):
        if ch == _P:
            out[:, pos] += shift
        elif ch == _R:
            out[:, pos] += shift * x[:, idx.index(_DMRS)]
        elif ch == _E:
            out[:, pos] += shift * x[:, idx.index(_DATA)]
        elif ch == _COMB:
            dmrs = (cfg or SimConfig()).dmrs_symbols
            out[:, pos, :, dmrs] += shift
    return out


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class PixelMLP(nn.Module):
    """Per-cell classifier for the ablation of spatial context.

    The same inputs as the U-Net, but only 1 x 1 convolutions: every cell is classified from its
    own features and masks, without seeing its neighbours (4 801 parameters for the fused
    variant with ``hidden=64``, 66 277 with ``hidden=252``, about as many as the U-Net).
    """

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[:, 0]


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int, base: int = 12):
        super().__init__()
        c1, c2, c3 = base, 2 * base, 4 * base
        self.enc1 = _conv_block(in_channels, c1)  # 24 x 16
        self.enc2 = _conv_block(c1, c2)  # 12 x 8
        self.bottom = _conv_block(c2, c3)  # 6 x 4
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _conv_block(2 * c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _conv_block(2 * c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_symbols = x.shape[-1]
        x = F.pad(x, (0, 16 - num_symbols))  # 14 -> 16 OFDM symbols

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottom(F.max_pool2d(e2, 2))

        # Decoder with skip connections
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)[:, 0, :, :num_symbols]


class _ResidualBlock(nn.Module):
    """Two 3 x 3 convolutions with batch norm, added to the block's input (He et al., 2016)."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.body(x))


class SlotResNet(nn.Module):
    """A residual CNN that keeps the full 24 x 14 resolution, for the ablation of the U-Net's
    multi-scale path.

    One 3 x 3 convolution lifts the inputs to ``width`` channels, ``blocks`` residual blocks
    follow, and a 1 x 1 convolution gives the logits. With ``width=30`` and ``blocks=4`` it has
    66 991 parameters for the fused inputs, about as many as the U-Net (66 553). Its nine 3 x 3
    convolutions see 19 x 19 cells around each cell, whereas the U-Net's pooling lets every cell
    see the whole slot.
    """

    def __init__(self, in_channels: int, width: int = 30, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))[:, 0]
