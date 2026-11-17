"""Small U-Net that turns the feature maps of a slot into a per-cell jamming map.

Input: a subset of the feature channels ``[B, C, 24, 14]`` (PRB x OFDM symbol). Output: one logit
per cell ``[B, 24, 14]``; ``sigmoid(logit)`` is the jamming probability.

Two resolution levels are enough for a 24 x 14 grid. The symbol axis is padded from 14 to 16 so
that it can be halved twice (24 x 16 -> 12 x 8 -> 6 x 4) and cropped back at the end.
"""

import torch
import torch.nn.functional as F
from torch import nn

from jamloc.features import FEATURE_NAMES

# Which feature channels each detector variant sees (all variants get the masks)
_P, _R, _E, _ALLOC, _DMRS, _DATA = range(len(FEATURE_NAMES))
VARIANTS = {
    "fused": [_P, _R, _E, _ALLOC, _DMRS, _DATA],
    "power": [_P, _ALLOC, _DMRS, _DATA],
    "evm": [_E, _ALLOC, _DMRS, _DATA],
    "no_residual": [_P, _E, _ALLOC, _DMRS, _DATA],
}
_DB_CHANNELS = (_P, _R, _E)


def select_inputs(features: torch.Tensor, variant: str) -> torch.Tensor:
    """Pick the variant's channels; dB features are divided by 10 to keep inputs near [-1, 4]."""
    idx = VARIANTS[variant]
    scale = torch.tensor([0.1 if i in _DB_CHANNELS else 1.0 for i in idx], device=features.device)
    return features[:, idx] * scale[None, :, None, None]


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


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
