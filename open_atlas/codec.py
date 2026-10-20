"""Frozen RGB image codec, adapted from MIT-licensed madebyollin/taesd.

Copyright (c) 2023 Ollin Boer Bohan. License: docs/asset-licenses/TAESD.txt.
Only the SD1/2 architecture is retained; its four latent channels are not RGBD.
"""

from pathlib import Path

import torch
from torch import nn


def _conv(inputs, outputs, **kwargs):
    return nn.Conv2d(inputs, outputs, 3, padding=1, **kwargs)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(_conv(64, 64), nn.ReLU(), _conv(64, 64), nn.ReLU(), _conv(64, 64))

    def forward(self, x):
        return (self.conv(x) + x).relu()


class _Clamp(nn.Module):
    def forward(self, x):
        return (x / 3).tanh() * 3


class ImageCodec(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        encoder = [_conv(3, 64), _Block()]
        for _ in range(3):
            encoder += [_conv(64, 64, stride=2, bias=False), _Block(), _Block(), _Block()]
        self.encoder = nn.Sequential(*encoder, _conv(64, 4))
        decoder = [_Clamp(), _conv(4, 64), nn.ReLU()]
        for _ in range(3):
            decoder += [_Block(), _Block(), _Block(), nn.Upsample(scale_factor=2), _conv(64, 64, bias=False)]
        self.decoder = nn.Sequential(*decoder, _Block(), _conv(64, 3))
        if weights is not None:
            for name in ["encoder", "decoder"]:
                state = torch.load(Path(weights) / f"taesd_{name}.pth", map_location="cpu", weights_only=True)
                getattr(self, name).load_state_dict(state)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, rgb):
        """RGB [0,1] → learned latents [B,4,H/8,W/8], without quantization."""
        return self.encoder(rgb)

    @torch.no_grad()
    def decode(self, latent):
        return self.decoder(latent).clamp(0, 1)
