"""Autoregressive image-latent DiT with camera conditioning and rectified flow."""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class AtlasConfig:
    latent_size: int = 8
    patch_size: int = 1
    dim: int = 128
    layers: int = 4
    heads: int = 4


class DiTBlock(nn.Module):
    """DiT 风格 adaLN-Zero；条件控制归一化及注意力、MLP 残差门。"""
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, x, condition, mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.condition(condition).chunk(6, dim=-1)
        normalized = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attention(normalized, normalized, normalized, attn_mask=mask, need_weights=False)[0]
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class AtlasMini(nn.Module):
    """学习下一张 RGB 图像的潜变量；四个潜通道均为图像特征，不含深度。"""
    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        if config.latent_size % config.patch_size or config.dim % config.heads or config.dim % 2:
            raise ValueError("Latent size must divide into patches; dim must be even and divide into heads")
        self.patch = nn.Conv2d(10, config.dim, config.patch_size, config.patch_size)
        tokens = (config.latent_size // config.patch_size)**2
        self.position = nn.Parameter(torch.randn(1, tokens, config.dim) * .02)
        self.kind = nn.Embedding(2, config.dim)
        self.time = nn.Sequential(nn.Linear(config.dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        self.frame_time = nn.Sequential(nn.Linear(1, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        self.blocks = nn.ModuleList([DiTBlock(config.dim, config.heads) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=1e-6)
        self.final_condition = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, config.dim * 2))
        self.output = nn.Linear(config.dim, 4 * config.patch_size**2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def time_embedding(self, time):
        half = self.config.dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=time.device) / max(half - 1, 1))
        angle = time[..., None] * frequencies * 1000
        return self.time(torch.cat([angle.sin(), angle.cos()], dim=-1))

    def forward(self, x, t, context, context_rays, target_rays, context_times):
        batch, views = context.shape[:2]
        count = self.position.shape[1]
        past = self.patch(torch.cat([context, context_rays], dim=2).flatten(0, 1)).flatten(2).transpose(1, 2)
        past = past.reshape(batch, views, count, -1) + self.position[:, None] + self.kind.weight[0]
        past = past + self.frame_time(context_times[..., None])[:, :, None]
        past = past.reshape(batch, views * count, -1)
        current = self.patch(torch.cat([x, target_rays], dim=1)).flatten(2).transpose(1, 2)
        current = current + self.position + self.kind.weight[1]
        current = current + self.frame_time(torch.zeros_like(t[:, None]))[:, None]
        # 干净历史 t=1，只有当前目标沿噪声到图像潜变量的流路径演化。
        target_condition = self.time_embedding(t)
        past_condition = self.time_embedding(torch.ones_like(t))[:, None].expand(-1, views * count, -1)
        condition = torch.cat([past_condition, target_condition[:, None].expand(-1, count, -1)], dim=1)
        features = torch.cat([past, current], dim=1)
        mask = torch.zeros(features.shape[1], features.shape[1], dtype=torch.bool, device=x.device)
        mask[:views * count, views * count:] = True
        for block in self.blocks:
            features = block(features, condition, mask)
        shift, scale = self.final_condition(target_condition).chunk(2, dim=-1)
        features = self.norm(features[:, -count:]) * (1 + scale[:, None]) + shift[:, None]
        size, patch = self.config.latent_size // self.config.patch_size, self.config.patch_size
        pixels = self.output(features).reshape(batch, size, size, 4, patch, patch)
        return pixels.permute(0, 3, 1, 4, 2, 5).reshape_as(x)


def flow_loss(model, batch, noise=None, time=None):
    """唯一目标是下一帧图像潜变量的速度；不读取深度或深度掩码。"""
    target = batch["target"]
    noise = torch.randn_like(target) if noise is None else noise
    time = torch.rand(target.shape[0], device=target.device) if time is None else time
    x = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * target
    velocity = model(x, time, batch["context"], batch["context_rays"], batch["target_rays"], batch["context_times"])
    return (velocity - (target - noise)).square().mean()


@torch.no_grad()
def sample_flow(model, context, context_rays, target_rays, context_times, steps=20, noise=None):
    if steps < 1:
        raise ValueError("Sampling steps must be positive")
    x = torch.randn_like(context[:, 0]) if noise is None else noise.clone()
    for step in range(steps):
        t = torch.full((x.shape[0],), step / steps, device=x.device)
        x += model(x, t, context, context_rays, target_rays, context_times) / steps
    return x
