"""Small RGBD DiT with camera-ray conditioning and a rectified-flow objective."""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class AtlasConfig:
    image_size: int = 32
    patch_size: int = 4
    dim: int = 128
    layers: int = 4
    heads: int = 4


class AtlasMini(nn.Module):
    """将 RGBD 与相机射线编码为 patch token，每次预测一帧的流速度。"""

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        if config.image_size % config.patch_size or config.dim % config.heads:
            raise ValueError("Image size must divide into patches; dim must divide into heads")
        # 4 个 RGBD 通道 + 6 个 Plücker 射线通道，共享同一 patch 编码器。
        self.patch = nn.Conv2d(10, config.dim, config.patch_size, config.patch_size)
        tokens = (config.image_size // config.patch_size) ** 2
        self.position = nn.Parameter(torch.randn(1, tokens, config.dim) * 0.02)
        self.kind = nn.Embedding(2, config.dim)
        self.time = nn.Sequential(nn.Linear(config.dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        block = nn.TransformerEncoderLayer(config.dim, config.heads, config.dim * 4,
                                           dropout=0., activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, config.layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(config.dim), nn.Linear(config.dim, 4 * config.patch_size**2))

    def time_embedding(self, time):
        half = self.config.dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=time.device) / max(half - 1, 1))
        angle = time[..., None] * frequencies * 1000
        return self.time(torch.cat([angle.sin(), angle.cos()], dim=-1))

    def forward(self, x, t, context, context_rays, target_rays):
        """x 为 [B,4,H,W]，历史为 [B,V,4,H,W]，每条射线有 6 个分量。"""
        batch, views = context.shape[:2]
        tokens_per_view = self.position.shape[1]
        past = torch.cat([context, context_rays], dim=2).flatten(0, 1)
        past = self.patch(past).flatten(2).transpose(1, 2) + self.position
        # t=1 表示无噪声历史；当前目标则使用实际的流时间 t。
        past = past + self.kind.weight[0] + self.time_embedding(torch.ones(batch * views, device=x.device))[:, None]
        past = past.reshape(batch, views * tokens_per_view, -1)
        target = self.patch(torch.cat([x, target_rays], dim=1)).flatten(2).transpose(1, 2)
        target = target + self.position + self.kind.weight[1] + self.time_embedding(t)[:, None]
        length = past.shape[1] + tokens_per_view
        mask = torch.zeros(length, length, dtype=torch.bool, device=x.device)
        # 历史不能读取当前目标；当前目标可以读取全部历史。
        mask[:past.shape[1], past.shape[1]:] = True
        features = self.transformer(torch.cat([past, target], dim=1), mask=mask)[:, -tokens_per_view:]
        size = self.config.image_size // self.config.patch_size
        patches = self.output(features).reshape(batch, size, size, 4, self.config.patch_size, self.config.patch_size)
        # 交错排列 patch 网格与 patch 内像素，恢复 [B,4,H,W]。
        return patches.permute(0, 3, 1, 4, 2, 5).reshape_as(x)


def flow_loss(model, batch, noise=None, time=None):
    """沿 noise→target 的直线路径，回归恒定速度 target-noise。"""
    target = batch["target"]
    noise = torch.randn_like(target) if noise is None else noise
    time = torch.rand(target.shape[0], device=target.device) if time is None else time
    t = time[:, None, None, None]
    x = (1 - t) * noise + t * target
    velocity = model(x, time, batch["context"], batch["context_rays"], batch["target_rays"])
    # RGB 始终参与监督，缺失或超出量程的深度不计入损失。
    weights = torch.cat([torch.ones_like(target[:, :3]), batch["valid_depth"]], dim=1)
    return ((velocity - (target - noise)).square() * weights).sum() / weights.sum().clamp_min(1)


@torch.no_grad()
def sample_flow(model, context, context_rays, target_rays, steps=20, noise=None):
    """从 t=0 的噪声出发，用 Euler 积分生成 t=1 的 RGBD。"""
    if steps < 1:
        raise ValueError("Sampling steps must be positive")
    x = torch.randn_like(context[:, 0]) if noise is None else noise.clone()
    for step in range(steps):
        t = torch.full((x.shape[0],), step / steps, device=x.device)
        x += model(x, t, context, context_rays, target_rays) / steps
    return x.clamp(-1, 1)
