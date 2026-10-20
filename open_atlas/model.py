"""Autoregressive RGB-pixel DiT with camera conditioning and rectified flow."""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class AtlasConfig:
    image_size: int = 64
    patch_size: int = 4
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
    """从加噪目标与历史条件预测干净 RGB；无独立图像编解码器。"""
    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        if min(config.image_size, config.patch_size, config.dim, config.layers, config.heads) < 1:
            raise ValueError("Model dimensions must be positive")
        if config.image_size % config.patch_size or config.dim % config.heads or config.dim % 2:
            raise ValueError("Image size must divide into patches; dim must be even and divide into heads")
        self.patch = nn.Conv2d(9, config.dim, config.patch_size, config.patch_size)
        tokens = (config.image_size // config.patch_size)**2
        self.position = nn.Parameter(torch.randn(1, tokens, config.dim) * .02)
        self.kind = nn.Embedding(2, config.dim)
        self.time = nn.Sequential(nn.Linear(config.dim, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        self.frame_time = nn.Sequential(nn.Linear(1, config.dim), nn.SiLU(), nn.Linear(config.dim, config.dim))
        self.blocks = nn.ModuleList([DiTBlock(config.dim, config.heads) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=1e-6)
        self.final_condition = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, config.dim * 2))
        self.output = nn.Linear(config.dim, 3 * config.patch_size**2)
        nn.init.zeros_(self.final_condition[-1].weight)
        nn.init.zeros_(self.final_condition[-1].bias)
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
        # 干净历史 t=1，只有当前目标沿噪声到 RGB 像素的流路径演化。
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
        size, patch = self.config.image_size // self.config.patch_size, self.config.patch_size
        pixels = self.output(features).reshape(batch, size, size, 3, patch, patch)
        return pixels.permute(0, 3, 1, 4, 2, 5).reshape_as(x)


def flow_velocity(image, x, time):
    """JiT 式图像预测转速度；分母下限抑制干净端点的数值放大。"""
    return (image - x) / (1 - time[:, None, None, None]).clamp_min(.05)


def flow_loss(model, batch, noise=None, time=None):
    """网络预测干净下一帧，仅在速度空间计算 RGB 损失。"""
    target = batch["target"]
    noise = torch.randn_like(target) if noise is None else noise
    # 配合图像预测的速度损失，沿用 JiT 时间分布，避免干净端权重主导小批次。
    time = torch.sigmoid(torch.randn(target.shape[0], device=target.device) * .8 - .8) if time is None else time
    x = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * target
    prediction = model(x, time, batch["context"], batch["context_rays"], batch["target_rays"], batch["context_times"])
    # 两端使用相同分母；即使接近 t=1，预测等于真值时损失仍为零。
    return (flow_velocity(prediction, x, time) - flow_velocity(target, x, time)).square().mean()


@torch.no_grad()
def sample_flow(model, context, context_rays, target_rays, context_times, steps=20, noise=None):
    if steps < 1:
        raise ValueError("Sampling steps must be positive")
    x = torch.randn_like(context[:, 0]) if noise is None else noise.clone()
    for step in range(steps):
        t = torch.full((x.shape[0],), step / steps, device=x.device)
        image = model(x, t, context, context_rays, target_rays, context_times)
        # 最后直接取干净预测，避免高采样步数下分母保护留下噪声残差。
        if step == steps - 1:
            return image
        x += flow_velocity(image, x, t) / steps
