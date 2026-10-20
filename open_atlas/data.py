"""Load rendered Replica trajectories without target or future-frame leakage."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .geometry import rays


def encode_depth(depth, max_depth=10.):
    """把有效深度对数映射到 [-1,1]，同时保留监督掩码。"""
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    depth = np.where(valid, depth, 0)
    return (2 * np.log1p(depth) / np.log1p(max_depth) - 1).astype(np.float32), valid


def decode_depth(encoded, max_depth=10.):
    return np.expm1((np.clip(encoded, -1, 1) + 1) * 0.5 * np.log1p(max_depth))


def context_indices(length, count):
    # 单帧上下文必须取最近一帧；多帧则均匀覆盖可见历史。
    return np.linspace(0, length - 1, count).astype(int) if count > 1 else np.array([length - 1])


def load_rgb(directory, metadata, frame, image_size):
    """读取 RGB 和相机射线；上下文与推理路径不访问任何真值深度。"""
    with Image.open(directory / frame["rgb"]) as image:
        rgb = np.asarray(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32)
    K = np.array(metadata["K"], dtype=np.float64)
    # 图像尺寸变化时，焦距和主点必须按同样比例缩放。
    K[0] *= image_size / metadata["width"]
    K[1] *= image_size / metadata["height"]
    camera_rays = rays(K, np.array(frame["c2w"]), image_size, image_size)
    return torch.from_numpy((rgb / 127.5 - 1).transpose(2, 0, 1)), torch.from_numpy(camera_rays)


def load_frame(directory, metadata, frame, image_size, max_depth):
    """深度只用于目标监督，不作为参考帧条件。"""
    rgb, camera_rays = load_rgb(directory, metadata, frame, image_size)
    depth = np.load(directory / frame["depth"], allow_pickle=False)
    # 最近邻缩放避免把无效深度与前后景深度混合。
    depth = np.asarray(Image.fromarray(depth).resize((image_size, image_size), Image.Resampling.NEAREST))
    encoded, valid = encode_depth(depth, max_depth)
    rgbd = torch.cat([rgb, torch.from_numpy(encoded[None])])
    return rgbd, camera_rays, torch.from_numpy(valid[None].astype(np.float32))


class TrajectoryDataset(Dataset):
    def __init__(self, root, split="train", image_size=32, context_frames=2, max_depth=10.):
        self.image_size, self.context_frames, self.max_depth = image_size, context_frames, max_depth
        if context_frames < 1 or image_size < 1 or max_depth <= 0:
            raise ValueError("Image size, context frames and max depth must be positive")
        self.clips, self.samples = [], []
        # 扫描所有 split，确保同一场景不会同时用于训练与验证。
        scene_splits = {}
        for path in sorted(Path(root).rglob("transforms.json")):
            metadata = json.loads(path.read_text())
            scene = metadata["scene_id"]
            clip_split = metadata["split"]
            if scene in scene_splits and scene_splits[scene] != clip_split:
                raise ValueError(f"Scene {scene} appears in more than one split")
            scene_splits[scene] = clip_split
            if clip_split == split:
                index = len(self.clips)
                self.clips.append((path.parent, metadata))
                self.samples.extend((index, target) for target in range(1, len(metadata["frames"])))
        if not self.samples:
            raise ValueError(f"No {split} trajectory samples under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        clip, target_id = self.samples[index]
        directory, metadata = self.clips[clip]
        context_ids = context_indices(target_id, self.context_frames)
        history = [load_rgb(directory, metadata, metadata["frames"][int(i)], self.image_size)
                   for i in context_ids]
        target, target_rays, valid = load_frame(directory, metadata, metadata["frames"][target_id],
                                               self.image_size, self.max_depth)
        return {"context": torch.stack([h[0] for h in history]),
                "context_rays": torch.stack([h[1] for h in history]),
                "target": target, "target_rays": target_rays, "valid_depth": valid}
