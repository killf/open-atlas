"""Read only RGB and known cameras; the target is the next observed frame."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .geometry import rays


def context_indices(length, count):
    return np.linspace(0, length - 1, count).astype(int) if count > 1 else np.array([length - 1])


def load_image(directory, frame, image_size):
    with Image.open(directory / frame["rgb"]) as image:
        rgb = np.asarray(image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32)
    return torch.from_numpy((rgb / 255).transpose(2, 0, 1))


def camera_rays(metadata, frame, size):
    """始终以第一参考帧为坐标原点；世界坐标的刚体变换不应改变条件。"""
    reference = np.asarray(metadata["frames"][0]["c2w"], dtype=np.float64)
    relative = np.linalg.solve(reference, np.asarray(frame["c2w"], dtype=np.float64))
    K = np.array(metadata["K"], dtype=np.float64)
    K[0] *= size / metadata["width"]
    K[1] *= size / metadata["height"]
    return torch.from_numpy(rays(K, relative, size, size))


def history_times(metadata, target_id, indices):
    # 视频时间与扩散时间独立；保留目标与可见历史之间的真实时间间隔。
    def time(index):
        frame = metadata["frames"][int(index)]
        return frame.get("timestamp", frame.get("frame_id", int(index)) / metadata.get("fps", 12))
    return torch.tensor([time(index) - time(target_id) for index in indices], dtype=torch.float32)


class TrajectoryDataset(Dataset):
    def __init__(self, root, split="train", image_size=64, context_frames=2):
        if context_frames < 1 or image_size < 1:
            raise ValueError("Image size and context frames must be positive")
        self.image_size, self.context_frames = image_size, context_frames
        self.clips, self.samples = [], []
        scene_splits = {}
        for path in sorted(Path(root).rglob("transforms.json")):
            metadata = json.loads(path.read_text())
            scene, clip_split = metadata["scene_id"], metadata["split"]
            if scene in scene_splits and scene_splits[scene] != clip_split:
                raise ValueError(f"Scene {scene} appears in more than one split")
            scene_splits[scene] = clip_split
            if clip_split == split:
                self.clips.append((path.parent, metadata))
                self.samples.extend((len(self.clips) - 1, target) for target in range(1, len(metadata["frames"])))
        if not self.samples:
            raise ValueError(f"No {split} trajectory samples under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        clip, target_id = self.samples[index]
        directory, metadata = self.clips[clip]
        indices = context_indices(target_id, self.context_frames)
        frames = metadata["frames"]
        def image(frame_id):
            return load_image(directory, frames[int(frame_id)], self.image_size)
        return {"context": torch.stack([image(i) for i in indices]), "target": image(target_id),
                "context_rays": torch.stack([camera_rays(metadata, frames[int(i)], self.image_size) for i in indices]),
                "target_rays": camera_rays(metadata, frames[target_id], self.image_size),
                "context_times": history_times(metadata, target_id, indices)}
