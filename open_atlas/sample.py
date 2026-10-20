"""Generate video and predicted depth from RGB references and target camera poses."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from .data import context_indices, decode_depth, load_rgb
from .geometry import camera_directions, rays
from .model import AtlasConfig, AtlasMini, sample_flow


def save_point_cloud(path, colors, depths, poses, K):
    """相机 Z 深度乘以未归一化射线得到相机坐标，再变换到世界坐标。"""
    points, rgb = [], []
    for color, depth, pose in zip(colors, depths, poses):
        xyz = camera_directions(K, *depth.shape) * depth[..., None]
        valid = np.isfinite(depth) & (depth > .005)
        points.append(xyz[valid] @ pose[:3, :3].T + pose[:3, 3])
        rgb.append(color[valid])
    xyz, rgb = np.concatenate(points), np.concatenate(rgb)
    with Path(path).open("w") as handle:
        handle.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n"
                     "property float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(handle, np.concatenate([xyz, rgb], axis=1), fmt="%.6f %.6f %.6f %d %d %d")


@torch.no_grad()
def generate(checkpoint, trajectory, output, steps=20, seed=0, reference_frames=1, device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = torch.device(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("context_modality") != "rgb":
        raise ValueError("This checkpoint requires RGBD context; retrain with RGB-only context")
    config = AtlasConfig(**saved["config"])
    model = AtlasMini(config).to(device).eval()
    model.load_state_dict(saved["model"])
    trajectory, output = Path(trajectory), Path(output)
    metadata = json.loads(trajectory.read_text())
    frames = metadata["frames"]
    if not 1 <= reference_frames < len(frames):
        raise ValueError("Need at least one reference and one target frame")
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a fresh directory")
    (output / "rgb").mkdir(parents=True)
    (output / "depth").mkdir()
    size, max_depth = config.image_size, saved["max_depth"]
    K = np.array(metadata["K"], dtype=np.float64)
    K[0] *= size / metadata["width"]
    K[1] *= size / metadata["height"]
    history = []
    for frame in frames[:reference_frames]:
        rgb, ray = load_rgb(trajectory.parent, metadata, frame, size)
        history.append((rgb.to(device), ray.to(device)))
    colors, depths, poses, predicted_frames = [], [], [], []
    for index, frame in enumerate(frames[reference_frames:]):
        pose = np.array(frame["c2w"])
        target_rays = torch.from_numpy(rays(K, pose, size, size)).to(device)
        selected = context_indices(len(history), saved["context_frames"])
        context = torch.stack([history[i][0] for i in selected])[None]
        context_rays = torch.stack([history[i][1] for i in selected])[None]
        prediction = sample_flow(model, context, context_rays, target_rays[None], steps)[0]
        # 后续帧只使用生成的 RGB 作为历史，不读取或依赖任何真值深度。
        history.append((prediction[:3], target_rays))
        rgbd = prediction.cpu().numpy()
        color = np.clip(np.rint((rgbd[:3].transpose(1, 2, 0) + 1) * 127.5), 0, 255).astype(np.uint8)
        depth = decode_depth(rgbd[3], max_depth).astype(np.float32)
        Image.fromarray(color).save(output / "rgb" / f"{index:06d}.png")
        np.save(output / "depth" / f"{index:06d}.npy", depth)
        colors.append(color)
        depths.append(depth)
        poses.append(pose)
        predicted_frames.append({"frame_id": frame.get("frame_id", index + reference_frames),
                                 "timestamp": frame.get("timestamp", index / metadata.get("fps", 12)),
                                 "rgb": f"rgb/{index:06d}.png", "depth": f"depth/{index:06d}.npy",
                                 "c2w": pose.tolist(), "w2c": np.linalg.inv(pose).tolist()})
    np.savez_compressed(output / "rgbd.npz", rgb=np.stack(colors), depth=np.stack(depths), c2w=np.stack(poses), K=K)
    save_point_cloud(output / "points.ply", colors, depths, poses, K)
    result = {**metadata, "split": "generated", "width": size, "height": size, "K": K.tolist(),
              "reference_frames": reference_frames, "checkpoint": str(checkpoint), "seed": seed,
              "sampling_steps": steps, "frames": predicted_frames}
    (output / "transforms.json").write_text(json.dumps(result, indent=2) + "\n")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(metadata.get("fps", 12)),
                    "-i", str(output / "rgb/%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(output / "preview.mp4")], check=True)
    print(f"Generated {len(colors)} frames at {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True, help="transforms.json with reference RGB and target cameras")
    parser.add_argument("--output", type=Path, default=Path("outputs/sample"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-frames", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    generate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
