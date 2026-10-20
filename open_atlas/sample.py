"""Generate next RGB frames autoregressively at given camera poses."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from .data import camera_rays, context_indices, history_times, load_image
from .model import AtlasConfig, AtlasMini, sample_flow


def load_checkpoint(checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved.get("format_version") != 4 or saved.get("prediction") != "next_rgb"
            or saved.get("representation") != "rgb_pixels" or saved.get("parameterization") != "clean_rgb"):
        raise ValueError("Checkpoint is not a clean-RGB pixel model; retrain with the current code")
    model = AtlasMini(AtlasConfig(**saved["config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    return saved, model


@torch.no_grad()
def generate(checkpoint, trajectory, output, steps=20, seed=0, reference_frames=1, device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = torch.device(device)
    saved, model = load_checkpoint(checkpoint, device)
    trajectory, output = Path(trajectory), Path(output)
    metadata = json.loads(trajectory.read_text())
    frames = metadata["frames"]
    if not 1 <= reference_frames < len(frames):
        raise ValueError("Need at least one reference and one target frame")
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a fresh directory")
    (output / "rgb").mkdir(parents=True)
    size = saved["image_size"]
    K = np.array(metadata["K"], dtype=np.float64)
    K[0] *= size / metadata["width"]
    K[1] *= size / metadata["height"]
    history = []
    for frame in frames[:reference_frames]:
        history.append(load_image(trajectory.parent, frame, size).to(device))
    colors, poses, predicted_frames = [], [], []
    for target_id in range(reference_frames, len(frames)):
        frame = frames[target_id]
        indices = context_indices(len(history), saved["context_frames"])
        context = torch.stack([history[i] for i in indices])[None]
        context_rays = torch.stack([camera_rays(metadata, frames[int(i)], size) for i in indices])[None].to(device)
        target_rays = camera_rays(metadata, frame, size)[None].to(device)
        times = history_times(metadata, target_id, indices)[None].to(device)
        prediction = sample_flow(model, context, context_rays, target_rays, times, steps)[0].clamp(0, 1)
        # 积分结束后才裁剪到 RGB 范围；预测像素直接反馈，不读取目标图像。
        history.append(prediction)
        rgb = prediction.cpu().numpy().transpose(1, 2, 0)
        color = np.rint(rgb * 255).astype(np.uint8)
        index = len(colors)
        Image.fromarray(color).save(output / "rgb" / f"{index:06d}.png")
        pose = np.array(frame["c2w"])
        colors.append(color)
        poses.append(pose)
        predicted_frames.append({"frame_id": frame.get("frame_id", target_id),
                                 "timestamp": frame.get("timestamp", target_id / metadata.get("fps", 12)),
                                 "rgb": f"rgb/{index:06d}.png", "c2w": pose.tolist(),
                                 "w2c": np.linalg.inv(pose).tolist()})
    np.savez_compressed(output / "rgb.npz", rgb=np.stack(colors), c2w=np.stack(poses), K=K)
    result = {key: value for key, value in metadata.items() if key != "depth_convention"}
    result.update(split="generated", width=size, height=size, K=K.tolist(), reference_frames=reference_frames,
                  checkpoint=str(checkpoint), seed=seed, sampling_steps=steps, prediction="next_rgb",
                  camera_condition_frame="first_reference", frames=predicted_frames)
    (output / "transforms.json").write_text(json.dumps(result, indent=2) + "\n")
    # yuv420p 要求偶数尺寸；仅为预览补边，原始 RGB 与相机参数保持不变。
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(metadata.get("fps", 12)),
                    "-i", str(output / "rgb/%06d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(output / "preview.mp4")], check=True)
    print(f"Generated {len(colors)} RGB frames at {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/sample"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-frames", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    generate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
