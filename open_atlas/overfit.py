"""Memorize eight training frames and score single-step generation with real history."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .data import TrajectoryDataset
from .evaluate import _psnr
from .model import sample_flow
from .sample import load_checkpoint
from .train import train


def _rgb(tensor):
    return np.rint(tensor.detach().clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)


def overfit(trajectory, output, frames=8, steps=1000, image_size=64, sampling_steps=20, seed=42, device="cpu"):
    trajectory, output = Path(trajectory), Path(output)
    metadata = json.loads(trajectory.read_text())
    if metadata["split"] != "train":
        raise ValueError("Overfit diagnostics must use a train trajectory")
    if not 2 <= frames <= len(metadata["frames"]):
        raise ValueError("Need at least two available frames")
    if output.exists():
        raise ValueError("Choose a fresh overfit output directory")
    clip = output / "data/clip"
    (clip / "rgb").mkdir(parents=True)
    selected, hashes = [], []
    for index, frame in enumerate(metadata["frames"][:frames]):
        name = f"rgb/{index:06d}.png"
        source = trajectory.parent / frame["rgb"]
        shutil.copyfile(source, clip / name)
        selected.append({**{k: v for k, v in frame.items() if k != "depth"}, "rgb": name})
        hashes.append({"source_rgb": frame["rgb"], "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    subset = {k: v for k, v in metadata.items() if k != "depth_convention"}
    subset["frames"] = selected
    (clip / "transforms.json").write_text(json.dumps(subset, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps({"source": str(trajectory), "frames": hashes}, indent=2) + "\n")
    train(output / "data", output / "train", steps=steps, image_size=image_size,
          seed=seed, device=device)
    return evaluate_overfit(output, sampling_steps, device)


def evaluate_overfit(output, sampling_steps=20, device="cpu"):
    """重新加载检查点，用真实历史评测；可独立复核，无需再次训练。"""
    output = Path(output)
    checkpoint = output / "train/last.pt"
    manifest = json.loads((output / "manifest.json").read_text())
    saved, model = load_checkpoint(checkpoint, device)
    image_size = saved["image_size"]
    dataset = TrajectoryDataset(output / "data", image_size=image_size, context_frames=saved["context_frames"])
    batch = {k: v.to(device) for k, v in next(iter(DataLoader(dataset, batch_size=len(dataset)))).items()}
    truth = _rgb(batch["target"]).astype(np.float64) / 255
    previous = _rgb(batch["context"][:, -1]).astype(np.float64) / 255
    copy_error = np.square(previous - truth)
    records, squared, absolute = [], 0., 0.
    # 只测单步；每个目标的历史均是真实 RGB，三个独立噪声种子检查稳定性。
    for sampling_seed in [0, 1, 2]:
        torch.manual_seed(sampling_seed)
        prediction = sample_flow(model, batch["context"], batch["context_rays"], batch["target_rays"],
                                 batch["context_times"], steps=sampling_steps)
        rgb = _rgb(prediction)
        directory = output / "predictions" / f"seed_{sampling_seed}"
        directory.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(rgb, start=1):
            Image.fromarray(image).save(directory / f"{index:06d}.png")
        error = rgb.astype(np.float64) / 255 - truth
        total = float(np.square(error).sum())
        records.append({"seed": sampling_seed, "rgb_psnr": _psnr(total, error.size),
                        "rgb_mae": float(np.abs(error).mean())})
        squared += total
        absolute += float(np.abs(error).sum())
    pixels = truth.size * len(records)
    result = {"checkpoint": str(checkpoint), "source": manifest["source"], "frames": len(manifest["frames"]), "targets": len(dataset),
              "parameterization": saved["parameterization"], "time_sampling": saved.get("time_sampling", "uniform"),
              "training_steps": saved["step"], "training_seed": saved["seed"],
              "image_size": image_size, "sampling_steps": sampling_steps, "parameters": saved["parameters"], "model_config": saved["config"],
              "protocol": "training-set memorization only; one step with real past RGB; seeds 0,1,2; native resolution",
              "rgb_psnr": _psnr(squared, pixels), "rgb_mae": absolute / pixels,
              "copy_last_rgb_psnr": _psnr(float(copy_error.sum()), copy_error.size), "per_seed": records}
    result["passed"] = result["rgb_psnr"] >= 30 and result["rgb_psnr"] >= result["copy_last_rgb_psnr"] + 3
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name, default in [("frames", 8), ("steps", 1000), ("image-size", 64), ("sampling-steps", 20), ("seed", 42)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    overfit(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
