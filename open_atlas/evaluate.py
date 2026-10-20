"""Score free-running next-RGB generation against a repeated-reference baseline."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .sample import generate


def _image(path, size):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float64) / 255


def _psnr(squared_error, count):
    return -10 * math.log10(max(squared_error / count, 1e-12))


def evaluate(checkpoint, data, output, split="test", steps=20, seed=0, metric_size=32, device="cpu"):
    if metric_size < 1:
        raise ValueError("Metric size must be positive")
    data, output = Path(data), Path(output)
    paths = [p for p in sorted(data.rglob("transforms.json")) if json.loads(p.read_text())["split"] == split]
    if not paths:
        raise ValueError(f"No {split} clips under {data}")
    if output.exists():
        raise ValueError("Choose a fresh evaluation directory")
    output.mkdir(parents=True)
    records, total_squared, total_copy, total_absolute, count = [], 0., 0., 0., 0
    for path in paths:
        metadata = json.loads(path.read_text())
        destination = output / path.parent.relative_to(data)
        generate(checkpoint, path, destination, steps=steps, seed=seed, device=device)
        # 评分阶段才读取目标图像；生成器只看到首帧 RGB 与相机序列。
        reference = _image(path.parent / metadata["frames"][0]["rgb"], metric_size)
        squared, copy_squared, absolute, pixels = 0., 0., 0., 0
        for index, frame in enumerate(metadata["frames"][1:]):
            truth = _image(path.parent / frame["rgb"], metric_size)
            prediction = _image(destination / "rgb" / f"{index:06d}.png", metric_size)
            squared += float(np.square(prediction - truth).sum())
            copy_squared += float(np.square(reference - truth).sum())
            absolute += float(np.abs(prediction - truth).sum())
            pixels += truth.size
        records.append({"clip": str(path.parent.relative_to(data)), "generated_frames": len(metadata["frames"]) - 1,
                        "rgb_psnr": _psnr(squared, pixels), "rgb_mae": absolute / pixels,
                        "copy_first_rgb_psnr": _psnr(copy_squared, pixels)})
        total_squared += squared
        total_copy += copy_squared
        total_absolute += absolute
        count += pixels
    result = {"checkpoint": str(checkpoint), "split": split, "test_clips": len(paths),
              "generated_frames": sum(r["generated_frames"] for r in records), "metric_size": metric_size,
              "sampling_steps": steps, "seed": seed,
              "protocol": "one reference RGB, known cameras, free autoregressive rollout; no depth loaded",
              "rgb_psnr": _psnr(total_squared, count), "rgb_mae": total_absolute / count,
              "copy_first_rgb_psnr": _psnr(total_copy, count), "per_clip": records}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_clip"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    evaluate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
