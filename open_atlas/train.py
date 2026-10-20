"""Train next-frame RGB pixel generation without an image codec or depth supervision."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import TrajectoryDataset
from .model import AtlasConfig, AtlasMini, flow_loss


def train(data, output, steps=1000, image_size=64, patch_size=4, dim=128, layers=4, heads=4,
          batch_size=4, context_frames=2, lr=3e-4, seed=0, device="cpu", threads=2,
          validate=False):
    if steps < 1 or batch_size < 1 or lr <= 0 or image_size < 1:
        raise ValueError("Need positive steps, batch size, learning rate and image size")
    output = Path(output)
    if (output / "metrics.jsonl").exists() or (output / "last.pt").exists():
        raise ValueError("Choose a fresh training directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    device = torch.device(device)
    dataset = TrajectoryDataset(data, image_size=image_size, context_frames=context_frames)
    validation = TrajectoryDataset(data, split="val", image_size=image_size, context_frames=context_frames) if validate else None
    config = AtlasConfig(image_size, patch_size, dim, layers, heads)
    model = AtlasMini(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    count = sum(p.numel() for p in model.parameters())
    print(f"Training {count:,} parameters on {len(dataset)} next-RGB targets; device={device}", flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    iterator = iter(loader)
    with (output / "metrics.jsonl").open("w") as log:
        for step in range(1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            model.train()
            loss = flow_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record = {"step": step, "flow_loss": loss.item()}
            if step == 1 or step % 20 == 0 or step == steps:
                if validation is not None:
                    model.eval()
                    total, samples = 0., 0
                    devices = [device.index or 0] if device.type == "cuda" else []
                    # 固定验证噪声，恢复随机数状态以免改变后续训练序列。
                    with torch.no_grad(), torch.random.fork_rng(devices=devices):
                        torch.manual_seed(12345)
                        for values in DataLoader(validation, batch_size=batch_size):
                            size = values["target"].shape[0]
                            total += flow_loss(model, {k: v.to(device) for k, v in values.items()}).item() * size
                            samples += size
                    record["val_flow_loss"] = total / samples
                print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
    checkpoint = output / "last.pt"
    torch.save({"format_version": 4, "prediction": "next_rgb", "representation": "rgb_pixels",
                "parameterization": "clean_rgb",
                "time_sampling": "uniform",
                "camera_frame": "first_reference", "config": asdict(config), "model": model.state_dict(),
                "image_size": image_size, "step": steps, "context_frames": context_frames,
                "seed": seed, "parameters": count, "training_samples": len(dataset)}, checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/open-atlas"))
    for name, default in [("steps", 1000), ("image-size", 64), ("patch-size", 4), ("dim", 128),
                          ("layers", 4), ("heads", 4), ("batch-size", 4), ("context-frames", 2), ("seed", 0), ("threads", 2)]:
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate", action="store_true")
    print(f"Saved {train(**vars(parser.parse_args()))}")


if __name__ == "__main__":
    main()
