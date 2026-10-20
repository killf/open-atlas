"""Train RGB-conditioned video and depth generation with rectified flow."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import TrajectoryDataset
from .model import AtlasConfig, AtlasMini, flow_loss


def train(data, output, steps=1000, image_size=32, patch_size=4, dim=128, layers=4, heads=4,
          batch_size=4, context_frames=2, max_depth=10., lr=3e-4, seed=0, device="cpu", threads=2,
          validate=False):
    if steps < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Steps, batch size and learning rate must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    dataset = TrajectoryDataset(data, image_size=image_size, context_frames=context_frames, max_depth=max_depth)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation = (TrajectoryDataset(data, split="val", image_size=image_size,
                                    context_frames=context_frames, max_depth=max_depth) if validate else None)
    config = AtlasConfig(image_size, patch_size, dim, layers, heads)
    model = AtlasMini(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    count = sum(p.numel() for p in model.parameters())
    print(f"Training {count:,} parameters on {len(dataset)} target frames; context=RGB; output=RGBD; device={device}", flush=True)
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
                    losses = []
                    devices = [device.index or 0] if device.type == "cuda" else []
                    # 验证使用固定噪声，并在结束后恢复训练的随机数状态。
                    with torch.no_grad(), torch.random.fork_rng(devices=devices):
                        torch.manual_seed(12345)
                        for val_batch in DataLoader(validation, batch_size=batch_size):
                            losses.append(flow_loss(model, {k: v.to(device) for k, v in val_batch.items()}).item())
                    record["val_flow_loss"] = sum(losses) / len(losses)
                print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
    checkpoint = output / "last.pt"
    torch.save({"config": asdict(config), "model": model.state_dict(), "step": steps,
                "context_modality": "rgb", "output_modality": "rgbd",
                "context_frames": context_frames, "max_depth": max_depth, "seed": seed,
                "parameters": count, "training_samples": len(dataset)}, checkpoint)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/open-atlas"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-frames", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=10.)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--validate", action="store_true", help="Evaluate a separate scene-disjoint val split")
    args = parser.parse_args()
    print(f"Saved {train(**vars(args))}")


if __name__ == "__main__":
    main()
