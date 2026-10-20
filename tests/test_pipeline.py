import json

import numpy as np
import pytest
import torch

from test_data import write_clip


def remove_depth(clip):
    for path in clip.glob("*.npy"):
        path.unlink()
    path = clip / "transforms.json"
    metadata = json.loads(path.read_text())
    for frame in metadata["frames"]:
        del frame["depth"]
    path.write_text(json.dumps(metadata))


def test_rgb_only_training_and_rollout_without_future_truth(tmp_path, codec_dir):
    from open_atlas.train import train
    from open_atlas.sample import generate
    from open_atlas.codec import ImageCodec

    root = tmp_path / "data"
    root.mkdir()
    clip = write_clip(root, count=4)
    remove_depth(clip)
    checkpoint = train(root, tmp_path / "run", steps=2, image_size=16, dim=32, layers=1,
                       batch_size=2, context_frames=2, seed=7, codec_dir=codec_dir)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["step"] == 2 and saved["image_size"] == 16
    assert saved["prediction"] == "next_rgb" and saved["camera_frame"] == "first_reference"
    initial_codec = ImageCodec(codec_dir).state_dict()
    for key, value in saved["codec"].items():
        torch.testing.assert_close(value, initial_codec[key], rtol=0, atol=0)
    output = tmp_path / "generated"
    generate(checkpoint, clip / "transforms.json", output, steps=2, seed=3)
    with np.load(output / "rgb.npz", allow_pickle=False) as predictions:
        first = predictions["rgb"].copy()
        assert first.shape == (3, 16, 16, 3) and set(predictions.files) == {"rgb", "c2w", "K"}
    assert not (output / "depth").exists() and not (output / "points.ply").exists()
    for i in range(1, 4):
        (clip / f"{i}.png").unlink()
    generate(checkpoint, clip / "transforms.json", tmp_path / "no_future", steps=2, seed=3)
    with np.load(tmp_path / "no_future/rgb.npz", allow_pickle=False) as predictions:
        np.testing.assert_array_equal(first, predictions["rgb"])


def test_one_frame_context_uses_generated_latent(tmp_path, monkeypatch, codec_dir):
    from open_atlas.train import train
    from open_atlas.sample import generate

    root = tmp_path / "data"
    root.mkdir()
    clip = write_clip(root, count=4)
    remove_depth(clip)
    checkpoint = train(root, tmp_path / "run", steps=1, image_size=16, dim=32, layers=1,
                       context_frames=1, batch_size=1, codec_dir=codec_dir)
    previous = []

    def known_prediction(model, context, context_rays, target_rays, context_times, steps):
        if previous:
            torch.testing.assert_close(context[:, 0], previous[-1])
        result = torch.full_like(context[:, 0], .1 * (len(previous) + 1))
        previous.append(result)
        return result

    monkeypatch.setattr("open_atlas.sample.sample_flow", known_prediction)
    generate(checkpoint, clip / "transforms.json", tmp_path / "generated", steps=1)
    assert len(previous) == 3


def test_evaluation_scores_rollout_against_copy_baseline(tmp_path, codec_dir):
    from open_atlas.train import train
    from open_atlas.evaluate import evaluate

    root = tmp_path / "data"
    root.mkdir()
    remove_depth(write_clip(root, count=4))
    remove_depth(write_clip(root, scene="test_room", split="test", count=4))
    checkpoint = train(root, tmp_path / "run", steps=1, image_size=16, dim=32, layers=1,
                       batch_size=1, codec_dir=codec_dir)
    result = evaluate(checkpoint, root, tmp_path / "evaluation", steps=1, metric_size=8)
    assert result["generated_frames"] == 3 and result["test_clips"] == 1
    assert np.isfinite(result["rgb_psnr"])
    expected = -10 * np.log10(np.mean((np.array([20, 40, 60]) / 255.)**2))
    assert result["copy_first_rgb_psnr"] == pytest.approx(expected)
