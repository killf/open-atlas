import json

import numpy as np
import torch

from test_data import write_clip


def test_training_checkpoint_and_autoregressive_rollout(tmp_path):
    from open_atlas.train import train
    from open_atlas.sample import generate

    torch.set_num_threads(2)
    root = tmp_path / "data"
    root.mkdir()
    clip = write_clip(root, count=4)
    checkpoint = train(root, tmp_path / "run", steps=2, image_size=8, dim=32, layers=1,
                       batch_size=2, context_frames=2, seed=7)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["step"] == 2
    assert saved["config"]["image_size"] == 8
    assert saved["context_modality"] == "rgb"
    output = tmp_path / "generated"
    generate(checkpoint, clip / "transforms.json", output, steps=2, seed=3)
    predictions = np.load(output / "rgbd.npz", allow_pickle=False)
    assert predictions["rgb"].shape == (3, 8, 8, 3)
    assert np.isfinite(predictions["depth"]).all()
    assert (output / "points.ply").is_file()
    first = predictions["rgb"].copy()
    # 所有真值深度和目标 RGB 都只用于监督，推理只能读取参考 RGB 与相机。
    first_depth = predictions["depth"].copy()
    (clip / "0.npy").unlink()
    for i in range(1, 4):
        (clip / f"{i}.png").unlink()
        (clip / f"{i}.npy").unlink()
    metadata = json.loads((clip / "transforms.json").read_text())
    for frame in metadata["frames"]:
        del frame["depth"]
    (clip / "transforms.json").write_text(json.dumps(metadata))
    generate(checkpoint, clip / "transforms.json", tmp_path / "no_future", steps=2, seed=3)
    second = np.load(tmp_path / "no_future/rgbd.npz", allow_pickle=False)["rgb"]
    np.testing.assert_array_equal(first, second)
    second_depth = np.load(tmp_path / "no_future/rgbd.npz", allow_pickle=False)["depth"]
    np.testing.assert_array_equal(first_depth, second_depth)


def test_one_frame_context_rollout_consumes_generated_history(tmp_path, monkeypatch):
    from open_atlas.train import train
    from open_atlas.sample import generate

    root = tmp_path / "data"
    root.mkdir()
    clip = write_clip(root, count=4)
    checkpoint = train(root, tmp_path / "run", steps=1, image_size=8, dim=32, layers=1,
                       context_frames=1, batch_size=1)
    previous = []

    def known_prediction(model, context, context_rays, target_rays, steps):
        if previous:
            torch.testing.assert_close(context[:, 0], previous[-1][:, :3])
        result = context.new_full((context.shape[0], 4, *context.shape[-2:]), .1 * (len(previous) + 1))
        previous.append(result)
        return result

    monkeypatch.setattr("open_atlas.sample.sample_flow", known_prediction)
    generate(checkpoint, clip / "transforms.json", tmp_path / "generated", steps=1)
    assert len(previous) == 3
