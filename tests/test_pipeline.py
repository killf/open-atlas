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
    output = tmp_path / "generated"
    generate(checkpoint, clip / "transforms.json", output, steps=2, seed=3)
    predictions = np.load(output / "rgbd.npz", allow_pickle=False)
    assert predictions["rgb"].shape == (3, 8, 8, 3)
    assert np.isfinite(predictions["depth"]).all()
    assert (output / "points.ply").is_file()
    first = predictions["rgb"].copy()
    # Future reference RGBD is supervision only; deleting it must not affect sampling.
    for i in range(1, 4):
        (clip / f"{i}.png").unlink()
        (clip / f"{i}.npy").unlink()
    generate(checkpoint, clip / "transforms.json", tmp_path / "no_future", steps=2, seed=3)
    second = np.load(tmp_path / "no_future/rgbd.npz", allow_pickle=False)["rgb"]
    np.testing.assert_array_equal(first, second)


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
            torch.testing.assert_close(context[:, 0], previous[-1])
        result = torch.full_like(context[:, 0], .1 * (len(previous) + 1))
        previous.append(result)
        return result

    monkeypatch.setattr("open_atlas.sample.sample_flow", known_prediction)
    generate(checkpoint, clip / "transforms.json", tmp_path / "generated", steps=1)
    assert len(previous) == 3
