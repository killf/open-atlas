import json

import numpy as np
import pytest

from test_data import write_clip


def test_overfit_uses_only_selected_training_frames_and_scores_real_history(tmp_path):
    from open_atlas.overfit import overfit

    source = write_clip(tmp_path, count=4)
    original = (source / "transforms.json").read_bytes()
    result = overfit(source / "transforms.json", tmp_path / "run", frames=3, steps=1,
                     image_size=16, sampling_steps=2)
    assert result["frames"] == 3 and result["targets"] == 2
    assert result["parameterization"] == "clean_rgb"
    assert len(result["per_seed"]) == 3 and np.isfinite(result["rgb_psnr"])
    expected = -20 * np.log10(20 / 255)
    assert result["copy_last_rgb_psnr"] == pytest.approx(expected)
    copied = json.loads((tmp_path / "run/data/clip/transforms.json").read_text())
    assert len(copied["frames"]) == 3 and all("depth" not in f for f in copied["frames"])
    assert not list((tmp_path / "run/data").rglob("*.npy"))
    assert (source / "transforms.json").read_bytes() == original
    assert len(list((tmp_path / "run/predictions").rglob("*.png"))) == 6


def test_overfit_rejects_non_training_trajectory(tmp_path):
    from open_atlas.overfit import overfit

    source = write_clip(tmp_path, split="val")
    with pytest.raises(ValueError, match="train"):
        overfit(source / "transforms.json", tmp_path / "run", steps=1)
