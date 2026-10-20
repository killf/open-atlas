"""A numerical rendering contract, independent of Replica download availability."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest


@pytest.mark.skipif(not shutil.which("blender") or not shutil.which("ffmpeg"), reason="Blender and FFmpeg required")
def test_blender_depth_is_camera_z_and_rgb_preserves_vertex_color(tmp_path):
    mesh = tmp_path / "plane.npz"
    np.savez(mesh, vertices=np.array([[-10, 0, -10], [10, 0, -10], [10, 0, 10], [-10, 0, 10]]),
             faces=np.array([[0, 1, 2], [0, 2, 3]]), colors=np.tile([.25, .5, .75], (4, 1)), world_from_raw=np.eye(4))
    scene = tmp_path / "plane"
    scene.mkdir()
    script = Path(__file__).resolve().parents[1] / "scripts/render_replica.py"
    result = subprocess.run(["blender", "--background", "--factory-startup", "--threads", "2", "--python-exit-code", "1",
                             "--python", str(script), "--", "--worker-mesh", str(mesh), "--scene", str(scene),
                             "--output", str(tmp_path / "rendered"), "--trajectories", "pan", "--frames", "2",
                             "--size", "8", "--fov", "90", "--position", "0", "-2", "1", "--target", "0", "0", "1",
                             "--amplitude", "0.000001"], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata_path = next((tmp_path / "rendered").glob("*/transforms.json"))
    metadata = json.loads(metadata_path.read_text())
    for frame in metadata["frames"]:
        depth = np.load(metadata_path.parent / frame["depth"])
        np.testing.assert_allclose(depth, 2, atol=1e-5)
        rgb = np.asarray(Image.open(metadata_path.parent / frame["rgb"]))
        np.testing.assert_allclose(rgb.mean(axis=(0, 1)), [64, 128, 191], atol=2)
        np.testing.assert_allclose(np.array(frame["c2w"]) @ np.array(frame["w2c"]), np.eye(4), atol=1e-6)
