import json

import numpy as np
from PIL import Image
import pytest
import torch


def write_clip(root, scene="room_0", split="train", count=5):
    directory = root / f"{scene}_{split}"
    directory.mkdir()
    frames = []
    for i in range(count):
        Image.fromarray(np.full((8, 8, 3), 20 * i, dtype=np.uint8)).save(directory / f"{i}.png")
        np.save(directory / f"{i}.npy", np.full((8, 8), i + 1, dtype=np.float32))
        pose = np.eye(4)
        pose[0, 3] = i * 0.1
        frames.append({"rgb": f"{i}.png", "depth": f"{i}.npy", "c2w": pose.tolist(), "frame_id": i})
    metadata = {"scene_id": scene, "split": split, "width": 8, "height": 8,
                "K": [[8, 0, 4], [0, 8, 4], [0, 0, 1]], "frames": frames}
    (directory / "transforms.json").write_text(json.dumps(metadata))
    return directory


def test_camera_looks_at_target_and_projects_center():
    from open_atlas.geometry import look_at, intrinsics, rays

    pose = look_at(np.array([0., -2., 1.]), np.array([0., 0., 1.]))
    point = np.linalg.inv(pose) @ np.array([0., 0., 1., 1.])
    K = intrinsics(8, 8, 60)
    uv = K @ point[:3]
    np.testing.assert_allclose(uv[:2] / uv[2], [4, 4], atol=1e-6)
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
    result = rays(K, pose, 8, 8)
    np.testing.assert_allclose(np.linalg.norm(result[:3], axis=0), 1, atol=1e-6)
    np.testing.assert_allclose((result[:3] * result[3:]).sum(0), 0, atol=1e-6)


def test_dataset_uses_only_past_frames_and_scales_intrinsics(tmp_path):
    from open_atlas.data import TrajectoryDataset
    from open_atlas.geometry import rays

    write_clip(tmp_path)
    dataset = TrajectoryDataset(tmp_path, image_size=4, context_frames=2)
    assert len(dataset) == 4
    sample = dataset[2]
    assert sample["context"].shape == (2, 3, 4, 4)
    assert sample["target"].shape == (3, 4, 4)
    # 测试帧灰度随时间递增，直接检查实际上下文没有读到目标或未来图像。
    assert sample["context"][:, :3].max() < sample["target"][:3].min()
    expected = rays(np.array([[4, 0, 2], [0, 4, 2], [0, 0, 1]]), np.eye(4), 4, 4)
    np.testing.assert_allclose(sample["context_rays"][0, :3], expected[:3], atol=1e-6)
    assert torch.isfinite(sample["target"]).all()


def test_training_never_reads_depth_and_predicts_next_rgb(tmp_path):
    from open_atlas.data import TrajectoryDataset

    clip = write_clip(tmp_path)
    for path in clip.glob("*.npy"):
        path.unlink()
    metadata = json.loads((clip / "transforms.json").read_text())
    for frame in metadata["frames"]:
        del frame["depth"]
    (clip / "transforms.json").write_text(json.dumps(metadata))
    sample = TrajectoryDataset(tmp_path, image_size=8, context_frames=2)[2]
    assert sample["context"].shape == (2, 3, 8, 8)
    torch.testing.assert_close(sample["target"], torch.full((3, 8, 8), 60 / 255))
    assert "valid_depth" not in sample
    torch.testing.assert_close(sample["context_times"], torch.tensor([-3 / 12, -1 / 12]))


def test_camera_conditions_invariant_to_world_coordinates(tmp_path):
    from open_atlas.data import TrajectoryDataset

    clip = write_clip(tmp_path)
    original = TrajectoryDataset(tmp_path, image_size=8)[2]
    metadata = json.loads((clip / "transforms.json").read_text())
    transform = np.array([[0, -1, 0, 2], [1, 0, 0, -3], [0, 0, 1, 0.5], [0, 0, 0, 1]])
    for frame in metadata["frames"]:
        frame["c2w"] = (transform @ np.array(frame["c2w"])).tolist()
    (clip / "transforms.json").write_text(json.dumps(metadata))
    changed = TrajectoryDataset(tmp_path, image_size=8)[2]
    for key in ["context_rays", "target_rays"]:
        torch.testing.assert_close(original[key], changed[key], atol=1e-6, rtol=1e-6)


def test_reject_scene_leakage_across_splits(tmp_path):
    from open_atlas.data import TrajectoryDataset

    write_clip(tmp_path, "room_0", "train")
    write_clip(tmp_path, "room_0", "val")
    with pytest.raises(ValueError, match="split"):
        TrajectoryDataset(tmp_path)


def test_single_context_uses_immediately_previous_frame(tmp_path):
    from open_atlas.data import TrajectoryDataset

    write_clip(tmp_path)
    sample = TrajectoryDataset(tmp_path, context_frames=1)[2]
    torch.testing.assert_close(sample["context"][0, :3], torch.full((3, 64, 64), 40 / 255))


def test_trajectory_modes_are_smooth_and_have_invertible_cameras():
    from open_atlas.geometry import trajectory

    for kind in ["orbit", "dolly", "truck", "crane", "pan", "compound"]:
        poses = trajectory(np.array([0., -1., 1.]), np.array([0., 0., 1.]),
                           kind, frames=12, amplitude=0.1, seed=7)
        assert poses.shape == (12, 4, 4)
        np.testing.assert_allclose(poses @ np.linalg.inv(poses), np.broadcast_to(np.eye(4), poses.shape), atol=1e-6)
        assert np.max(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1)) < 0.1


def test_mesh_preparation_preserves_coordinates_colors_and_faces(tmp_path):
    from open_atlas.mesh import prepare_mesh

    ply = tmp_path / "mesh.ply"
    ply.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n"
                   "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                   "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                   "0 0 0 255 0 0\n1 0 0 255 0 0\n1 1 0 255 0 0\n0 1 0 255 0 0\n4 0 1 2 3\n")
    mesh = prepare_mesh(ply, voxel_size=.01, up_axis="y")
    assert mesh["vertices"].shape == (4, 3)
    assert mesh["faces"].shape == (2, 3)
    np.testing.assert_allclose(mesh["colors"], np.tile([1, 0, 0], (4, 1)))
    raw = np.array([[0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 0, 1]])
    transformed = (raw @ mesh["world_from_raw"].T)[:, :3]
    np.testing.assert_allclose(np.sort(mesh["vertices"], axis=0), np.sort(transformed, axis=0))
