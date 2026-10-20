"""Prepare Replica's vertex-colored PLY with optional voxel simplification."""

import numpy as np
from plyfile import PlyData


def prepare_mesh(path, voxel_size=.025, up_axis="z"):
    try:
        ply = PlyData.read(path, known_list_len={"face": {"vertex_indices": 4}})
        polygons = ply["face"]["vertex_indices"]
    except ValueError:
        ply = PlyData.read(path)
        polygons = ply["face"]["vertex_indices"]
    if polygons.dtype == object:
        polygons = np.stack(polygons)
    raw = np.stack([ply["vertex"][name] for name in "xyz"], axis=1).astype(np.float32)
    colors = np.stack([ply["vertex"][name] for name in ("red", "green", "blue")], axis=1) / 255.
    rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) if up_axis == "y" else np.eye(3)
    center = (raw.min(0) + raw.max(0)) / 2
    scale = 4 / max(float(np.ptp(raw, axis=0).max()), 1e-6)
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = -scale * rotation @ center
    vertices = raw @ transform[:3, :3].T + transform[:3, 3]
    if voxel_size > 0:
        # 同一体素的顶点与颜色取均值，面索引同步映射到合并后的顶点。
        cells = np.floor(vertices / voxel_size).astype(np.int64)
        cells -= cells.min(0)
        dims = cells.max(0) + 1
        keys = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
        _, inverse = np.unique(keys, return_inverse=True)
        counts = np.bincount(inverse)
        vertices = np.stack([np.bincount(inverse, weights=vertices[:, i]) / counts for i in range(3)], axis=1)
        colors = np.stack([np.bincount(inverse, weights=colors[:, i]) / counts for i in range(3)], axis=1)
        polygons = inverse[polygons]
    # 多边形扇形三角化后，去掉顶点合并产生的退化面和重复面。
    faces = np.concatenate([polygons[:, [0, i, i + 1]] for i in range(1, polygons.shape[1] - 1)])
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    faces = faces[distinct]
    _, unique_ids = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return {"vertices": vertices.astype(np.float32), "faces": faces[unique_ids].astype(np.int32),
            "colors": colors.astype(np.float32), "world_from_raw": transform}
