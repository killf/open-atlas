"""OpenCV cameras: x right, y down, z forward; matrices map camera to world."""

import numpy as np


def look_at(position, target, up=(0., 0., 1.)):
    """构造 OpenCV 相机到世界的变换，旋转列向量依次为右、下、前。"""
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("Camera direction must not be parallel to up")
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, np.cross(forward, right), forward], axis=1)
    pose[:3, 3] = position
    return pose


def intrinsics(width, height, fov_degrees):
    if not 0 < fov_degrees < 180:
        raise ValueError("Horizontal field of view must lie in (0, 180)")
    focal = width / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])


def camera_directions(K, height, width):
    y, x = np.mgrid[:height, :width]
    pixels = np.stack([x + 0.5, y + 0.5, np.ones_like(x)], axis=-1)
    return pixels @ np.linalg.inv(K).T


def rays(K, c2w, height, width):
    """返回世界坐标中的 Plücker 射线 [d, o×d]，形状为 [6,H,W]。"""
    direction = camera_directions(K, height, width) @ c2w[:3, :3].T
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    moment = np.cross(np.broadcast_to(c2w[:3, 3], direction.shape), direction)
    return np.concatenate([direction, moment], axis=-1).transpose(2, 0, 1).astype(np.float32)


def trajectory(position, target, kind, frames=48, amplitude=0.1, seed=0):
    """生成平滑相机位姿；平移幅度用场景单位，orbit/pan 幅度用弧度。"""
    if frames < 2 or amplitude <= 0:
        raise ValueError("A trajectory needs at least two frames and positive amplitude")
    base = look_at(position, target)
    right, up, forward = base[:3, 0], -base[:3, 1], base[:3, 2]
    phase = np.random.default_rng(seed).uniform(-0.2, 0.2)
    times = np.linspace(-1, 1, frames) + phase
    poses = []
    for t in times:
        eye, focus = np.array(position, dtype=float), np.array(target, dtype=float)
        if kind == "orbit":
            offset = eye - focus
            angle = t * amplitude
            eye = focus + offset * np.cos(angle) + np.cross(up, offset) * np.sin(angle)
        elif kind == "dolly":
            eye += t * amplitude * forward
            focus += t * amplitude * forward
        elif kind == "truck":
            eye += t * amplitude * right
            focus += t * amplitude * right
        elif kind == "crane":
            eye += t * amplitude * up
            focus += t * amplitude * up
        elif kind == "pan":
            focus = eye + forward * np.cos(t * amplitude) + right * np.sin(t * amplitude)
        elif kind == "compound":
            eye += amplitude * (t * right + 0.3 * np.sin(t * np.pi) * up + 0.2 * t * forward)
        else:
            raise ValueError(f"Unknown trajectory: {kind}")
        poses.append(look_at(eye, focus, up))
    return np.stack(poses)
