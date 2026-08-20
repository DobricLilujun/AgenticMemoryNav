"""Build shared-world ground-truth point clouds from calibrated depth frames."""

from __future__ import annotations

import numpy as np


def backproject_depth_to_world(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    stride: int = 1,
    min_depth_m: float = 0.05,
    max_depth_m: float = 50.0,
) -> np.ndarray:
    """Back-project a metric depth image through a $4\times4$ c2w transform."""
    image = np.asarray(depth, dtype=np.float32)
    matrix = np.asarray(intrinsics, dtype=np.float32)
    transform = np.asarray(camera_to_world, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("depth must be a 2D metric image")
    if matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if transform.shape != (4, 4):
        raise ValueError("camera_to_world must have shape (4, 4)")
    if stride <= 0:
        raise ValueError("stride must be positive")

    rows, cols = np.mgrid[0 : image.shape[0] : stride, 0 : image.shape[1] : stride]
    values = image[rows, cols]
    valid = np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)
    rows, cols, values = rows[valid], cols[valid], values[valid]
    if len(values) == 0:
        return np.empty((0, 3), dtype=np.float32)
    x = (cols.astype(np.float32) - matrix[0, 2]) * values / matrix[0, 0]
    y = (rows.astype(np.float32) - matrix[1, 2]) * values / matrix[1, 1]
    camera_points = np.column_stack((x, y, values, np.ones_like(values)))
    return (camera_points @ transform.T)[:, :3].astype(np.float32)
