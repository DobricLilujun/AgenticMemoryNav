"""Back-project binary instance masks into world-coordinate point clouds."""

from __future__ import annotations

import numpy as np

from agentic_memory_nav.common.types import CameraIntrinsics, MappingUpdate


def backproject_mask(
    mask: np.ndarray,
    mapping: MappingUpdate,
    intrinsics: CameraIntrinsics,
    *,
    min_confidence: float = 0.5,
    min_depth_m: float = 0.05,
    max_depth_m: float = 20.0,
) -> np.ndarray:
    """Return finite world points selected by an image-space instance mask."""
    binary_mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(mapping.depth, dtype=np.float32)
    confidence = np.asarray(mapping.confidence, dtype=np.float32)
    if binary_mask.shape != depth.shape or confidence.shape != depth.shape:
        raise ValueError("Mask, depth, and confidence must have identical shapes")

    rows, cols = np.nonzero(binary_mask)
    selected_depth = depth[rows, cols]
    selected_confidence = confidence[rows, cols]
    valid = (
        np.isfinite(selected_depth)
        & np.isfinite(selected_confidence)
        & (selected_depth >= min_depth_m)
        & (selected_depth <= max_depth_m)
        & (selected_confidence >= min_confidence)
    )
    rows, cols, selected_depth = rows[valid], cols[valid], selected_depth[valid]
    if len(selected_depth) == 0:
        return np.empty((0, 3), dtype=np.float32)

    x = (cols.astype(np.float32) - intrinsics.cx) * selected_depth / intrinsics.fx
    y = (rows.astype(np.float32) - intrinsics.cy) * selected_depth / intrinsics.fy
    local = np.column_stack((x, y, selected_depth))
    offset = np.asarray(mapping.camera_pose.position, dtype=np.float32)
    return (local + offset).astype(np.float32)
