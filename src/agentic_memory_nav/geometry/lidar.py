"""Convert calibrated 2D LiDAR flat scans into metric point clouds."""

from __future__ import annotations

import numpy as np


def flat_scan_to_points(
    linear_depth_m: np.ndarray,
    azimuth_range_rad: np.ndarray,
    *,
    min_range_m: float = 0.05,
    max_range_m: float = 50.0,
) -> np.ndarray:
    """Return local XY-plane points from a uniformly sampled 2D LiDAR scan.

    Isaac RTX flat-scan annotators label the field `azimuth_range` but return values
    in degrees. The helper accepts both radians and the Isaac degree convention.
    """
    depths = np.asarray(linear_depth_m, dtype=np.float32).reshape(-1)
    azimuth = np.asarray(azimuth_range_rad, dtype=np.float32).reshape(-1)
    if azimuth.shape != (2,):
        raise ValueError("azimuth_range_rad must contain exactly [start, end]")
    if np.max(np.abs(azimuth)) > 2.0 * np.pi:
        azimuth = np.deg2rad(azimuth)
    angles = np.linspace(azimuth[0], azimuth[1], num=len(depths), dtype=np.float32)
    valid = np.isfinite(depths) & (depths >= min_range_m) & (depths <= max_range_m)
    depths, angles = depths[valid], angles[valid]
    return np.column_stack(
        (depths * np.cos(angles), depths * np.sin(angles), np.zeros_like(depths))
    ).astype(np.float32)
