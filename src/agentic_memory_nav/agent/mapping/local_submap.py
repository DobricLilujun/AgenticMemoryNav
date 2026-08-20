"""Bounded local submaps built from depth-plus-pose point-cloud updates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from agentic_memory_nav.common.types import MappingUpdate


@dataclass(frozen=True, slots=True)
class LocalSubmap:
    frame_ids: list[str]
    points: np.ndarray
    geometric_residual_m: float
    stable: bool
    timestamp: float


class LocalSubmapBuilder:
    """Commit bounded geometry windows only after a simple consistency check.

    Each accepted update contributes a local cloud generated from predicted depth and
    c2w pose. A completed window is stable only when adjacent local clouds have a
    bounded symmetric nearest-neighbor overlap residual; callers should persist only
    stable windows. This tolerates normal camera motion and changing view centroids.
    """

    def __init__(
        self,
        window_frames: int = 300,
        frame_stride: int = 2,
        stability_threshold_m: float = 0.50,
        max_points_per_frame: int = 20_000,
    ) -> None:
        if window_frames <= 0 or frame_stride <= 0 or stability_threshold_m < 0:
            raise ValueError("window_frames, frame_stride, and stability threshold must be valid")
        self.window_frames = window_frames
        self.frame_stride = frame_stride
        self.stability_threshold_m = stability_threshold_m
        self.max_points_per_frame = max_points_per_frame
        self._updates: deque[MappingUpdate] = deque(maxlen=window_frames)
        self._seen = 0

    def add(self, update: MappingUpdate) -> LocalSubmap | None:
        self._seen += 1
        if (self._seen - 1) % self.frame_stride:
            return None
        self._updates.append(update)
        if len(self._updates) < self.window_frames:
            return None
        submap = self._build()
        self._updates.clear()
        return submap

    def flush(self, min_frames: int = 1) -> LocalSubmap | None:
        if len(self._updates) < min_frames:
            return None
        submap = self._build()
        self._updates.clear()
        return submap

    def _build(self) -> LocalSubmap:
        updates = list(self._updates)
        clouds = [self._sample(update.local_pointcloud) for update in updates]
        residual = (
            max(
                self._symmetric_nn_residual(left, right)
                for left, right in zip(clouds[:-1], clouds[1:], strict=True)
            )
            if len(clouds) > 1
            else 0.0
        )
        return LocalSubmap(
            frame_ids=[update.frame_id for update in updates],
            points=np.concatenate(clouds, axis=0),
            geometric_residual_m=residual,
            stable=residual <= self.stability_threshold_m,
            timestamp=updates[-1].timestamp,
        )

    def _sample(self, points: np.ndarray) -> np.ndarray:
        cloud = np.asarray(points, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
            raise ValueError("Mapping update must contain a non-empty local point cloud")
        if len(cloud) <= self.max_points_per_frame:
            return cloud
        indices = np.linspace(0, len(cloud) - 1, self.max_points_per_frame, dtype=np.int64)
        return cloud[indices]

    @staticmethod
    def _symmetric_nn_residual(left: np.ndarray, right: np.ndarray) -> float:
        return 0.5 * (
            LocalSubmapBuilder._mean_nn_distance(left, right)
            + LocalSubmapBuilder._mean_nn_distance(right, left)
        )

    @staticmethod
    def _mean_nn_distance(source: np.ndarray, target: np.ndarray, batch_size: int = 256) -> float:
        distances = []
        for start in range(0, len(source), batch_size):
            batch = source[start : start + batch_size]
            squared = np.sum((batch[:, None, :] - target[None, :, :]) ** 2, axis=2)
            distances.append(np.sqrt(squared.min(axis=1)))
        return float(np.concatenate(distances).mean())
