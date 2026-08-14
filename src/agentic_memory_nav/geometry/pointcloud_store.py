"""Persistent storage for per-instance point-cloud artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agentic_memory_nav.common.types import InstanceGeometry


class PointCloudStore:
    def __init__(self, root: Path, max_points: int = 250_000) -> None:
        self.root = root
        self.max_points = max_points
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        instance_id: str,
        points: np.ndarray,
        *,
        coordinate_frame: str,
        confidence: float,
        mask_provenance: list[str] | None = None,
    ) -> InstanceGeometry:
        cloud = self._validate(points)
        path = self.root / f"{instance_id}.npz"
        np.savez_compressed(path, points=cloud)
        minimum = cloud.min(axis=0)
        maximum = cloud.max(axis=0)
        return InstanceGeometry(
            instance_id=instance_id,
            artifact_path=str(path),
            point_count=len(cloud),
            centroid_3d=tuple(cloud.mean(axis=0).tolist()),  # type: ignore[arg-type]
            dimensions_3d=tuple((maximum - minimum).tolist()),  # type: ignore[arg-type]
            coordinate_frame=coordinate_frame,
            confidence=float(confidence),
            mask_provenance=mask_provenance or [],
        )

    def get(self, geometry: InstanceGeometry) -> np.ndarray:
        if geometry.artifact_path is None:
            raise ValueError(f"Instance {geometry.instance_id} has no point-cloud artifact")
        with np.load(geometry.artifact_path) as artifact:
            return self._validate(artifact["points"])

    def _validate(self, points: np.ndarray) -> np.ndarray:
        cloud = np.asarray(points, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError("Point cloud must have shape (N, 3)")
        if len(cloud) == 0:
            raise ValueError("Point cloud must not be empty")
        if len(cloud) > self.max_points:
            raise ValueError(f"Point cloud exceeds max_points={self.max_points}")
        if not np.isfinite(cloud).all():
            raise ValueError("Point cloud contains non-finite coordinates")
        return cloud
