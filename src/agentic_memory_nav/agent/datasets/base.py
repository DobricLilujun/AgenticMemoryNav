"""Unified frame source contract and lightweight adapters."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from agentic_memory_nav.common.types import CameraIntrinsics, FrameObservation


class DatasetAdapter(Protocol):
    @property
    def limitations(self) -> str: ...
    def frames(self) -> Iterator[FrameObservation]: ...


class NumpyRGBDSequence:
    """Portable fixture format: rgb_*.npy and optional depth_*.npy files."""

    def __init__(self, root: Path, intrinsics: CameraIntrinsics | None = None) -> None:
        self.root = root
        self.intrinsics = intrinsics

    @property
    def limitations(self) -> str:
        return "Camera poses are unavailable unless supplied by a mapping backend."

    def frames(self) -> Iterator[FrameObservation]:
        for index, rgb_path in enumerate(sorted(self.root.glob("rgb_*.npy"))):
            depth_path = self.root / rgb_path.name.replace("rgb_", "depth_")
            yield FrameObservation(
                frame_id=rgb_path.stem,
                timestamp=float(index),
                rgb=np.load(rgb_path).astype(np.uint8),
                depth=np.load(depth_path).astype(np.float32) if depth_path.exists() else None,
                camera_intrinsics=self.intrinsics,
                source="numpy_rgbd",
                provenance=[rgb_path.name],
            )

