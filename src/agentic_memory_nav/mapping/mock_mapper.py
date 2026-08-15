"""Deterministic CPU mapper for tests and fallback execution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agentic_memory_nav.common.types import FrameObservation, MappingUpdate, Pose3D
from agentic_memory_nav.mapping.streaming_buffer import StreamingBuffer


class MockMapper:
    def __init__(self, keyframe_interval: int = 2, depth_m: float = 2.0) -> None:
        self.keyframe_interval = max(1, keyframe_interval)
        self.depth_m = depth_m
        self.buffer = StreamingBuffer()
        self._started = False
        self._version = 0
        self._latest: MappingUpdate | None = None
        self._chunks: list[np.ndarray] = []

    def start(self) -> None:
        self._started = True

    def update(self, frame: FrameObservation) -> MappingUpdate:
        if not self._started:
            raise RuntimeError("Mapper must be started before update")
        height, width = frame.rgb.shape[:2]
        pose = frame.camera_pose or Pose3D(position=(self._version * 0.35, 0.0, 0.0))
        depth = (
            np.asarray(frame.depth, dtype=np.float32)
            if frame.depth is not None
            else np.full((height, width), self.depth_m, dtype=np.float32)
        )
        confidence = np.ones_like(depth, dtype=np.float32)
        local = self._backproject(depth, pose)
        self._chunks.append(local[:: max(1, len(local) // 128)])
        self._version += 1
        is_keyframe = (self._version - 1) % self.keyframe_interval == 0
        self.buffer.add(frame, is_keyframe)
        global_cloud = np.concatenate(self._chunks, axis=0)
        self._latest = MappingUpdate(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            camera_pose=pose,
            depth=depth,
            confidence=confidence,
            local_pointcloud=local,
            global_pointcloud=global_cloud,
            is_keyframe=is_keyframe,
            map_version=self._version,
            provenance=[frame.frame_id, "mock_mapper"],
        )
        return self._latest

    @staticmethod
    def _backproject(depth: np.ndarray, pose: Pose3D) -> np.ndarray:
        height, width = depth.shape
        ys, xs = np.mgrid[0:height:4, 0:width:4]
        # Real sensors (e.g. Isaac Sim) report inf/nan depth for sky/no-hit pixels.
        z = np.nan_to_num(depth[ys, xs], nan=0.0, posinf=0.0, neginf=0.0)
        x = (xs - width / 2) / max(width, 1) * z + pose.position[0]
        y = (ys - height / 2) / max(height, 1) * z + pose.position[1]
        return np.stack((x, y, z + pose.position[2]), axis=-1).reshape(-1, 3).astype(np.float32)

    def get_latest_pose(self) -> Pose3D | None:
        return self._latest.camera_pose if self._latest else None

    def get_local_pointcloud(self) -> np.ndarray:
        return self._latest.local_pointcloud if self._latest else np.empty((0, 3), np.float32)

    def get_global_pointcloud(self) -> np.ndarray:
        return self._latest.global_pointcloud if self._latest else np.empty((0, 3), np.float32)

    def reset(self) -> None:
        self.buffer.clear()
        self._chunks.clear()
        self._latest = None
        self._version = 0
        self._started = False

    def save_state(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "pointcloud.npy", self.get_global_pointcloud())
        manifest = {
            "version": self._version,
            "keyframes": [f.frame_id for f in self.buffer.keyframes],
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load_state(self, path: Path) -> None:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        cloud = np.load(path / "pointcloud.npy")
        self._version = int(manifest["version"])
        self._chunks = [cloud]
        self._started = True
