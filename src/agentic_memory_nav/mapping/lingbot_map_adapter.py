"""Optional LingBot-Map integration boundary.

The adapter deliberately fails with installation guidance when the optional backend is
unavailable. MockMapper remains the CPU fallback.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from agentic_memory_nav.common.types import FrameObservation, MappingUpdate, Pose3D
from agentic_memory_nav.geometry.ground_truth import backproject_depth_to_world
from agentic_memory_nav.mapping.local_submap import LocalSubmap, LocalSubmapBuilder

LingBotPredictor = Callable[[FrameObservation], dict[str, Any]]


class LingBotMapAdapter:
    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        *,
        predictor: LingBotPredictor | None = None,
        keyframe_interval: int = 6,
        local_submap_frames: int = 300,
        local_submap_stride: int = 2,
        local_submap_stability_threshold_m: float = 0.50,
        local_submap_max_points_per_frame: int = 20_000,
        **_: object,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.predictor = predictor
        self.keyframe_interval = max(1, keyframe_interval)
        self.local_submaps = LocalSubmapBuilder(
            window_frames=local_submap_frames,
            frame_stride=local_submap_stride,
            stability_threshold_m=local_submap_stability_threshold_m,
            max_points_per_frame=local_submap_max_points_per_frame,
        )
        self._available = importlib.util.find_spec("lingbot_map") is not None
        self._started = False
        self._version = 0
        self._latest: MappingUpdate | None = None
        self._last_committed_submap: LocalSubmap | None = None

    def start(self) -> None:
        if self.predictor is not None:
            self._started = True
            return
        if not self._available:
            raise RuntimeError(
                "LingBot-Map is not installed. Install the optional backend "
                "or select mapping.backend=mock."
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"LingBot-Map checkpoint not found: {self.checkpoint}")
        raise NotImplementedError(
            "LingBot-Map runtime is gated pending pose-convention validation; "
            "use MockMapper for the MVP."
        )

    def update(self, frame: FrameObservation) -> MappingUpdate:
        if not self._started:
            raise RuntimeError("Mapper must be started before update")
        if self.predictor is None:
            raise RuntimeError(
                "LingBot-Map runtime worker is not configured. "
                "Configure a worker predictor in the isolated LingBot environment."
            )
        update = self._normalize_result(frame, self.predictor(frame))
        submap = self.local_submaps.add(update)
        if submap is not None and submap.stable:
            self._last_committed_submap = submap
        return update

    def get_last_committed_submap(self) -> LocalSubmap | None:
        return self._last_committed_submap

    def get_latest_pose(self) -> Pose3D | None:
        return self._latest.camera_pose if self._latest else None

    def get_local_pointcloud(self) -> np.ndarray:
        return self._latest.local_pointcloud if self._latest else np.empty((0, 3), dtype=np.float32)

    def get_global_pointcloud(self) -> np.ndarray:
        return (
            self._latest.global_pointcloud if self._latest else np.empty((0, 3), dtype=np.float32)
        )

    def reset(self) -> None:
        self._started = False
        self._version = 0
        self._latest = None
        self._last_committed_submap = None

    def save_state(self, path: Path) -> None:
        if self._latest is None:
            raise RuntimeError("Cannot save LingBot-Map state before the first update")
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "lingbot_map_state.npz",
            depth=self._latest.depth,
            confidence=self._latest.confidence,
            local_pointcloud=self._latest.local_pointcloud,
            global_pointcloud=self._latest.global_pointcloud,
        )
        manifest = {
            "frame_id": self._latest.frame_id,
            "timestamp": self._latest.timestamp,
            "camera_pose": {
                "position": self._latest.camera_pose.position,
                "yaw": self._latest.camera_pose.yaw,
            },
            "is_keyframe": self._latest.is_keyframe,
            "map_version": self._latest.map_version,
            "provenance": self._latest.provenance,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load_state(self, path: Path) -> None:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        with np.load(path / "lingbot_map_state.npz") as state:
            camera = manifest["camera_pose"]
            self._latest = MappingUpdate(
                frame_id=str(manifest["frame_id"]),
                timestamp=float(manifest["timestamp"]),
                camera_pose=Pose3D(
                    position=tuple(camera["position"]),  # type: ignore[arg-type]
                    yaw=float(camera["yaw"]),
                ),
                depth=np.asarray(state["depth"], dtype=np.float32),
                confidence=np.asarray(state["confidence"], dtype=np.float32),
                local_pointcloud=np.asarray(state["local_pointcloud"], dtype=np.float32),
                global_pointcloud=np.asarray(state["global_pointcloud"], dtype=np.float32),
                is_keyframe=bool(manifest["is_keyframe"]),
                map_version=int(manifest["map_version"]),
                provenance=list(manifest["provenance"]),
            )
        self._version = self._latest.map_version
        self._started = True

    def _normalize_result(
        self,
        frame: FrameObservation,
        result: dict[str, Any],
    ) -> MappingUpdate:
        pose, c2w = self._pose(result.get("camera_pose"))
        depth = self._image(result.get("depth"), "depth")
        confidence = self._image(result.get("confidence", result.get("depth_conf")), "confidence")
        local = self._depth_pose_pointcloud(depth, confidence, result.get("intrinsics"), c2w)
        if len(local) == 0:
            local = self._pointcloud(
                result.get("local_pointcloud", result.get("world_points")), "local"
            )
        global_cloud = self._pointcloud(result.get("global_pointcloud", local), "global")
        self._version += 1
        update = MappingUpdate(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            camera_pose=pose,
            depth=depth,
            confidence=confidence,
            local_pointcloud=local,
            global_pointcloud=global_cloud,
            is_keyframe=bool(
                result.get("is_keyframe", (self._version - 1) % self.keyframe_interval == 0)
            ),
            map_version=self._version,
            provenance=[frame.frame_id, "lingbot_map"],
        )
        self._latest = update
        return update

    @staticmethod
    def _pose(value: object) -> tuple[Pose3D, np.ndarray | None]:
        if not isinstance(value, dict) or not isinstance(value.get("position"), (list, tuple)):
            raise ValueError("LingBot result requires camera_pose.position")
        position = tuple(float(component) for component in value["position"])
        if len(position) != 3:
            raise ValueError("LingBot camera pose must contain three position values")
        c2w_value = value.get("camera_to_world")
        c2w = np.asarray(c2w_value, dtype=np.float32) if c2w_value is not None else None
        if c2w is not None and c2w.shape != (4, 4):
            raise ValueError("LingBot camera_to_world must have shape (4, 4)")
        return Pose3D(position=position, yaw=float(value.get("yaw", 0.0))), c2w  # type: ignore[arg-type]

    @staticmethod
    def _depth_pose_pointcloud(
        depth: np.ndarray,
        confidence: np.ndarray,
        intrinsics_value: object,
        camera_to_world: np.ndarray | None,
    ) -> np.ndarray:
        if camera_to_world is None or intrinsics_value is None:
            return np.empty((0, 3), dtype=np.float32)
        intrinsics = np.asarray(intrinsics_value, dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError("LingBot intrinsics must have shape (3, 3)")
        filtered_depth = np.where(confidence > 0, depth, np.nan)
        return backproject_depth_to_world(
            filtered_depth,
            intrinsics,
            camera_to_world,
            stride=4,
        )

    @staticmethod
    def _image(value: object, name: str) -> np.ndarray:
        image = np.asarray(value, dtype=np.float32)
        if image.ndim != 2 or not np.isfinite(image).all():
            raise ValueError(f"LingBot {name} must be a finite 2D array")
        return image

    @staticmethod
    def _pointcloud(value: object, name: str) -> np.ndarray:
        cloud = np.asarray(value, dtype=np.float32)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            raise ValueError(f"LingBot {name} point cloud must be a finite (N, 3) array")
        return cloud
