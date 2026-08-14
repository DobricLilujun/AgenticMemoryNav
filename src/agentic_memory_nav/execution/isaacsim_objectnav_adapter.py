"""Optional Isaac Sim ObjectNav executor boundary (InteriorAgent-style scenes).

Independent reimplementation of the InteriorAgent `experiments.json` workflow
described by github.com/learnsyslab/isaac-objnav-semistatic-eval; no code from
that repository is used (see docs/dependency-decisions.md for the license check
that led to this decision). Not yet runtime-verified: it requires the
InteriorAgent dataset, which was not available in this environment.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import numpy as np

from agentic_memory_nav.common.types import (
    ActionIntent,
    CameraIntrinsics,
    ExecutionFeedback,
    FrameObservation,
    Pose3D,
    Vector3,
)
from agentic_memory_nav.datasets.objectnav import ObjectNavExperiment
from agentic_memory_nav.execution.isaacsim_adapter import _ensure_simulation_app
from agentic_memory_nav.execution.safety_controller import SafetyController, SafetyError


class IsaacSimObjectNavExecutor:
    """RobotBackend-compatible wrapper that loads an InteriorAgent-style USD scene,
    applies experiment-driven asset removal, and exposes goal-asset positions.
    """

    def __init__(
        self,
        scene_root: str,
        experiment: ObjectNavExperiment,
        safety: SafetyController,
        max_speed: float = 0.5,
        dt: float = 0.1,
        camera_resolution: tuple[int, int] = (64, 96),
        headless: bool = True,
    ) -> None:
        if importlib.util.find_spec("isaacsim") is None:
            raise RuntimeError("isaacsim is not importable in this Python environment")
        scene_path = Path(scene_root).expanduser() / experiment.scene
        if not scene_path.exists():
            raise FileNotFoundError(
                f"ObjectNav scene not found: {scene_path} (InteriorAgent dataset not downloaded?)"
            )

        self.experiment = experiment
        self.safety = safety
        self.max_speed = min(max_speed, safety.max_speed)
        self.dt = dt
        self._frame_index = 0
        self._collision = False
        self._stopped = True
        self._yaw = 0.0
        self._camera_resolution = camera_resolution

        self._simulation_app = _ensure_simulation_app(headless=headless)

        # Imported lazily so mock-only workflows never require Isaac Sim dependencies.
        from isaacsim.core.api import World  # type: ignore[import-not-found]
        from isaacsim.core.api.objects import DynamicCuboid  # type: ignore[import-not-found]
        from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

        self._world = World(stage_units_in_meters=1.0)
        self._world.stage.GetRootLayer().subLayerPaths.append(str(scene_path))
        self._remove_assets(experiment.remove_assets, experiment.exclude_remove_assets)

        start = experiment.robot_start
        self._robot = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/robot",
                name="robot",
                position=np.array([start[0], start[1], start[2] + 0.15], dtype=np.float32),
                scale=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            )
        )
        height, width = camera_resolution
        self._camera = Camera(
            prim_path="/World/robot/camera",
            position=np.array([0.0, 0.0, 0.35], dtype=np.float32),
            resolution=(width, height),
        )
        self._world.reset()
        self._camera.initialize()
        self._camera.add_distance_to_image_plane_to_frame()
        for _ in range(3):
            self._world.step(render=True)

    def _remove_assets(self, remove_substrings: list[str], exclude_substrings: list[str]) -> None:
        if not remove_substrings:
            return
        stage = self._world.stage
        for prim in list(stage.Traverse()):
            name = prim.GetName()
            if any(token in name for token in remove_substrings) and not any(
                token in name for token in exclude_substrings
            ):
                stage.RemovePrim(prim.GetPath())

    def goal_positions(self) -> dict[str, Vector3]:
        """World positions of prims matching `experiment.goal.asset` (search task)."""
        asset = self.experiment.goal.asset
        if asset is None:
            return {}
        tokens = [asset] if isinstance(asset, str) else list(asset)
        from pxr import UsdGeom  # type: ignore[import-not-found]

        stage = self._world.stage
        positions: dict[str, Vector3] = {}
        for prim in stage.Traverse():
            name = prim.GetName()
            if any(token in name for token in tokens) and prim.IsA(UsdGeom.Xformable):
                transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
                translation = transform.ExtractTranslation()
                positions[name] = self._usd_to_pose(np.array(translation), 0.0).position
        return positions

    def shortest_distance_to_goal(self) -> float | None:
        """Euclidean distance to the nearest goal asset.

        This is a lower-bound approximation, not a geodesic/occupancy-aware distance:
        deriving a reliable floor-plan occupancy grid from arbitrary InteriorAgent
        scene geometry needs calibration against real scenes, which requires the
        dataset (not available in this environment). Revisit once it is downloaded.
        """
        goal_positions = self.goal_positions()
        if not goal_positions:
            return None
        state = self.get_state()
        return min(
            math.dist((state.position[0], state.position[2]), (position[0], position[2]))
            for position in goal_positions.values()
        )

    def _usd_to_pose(self, position_usd: np.ndarray, yaw: float) -> Pose3D:
        x, y_north, z_up = (float(value) for value in position_usd)
        return Pose3D(position=(x, z_up, y_north), yaw=yaw)

    def _pose_to_usd(self, position: Vector3) -> np.ndarray:
        x, z_up, y_north = position
        return np.array([x, y_north, z_up], dtype=np.float32)

    def reset(self) -> None:
        self._frame_index = 0
        self._collision = False
        self._stopped = True
        self._yaw = 0.0
        start = self.experiment.robot_start
        self._robot.set_world_pose(
            position=np.array([start[0], start[1], start[2] + 0.15], dtype=np.float32)
        )
        self._world.reset()

    def get_state(self) -> Pose3D:
        position, _ = self._robot.get_world_pose()
        return self._usd_to_pose(position, self._yaw)

    def get_observation(self) -> FrameObservation:
        self._world.step(render=True)
        rgba = self._camera.get_rgba()
        height, width = self._camera_resolution
        if rgba is not None and rgba.size:
            rgb = rgba[:, :, :3].astype(np.uint8)
        else:
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth_frame = self._camera.get_current_frame().get("distance_to_image_plane")
        depth = np.asarray(depth_frame, dtype=np.float32) if depth_frame is not None else None
        frame = FrameObservation(
            frame_id=f"frame_{self._frame_index:04d}",
            timestamp=float(self._frame_index),
            rgb=rgb,
            depth=depth,
            camera_intrinsics=CameraIntrinsics(80.0, 80.0, width / 2, height / 2, width, height),
            camera_pose=self.get_state(),
            robot_pose=self.get_state(),
            provenance=["isaacsim_objectnav"],
        )
        self._frame_index += 1
        return frame

    def send_velocity_command(self, vx: float, vy: float, wz: float) -> ExecutionFeedback:
        started = time.perf_counter()
        speed = math.hypot(vx, vy)
        if speed > self.max_speed or abs(wz) > self.safety.max_angular_speed:
            self.stop()
            return ExecutionFeedback(
                "velocity", False, self.get_state(), False, 0.0, "velocity limit exceeded"
            )

        position, _ = self._robot.get_world_pose()
        position = position + np.array([vx * self.dt, vy * self.dt, 0.0], dtype=np.float32)
        self._robot.set_world_pose(position=position)
        self._yaw += wz * self.dt
        self._stopped = False
        self._world.step(render=False)
        return ExecutionFeedback(
            "velocity",
            True,
            self.get_state(),
            self._collision,
            time.perf_counter() - started,
            "executed",
        )

    def send_waypoint(self, waypoint: Vector3, intent: ActionIntent) -> ExecutionFeedback:
        started = time.perf_counter()
        try:
            self.safety.validate(intent, self.get_state(), self._collision)
        except SafetyError as error:
            self.stop()
            return ExecutionFeedback(
                intent.action_id,
                False,
                self.get_state(),
                True,
                0.0,
                str(error),
            )

        current_usd, _ = self._robot.get_world_pose()
        target_usd = self._pose_to_usd(waypoint)
        delta = target_usd - current_usd
        distance = float(np.linalg.norm(delta[:2]))
        if distance == 0.0:
            self.stop()
            return ExecutionFeedback(
                intent.action_id,
                True,
                self.get_state(),
                False,
                0.0,
                "already at waypoint",
            )

        travel = min(distance, self.max_speed * intent.duration)
        direction = delta / max(distance, 1e-9)
        destination = current_usd + direction * travel
        self._robot.set_world_pose(position=destination)
        self._yaw = math.atan2(float(delta[1]), float(delta[0]))
        self._stopped = False
        self._world.step(render=False)
        reached = travel >= distance - 1e-6
        self.stop()
        return ExecutionFeedback(
            intent.action_id,
            reached,
            self.get_state(),
            self._collision,
            time.perf_counter() - started,
            "waypoint reached" if reached else "action timeout before waypoint",
        )

    def stop(self) -> None:
        self._stopped = True

    def emergency_stop(self) -> None:
        self.safety.emergency_stop()
        self.stop()

    def is_collision(self) -> bool:
        return self._collision

    def close(self) -> None:
        self._world.stop()
        self._simulation_app.close()
