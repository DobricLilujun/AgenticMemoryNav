"""Optional Isaac Sim executor boundary."""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from agentic_memory_nav.common.types import (
    ActionIntent,
    CameraIntrinsics,
    ExecutionFeedback,
    FrameObservation,
    Pose3D,
    Vector3,
)
from agentic_memory_nav.execution.safety_controller import SafetyController, SafetyError

_SIMULATION_APP: Any | None = None


def _ensure_simulation_app(headless: bool = True) -> Any:
    """Isaac Sim allows only one SimulationApp per process; reuse it if already started."""
    global _SIMULATION_APP
    if _SIMULATION_APP is None:
        if importlib.util.find_spec("isaacsim") is None:
            raise RuntimeError("isaacsim is not importable; select execution.backend=unitree_sim")
        from isaacsim import SimulationApp  # type: ignore[import-not-found]

        _SIMULATION_APP = SimulationApp({"headless": headless})
    return _SIMULATION_APP


class IsaacSimAdapter:
    """Availability boundary mirroring `HabitatAdapter` for the isaacsim backend."""

    def __init__(self, scene: str | None) -> None:
        self.scene = scene
        self.available = importlib.util.find_spec("isaacsim") is not None

    def start(self) -> None:
        if not self.available:
            raise RuntimeError("Isaac Sim is not installed; select execution.backend=unitree_sim")
        raise NotImplementedError("Use IsaacSimExecutor for a validated scene and robot")


class IsaacSimExecutor:
    """Thin RobotBackend-compatible wrapper over Isaac Sim for MVP integration.

    Scenes are procedurally built (ground plane + obstacles) or loaded from a
    provided USD file; Nucleus-hosted sample environments are not available in
    this standalone installation. USD is z-up; this project's `Pose3D` stores
    height at index 1 (matching the Habitat adapter's convention), so
    coordinates are remapped at the boundary rather than left in native USD axes.
    """

    def __init__(
        self,
        scene: str | None,
        safety: SafetyController,
        max_speed: float = 0.5,
        dt: float = 0.1,
        camera_resolution: tuple[int, int] = (64, 96),
        headless: bool = True,
    ) -> None:
        if importlib.util.find_spec("isaacsim") is None:
            raise RuntimeError("isaacsim is not importable in this Python environment")
        if scene and not Path(scene).expanduser().exists():
            raise FileNotFoundError(f"Isaac Sim scene not found: {scene}")

        self.scene = str(Path(scene).expanduser().resolve()) if scene else None
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
        self._world.scene.add_default_ground_plane()
        if self.scene:
            self._load_scene_layer(self.scene)
        else:
            self._add_procedural_obstacles()

        self._robot = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/robot",
                name="robot",
                position=np.array([0.0, 0.0, 0.15], dtype=np.float32),
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

    def _load_scene_layer(self, scene: str) -> None:
        stage = self._world.stage
        stage.GetRootLayer().subLayerPaths.append(scene)

    def _add_procedural_obstacles(self) -> None:
        from isaacsim.core.api.objects import FixedCuboid  # type: ignore[import-not-found]

        for index, (x, y) in enumerate([(1.5, 0.0), (-1.5, 1.0), (0.5, -1.5)]):
            self._world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/obstacle_{index}",
                    name=f"obstacle_{index}",
                    position=np.array([x, y, 0.25], dtype=np.float32),
                    scale=np.array([0.4, 0.4, 0.5], dtype=np.float32),
                )
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
        self._robot.set_world_pose(position=np.array([0.0, 0.0, 0.15], dtype=np.float32))
        self._world.reset()

    def get_state(self) -> Pose3D:
        position, _ = self._robot.get_world_pose()
        return self._usd_to_pose(position, self._yaw)

    def teleport(self, position: Vector3) -> None:
        """Place the robot at an arbitrary pose; used by benchmark harnesses only."""
        self._robot.set_world_pose(position=self._pose_to_usd(position))
        self._yaw = 0.0
        self._collision = False
        self._frame_index = 0

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
            provenance=["isaacsim"],
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
