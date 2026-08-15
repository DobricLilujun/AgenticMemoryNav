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

# Camera() defaults to looking down its local -Z axis; on this z-up stage that means
# straight at the floor unless rotated to face the robot's +X direction of travel.
_FORWARD_CAMERA_ORIENTATION_WXYZ = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float32)


def _ensure_simulation_app(
    headless: bool = True,
    livestream_args: list[str] | None = None,
    window_resolution: tuple[int, int] | None = None,
) -> Any:
    """Isaac Sim allows only one SimulationApp per process; reuse it if already started."""
    global _SIMULATION_APP
    if _SIMULATION_APP is None:
        if importlib.util.find_spec("isaacsim") is None:
            raise RuntimeError("isaacsim is not importable; select execution.backend=unitree_sim")
        from isaacsim import SimulationApp  # type: ignore[import-not-found]

        if livestream_args is not None:
            # The trimmed-down "base.python" experience has no WebRTC livestream
            # extension; the full streaming experience is required to serve a stream.
            experience = str(Path("~/isaacsim/apps/isaacsim.exp.full.streaming.kit").expanduser())
            config: dict[str, Any] = {
                "headless": headless,
                "hide_ui": False,
                "extra_args": livestream_args,
            }
            if window_resolution is not None:
                # The WebRTC client negotiates a max frame size on connect (often
                # 1280x720); a mismatched window/render resolution makes the plugin
                # drop every frame with "exceeds the max of ..." warnings.
                width, height = window_resolution
                config["width"] = width
                config["height"] = height
                config["extra_args"] = [
                    *livestream_args,
                    f"--/app/window/width={width}",
                    f"--/app/window/height={height}",
                    f"--/app/renderer/resolution/width={width}",
                    f"--/app/renderer/resolution/height={height}",
                ]
            _SIMULATION_APP = SimulationApp(config, experience=experience)
        else:
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
        livestream_args: list[str] | None = None,
        window_resolution: tuple[int, int] | None = None,
    ) -> None:
        if importlib.util.find_spec("isaacsim") is None:
            raise RuntimeError("isaacsim is not importable in this Python environment")
        # NVIDIA's SimReady environments (Warehouse/Office/...) are served from a remote
        # Nucleus/CDN URL rather than a local path; only local paths must exist on disk.
        is_remote = isinstance(scene, str) and "://" in scene
        if scene and not is_remote and not Path(scene).expanduser().exists():
            raise FileNotFoundError(f"Isaac Sim scene not found: {scene}")

        self.scene = (
            scene
            if (scene and is_remote)
            else (str(Path(scene).expanduser().resolve()) if scene else None)
        )
        self.safety = safety
        self.max_speed = min(max_speed, safety.max_speed)
        self.dt = dt
        self._frame_index = 0
        self._collision = False
        self._stopped = True
        self._yaw = 0.0
        self._camera_resolution = camera_resolution

        self._simulation_app = _ensure_simulation_app(
            headless=headless,
            livestream_args=livestream_args,
            window_resolution=window_resolution,
        )

        # Imported lazily so mock-only workflows never require Isaac Sim dependencies.
        from isaacsim.core.api import World  # type: ignore[import-not-found]
        from isaacsim.core.api.objects import (  # type: ignore[import-not-found]
            DynamicCuboid,
            GroundPlane,
        )
        from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

        self._world = World(stage_units_in_meters=1.0)
        if self.scene:
            # SimReady environments (e.g. NVIDIA Warehouse/Office) ship their own floor,
            # walls, and lighting; adding our procedural ones on top only causes
            # z-fighting and washes out their authored materials.
            self._load_scene_layer(self.scene)
        else:
            # `add_default_ground_plane()` references a large "Grid" reference environment
            # asset with horizon markings; at robot-eye-level that dominates the whole
            # forward-facing frame. A plain flat plane keeps the same physics collider
            # without that distracting backdrop.
            self._world.scene.add(
                GroundPlane(
                    prim_path="/World/groundPlane",
                    z_position=0.0,
                    color=np.array([0.4, 0.4, 0.4], dtype=np.float32),
                )
            )
            # A bare stage has no usable illumination; match the verified recorder setup
            # (scripts/record_isaacsim_sequence.py) so RGB frames aren't near-black.
            from pxr import UsdLux  # type: ignore[import-not-found]

            UsdLux.DomeLight.Define(
                self._world.stage, "/World/realtime_agent_dome_light"
            ).CreateIntensityAttr(1000.0)
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
            orientation=_FORWARD_CAMERA_ORIENTATION_WXYZ,
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

    def spawn_object(
        self,
        name: str,
        position: Vector3,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        scale: float = 0.15,
    ) -> None:
        """Place a real, physically-simulated target cuboid for ObjectNav-style search tasks."""
        from isaacsim.core.api.objects import DynamicCuboid  # type: ignore[import-not-found]

        self._world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/{name}",
                name=name,
                position=self._pose_to_usd(position),
                scale=np.array([scale, scale, scale], dtype=np.float32),
                color=np.array(color, dtype=np.float32),
            )
        )
        self._world.reset()

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
