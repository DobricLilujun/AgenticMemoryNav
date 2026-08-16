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

# Body origin height and head-mounted camera offset for the Unitree Go2 asset, whose
# authored default pose is standing. The camera sits clear of the head mesh, which
# otherwise occludes the lower half of the frame.
_GO2_BASE_HEIGHT_M = 0.40
_GO2_CAMERA_OFFSET_M = (0.42, 0.0, 0.14)
_GO2_STANDING_HALF_EXTENTS_M = (0.34, 0.20, 0.30)
_CUBOID_BASE_HEIGHT_M = 0.15


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return np.array(
        [
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ],
        dtype=np.float32,
    )


def _axis_angle_quat_wxyz(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    half = angle * 0.5
    scale = math.sin(half)
    return np.array(
        [math.cos(half), axis[0] * scale, axis[1] * scale, axis[2] * scale],
        dtype=np.float32,
    )


def _freeze_articulation(stage: Any, root_path: str) -> None:
    """Keep a referenced robot rigid and upright without a locomotion controller.

    The Go2 asset is a physics articulation; without a gait controller it collapses
    under gravity, so its bodies are made kinematic and driven by the parent Xform.
    """
    from pxr import Sdf, Usd, UsdPhysics  # type: ignore[import-not-found]

    root = stage.GetPrimAtPath(root_path)
    if not root:
        return
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            attribute = prim.GetAttribute("physics:articulationEnabled")
            if not attribute:
                attribute = prim.CreateAttribute(
                    "physics:articulationEnabled", Sdf.ValueTypeNames.Bool
                )
            attribute.Set(False)
        if prim.IsA(UsdPhysics.Joint):
            # Kinematic bodies can't be jointed; leaving these on spams PhysX errors.
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)


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
        robot_usd: str | None = None,
        bind_viewport_to_camera: bool = False,
        scene_up_axis: str = "z",
        overhead_camera_position: Vector3 | None = None,
        head_scan_yaw_deg: float = 0.0,
        head_scan_pitch_deg: float = 0.0,
        head_scan_period_frames: int = 1,
        validate_initial_placement: bool = False,
        initial_robot_position: Vector3 | None = None,
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
        if robot_usd and not Path(robot_usd).expanduser().exists():
            raise FileNotFoundError(f"Isaac Sim robot USD not found: {robot_usd}")
        self.robot_usd = str(Path(robot_usd).expanduser().resolve()) if robot_usd else None
        self.safety = safety
        self.max_speed = min(max_speed, safety.max_speed)
        self.dt = dt
        self._frame_index = 0
        self._collision = False
        self._stopped = True
        self._yaw = 0.0
        self._camera_resolution = camera_resolution
        self._head_scan_yaw_rad = math.radians(head_scan_yaw_deg)
        self._head_scan_pitch_rad = math.radians(head_scan_pitch_deg)
        self._head_scan_period_frames = max(1, head_scan_period_frames)
        self._validate_initial_placement = validate_initial_placement
        self._initial_robot_position_usd = (
            self._pose_to_usd(initial_robot_position)
            if initial_robot_position is not None
            else np.array([0.0, 0.0, _GO2_BASE_HEIGHT_M], dtype=np.float32)
        )

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
            self._load_scene_layer(self.scene, scene_up_axis)
            self._ensure_scene_lighting()
            if self._validate_initial_placement:
                self._enable_scene_collisions()
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

        if self.robot_usd:
            from isaacsim.core.prims import SingleXFormPrim  # type: ignore[import-not-found]
            from isaacsim.core.utils.stage import (  # type: ignore[import-not-found]
                add_reference_to_stage,
            )

            self._base_height = _GO2_BASE_HEIGHT_M
            camera_offset = np.array(_GO2_CAMERA_OFFSET_M, dtype=np.float32)
            add_reference_to_stage(self.robot_usd, "/World/robot")
            _freeze_articulation(self._world.stage, "/World/robot")
            self._robot = SingleXFormPrim(
                "/World/robot",
                name="robot",
                position=self._initial_robot_position_usd,
            )
        else:
            self._base_height = _CUBOID_BASE_HEIGHT_M
            camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float32)
            self._robot = self._world.scene.add(
                DynamicCuboid(
                    prim_path="/World/robot",
                    name="robot",
                    position=self._initial_robot_position_usd,
                    scale=np.array([0.3, 0.3, 0.3], dtype=np.float32),
                )
            )
        height, width = camera_resolution
        # Parented to the robot Xform, so the lens follows the body's position and yaw.
        self._camera = Camera(
            prim_path="/World/robot/agent_camera",
            position=camera_offset,
            orientation=_FORWARD_CAMERA_ORIENTATION_WXYZ,
            resolution=(width, height),
        )
        self._overhead_camera = None
        if overhead_camera_position is not None:
            self._overhead_camera = Camera(
                prim_path="/World/overhead_camera",
                position=self._pose_to_usd(overhead_camera_position),
                # Camera() looks along local -Z, so identity is a vertical top-down view.
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                resolution=(width, height),
            )
        self._world.reset()
        self._set_robot_pose(self._initial_robot_position_usd, yaw=0.0)
        for _ in range(2):
            self._world.step(render=False)
        if self._validate_initial_placement:
            self._validate_robot_placement()
        self._camera.initialize()
        self._camera.add_distance_to_image_plane_to_frame()
        if self._overhead_camera is not None:
            self._overhead_camera.initialize()
        if bind_viewport_to_camera:
            self._bind_viewport("/World/robot/agent_camera")
        for _ in range(3):
            self._world.step(render=True)

    def _bind_viewport(self, camera_path: str) -> None:
        """Point the streamed viewport at the robot's own lens instead of the default persp camera."""
        from omni.kit.viewport.utility import (  # type: ignore[import-not-found]
            get_active_viewport,
        )

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path

    def _load_scene_layer(self, scene: str, up_axis: str = "z") -> None:
        if up_axis.lower() == "y":
            from isaacsim.core.prims import SingleXFormPrim  # type: ignore[import-not-found]
            from isaacsim.core.utils.stage import (  # type: ignore[import-not-found]
                add_reference_to_stage,
            )

            # A Y-up asset (e.g. GLB converted to USD) needs +90 deg about X on this Z-up stage.
            add_reference_to_stage(scene, "/World/scene")
            SingleXFormPrim(
                "/World/scene",
                name="scene",
                orientation=np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32),
            )
            return
        stage = self._world.stage
        stage.GetRootLayer().subLayerPaths.append(scene)

    def _ensure_scene_lighting(self) -> None:
        """Some environment assets (e.g. Modular_Warehouse props) ship geometry but no lights."""
        from pxr import UsdLux  # type: ignore[import-not-found]

        stage = self._world.stage
        if any(prim.HasAPI(UsdLux.LightAPI) for prim in stage.Traverse()):
            return
        UsdLux.DomeLight.Define(
            stage, "/World/realtime_agent_dome_light"
        ).CreateIntensityAttr(1000.0)

    def _enable_scene_collisions(self) -> None:
        """Make imported render meshes queryable as static PhysX environment colliders."""
        from pxr import UsdGeom, UsdPhysics  # type: ignore[import-not-found]

        for prim in self._world.stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Mesh(prim).GetPointsAttr().Get():
                UsdPhysics.CollisionAPI.Apply(prim)

    def _validate_robot_placement(self) -> None:
        """Abort before the loop when the Go2 standing volume intersects the environment."""
        import carb  # type: ignore[import-not-found]
        from omni.physx import get_physx_scene_query_interface  # type: ignore[import-not-found]

        position, _ = self._robot.get_world_pose()
        hits: list[str] = []

        def report_overlap(hit: Any) -> bool:
            collider = str(hit.rigid_body)
            if not collider.startswith("/World/robot"):
                hits.append(collider)
            return True

        half_x, half_y, half_z = _GO2_STANDING_HALF_EXTENTS_M
        get_physx_scene_query_interface().overlap_box(
            carb.Float3(half_x, half_y, half_z),
            carb.Float3(*position),
            carb.Float4(0.0, 0.0, 0.0, 1.0),
            report_overlap,
            False,
        )
        if hits:
            raise RuntimeError(
                "Go2 initial placement overlaps environment colliders: "
                + ", ".join(sorted(set(hits))[:5])
            )

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

    def _set_robot_pose(self, position_usd: np.ndarray, yaw: float | None = None) -> None:
        if yaw is not None:
            self._yaw = yaw
        self._robot.set_world_pose(
            position=position_usd, orientation=_yaw_to_quat_wxyz(self._yaw)
        )

    def _update_head_camera_scan(self) -> None:
        phase = 2.0 * math.pi * self._frame_index / self._head_scan_period_frames
        scan_yaw = self._head_scan_yaw_rad * math.sin(phase)
        scan_pitch = self._head_scan_pitch_rad * math.sin(phase * 0.5)
        orientation = _quat_multiply_wxyz(
            _FORWARD_CAMERA_ORIENTATION_WXYZ,
            _quat_multiply_wxyz(
                _axis_angle_quat_wxyz((0.0, 0.0, 1.0), scan_yaw),
                _axis_angle_quat_wxyz((1.0, 0.0, 0.0), scan_pitch),
            ),
        )
        self._camera.set_local_pose(orientation=orientation)

    def get_overhead_rgb(self) -> np.ndarray | None:
        """Return the observer camera frame; this is never passed to the agent."""
        if self._overhead_camera is None:
            return None
        rgba = self._overhead_camera.get_rgba()
        if rgba is None or not rgba.size:
            return None
        return rgba[:, :, :3].astype(np.uint8)

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
        self._set_robot_pose(
            self._initial_robot_position_usd, yaw=0.0
        )
        self._world.reset()

    def get_state(self) -> Pose3D:
        position, _ = self._robot.get_world_pose()
        return self._usd_to_pose(position, self._yaw)

    def teleport(self, position: Vector3) -> None:
        """Place the robot at an arbitrary pose; used by benchmark harnesses only."""
        self._set_robot_pose(self._pose_to_usd(position), yaw=0.0)
        self._collision = False
        self._frame_index = 0

    def get_observation(self) -> FrameObservation:
        self._update_head_camera_scan()
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
        self._set_robot_pose(position, yaw=self._yaw + wz * self.dt)
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
        self._set_robot_pose(destination, yaw=math.atan2(float(delta[1]), float(delta[0])))
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
