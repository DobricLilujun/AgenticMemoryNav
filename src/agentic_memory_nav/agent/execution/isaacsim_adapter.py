"""Optional Isaac Sim executor boundary.

Thin RobotBackend wrapper over Isaac Sim: scene/robot setup, kinematic driving,
collision checks, and rendered frames. Isaac Sim is imported lazily so mock-only
workflows do not depend on it.

Supported constraints:
- robot_motion_mode='kinematic' (no gait controller)
- light_rig='gray_studio'
- referenced robots are frozen as rigid bodies to stay upright
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

from agentic_memory_nav.agent.execution.discrete_actions import (
    LOOK_PITCH_LIMIT_RAD,
    LOOK_STEP_RAD,
    MOVE_STEP_M,
    TURN_BIG_STEP_RAD,
    TURN_STEP_RAD,
    DiscreteAction,
)
from agentic_memory_nav.agent.execution.safety_controller import SafetyController, SafetyError
from agentic_memory_nav.common.types import (
    ActionIntent,
    CameraIntrinsics,
    ExecutionFeedback,
    FrameObservation,
    Pose3D,
    Vector3,
)

_SIMULATION_APP: Any | None = None

# Camera axes (Isaac Sim optical: -Z forward, +Y up, +X right) are aligned to
# base_link (ROS REP 105: +X forward, +Y left, +Z up) via the fixed offset below.
# The head camera is parented to the robot; its local pose is set once at
# construction and the robot world pose carries it automatically.

# Go2 standing body origin and head-camera offset in base_link coordinates.
_OPTICAL_TO_BASE_OFFSET_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Camera-link translation in base_link coordinates: (forward, left, up), metres.
_GO2_BASE_HEIGHT_M = 0.40
_GO2_CAMERA_OFFSET_M = (0.25, 0.0, 0.20)
# Full body half extents used for placement validation feedback only.
_GO2_STANDING_HALF_EXTENTS_M = (0.34, 0.20, 0.30)
# Torso-level half extents for movement checks. The feet must be allowed to
# touch/intersect the floor mesh, otherwise every forward step is reported as
# an obstacle.
_GO2_TORSO_HALF_EXTENTS_M = (0.34, 0.20, 0.10)
_CUBOID_BASE_HEIGHT_M = 0.15

# Swept collision-check spacing: sample the movement segment so thin walls
# cannot be stepped through. Keeps the check cheap (< 10 overlap queries).
_SWEPT_CHECK_SAMPLE_SPACING_M = 0.05
_SWEPT_CHECK_MAX_SAMPLES = 5


def _normalize_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("Quaternion norm must be non-zero")
    return quaternion / norm


def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _euler_xyz_deg_to_quat_wxyz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    """Convert X/Y/Z Euler angles in degrees into a normalized WXYZ quaternion."""
    x_rad, y_rad, z_rad = (math.radians(angle) for angle in (x_deg, y_deg, z_deg))
    cos_x, sin_x = math.cos(x_rad * 0.5), math.sin(x_rad * 0.5)
    cos_y, sin_y = math.cos(y_rad * 0.5), math.sin(y_rad * 0.5)
    cos_z, sin_z = math.cos(z_rad * 0.5), math.sin(z_rad * 0.5)
    return np.array(
        [
            cos_x * cos_y * cos_z + sin_x * sin_y * sin_z,
            sin_x * cos_y * cos_z - cos_x * sin_y * sin_z,
            cos_x * sin_y * cos_z + sin_x * cos_y * sin_z,
            cos_x * cos_y * sin_z - sin_x * sin_y * cos_z,
        ],
        dtype=np.float32,
    )


def _xyz_degrees(value: Vector3 | None, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float32)
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three X/Y/Z degree values")
    return np.asarray(value, dtype=np.float32)


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
    from pxr import Usd, UsdPhysics  # type: ignore[import-not-found]

    root = stage.GetPrimAtPath(root_path)
    if not root:
        return
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
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
            isaacsim_path = os.environ.get("ISAACSIM_PATH", "")
            if not isaacsim_path:
                isaacsim_path = Path("~/IsaacSim/_build/linux-aarch64/release").expanduser()
            experience = str(Path(isaacsim_path) / "apps" / "isaacsim.exp.full.streaming.kit")
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
        go2_camera_orient: Vector3 | None = None,
        head_scan_yaw_deg: float = 0.0,
        head_scan_pitch_deg: float = 0.0,
        head_scan_period_frames: int = 1,
        validate_initial_placement: bool = False,
        initial_robot_position: Vector3 | None = None,
        initial_robot_yaw_deg: float = 0.0,
        camera_fps: int = 30,
        camera_focal_length: float = 12.0,
        environment_planes: dict[str, Any] | None = None,
        robot_motion_mode: str = "kinematic",
        light_rig: str = "gray_studio",
        turn_step_deg: float | None = None,
        turn_big_step_deg: float | None = None,
        move_step_m: float | None = None,
        look_step_deg: float | None = None,
        camera_offset: Vector3 | None = None,
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
        self._initial_robot_yaw_rad = math.radians(initial_robot_yaw_deg)
        self._yaw = self._initial_robot_yaw_rad
        self._camera_resolution = camera_resolution
        if not 30 <= camera_fps <= 60:
            raise ValueError(f"camera_fps must be between 30 and 60, got {camera_fps}")
        if camera_focal_length <= 0.0:
            raise ValueError(f"camera_focal_length must be positive, got {camera_focal_length}")
        if robot_motion_mode != "kinematic":
            raise ValueError(
                "Only robot_motion_mode='kinematic' is supported "
                "until a Go2 gait controller is added"
            )
        if light_rig != "gray_studio":
            raise ValueError(f"Only light_rig='gray_studio' is supported, got {light_rig!r}")
        self._camera_fps = camera_fps
        self._camera_focal_length = float(camera_focal_length)
        self._robot_motion_mode = robot_motion_mode
        self._head_scan_yaw_rad = math.radians(head_scan_yaw_deg)
        self._head_scan_pitch_rad = math.radians(head_scan_pitch_deg)
        self._head_scan_period_frames = max(1, head_scan_period_frames)
        # Armed look-around window: 0 => the head lens holds still and points
        # straight ahead with the body. A scan only runs while this is > 0.
        self._head_scan_frames = 0
        # Persistent manual look_up/look_down offset, set via apply_discrete_action.
        self._manual_pitch_offset_rad = 0.0
        # Per-call overrides for the standard discrete-action step sizes (turn angle,
        # forward step, look tilt); default to the module-wide standard constants.
        self._turn_step_rad = (
            math.radians(turn_step_deg) if turn_step_deg is not None else TURN_STEP_RAD
        )
        self._turn_big_step_rad = (
            math.radians(turn_big_step_deg) if turn_big_step_deg is not None else TURN_BIG_STEP_RAD
        )
        self._move_step_m = move_step_m if move_step_m is not None else MOVE_STEP_M
        self._look_step_rad = (
            math.radians(look_step_deg) if look_step_deg is not None else LOOK_STEP_RAD
        )
        self._validate_initial_placement = validate_initial_placement
        self._environment_planes = environment_planes or {}
        self._environment_plane_paths: set[str] = set()
        # Optional camera orientation offset applied on top of the forward look-at pose.
        self._go2_camera_orient_wxyz = _euler_xyz_deg_to_quat_wxyz(
            *_xyz_degrees(go2_camera_orient, "go2_camera_orient")
        )
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

        if self.scene:
            self._open_scene_stage(self.scene, scene_up_axis)
            self._assert_world_identity()

        # Imported lazily so mock-only workflows never require Isaac Sim dependencies.
        from isaacsim.core.api import World  # type: ignore[import-not-found]
        from isaacsim.core.api.objects import (  # type: ignore[import-not-found]
            DynamicCuboid,
            GroundPlane,
        )
        from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]

        self._world = World(stage_units_in_meters=1.0)
        if self.scene:
            self._environment_bounds = self._get_environment_bounds()
            self._add_environment_planes(self._environment_planes)
            self._apply_builtin_grey_studio_light_rig()
            # Add a scene-level dome light so the camera-rendered frame is lit
            # (the viewport lighting rig alone does not illuminate the camera sensor).
            self._add_interior_dome_light(intensity=1000.0)
            # Scene meshes must always be registered as static colliders so that
            # _can_move_to overlap queries can hit walls/furniture. Tying this to
            # validate_initial_placement is a bug: it silently disables collisions
            # for quick-test configs and the robot walks through geometry.
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
            self._add_interior_dome_light(intensity=1000.0)
            self._add_procedural_obstacles()

        if self.robot_usd:
            from isaacsim.core.prims import SingleXFormPrim  # type: ignore[import-not-found]
            from isaacsim.core.utils.stage import (  # type: ignore[import-not-found]
                add_reference_to_stage,
            )

            self._base_height = _GO2_BASE_HEIGHT_M
            self._robot_prim_path = "/World/robot"
            camera_offset = (
                np.array(_xyz_degrees(camera_offset, "camera_offset"), dtype=np.float32)
                if camera_offset is not None
                else np.array(_GO2_CAMERA_OFFSET_M, dtype=np.float32)
            )
            add_reference_to_stage(self.robot_usd, "/World/robot")
            _freeze_articulation(self._world.stage, "/World/robot")
            self._robot = SingleXFormPrim(
                self._robot_prim_path,
                name="robot",
                position=self._initial_robot_position_usd,
            )
        else:
            self._base_height = _CUBOID_BASE_HEIGHT_M
            self._robot_prim_path = "/World/robot"
            camera_offset = np.array([0.0, 0.0, 0.35], dtype=np.float32)
            self._robot = self._world.scene.add(
                DynamicCuboid(
                    prim_path="/World/robot",
                    name="robot",
                    position=self._initial_robot_position_usd,
                    scale=np.array([0.3, 0.3, 0.3], dtype=np.float32),
                )
            )
        # ------------------------------------------------------------------
        # The head ("agent") camera is the only lens: a child prim of the robot
        # Xform, so it follows the body's world pose for free once its local
        # pose is set at construction time; no per-frame repositioning is
        # needed. An on-demand look-around scan perturbs only its local
        # orientation (see _update_head_camera_local_pose).
        # ------------------------------------------------------------------
        height, width = camera_resolution

        # The camera prim is created beneath the robot Xform. Its transform is therefore
        # local to base_link/robot, and the robot world pose carries it automatically.
        self._head_camera_path = f"{self._robot_prim_path}/head_camera"
        self._camera = Camera(
            prim_path=self._head_camera_path,
            resolution=(width, height),
            frequency=self._camera_fps,
        )
        self._camera.initialize()
        self._camera.set_focal_length(self._camera_focal_length)
        self._camera.set_clipping_range(0.1, 20.0)
        self._camera.add_distance_to_image_plane_to_frame()

        # camera_offset is already expressed in the robot local frame.
        self._camera_offset = np.asarray(camera_offset, dtype=np.float32)

        # q_base_camera maps camera axes to base_link axes. The user correction is
        # camera-local, so it is right-multiplied:
        # q_base_camera = q_base_optical * q_optical_correction.
        self._camera_base_orientation_wxyz = _normalize_quat_wxyz(
            _quat_multiply_wxyz(
                _OPTICAL_TO_BASE_OFFSET_WXYZ,
                self._go2_camera_orient_wxyz,
            )
        )

        # In Isaac Sim Camera / XFormPrim APIs, local translation is named `translation`.
        # This is deliberately set once; later updates affect only local orientation for scans.
        self._camera.set_local_pose(
            translation=self._camera_offset,
            orientation=self._camera_base_orientation_wxyz,
        )

        # Fail early if a version/API change creates the camera outside the intended robot Xform.
        camera_prim = self._world.stage.GetPrimAtPath(self._head_camera_path)
        if not camera_prim.IsValid():
            raise RuntimeError(f"Head camera prim was not created: {self._head_camera_path}")
        parent_prim = camera_prim.GetParent()
        if not parent_prim.IsValid() or parent_prim.GetPath().pathString != self._robot_prim_path:
            actual_parent = parent_prim.GetPath().pathString if parent_prim.IsValid() else None
            raise RuntimeError(
                f"Head camera must be parented to {self._robot_prim_path}; "
                f"actual parent is {actual_parent}"
            )

        self._world.reset()
        self._set_robot_pose(self._initial_robot_position_usd, yaw=self._initial_robot_yaw_rad)
        for _ in range(2):
            self._world.step(render=False)
        if self._validate_initial_placement:
            self._validate_robot_placement()

        # First explicit application so the streamed viewport shows a valid frame
        # (a no-op scan perturbation since _head_scan_frames starts at 0).
        self._update_head_camera_local_pose()
        if bind_viewport_to_camera:
            self._bind_viewport(self._head_camera_path)
        for _ in range(3):
            self._world.step(render=True)

    def _bind_viewport(self, camera_path: str) -> None:
        """Point the streamed viewport at the robot's camera."""
        from omni.kit.viewport.utility import (  # type: ignore[import-not-found]
            get_active_viewport,
        )

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path

    def _open_scene_stage(self, scene: str, up_axis: str) -> None:
        """Open the environment through SimulationApp's USD context before World setup.

        The InternScenes conversion writes scene content under top-level prims
        (/Objects for geometry, /Look for materials) with a Y up-axis and NO
        /World prim. A wrapper that references @scene@</World> therefore pulls in
        nothing and the rendered scene is empty (only the robot/lights added by the
        adapter remain). We instead reference the real content prim and apply a
        90deg Y-up->Z-up rotation on the wrapper when the asset is Y-up. The rotation
        is applied here (never baked into the asset) so the robot placement
        coordinates in scene.json stay valid in the original asset frame.
        """
        stem = Path(scene).stem
        wrapper = Path(tempfile.gettempdir()) / f"agentic_memory_nav_{stem}_wrapped.usda"
        wrapper.write_text(self._build_scene_wrapper_text(scene, up_axis))
        scene_to_open = str(wrapper)

        context = self._simulation_app.context
        if not context.open_stage(scene_to_open):
            raise RuntimeError(f"Isaac Sim failed to open scene stage: {scene_to_open}")
        while context.get_stage_loading_status()[2] > 0:
            self._simulation_app.update()

    def _build_scene_wrapper_text(self, scene: str, up_axis: str) -> str:
        """Build a wrapper that pulls the scene's real content into /World/scene.

        References @scene@</Objects> (geometry) when present, else falls back to the
        whole-layer reference, so both old (/World-rooted) and new (/Objects-rooted)
        InternScenes assets load. A Y-up asset is rotated 90deg about X so it stands
        upright in the Z-up Isaac Sim world.
        """
        asset_up, ref = "", None
        # Prefer a direct pxr read of the asset to pick the reference path that
        # actually holds geometry. New pipeline output nests content under /World
        # (/World/Objects); older assets keep it at the top level (/Objects). If
        # pxr is unavailable at this point we still never revert to the empty
        # whole-layer reference -- we default to the InternScenes-standard
        # /Objects layout instead of the broken /World assumption.
        try:
            from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

            probe = Usd.Stage.Open(scene)
            asset_up = str(UsdGeom.GetStageUpAxis(probe)).lower()
            for candidate in ("/World/Objects", "/Objects", "/World"):
                prim = probe.GetPrimAtPath(candidate)
                if prim.IsValid() and len(prim.GetChildren()) > 0:
                    ref = f"@{scene}@<{candidate}>"
                    break
        except Exception:
            pass
        if ref is None:
            ref = f"@{scene}@</Objects>"
        rotate = asset_up == "y" if asset_up else up_axis.lower() == "y"
        rotate_lines = (
            (
                "        quatd xformOp:orient = (0.70710678, 0.70710678, 0, 0)\n"
                '        uniform token[] xformOpOrder = ["xformOp:orient"]\n'
            )
            if rotate
            else ""
        )
        return (
            '#usda 1.0\n(\n    defaultPrim = "World"\n    upAxis = "Z"\n)\n\n'
            'def Xform "World"\n{\n'
            f'    def Xform "scene" (\n        prepend references = {ref}\n    )\n'
            "    {\n" + rotate_lines + "    }\n}\n"
        )

    def _assert_world_identity(self) -> None:
        world = self._simulation_app.context.get_stage().GetPrimAtPath("/World")
        if not world.IsValid():
            raise RuntimeError("Scene did not produce a /World root")

    def _apply_builtin_grey_studio_light_rig(self) -> None:
        """Apply Isaac Sim's built-in Grey Studio rig through its viewport lighting API."""
        import carb  # type: ignore[import-not-found]
        import omni.kit.app  # type: ignore[import-not-found]

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate(
            "omni.kit.viewport.menubar.lighting", True
        )
        import omni.kit.viewport.menubar.lighting as lighting  # type: ignore[import-not-found]

        if lighting.__file__ is None:
            raise RuntimeError(
                "Isaac Sim lighting module has no __file__; cannot resolve Grey Studio asset"
            )
        rig_directory = Path(lighting.__file__).resolve().parents[5] / "data/usd"
        if not (rig_directory / "Grey_Studio.usda").is_file():
            raise RuntimeError(f"Isaac Sim built-in Grey Studio asset is missing: {rig_directory}")
        carb.settings.get_settings().set(
            "/exts/omni.kit.viewport.menubar.lighting/rigs", str(rig_directory)
        )
        from omni.kit.viewport.menubar.lighting.actions import (  # type: ignore[import-not-found]
            _set_lighting_mode,
        )

        success, _, _ = _set_lighting_mode("Grey Studio", usd_context=self._simulation_app.context)
        if not success:
            raise RuntimeError("Isaac Sim failed to apply built-in Grey_Studio light rig")

    def _add_interior_lighting_rig(
        self,
        intensity: float | None = None,
        dome_intensity: float = 100000.0,
        dome_exposure: float = 2.0,
        distant_intensity: float = 10000.0,
        sphere_intensity: float = 500000.0,
    ) -> None:
        """Add a UsdLux rig that actually illuminates the camera-rendered RGB frame.

        InternScenes conversions contain no UsdLux lights. The viewport lighting
        rig only lights the streamed/editor viewport; the camera sensor sees the
        actual stage illumination, which is otherwise black indoors. We place a
        bright omni SphereLight on the ceiling inside the room plus a very strong
        DomeLight so the renderer has light both from inside and outside the
        geometry.

        ``intensity`` is accepted as a backwards-compatible alias for
        ``dome_intensity`` because older call sites pass it as a keyword.
        """
        from pxr import Gf, Sdf, UsdLux  # type: ignore[import-not-found]

        if intensity is not None:
            dome_intensity = intensity

        # Very bright ambient fill so light reaches the camera even through walls.
        dome = UsdLux.DomeLight.Define(self._world.stage, "/World/realtime_agent_dome_light")
        dome.CreateIntensityAttr(float(dome_intensity))
        dome.CreateExposureAttr(float(dome_exposure))
        dome.CreateColorAttr((1.0, 1.0, 1.0))
        dome.CreateColorTemperatureAttr(6500.0)
        dome.CreateTextureFileAttr("")

        # A directional sun gives the interior shape/definiton and prevents the
        # scene from looking like a flat ambient blob. The rotation points it
        # slightly downward through the room.
        sun = UsdLux.DistantLight.Define(self._world.stage, "/World/realtime_agent_sun_light")
        sun.CreateIntensityAttr(float(distant_intensity))
        sun.CreateAngleAttr(34.3)
        sun.CreateColorAttr((1.0, 1.0, 1.0))
        # DistantLight has no orient attr; rotate the xform.
        sun_xform = self._world.stage.GetPrimAtPath("/World/realtime_agent_sun_light")
        if sun_xform.IsValid():
            rot_op = sun_xform.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3)
            rot_op.Set(Gf.Vec3d(255.0, 0.0, 0.0))
            order = sun_xform.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray)
            order.Set(["xformOp:rotateXYZ"])

        # Ceiling-mounted omni light that actually lives inside the room.
        self._add_ceiling_sphere_light(intensity=float(sphere_intensity))

    def _find_ceiling_z(self) -> float | None:
        """Return the lowest ceiling-ish prim's world Z, or None if not found."""
        from pxr import UsdGeom  # type: ignore[import-not-found]

        stage = self._world.stage
        scene = stage.GetPrimAtPath("/World/scene")
        if not scene.IsValid():
            scene = stage.GetPrimAtPath("/World")
        if not scene.IsValid():
            return None

        cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        best_z: float | None = None
        scene_path_str = str(scene.GetPath())
        for prim in self._world.stage.Traverse():
            if not str(prim.GetPath()).startswith(scene_path_str):
                continue
            name = prim.GetName().lower()
            if "ceiling" not in name and "roof" not in name:
                continue
            bound = cache.ComputeWorldBound(prim)
            aligned = bound.ComputeAlignedRange()
            min_z = float(aligned.GetMin()[2])
            # Use the lower face of the ceiling slab as the mounting height.
            candidate = min_z
            if best_z is None or candidate < best_z:
                best_z = candidate
        return best_z

    def _add_ceiling_sphere_light(
        self,
        prim_path: str = "/World/realtime_agent_ceiling_light/sphere_light",
        intensity: float = 500000.0,
        radius: float = 0.15,
    ) -> None:
        """Add an omni SphereLight on the room ceiling, shining in all directions.

        If the USD contains a prim whose name includes 'ceiling' or 'roof', the
        light is placed just underneath its lowest world-Z face and centered in
        X/Y. Otherwise it falls back to the top of the scene bounding box.
        """
        from pxr import Gf, UsdLux  # type: ignore[import-not-found]

        lower, upper = self._environment_bounds
        ceiling_z = self._find_ceiling_z()
        if ceiling_z is None:
            ceiling_z = float(upper[2])

        x_center = (float(lower[0]) + float(upper[0])) * 0.5
        y_center = (float(lower[1]) + float(upper[1])) * 0.5
        # Hang the omni light slightly below the ceiling so it is inside the room.
        z_mount = ceiling_z - 0.1

        light = UsdLux.SphereLight.Define(self._world.stage, prim_path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateExposureAttr(1.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        light.CreateColorTemperatureAttr(5500.0)
        light.CreateRadiusAttr(float(radius))
        # SphereLight radiates in all directions, so it only needs translation.
        light.AddTranslateOp().Set(Gf.Vec3d(x_center, y_center, z_mount))

    # Backwards-compatible alias kept in case anything still calls it by name.
    _add_interior_dome_light = _add_interior_lighting_rig

    def _get_environment_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        from pxr import UsdGeom  # type: ignore[import-not-found]

        scene = self._world.stage.GetPrimAtPath("/World/scene")
        if not scene.IsValid():
            scene = self._world.stage.GetPrimAtPath("/World")
        if not scene.IsValid():
            raise RuntimeError("Opened scene does not define a usable /World or /World/scene root")
        bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(scene)
        aligned = bounds.ComputeAlignedRange()
        return np.array(aligned.GetMin(), dtype=np.float32), np.array(
            aligned.GetMax(), dtype=np.float32
        )

    def _add_environment_planes(self, config: dict[str, Any]) -> None:
        # An explicit empty config means "this scene already has its own authored
        # ground/ceiling geometry". Creating default planes here for a prebuilt USD
        # asset causes duplicate static colliders and can trip PhysX with invalid
        # broad-phase bounds.
        if not config:
            return
        lower, upper = self._environment_bounds
        for name, z, default_color in (
            ("ground", float(lower[2]), (0.18, 0.20, 0.22)),
            ("ceiling", float(upper[2]), (0.82, 0.84, 0.88)),
        ):
            settings = dict(config.get(name, {}))
            if settings.get("enabled", True):
                self._add_static_plane(name, z, lower, upper, default_color, settings)

    def _add_static_plane(
        self,
        name: str,
        z: float,
        lower: np.ndarray,
        upper: np.ndarray,
        default_color: tuple[float, float, float],
        config: dict[str, Any],
    ) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade  # type: ignore[import-not-found]

        margin = float(config.get("margin_m", 0.5))
        if margin < 0.0:
            raise ValueError(f"environment_planes.{name}.margin_m must be non-negative")
        color = tuple(float(value) for value in config.get("color", default_color))
        if len(color) != 3:
            raise ValueError(f"environment_planes.{name}.color must have 3 values")
        path = str(config.get("prim_path", f"/World/{name}_plane"))

        lower_f = np.asarray(lower, dtype=np.float64)
        upper_f = np.asarray(upper, dtype=np.float64)
        if not np.all(np.isfinite(lower_f)) or not np.all(np.isfinite(upper_f)):
            raise ValueError(
                f"environment_planes.{name} has non-finite bounds: lower={lower!r}, upper={upper!r}"
            )

        position = np.array(
            [
                (float(lower_f[0]) + float(upper_f[0])) / 2.0,
                (float(lower_f[1]) + float(upper_f[1])) / 2.0,
                float(z) + float(config.get("z_offset_m", 0.0)),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(position)):
            raise ValueError(
                f"environment_planes.{name} has non-finite plane position: {position!r}"
            )

        width = float(upper_f[0] - lower_f[0]) + 2.0 * margin
        length = float(upper_f[1] - lower_f[1]) + 2.0 * margin
        if not math.isfinite(width) or not math.isfinite(length) or width <= 0.0 or length <= 0.0:
            raise ValueError(
                f"environment_planes.{name} generated invalid plane dimensions: "
                f"width={width}, length={length}, lower={lower!r}, upper={upper!r}"
            )

        plane = UsdGeom.Plane.Define(self._world.stage, path)
        plane.CreateAxisAttr("Z")
        plane.CreateDoubleSidedAttr(True)
        plane.CreateWidthAttr(width)
        plane.CreateLengthAttr(length)
        plane.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in position)))
        material = UsdShade.Material.Define(self._world.stage, f"{path}_material")
        shader = UsdShade.Shader.Define(self._world.stage, f"{path}_material/preview")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(plane).Bind(material)
        collision = UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
        collision.CreateCollisionEnabledAttr(bool(config.get("collision_enabled", True)))
        body = UsdPhysics.RigidBodyAPI.Apply(plane.GetPrim())
        body.CreateKinematicEnabledAttr(True)
        body.CreateRigidBodyEnabledAttr(True)
        self._environment_plane_paths.add(path)

    def _enable_scene_collisions(self) -> None:
        """Make imported render meshes queryable as static PhysX environment colliders.

        Solid room-shell meshes (low vertex count, large volume) are replaced by
        six thin slab colliders so the robot can navigate inside the room.
        """
        from pxr import UsdGeom  # type: ignore[import-not-found]

        room_shell_paths: list[str] = []
        for prim in self._world.stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            if not points:
                continue
            name = str(prim.GetName())
            path = str(prim.GetPath())

            pts = np.asarray(points)
            lower = pts.min(axis=0)
            upper = pts.max(axis=0)
            sizes = upper - lower
            volume = float(np.prod(sizes))

            # Heuristic: InternScenes room shells are simple boxes that occupy
            # the whole scene volume. Treating them as solid blocks navigation.
            if self._is_room_shell_mesh(prim, pts):
                room_shell_paths.append(path)
                continue

            # Skip floor/ceiling slabs. A vertical wall has a large Z extent;
            # a floor/ceiling has a very small Z extent relative to its
            # horizontal span. Using a 10 cm threshold catches typical floors
            # while leaving walls and furniture collidable.
            if sizes[2] < 0.10 or sizes[2] / max(max(sizes[0], sizes[1]), 1e-6) < 0.05:
                continue

            self._apply_mesh_collision(prim)

        # Always add a bounds shell as a safety net. If no room shell was found
        # this prevents the robot from walking out of the scene entirely; if a
        # room shell was found the bounds shell is added in addition to the
        # detected shell for redundancy.
        self._add_hollow_room_colliders_from_bounds()

    def _apply_mesh_collision(self, prim: Any) -> None:
        """Apply a PhysX collision API to a mesh."""
        from pxr import UsdPhysics  # type: ignore[import-not-found]

        UsdPhysics.CollisionAPI.Apply(prim)
        try:
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set("convexDecomposition")
        except Exception:
            pass

    def _is_room_shell_mesh(self, prim: Any, pts: np.ndarray) -> bool:
        """Detect solid room bounding-box meshes from InternScenes conversions."""
        name = str(prim.GetName())
        if not (name.startswith("world_") or name.startswith("geometry_")):
            return False
        if pts.shape[0] < 8:
            return False
        lower = pts.min(axis=0)
        upper = pts.max(axis=0)
        sizes = upper - lower
        if np.any(sizes <= 0.0):
            return False
        volume = float(np.prod(sizes))
        # A room shell is big (tens of cubic metres) and box-like. Allow more
        # vertices than a perfect cube because some conversions tessellate the
        # six faces. Also compare against the authored scene bounds: if the mesh
        # spans the whole environment it is the room shell, not furniture.
        if volume <= 5.0 or np.any(sizes < 0.3):
            return False
        env_lower, env_upper = self._environment_bounds
        env_sizes = env_upper - env_lower
        spans_env = np.all(sizes >= env_sizes * 0.85)
        return bool(pts.shape[0] <= 2000 or spans_env)

    def _add_hollow_room_colliders_from_bounds(self) -> None:
        """Add six thin box colliders for walls/floor/ceiling around /World/scene."""

        lower, upper = self._environment_bounds
        thickness = 0.2
        x_mid = (float(lower[0]) + float(upper[0])) / 2.0
        y_mid = (float(lower[1]) + float(upper[1])) / 2.0
        z_mid = (float(lower[2]) + float(upper[2])) / 2.0
        x_size = float(upper[0] - lower[0])
        y_size = float(upper[1] - lower[1])
        z_size = float(upper[2] - lower[2])
        wall_specs = [
            # name, center (x,y,z), half extents (x,y,z)
            (
                "wall_min_x",
                (float(lower[0]) - thickness / 2, y_mid, z_mid),
                (thickness / 2, y_size / 2, z_size / 2),
            ),
            (
                "wall_max_x",
                (float(upper[0]) + thickness / 2, y_mid, z_mid),
                (thickness / 2, y_size / 2, z_size / 2),
            ),
            (
                "wall_min_y",
                (x_mid, float(lower[1]) - thickness / 2, z_mid),
                (x_size / 2, thickness / 2, z_size / 2),
            ),
            (
                "wall_max_y",
                (x_mid, float(upper[1]) + thickness / 2, z_mid),
                (x_size / 2, thickness / 2, z_size / 2),
            ),
            (
                "floor",
                (x_mid, y_mid, float(lower[2]) - thickness / 2),
                (x_size / 2, y_size / 2, thickness / 2),
            ),
            (
                "ceiling",
                (x_mid, y_mid, float(upper[2]) + thickness / 2),
                (x_size / 2, y_size / 2, thickness / 2),
            ),
        ]
        for name, center, half_extents in wall_specs:
            path = f"/World/{name}_collider"
            self._add_box_mesh_collider(path, center, half_extents)
            self._environment_plane_paths.add(path)

    def _add_box_mesh_collider(
        self,
        path: str,
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
    ) -> None:
        """Create a static box mesh collider with explicit vertices.

        Using an explicit triangle mesh avoids the ambiguous scale semantics of
        Isaac Sim's high-level cuboid wrappers and ensures overlap_box queries
        see the expected bounds.
        """
        from pxr import Gf, UsdGeom, UsdPhysics  # type: ignore[import-not-found]

        cx, cy, cz = (float(v) for v in center)
        hx, hy, hz = (float(v) for v in half_extents)
        points = [
            Gf.Vec3f(cx - hx, cy - hy, cz - hz),
            Gf.Vec3f(cx + hx, cy - hy, cz - hz),
            Gf.Vec3f(cx + hx, cy + hy, cz - hz),
            Gf.Vec3f(cx - hx, cy + hy, cz - hz),
            Gf.Vec3f(cx - hx, cy - hy, cz + hz),
            Gf.Vec3f(cx + hx, cy - hy, cz + hz),
            Gf.Vec3f(cx + hx, cy + hy, cz + hz),
            Gf.Vec3f(cx - hx, cy + hy, cz + hz),
        ]
        # 12 triangles (two per face)
        indices = [
            0,
            2,
            1,
            0,
            3,
            2,  # bottom
            4,
            5,
            6,
            4,
            6,
            7,  # top
            0,
            1,
            5,
            0,
            5,
            4,  # front
            2,
            3,
            7,
            2,
            7,
            6,  # back
            0,
            4,
            7,
            0,
            7,
            3,  # left
            1,
            2,
            6,
            1,
            6,
            5,  # right
        ]
        mesh = UsdGeom.Mesh.Define(self._world.stage, path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * 12)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateExtentAttr(
            [
                Gf.Vec3f(cx - hx, cy - hy, cz - hz),
                Gf.Vec3f(cx + hx, cy + hy, cz + hz),
            ]
        )
        collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        body = UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
        body.CreateKinematicEnabledAttr(True)
        body.CreateRigidBodyEnabledAttr(True)

    def _validate_robot_placement(self) -> None:
        """Abort before the loop when the robot torso intersects walls/furniture.

        Standing on the floor is expected, so the vertical check height is
        shrunk to torso level only. This avoids false positives where the
        robot's feet briefly intersect the authored floor mesh.
        """
        position, _ = self._robot.get_world_pose()
        hits = self._environment_overlap_hits(
            position,
            self._yaw,
            half_extents=(0.34, 0.20, 0.05),
        )
        if hits:
            raise RuntimeError(
                "Robot initial placement overlaps environment colliders: " + ", ".join(hits[:5])
            )

    def _environment_overlap_hits(
        self,
        position: np.ndarray,
        yaw: float,
        half_extents: tuple[float, float, float] | None = None,
    ) -> list[str]:
        import carb  # type: ignore[import-not-found]
        from omni.physx import get_physx_scene_query_interface  # type: ignore[import-not-found]

        hits: list[str] = []

        def report_overlap(hit: Any) -> bool:
            collider = str(hit.rigid_body)
            if collider.startswith("/World/robot"):
                return True
            # Ignore the viewport lighting rig's ground plane and any other
            # non-scene colliders that leak into overlap queries.
            if collider.startswith("/OmniKit_Viewport_LightRig"):
                return True
            if collider in self._environment_plane_paths:
                return True
            hits.append(collider)
            return True

        if half_extents is None:
            half_extents = _GO2_TORSO_HALF_EXTENTS_M
        half_x, half_y, half_z = half_extents
        half_yaw = yaw * 0.5
        get_physx_scene_query_interface().overlap_box(
            carb.Float3(half_x, half_y, half_z),
            carb.Float3(*position),
            carb.Float4(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            report_overlap,
            False,
        )
        return sorted(set(hits))

    def _can_move_to(
        self,
        position: np.ndarray,
        yaw: float,
        from_position: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        """Check the destination pose and a few samples along the path.

        A single overlap test at the destination can miss thin walls whose
        thickness is smaller than the movement step. Sampling the travelled
        segment catches those cases while still being cheap (at most
        ``_SWEPT_CHECK_MAX_SAMPLES`` extra queries).
        """
        hits = self._environment_overlap_hits(position, yaw)
        if hits:
            return False, "collision predicted with " + ", ".join(hits[:3])

        if from_position is not None:
            delta = position - from_position
            horizontal_distance = float(np.linalg.norm(delta[:2]))
            if horizontal_distance > 1e-6:
                num_substeps = min(
                    _SWEPT_CHECK_MAX_SAMPLES,
                    max(1, int(horizontal_distance / _SWEPT_CHECK_SAMPLE_SPACING_M)),
                )
                for i in range(1, num_substeps + 1):
                    t = i / (num_substeps + 1)
                    sample_pos = from_position + delta * t
                    hits = self._environment_overlap_hits(sample_pos, yaw)
                    if hits:
                        return False, "collision predicted with " + ", ".join(hits[:3])
        return True, "clear"

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
        x, y, z = (float(value) for value in position_usd)
        return Pose3D(position=(x, y, z), yaw=yaw)

    def _pose_to_usd(self, position: Vector3) -> np.ndarray:
        return np.array([position[0], position[1], position[2]], dtype=np.float32)

    def _set_robot_pose(self, position_usd: np.ndarray, yaw: float | None = None) -> None:
        if yaw is not None:
            self._yaw = yaw
        self._robot.set_world_pose(position=position_usd, orientation=_yaw_to_quat_wxyz(self._yaw))

    def trigger_head_scan(self, frames: int) -> None:
        """Arm a bounded head look-around; the lens is static until this is called.

        The agent camera holds still and points straight ahead with the body by
        default. A caller invokes this only when a look-around is genuinely needed
        (e.g. a new object just entered the scene graph); until the armed window
        expires the view never drifts.
        """
        self._head_scan_frames = max(0, int(frames))

    def _update_head_camera_local_pose(self) -> None:
        """Update only the parented head camera's local orientation.

        The camera translation is fixed in base_link coordinates. The robot's world
        transform propagates to the camera through the USD hierarchy. A scan applies
        base_link-frame yaw/pitch perturbations on the left of the fixed camera-to-base
        orientation, preserving the same physical semantics as the former world look-at.
        """
        scan_yaw = 0.0
        scan_pitch = self._manual_pitch_offset_rad
        if self._head_scan_frames > 0:
            self._head_scan_frames -= 1
            phase = 2.0 * math.pi * self._frame_index / self._head_scan_period_frames
            scan_yaw = self._head_scan_yaw_rad * math.sin(phase)
            scan_pitch += self._head_scan_pitch_rad * math.sin(phase * 0.5)

        if scan_yaw == 0.0 and scan_pitch == 0.0:
            orientation = self._camera_base_orientation_wxyz
        else:
            # Base +Z is yaw-up. Base -Y is the rightward pitch axis, so positive
            # scan_pitch lifts the viewing direction for a camera whose -Z is forward.
            q_scan_in_base = _quat_multiply_wxyz(
                _axis_angle_quat_wxyz((0.0, 0.0, 1.0), scan_yaw),
                _axis_angle_quat_wxyz((0.0, -1.0, 0.0), scan_pitch),
            )
            orientation = _normalize_quat_wxyz(
                _quat_multiply_wxyz(q_scan_in_base, self._camera_base_orientation_wxyz)
            )

        # Do not pass translation here: its fixed local extrinsic must not be reset.
        self._camera.set_local_pose(orientation=orientation)

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
        self._set_robot_pose(self._initial_robot_position_usd, yaw=self._initial_robot_yaw_rad)
        self._world.reset()

    def get_state(self) -> Pose3D:
        position, _ = self._robot.get_world_pose()
        return self._usd_to_pose(position, self._yaw)

    def get_observation(self) -> FrameObservation:
        """Render the parented head camera and return RGB/depth observation."""
        self._update_head_camera_local_pose()
        self._world.step(render=True)

        rgba = self._camera.get_rgba()
        height, width = self._camera_resolution
        if rgba is not None and rgba.size:
            rgb = rgba[:, :, :3].astype(np.uint8)
        else:
            rgb = np.zeros((height, width, 3), dtype=np.uint8)

        depth_frame = self._camera.get_current_frame().get("distance_to_image_plane")
        depth = np.asarray(depth_frame, dtype=np.float32) if depth_frame is not None else None

        # camera_pose currently follows the backend FrameObservation contract, which
        # exposes Pose3D (robot pose). If the contract is extended, populate it from
        # self._camera.get_world_pose() rather than reusing robot_pose.
        robot_pose = self.get_state()
        frame = FrameObservation(
            frame_id=f"frame_{self._frame_index:04d}",
            timestamp=float(self._frame_index),
            rgb=rgb,
            depth=depth,
            camera_intrinsics=CameraIntrinsics(80.0, 80.0, width / 2, height / 2, width, height),
            camera_pose=robot_pose,
            robot_pose=robot_pose,
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

        # Rotate local velocity into world coordinates.
        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)
        vx_world = vx * cos_yaw - vy * sin_yaw
        vy_world = vx * sin_yaw + vy * cos_yaw

        destination = position + np.array(
            [vx_world * self.dt, vy_world * self.dt, 0.0], dtype=np.float32
        )
        yaw = self._yaw + wz * self.dt

        can_move, message = self._can_move_to(destination, yaw, from_position=position)
        if not can_move:
            self.stop()
            self._collision = True
            return ExecutionFeedback("velocity", False, self.get_state(), True, 0.0, message)

        self._set_robot_pose(destination, yaw=yaw)
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

    def apply_discrete_action(self, action: DiscreteAction | str) -> ExecutionFeedback:
        started = time.perf_counter()
        action = DiscreteAction(action)

        if action is DiscreteAction.STOP:
            self.stop()
            return ExecutionFeedback(
                action.value, True, self.get_state(), self._collision, 0.0, "stopped"
            )

        if action in (DiscreteAction.LOOK_UP, DiscreteAction.LOOK_DOWN):
            sign = 1.0 if action is DiscreteAction.LOOK_UP else -1.0
            new_pitch = self._manual_pitch_offset_rad + sign * self._look_step_rad
            self._manual_pitch_offset_rad = max(
                -LOOK_PITCH_LIMIT_RAD, min(LOOK_PITCH_LIMIT_RAD, new_pitch)
            )
            self._stopped = False
            self._world.step(render=False)
            return ExecutionFeedback(
                action.value,
                True,
                self.get_state(),
                self._collision,
                time.perf_counter() - started,
                "executed",
            )

        if action in (
            DiscreteAction.TURN_LEFT,
            DiscreteAction.TURN_RIGHT,
            DiscreteAction.TURN_LEFT_BIG,
            DiscreteAction.TURN_RIGHT_BIG,
        ):
            sign = (
                1.0 if action in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_LEFT_BIG) else -1.0
            )
            step_rad = (
                self._turn_big_step_rad
                if action in (DiscreteAction.TURN_LEFT_BIG, DiscreteAction.TURN_RIGHT_BIG)
                else self._turn_step_rad
            )
            position, _ = self._robot.get_world_pose()
            yaw = self._yaw + sign * step_rad
            self._set_robot_pose(position, yaw=yaw)
            self._stopped = False
            self._world.step(render=False)
            return ExecutionFeedback(
                action.value,
                True,
                self.get_state(),
                self._collision,
                time.perf_counter() - started,
                "executed",
            )

        if action in (DiscreteAction.MOVE_FORWARD, DiscreteAction.MOVE_BACKWARD):
            sign = 1.0 if action is DiscreteAction.MOVE_FORWARD else -1.0
            position, _ = self._robot.get_world_pose()
            destination = position + np.array(
                [
                    sign * self._move_step_m * math.cos(self._yaw),
                    sign * self._move_step_m * math.sin(self._yaw),
                    0.0,
                ],
                dtype=np.float32,
            )
            can_move, message = self._can_move_to(
                destination, self._yaw, from_position=position
            )
            if not can_move:
                self.stop()
                self._collision = True
                return ExecutionFeedback(action.value, False, self.get_state(), True, 0.0, message)
            self._set_robot_pose(destination, yaw=self._yaw)
            self._stopped = False
            self._world.step(render=False)
            return ExecutionFeedback(
                action.value,
                True,
                self.get_state(),
                self._collision,
                time.perf_counter() - started,
                "executed",
            )

        # Unreachable: all valid actions are handled above.
        self.stop()
        return ExecutionFeedback(
            action.value, False, self.get_state(), self._collision, 0.0, "unknown action"
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
        yaw = math.atan2(float(delta[1]), float(delta[0]))
        can_move, message = self._can_move_to(destination, yaw)
        if not can_move:
            self.stop()
            self._collision = True
            return ExecutionFeedback(intent.action_id, False, self.get_state(), True, 0.0, message)
        self._set_robot_pose(destination, yaw=yaw)
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

    def stream_until_interrupted(self) -> None:
        """Keep the render loop alive after a completed run so WebRTC stays connected."""
        try:
            while True:
                self._world.step(render=True)
                time.sleep(1.0 / self._camera_fps)
        except KeyboardInterrupt:
            return

    def close(self) -> None:
        self._world.stop()
        self._simulation_app.close()
