"""Optional Isaac Sim executor boundary."""

# 【模块】可选的 Isaac Sim 执行器边界（核心后端）。
# 【作用】RobotBackend 兼容的 Isaac Sim 薄封装：构建/加载场景、放置机器人与相机、
#         运动学驱动、碰撞预检、渲染帧输出，供实时导航流水线调用。
#
# 【坐标系 —— 已标准化，全部定义如下，请勿再混用】
#   * odom（world）  : 全局固定坐标系，Z 轴朝上。就是 Isaac Sim 的 USD 世界坐标。
#                      Pose3D 直接存 odom (x, y, z)，与 USD 完全一致，不再做任何重映射。
#   * base_link     : 固定在机器狗躯干上的本体坐标系。X 朝前、Y 朝左、Z 朝上
#                      （ROS REP 105 标准）。机器人相对 odom 的偏航 yaw 绕 odom 的 Z 轴，
#                      yaw=0 时 base_link 的 X（前向）与 odom 的 X 对齐。
#   * camera_link   : 固定在机器狗上的相机安装坐标系，随 base_link 一起运动。
#                      其相对 base_link 的偏移用 _GO2_CAMERA_OFFSET_M 表达（base_link 系）。
#   * 相机光学轴     : Isaac Sim 默认沿本体系 -Z 看、+Y 朝上（本文件所有 look-at 均按此）；
#                      若需 ROS 光学系（+Z 看、+Y 朝下），在输出帧处再加一次固定变换。
#
# 【依赖】isaacsim 惰性导入；纯 mock 流程不要求安装 Isaac Sim。
# 【进程】SimulationApp 每进程只能一个，_ensure_simulation_app 复用；close() 会结束进程。
# 【注意】本文件当前只支持 robot_motion_mode='kinematic'（无步态/关节控制），
#         light_rig 仅 'gray_studio'；Go2 资产被冻结为刚体以保持直立。

from __future__ import annotations

import importlib.util
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from agentic_memory_nav.agent.execution.discrete_actions import (
    LOOK_PITCH_LIMIT_RAD,
    LOOK_STEP_RAD,
    MOVE_STEP_M,
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

# ------------------------------------------------------------------
# 相机光学系 vs 本体系的对应（关键，务必保持一致）。
#
# Isaac Sim 的 Camera 本体系：沿 -Z 看、+Y 朝上、+X 朝右（标准 Isaac 光学系）。
# 本项目 base_link（机器狗本体系，ROS REP 105）：+X 前、+Y 左、+Z 上。
# 两者朝向不同，所以“把镜头对准 base_link 前向”需要一次固定的旋转对齐：
#   让镜头 +X（右）→ base_link -Y（右，因 base +Y 是左），
#   让镜头 +Y（上）→ base_link +Z（上），
#   让镜头 +Z（后）→ base_link -X（前向的反向），
#   即镜头 -Z（看的方向）→ base_link +X（前向）。
# 该四元数经数值验证：optical +X→(0,-1,0)、+Y→(0,0,1)、+Z→(-1,0,0)。
# ------------------------------------------------------------------
# head 镜头现在是机器人的子节点(parented)：这个常量就是它的固定局部朝向，
# 只在构造时通过 set_local_pose 设一次，之后靠父节点(机器人)的世界位姿带动，
# 不再每帧重新 look-at。

# Body origin height and head-mounted camera offset for the Unitree Go2 asset, whose
# authored default pose is standing. The camera sits clear of the head mesh, which
# otherwise occludes the lower half of the frame.
#
# _GO2_CAMERA_OFFSET_M 是 camera_link 相对 base_link 的偏移，单位 base_link 系：
#   (forward=前向, left=左向, up=上) = (0.42, 0.0, 0.14) m。
#   前向 0.42m 让镜头在 Go2 机头前方、左向 0、上向 0.14m 抬高避免被机头遮挡。

_OPTICAL_TO_BASE_OFFSET_WXYZ = np.array(
    [1.0, 0.0, 0.0, 0.0], dtype=np.float32
)

# Camera-link translation in base_link coordinates: (forward, left, up), metres.
_GO2_BASE_HEIGHT_M = 0.40
_GO2_CAMERA_OFFSET_M = (0.25, 0.0, 0.20)
_GO2_STANDING_HALF_EXTENTS_M = (0.34, 0.20, 0.30)
_CUBOID_BASE_HEIGHT_M = 0.15


def _normalize_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("Quaternion norm must be non-zero")
    return quaternion / norm

# 【函数】偏航角 → WXYZ 四元数（绕世界 +Z 轴旋转）。
# 【原因】导航偏航 yaw 定义为绕 +Z 轴；cos/sin 半角构成四元数。
def _yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


# 【函数】X/Y/Z 欧拉角(度) → 归一化 WXYZ 四元数。
# 【原因】把配置里的 roll/pitch/yaw(度) 转成四元数偏移，供相机姿态叠加。
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


# 【函数】把可选的 X/Y/Z 度数向量转成 float32 数组；None → 零向量。
# 【原因】统一相机朝向偏移输入；长度必须为 3，否则抛错。
def _xyz_degrees(value: Vector3 | None, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float32)
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three X/Y/Z degree values")
    return np.asarray(value, dtype=np.float32)


# 【函数】WXYZ 四元数乘法 left⊗right。
# 【原因】把相机 look-at 姿态与配置朝向偏移组合（先 look-at 再叠加偏移）。
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


# 【函数】轴角 → WXYZ 四元数。
# 【原因】绕给定轴旋转指定角度；供朝向扰动使用。
def _axis_angle_quat_wxyz(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    half = angle * 0.5
    scale = math.sin(half)
    return np.array(
        [math.cos(half), axis[0] * scale, axis[1] * scale, axis[2] * scale],
        dtype=np.float32,
    )


# 【函数】把被引用的机器人(如 Go2)冻结为刚体、保持直立。
# 【原因】Go2 是物理 articulation，无步态控制器会在重力下塌陷；
#        故删除 ArticulationRootAPI、关闭关节、把刚体设为 kinematic，由父 Xform 驱动。
# 【结果】机器人作为刚体跟随世界位姿，不会自己倒下。
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


# 【函数】获取/创建 SimulationApp(每进程唯一，复用全局 _SIMULATION_APP)。
# 【分支】① 无 isaacsim → 抛错(提示改用 unitree_sim)；
#        ② 有 livestream_args → 用完整流媒体 experience，并按 window_resolution
#           同步窗口/渲染分辨率(否则 WebRTC 客户端会因分辨率超限丢帧)；
#        ③ 否则 headless 启动。
# 【原因】SimulationApp 每进程只能实例化一次；流媒体需专门的 experience。
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


# 【类】Isaac Sim 可用性边界(镜像 HabitatAdapter)。
# 【原因】available 检测 isaacsim 是否可导入；未安装则 start() 抛错。
class IsaacSimAdapter:
    """Availability boundary mirroring `HabitatAdapter` for the isaacsim backend."""

    # 【方法】记录场景路径；available = 能否导入 isaacsim。
    def __init__(self, scene: str | None) -> None:
        self.scene = scene
        self.available = importlib.util.find_spec("isaacsim") is not None
        
        
    # 【方法】未安装则抛错；已安装则提示用 IsaacSimExecutor(需已验证场景与机器人)。
    def start(self) -> None:
        if not self.available:
            raise RuntimeError("Isaac Sim is not installed; select execution.backend=unitree_sim")
        raise NotImplementedError("Use IsaacSimExecutor for a validated scene and robot")


# 【类】Isaac Sim 执行器(核心 RobotBackend 实现)。
# 【作用】构建/加载场景、放置机器人与 head 相机、运动学驱动、碰撞预检、渲染。
# 【坐标】USD 为 Z 轴朝上；Pose3D 高度在第 2 个分量，边界处重映射。
# 【相机】仅 head(=agent 相机)，parented 到机器人、局部位姿只设一次、随机器人
#        世界位姿自动跟随。
class IsaacSimExecutor:
    """Thin RobotBackend-compatible wrapper over Isaac Sim for MVP integration.

    Scenes are procedurally built (ground plane + obstacles) or loaded from a
    provided USD file; Nucleus-hosted sample environments are not available in
    this standalone installation. USD is z-up; this project's `Pose3D` stores
    height at index 1 (matching the Habitat adapter's convention), so
    coordinates are remapped at the boundary rather than left in native USD axes.
    """

    # 【方法】执行器构造(参数众多)。
    # 【流程】① 校验(场景路径/机器人 USD/相机 fps/焦距/流媒体相机/运动模式/灯光)；
    #        ② 解析场景(本地或 Nucleus 远程 URL)与机器人 USD；
    #        ③ _ensure_simulation_app 启动；④ 有场景则 _open_scene_stage + _assert_world_identity；
    #        ⑤ 构建 World：有场景→环境包围盒+平面+灰棚灯光(+可选碰撞校验)；
    #           无场景→平地+穹顶灯光+程序化障碍物；
    #        ⑥ 放置机器人(Go2 冻结为刚体 或 DynamicCuboid 占位)；
    #        ⑦ 创建 head 相机(parented 到机器人，局部位姿只设一次)；
    #        ⑧ 复位、设初始位姿、渲染若干帧预热。
    # 【关键状态】_head_scan_frames 默认 0(镜头静止)，仅 arm 后扫视。
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
        move_step_m: float | None = None,
        look_step_deg: float | None = None,
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
            raise ValueError(
                f"camera_focal_length must be positive, got {camera_focal_length}"
            )
        if robot_motion_mode != "kinematic":
            raise ValueError(
                "Only robot_motion_mode='kinematic' is supported until a Go2 gait controller is added"
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
        self._move_step_m = move_step_m if move_step_m is not None else MOVE_STEP_M
        self._look_step_rad = (
            math.radians(look_step_deg) if look_step_deg is not None else LOOK_STEP_RAD
        )
        self._validate_initial_placement = validate_initial_placement
        self._environment_planes = environment_planes or {}
        self._environment_plane_paths: set[str] = set()
        # go2 相机朝向偏移(欧拉角→四元数)：仅当外部显式传入非零偏移时才生效。
        # 当前 look-at 已直接朝前，默认取恒等 [0,0,0]。
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
            self._robot_prim_path = "/World/robot"
            camera_offset = np.array(_GO2_CAMERA_OFFSET_M, dtype=np.float32)
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
        # camera-local, so it is right-multiplied: q_base_camera = q_base_optical q_optical_correction.
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

    # 【方法】把 WebRTC 流媒体视口指向机器人自己的镜头(而非默认 persp 相机)。
    # 【原因】让用户在直播里看到的是 agent 视角。
    def _bind_viewport(self, camera_path: str) -> None:
        """Point the streamed viewport at the robot's own lens instead of the default persp camera."""
        from omni.kit.viewport.utility import (  # type: ignore[import-not-found]
            get_active_viewport,
        )

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path

    # 【方法】在 World 建立前，经 SimulationApp 的 USD context 打开环境场景。
    # 【分支】up_axis='y' 时：写一个临时 .usda 包装层，把场景放入 /World/scene 子层并
    #        加 90° 旋转(0.707,0.707,0,0)，把 Y-up 场景转成 Z-up；否则直接打开。
    # 【原因】Isaac Sim 要求 Z-up；包装层保留 /World 恒等变换(场景对齐放在 /World/scene)。
    def _open_scene_stage(self, scene: str, up_axis: str) -> None:
        """Open the environment through SimulationApp's USD context before World setup."""
        scene_to_open = scene
        if up_axis.lower() == "y":
            wrapper = Path(tempfile.gettempdir()) / f"agentic_memory_nav_{Path(scene).stem}_zup.usda"
            wrapper.write_text(
                "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Z\"\n)\n\n"
                "def Xform \"World\"\n{\n"
                f"    def Xform \"scene\" (\n        prepend references = @{scene}@</World>\n    )\n"
                "    {\n        quatd xformOp:orient = (0.70710678, 0.70710678, 0, 0)\n"
                "        uniform token[] xformOpOrder = [\"xformOp:orient\"]\n    }\n}\n"
            )
            scene_to_open = str(wrapper)

        context = self._simulation_app.context
        if not context.open_stage(scene_to_open):
            raise RuntimeError(f"Isaac Sim failed to open scene stage: {scene_to_open}")
        while context.get_stage_loading_status()[2] > 0:
            self._simulation_app.update()

    # 【方法】校验 /World 必须为恒等变换(无 xformOp)。
    # 【原因】场景对齐应放在 /World/scene 下；/World 若被变换会破坏坐标一致性。
    def _assert_world_identity(self) -> None:
        from pxr import UsdGeom  # type: ignore[import-not-found]

        world = self._simulation_app.context.get_stage().GetPrimAtPath("/World")
        if not world.IsValid() or UsdGeom.Xformable(world).GetOrderedXformOps():
            raise RuntimeError("/World must have identity transform; put scene alignment under /World/scene")

    # 【方法】应用 Isaac Sim 内置 Grey Studio 灯光 rig(经视口灯光 API)。
    # 【原因】真实场景需要稳定照明；解析 lighting 扩展路径找到 Grey_Studio.usda 并启用，
    #        失败则抛错。
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
            raise RuntimeError("Isaac Sim lighting module has no __file__; cannot resolve Grey Studio asset")
        rig_directory = Path(lighting.__file__).resolve().parents[5] / "data/usd"
        if not (rig_directory / "Grey_Studio.usda").is_file():
            raise RuntimeError(f"Isaac Sim built-in Grey Studio asset is missing: {rig_directory}")
        carb.settings.get_settings().set(
            "/exts/omni.kit.viewport.menubar.lighting/rigs", str(rig_directory)
        )
        from omni.kit.viewport.menubar.lighting.actions import (  # type: ignore[import-not-found]
            _set_lighting_mode,
        )

        success, _, _ = _set_lighting_mode(
            "Grey Studio", usd_context=self._simulation_app.context
        )
        if not success:
            raise RuntimeError("Isaac Sim failed to apply built-in Grey_Studio light rig")

    # 【方法】计算 /World/scene 的世界包围盒(对齐范围 min/max)。
    # 【原因】供平面/俯视相机放置、碰撞校验使用。
    def _get_environment_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        from pxr import UsdGeom  # type: ignore[import-not-found]

        scene = self._world.stage.GetPrimAtPath("/World/scene")
        if not scene.IsValid():
            raise RuntimeError("Opened scene does not define /World/scene")
        bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(scene)
        aligned = bounds.ComputeAlignedRange()
        return np.array(aligned.GetMin(), dtype=np.float32), np.array(aligned.GetMax(), dtype=np.float32)

    # 【方法】按配置添加地面/天花板静态平面(默认颜色+包围盒范围)。
    def _add_environment_planes(self, config: dict[str, Any]) -> None:
        lower, upper = self._environment_bounds
        for name, z, default_color in (
            ("ground", float(lower[2]), (0.18, 0.20, 0.22)),
            ("ceiling", float(upper[2]), (0.82, 0.84, 0.88)),
        ):
            settings = dict(config.get(name, {}))
            if settings.get("enabled", True):
                self._add_static_plane(name, z, lower, upper, default_color, settings)

    # 【方法】创建单个静态平面(UsdGeom.Plane + 材质 + 碰撞 + 刚体 API)。
    # 【参数】margin 外扩边距、color 颜色、z_offset 高度偏移；碰撞/运动学刚体 API。
    # 【原因】给真实场景补地/顶，并让其成为静态 PhysX 碰撞体。
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
        position = np.array(
            [
                (float(lower[0]) + float(upper[0])) / 2.0,
                (float(lower[1]) + float(upper[1])) / 2.0,
                z + float(config.get("z_offset_m", 0.0)),
            ],
            dtype=np.float32,
        )
        width = float(upper[0] - lower[0]) + 2.0 * margin
        length = float(upper[1] - lower[1]) + 2.0 * margin
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

    # 【方法】把导入的渲染网格(mesh)注册为静态 PhysX 环境碰撞体。
    # 【原因】只有带点的 mesh 才可查询碰撞；供 _can_move_to 预检使用。
    def _enable_scene_collisions(self) -> None:
        """Make imported render meshes queryable as static PhysX environment colliders."""
        from pxr import UsdGeom, UsdPhysics  # type: ignore[import-not-found]

        for prim in self._world.stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and UsdGeom.Mesh(prim).GetPointsAttr().Get():
                UsdPhysics.CollisionAPI.Apply(prim)

    # 【方法】循环前校验：Go2 站立体积与环境碰撞体相交则中止(抛错)。
    # 【原因】避免机器人初始嵌入墙/地板。
    def _validate_robot_placement(self) -> None:
        """Abort before the loop when the Go2 standing volume intersects the environment."""
        position, _ = self._robot.get_world_pose()
        hits = self._environment_overlap_hits(position, self._yaw)
        if hits:
            raise RuntimeError(
                "Go2 initial placement overlaps environment colliders: "
                + ", ".join(hits[:5])
            )

    # 【方法】用 PhysX overlap_box 查询机器人站立盒与环境的碰撞命中。
    # 【原因】半尺寸取 Go2 站立包围盒；旋转用 yaw 半角四元数；排除机器人自身与已加平面。
    # 【结果】返回命中的环境碰撞体路径列表(去重排序)。
    def _environment_overlap_hits(self, position: np.ndarray, yaw: float) -> list[str]:
        import carb  # type: ignore[import-not-found]
        from omni.physx import get_physx_scene_query_interface  # type: ignore[import-not-found]

        hits: list[str] = []

        def report_overlap(hit: Any) -> bool:
            collider = str(hit.rigid_body)
            if not collider.startswith("/World/robot") and collider not in self._environment_plane_paths:
                hits.append(collider)
            return True

        half_x, half_y, half_z = _GO2_STANDING_HALF_EXTENTS_M
        half_yaw = yaw * 0.5
        get_physx_scene_query_interface().overlap_box(
            carb.Float3(half_x, half_y, half_z),
            carb.Float3(*position),
            carb.Float4(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            report_overlap,
            False,
        )
        return sorted(set(hits))

    # 【方法】碰撞预检：到目标位置/朝向是否会碰撞。
    # 【原因】基于 _environment_overlap_hits；命中→(False, 原因)，否则→(True, 'clear')。
    def _can_move_to(self, position: np.ndarray, yaw: float) -> tuple[bool, str]:
        hits = self._environment_overlap_hits(position, yaw)
        if hits:
            return False, "collision predicted with " + ", ".join(hits[:3])
        return True, "clear"

    # 【方法】无场景时添加 3 个程序化固定立方体(占位障碍物)。
    # 【原因】平地场景无几何，加几个障碍物便于目视/测试。
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

    # 【方法】USD 世界坐标 → Pose3D。odom 与 USD 完全一致，故直接 (x, y, z) 无重映射。
    # 【原因】标准化后 Pose3D 就是 odom 系(Z 轴朝上)，不再交换 north/height。
    def _usd_to_pose(self, position_usd: np.ndarray, yaw: float) -> Pose3D:
        x, y, z = (float(value) for value in position_usd)
        return Pose3D(position=(x, y, z), yaw=yaw)

    # 【方法】Pose3D (x, y, z) → USD 世界坐标，无重映射。
    def _pose_to_usd(self, position: Vector3) -> np.ndarray:
        return np.array([position[0], position[1], position[2]], dtype=np.float32)

    # 【方法】设置机器人世界位姿(位置 + 由 yaw 生成的四元数)。
    def _set_robot_pose(self, position_usd: np.ndarray, yaw: float | None = None) -> None:
        if yaw is not None:
            self._yaw = yaw
        self._robot.set_world_pose(
            position=position_usd, orientation=_yaw_to_quat_wxyz(self._yaw)
        )

    # 【方法】arm 一个有界 head 扫视窗口；在此之前镜头静止朝正前方。
    # 【原因】默认静止；仅当真正需要(如场景图新增物体)才调用，扫视窗口到期后恢复静止。
    def trigger_head_scan(self, frames: int) -> None:
        """Arm a bounded head look-around; the lens is static until this is called.

        The agent camera holds still and points straight ahead with the body by
        default. A caller invokes this only when a look-around is genuinely needed
        (e.g. a new object just entered the scene graph); until the armed window
        expires the view never drifts.
        """
        self._head_scan_frames = max(0, int(frames))

    # 【方法】仅更新 head(agent) 镜头的局部朝向(平移在构造时已固定，永不再变)。
    # 【坐标系】相机已 parented 到机器人 Xform，父子关系自动带来世界跟随；
    #        本方法只在 base_link 局部系里叠加一个小的 yaw/pitch 扫视扰动。
    # 【算法】① 未 arm 扫视 → 直接用固定的 _camera_base_orientation_wxyz；
    #        ② 已 arm → 扰动 = 绕 base_link +Z 的 yaw 扰动 ⊗ 绕 base_link 右手轴的
    #           pitch 扰动(均由 _axis_angle_quat_wxyz 构造)，叠加在基准朝向之上；
    #        ③ set_local_pose(orientation=...)，不动 translation。
    # 【原因】扰动完全在局部系表达，不再需要每帧读取机器人世界位姿或做 look-at。
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

    # 【方法】放置一个真实物理模拟的目标立方体(用于 ObjectNav 类寻找任务)。
    # 【原因】给寻找任务一个可碰撞、可渲染的真实目标；随后 world.reset() 生效。
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

    # 【方法】重置：帧计数/碰撞/停状态归零，机器人回到初始位姿，world.reset()。
    def reset(self) -> None:
        self._frame_index = 0
        self._collision = False
        self._stopped = True
        self._set_robot_pose(
            self._initial_robot_position_usd, yaw=self._initial_robot_yaw_rad
        )
        self._world.reset()

    # 【方法】返回机器人当前位姿(经坐标重映射)。
    def get_state(self) -> Pose3D:
        position, _ = self._robot.get_world_pose()
        return self._usd_to_pose(position, self._yaw)

    # 【方法】按需更新 head 镜头的局部扫视扰动，再渲染一帧，取 RGB+深度。
    # 【原因】head 镜头已 parented 到机器人，随其世界位姿自动跟随，无需每帧重新
    #        放置；内参固定 f=80、主点在中心；provenance='isaacsim'。
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
            camera_intrinsics=CameraIntrinsics(
                80.0, 80.0, width / 2, height / 2, width, height
            ),
            camera_pose=robot_pose,
            robot_pose=robot_pose,
            provenance=["isaacsim"],
        )
        self._frame_index += 1
        return frame


    # 【方法】速度指令。
    # 【流程】① 速度/角速度超限→急停；② 把局部速度(vx,vy)旋转到世界系(按 yaw 旋转)；
    #        ③ 按 dt 积分得到目标位置；④ _can_move_to 碰撞预检(命中→急停并置碰撞)；
    #        ⑤ 设机器人位姿、渲染一帧。
    # 【原因】速度超限时 fail-closed；碰撞预检防止穿墙。
    def send_velocity_command(self, vx: float, vy: float, wz: float) -> ExecutionFeedback:
        started = time.perf_counter()
        speed = math.hypot(vx, vy)
        if speed > self.max_speed or abs(wz) > self.safety.max_angular_speed:
            self.stop()
            return ExecutionFeedback(
                "velocity", False, self.get_state(), False, 0.0, "velocity limit exceeded"
            )

        position, _ = self._robot.get_world_pose()
        
        # 将局部速度旋转到世界坐标系
        cos_yaw = math.cos(self._yaw)
        sin_yaw = math.sin(self._yaw)
        vx_world = vx * cos_yaw - vy * sin_yaw
        vy_world = vx * sin_yaw + vy * cos_yaw
        
        destination = position + np.array([vx_world * self.dt, vy_world * self.dt, 0.0], dtype=np.float32)
        yaw = self._yaw + wz * self.dt
        
        can_move, message = self._can_move_to(destination, yaw)
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

    # 【方法】标准 6-action 离散指令(turn_left/turn_right/move_forward/stop/look_up/look_down)。
    # 【流程】转向/看向：原地修改 yaw 或 head pitch(不做碰撞预检)；
    #        前进：按当前 yaw 前移 MOVE_STEP_M，复用 _can_move_to 碰撞预检。
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
                action.value, True, self.get_state(), self._collision, time.perf_counter() - started, "executed"
            )

        if action in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT):
            sign = 1.0 if action is DiscreteAction.TURN_LEFT else -1.0
            position, _ = self._robot.get_world_pose()
            yaw = self._yaw + sign * self._turn_step_rad
            self._set_robot_pose(position, yaw=yaw)
            self._stopped = False
            self._world.step(render=False)
            return ExecutionFeedback(
                action.value, True, self.get_state(), self._collision, time.perf_counter() - started, "executed"
            )

        # move_forward
        position, _ = self._robot.get_world_pose()
        destination = position + np.array(
            [
                self._move_step_m * math.cos(self._yaw),
                self._move_step_m * math.sin(self._yaw),
                0.0,
            ],
            dtype=np.float32,
        )
        can_move, message = self._can_move_to(destination, self._yaw)
        if not can_move:
            self.stop()
            self._collision = True
            return ExecutionFeedback(action.value, False, self.get_state(), True, 0.0, message)
        self._set_robot_pose(destination, yaw=self._yaw)
        self._stopped = False
        self._world.step(render=False)
        return ExecutionFeedback(
            action.value, True, self.get_state(), self._collision, time.perf_counter() - started, "executed"
        )

    # 【方法】航点指令。
    # 【流程】① 安全校验(失败→急停)；② 计算到目标 USD 距离(X/Y)；③ 距离 0→已到；
    #        ④ 限速 travel=min(距离, 速度×时长)；⑤ 沿方向移动 travel；
    #        ⑥ 碰撞预检(命中→急停)；⑦ 设位姿、渲染；⑧ 到达判定 travel≈距离。
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

    # 【方法】置 _stopped=True。
    def stop(self) -> None:
        self._stopped = True

    # 【方法】安全器急停 + 自身 stop。
    def emergency_stop(self) -> None:
        self.safety.emergency_stop()
        self.stop()

    # 【方法】返回当前碰撞标志。
    def is_collision(self) -> bool:
        return self._collision

    # 【方法】运行完成后保持渲染循环，使 WebRTC 保持连接(直到 Ctrl-C)。
    # 【原因】流媒体需持续出帧；按 camera_fps 节律渲染。
    def stream_until_interrupted(self) -> None:
        """Keep the render loop alive after a completed run so WebRTC stays connected."""
        try:
            while True:
                self._world.step(render=True)
                time.sleep(1.0 / self._camera_fps)
        except KeyboardInterrupt:
            return

    # 【方法】停止 World 并关闭 SimulationApp，释放资源(会结束进程)。
    def close(self) -> None:
        self._world.stop()
        self._simulation_app.close()
