"""Optional Isaac Sim ObjectNav executor boundary (InteriorAgent-style scenes).

Independent reimplementation of the InteriorAgent `experiments.json` workflow
described by github.com/learnsyslab/isaac-objnav-semistatic-eval; no code from
that repository is used (see docs/dependency-decisions.md for the license check
that led to this decision). Not yet runtime-verified: it requires the
InteriorAgent dataset, which was not available in this environment.
"""

# 【模块】可选的 Isaac Sim ObjectNav 执行器边界（InteriorAgent 风格场景）。
# 【作用】加载 InteriorAgent 风格 USD 场景、按实验移除/排除资产、暴露目标资产位置，
#         用于物体导航(寻找)任务；完全独立复现(未用该仓库代码，见 docs/dependency-decisions.md)。
# 【注意】尚未在运行时验证：需要 InteriorAgent 数据集(本环境不可用)。

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
from agentic_memory_nav.agent.datasets.objectnav import ObjectNavExperiment
from agentic_memory_nav.agent.execution.isaacsim_adapter import (
    _FORWARD_CAMERA_ORIENTATION_WXYZ,
    _ensure_simulation_app,
)
from agentic_memory_nav.agent.execution.safety_controller import SafetyController, SafetyError


# 【类】ObjectNav 执行器（RobotBackend 实现）。
# 【原因】加载 InteriorAgent 场景 USD，按 experiment 移除/排除资产，
#        暴露目标(goal)位置与到最近目标的距离；坐标同 Isaac Sim 适配器。
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
            orientation=_FORWARD_CAMERA_ORIENTATION_WXYZ,
            resolution=(width, height),
        )
        self._world.reset()
        self._camera.initialize()
        self._camera.add_distance_to_image_plane_to_frame()
        for _ in range(3):
            self._world.step(render=True)

    # 【方法】按子串匹配移除场景中的资产(排除 exclude 子串)。
    # 【原因】InteriorAgent 实验要求移除某些干扰物(如可移动家具)。
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

    # 【方法】返回匹配 experiment.goal.asset 的子 prim 的世界位置字典。
    # 【原因】寻找任务需要知道目标物体在哪；坐标经 _usd_to_pose 重映射。
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

    # 【方法】到最近目标的欧氏距离(X/Z 平面)。
    # 【注意】这是下界近似，非测地/占用感知距离；要从任意 InteriorAgent
    #        场景可靠推导地板占用栅格需标定，待数据集可用后再完善。
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

    # 【方法】USD 坐标 → Pose3D：USD(x,y,z_up) → (x, z_up, y)。
    def _usd_to_pose(self, position_usd: np.ndarray, yaw: float) -> Pose3D:
        x, y_north, z_up = (float(value) for value in position_usd)
        return Pose3D(position=(x, z_up, y_north), yaw=yaw)

    # 【方法】Pose3D(x, z_up, y_north) → USD(x, y_north, z_up)。
    def _pose_to_usd(self, position: Vector3) -> np.ndarray:
        x, z_up, y_north = position
        return np.array([x, y_north, z_up], dtype=np.float32)

    # 【方法】重置：机器人回到实验初始位姿(加 0.15m 高度)，清状态。
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

    # 【方法】返回机器人位姿(经坐标重映射)。
    def get_state(self) -> Pose3D:
        position, _ = self._robot.get_world_pose()
        return self._usd_to_pose(position, self._yaw)

    # 【方法】渲染一帧并取 RGB+深度；内参固定 f=80、主点在中心。
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

    # 【方法】速度指令（运动学积分）：超限急停；否则按 dt 积分位置与偏航。
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

    # 【方法】航点指令：安全校验 → 距离 → 限速 travel → 沿方向移动 → 偏航指向目标 → 到达判定。
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

    # 【方法】停止 World 并关闭 SimulationApp，释放资源。
    def close(self) -> None:
        self._world.stop()
        self._simulation_app.close()
