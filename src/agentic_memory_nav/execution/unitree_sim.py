"""Planar Unitree-like waypoint executor for the CPU MVP."""

# 【模块】面向 CPU MVP 的平面(Unitree 类)航点执行器。
# 【作用】纯 CPU、无需 GPU/Isaac 的轻量后端，用于快速验证导航流水线；
#         运动学积分(非物理)，深度固定 2.0m，RGB 为占位画面。

from __future__ import annotations

import math
import time

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


# 【类】平面 Unitree 执行器（RobotBackend 实现）。
# 【原因】MVP 阶段无需高保真物理，用简单运动学积分即可跑通端到端。
# 【状态】_state 位姿；_collision 是否碰撞；_stopped 是否已停；
#        _frame_index 帧计数；max_speed/dt 由安全器与配置给定。
class UnitreeSimExecutor:
    def __init__(self, safety: SafetyController, max_speed: float = 0.5, dt: float = 0.1) -> None:
        self.safety = safety
        self.max_speed = min(max_speed, safety.max_speed)
        self.dt = dt
        self._state = Pose3D()
        self._collision = False
        self._stopped = True
        self._frame_index = 0

    # 【方法】重置：位姿归零、清碰撞/停状态、帧计数归零。
    def reset(self) -> None:
        self._state = Pose3D()
        self._collision = False
        self._stopped = True
        self._frame_index = 0

    # 【方法】返回当前位姿 Pose3D。
    def get_state(self) -> Pose3D:
        return self._state

    # 【方法】生成一帧占位观察：RGB 为 64×96 灰底(绿通道=40)，
    #        第 2 帧起在中心画红块(220,20,20)便于目视定位；深度固定 2.0m。
    # 【原因】MVP 无真实渲染，用合成画面占位以打通观察接口。
    def get_observation(self) -> FrameObservation:
        height, width = 64, 96
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 1] = 40
        if self._frame_index >= 1:
            rgb[24:48, 42:58] = (220, 20, 20)
        frame = FrameObservation(
            frame_id=f"frame_{self._frame_index:04d}",
            timestamp=float(self._frame_index),
            rgb=rgb,
            depth=np.full((height, width), 2.0, dtype=np.float32),
            camera_intrinsics=CameraIntrinsics(80.0, 80.0, width / 2, height / 2, width, height),
            camera_pose=self._state,
            robot_pose=self._state,
            provenance=["unitree_sim"],
        )
        self._frame_index += 1
        return frame

    # 【方法】速度指令（运动学积分）。
    # 【原因】速度/角速度超限时急停并返回失败；否则按 vx,vy 积分位置、
    #        按 wz 积分偏航，时间步长 dt。
    def send_velocity_command(self, vx: float, vy: float, wz: float) -> ExecutionFeedback:
        started = time.perf_counter()
        speed = math.hypot(vx, vy)
        if speed > self.max_speed or abs(wz) > self.safety.max_angular_speed:
            self.stop()
            return ExecutionFeedback(
                "velocity", False, self._state, False, 0.0, "velocity limit exceeded"
            )
        self._stopped = False
        self._state = Pose3D(
            position=(
                self._state.position[0] + vx * self.dt,
                self._state.position[1] + vy * self.dt,
                self._state.position[2],
            ),
            yaw=self._state.yaw + wz * self.dt,
        )
        return ExecutionFeedback(
            "velocity", True, self._state, False, time.perf_counter() - started, "executed"
        )

    # 【方法】航点指令。
    # 【流程】① 安全校验(失败→急停)；② 计算到目标距离(X/Z 平面)；
    #        ③ 距离为 0 → 已到；④ 限速 travel=min(距离, 速度×时长)；
    #        ⑤ 沿方向移动 travel，偏航指向目标；⑥ 到达判定 travel≈距离。
    def send_waypoint(self, waypoint: Vector3, intent: ActionIntent) -> ExecutionFeedback:
        started = time.perf_counter()
        try:
            self.safety.validate(intent, self._state, self._collision)
        except SafetyError as error:
            self.stop()
            return ExecutionFeedback(
                intent.action_id, False, self._state, self._collision, 0.0, str(error)
            )
        self._stopped = False
        delta = np.asarray(waypoint) - np.asarray(self._state.position)
        distance = float(np.linalg.norm(delta[[0, 2]]))
        if distance == 0:
            self.stop()
            return ExecutionFeedback(
                intent.action_id, True, self._state, False, 0.0, "already at waypoint"
            )
        travel = min(distance, self.max_speed * intent.duration)
        direction = delta / max(distance, 1e-9)
        destination = np.asarray(self._state.position) + direction * travel
        yaw = math.atan2(float(delta[0]), float(delta[2]))
        self._state = Pose3D(tuple(destination.tolist()), yaw)  # type: ignore[arg-type]
        reached = travel >= distance - 1e-6
        self.stop()
        return ExecutionFeedback(
            intent.action_id,
            reached,
            self._state,
            self._collision,
            time.perf_counter() - started,
            "waypoint reached" if reached else "action timeout before waypoint",
        )

    # 【方法】置 _stopped=True（本实现无物理，仅标记状态）。
    def stop(self) -> None:
        self._stopped = True

    # 【方法】调用安全器急停 + 自身 stop。
    def emergency_stop(self) -> None:
        self.safety.emergency_stop()
        self.stop()

    # 【方法】返回当前碰撞标志。
    def is_collision(self) -> bool:
        return self._collision
