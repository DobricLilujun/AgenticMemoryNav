"""Robot backend protocol."""

# 【模块】机器人执行后端协议（RobotBackend）。
# 【作用】定义所有执行后端（Unitree / Habitat / Isaac Sim 等）必须实现的最小接口，
#         使上层导航流水线与具体仿真/真机解耦（依赖倒置）。

from __future__ import annotations

from typing import Protocol

from agentic_memory_nav.common.types import (
    ActionIntent,
    ExecutionFeedback,
    FrameObservation,
    Pose3D,
    Vector3,
)


# 【类 RobotBackend】机器人后端协议（Protocol）。
# 【原因】上层 pipeline 只依赖本接口，不关心底层是仿真还是真机。
# 【方法】reset 重置；get_state 取位姿；get_observation 取一帧观察；
#        send_velocity_command 速度指令；send_waypoint 航点指令；
#        stop/ emergency_stop 停/急停；is_collision 是否碰撞。
class RobotBackend(Protocol):
    def reset(self) -> None: ...
    def get_state(self) -> Pose3D: ...
    def get_observation(self) -> FrameObservation: ...
    def send_velocity_command(self, vx: float, vy: float, wz: float) -> ExecutionFeedback: ...
    def send_waypoint(self, waypoint: Vector3, intent: ActionIntent) -> ExecutionFeedback: ...
    def stop(self) -> None: ...
    def emergency_stop(self) -> None: ...
    def is_collision(self) -> bool: ...
