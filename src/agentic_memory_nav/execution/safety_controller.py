"""Fail-closed action validation."""

# 【模块】fail-closed（失败即安全）动作校验。
# 【作用】在执行任何移动动作前，对意图/状态做安全门禁；任一条件不满足即抛错，
#         保证机器人不会在异常状态下执行危险动作。

from __future__ import annotations

import math

from agentic_memory_nav.common.types import ActionIntent, ActionType, Pose3D


# 【类】安全异常。任何校验失败都抛此异常，使执行层能统一捕获并急停。
class SafetyError(RuntimeError):
    pass


# 【类】安全控制器。
# 【原因】集中管理速度/角速度/超时上限与急停锁存状态。
# 【字段】max_speed 最大线速度；max_angular_speed 最大角速度；
#        max_timeout 最大动作时长；emergency_latched 急停是否被锁存（触发后需手动复位）。
class SafetyController:
    def __init__(
        self, max_speed: float = 0.5, max_angular_speed: float = 1.0, max_timeout: float = 15.0
    ) -> None:
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.max_timeout = max_timeout
        self.emergency_latched = False

    # 【方法】校验动作意图（在移动前调用）。
    # 【原因】fail-closed：任一条件不满足即抛 SafetyError。
    # 【检查】① 急停已锁存 → 抛错；② 处于碰撞中 → 抛错；
    #        ③ 动作时长越界 → 抛错；④ 移动类动作缺航点 → 抛错；
    #        ⑤ 航点坐标非有限数(NaN/Inf) → 抛错；
    #        ⑥ EXPLORE 动作缺少 reduced_exploration_speed 约束 → 抛错。
    def validate(self, intent: ActionIntent, state: Pose3D, collision: bool = False) -> None:
        if self.emergency_latched:
            raise SafetyError("Emergency stop is latched")
        if collision:
            raise SafetyError("Collision is active")
        if intent.duration <= 0 or intent.duration > self.max_timeout:
            raise SafetyError("Action timeout is outside configured limits")
        if intent.action_type != ActionType.STOP and intent.waypoint is None:
            raise SafetyError("Moving action requires a waypoint")
        if intent.waypoint and not all(math.isfinite(value) for value in intent.waypoint):
            raise SafetyError("Waypoint contains non-finite coordinates")
        if (
            intent.action_type == ActionType.EXPLORE
            and "reduced_exploration_speed" not in intent.safety_constraints
        ):
            raise SafetyError("Exploration action lacks reduced speed constraint")

    # 【方法】急停：锁存 emergency_latched=True，此后 validate 一律拒绝执行。
    # 【原因】一旦急停，必须显式复位才能再动，防止误恢复。
    def emergency_stop(self) -> None:
        self.emergency_latched = True
