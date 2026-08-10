"""Fail-closed action validation."""

from __future__ import annotations

import math

from agentic_memory_nav.common.types import ActionIntent, ActionType, Pose3D


class SafetyError(RuntimeError):
    pass


class SafetyController:
    def __init__(
        self, max_speed: float = 0.5, max_angular_speed: float = 1.0, max_timeout: float = 15.0
    ) -> None:
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.max_timeout = max_timeout
        self.emergency_latched = False

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

    def emergency_stop(self) -> None:
        self.emergency_latched = True
