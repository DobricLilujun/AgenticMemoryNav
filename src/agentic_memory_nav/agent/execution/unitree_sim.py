"""Planar Unitree-like waypoint executor for the CPU MVP."""

from __future__ import annotations

import math
import time

import numpy as np

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


class UnitreeSimExecutor:
    """Kinematic CPU backend for fast pipeline validation."""

    def __init__(self, safety: SafetyController, max_speed: float = 0.5, dt: float = 0.1) -> None:
        self.safety = safety
        self.max_speed = min(max_speed, safety.max_speed)
        self.dt = dt
        self._state = Pose3D()
        self._collision = False
        self._stopped = True
        self._frame_index = 0
        self._manual_pitch_offset_rad = 0.0

    def reset(self) -> None:
        """Reset pose, collision state, and frame counter."""
        self._state = Pose3D()
        self._collision = False
        self._stopped = True
        self._frame_index = 0
        self._manual_pitch_offset_rad = 0.0

    def get_state(self) -> Pose3D:
        return self._state

    def get_observation(self) -> FrameObservation:
        """Return a placeholder RGB-D frame (fixed 2.0 m depth)."""
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

    def send_velocity_command(self, vx: float, vy: float, wz: float) -> ExecutionFeedback:
        """Integrate velocity kinematically."""
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

    def apply_discrete_action(self, action: DiscreteAction | str) -> ExecutionFeedback:
        """Execute one of the standard discrete actions."""
        started = time.perf_counter()
        action = DiscreteAction(action)

        if action is DiscreteAction.STOP:
            self.stop()
            return ExecutionFeedback(
                action.value, True, self._state, self._collision, 0.0, "stopped"
            )

        if action in (DiscreteAction.LOOK_UP, DiscreteAction.LOOK_DOWN):
            sign = 1.0 if action is DiscreteAction.LOOK_UP else -1.0
            self._manual_pitch_offset_rad = max(
                -LOOK_PITCH_LIMIT_RAD,
                min(LOOK_PITCH_LIMIT_RAD, self._manual_pitch_offset_rad + sign * LOOK_STEP_RAD),
            )
            self._stopped = False
            return ExecutionFeedback(
                action.value,
                True,
                self._state,
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
                TURN_BIG_STEP_RAD
                if action in (DiscreteAction.TURN_LEFT_BIG, DiscreteAction.TURN_RIGHT_BIG)
                else TURN_STEP_RAD
            )
            self._state = Pose3D(
                position=self._state.position, yaw=self._state.yaw + sign * step_rad
            )
            self._stopped = False
            return ExecutionFeedback(
                action.value,
                True,
                self._state,
                self._collision,
                time.perf_counter() - started,
                "executed",
            )

        if action in (DiscreteAction.MOVE_FORWARD, DiscreteAction.MOVE_BACKWARD):
            sign = 1.0 if action is DiscreteAction.MOVE_FORWARD else -1.0
            yaw = self._state.yaw
            self._state = Pose3D(
                position=(
                    self._state.position[0] + sign * MOVE_STEP_M * math.cos(yaw),
                    self._state.position[1] + sign * MOVE_STEP_M * math.sin(yaw),
                    self._state.position[2],
                ),
                yaw=yaw,
            )
            self._stopped = False
            return ExecutionFeedback(
                action.value,
                True,
                self._state,
                self._collision,
                time.perf_counter() - started,
                "executed",
            )

        # Unreachable: all valid actions are handled above.
        self.stop()
        return ExecutionFeedback(
            action.value, False, self._state, self._collision, 0.0, "unknown action"
        )

    def send_waypoint(self, waypoint: Vector3, intent: ActionIntent) -> ExecutionFeedback:
        """Move toward a waypoint with safety checks and stop on arrival."""
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

    def stop(self) -> None:
        """Mark the executor as stopped."""
        self._stopped = True

    def emergency_stop(self) -> None:
        """Latch the safety controller and stop."""
        self.safety.emergency_stop()
        self.stop()

    def is_collision(self) -> bool:
        return self._collision
