"""Planar Unitree-like waypoint executor for the CPU MVP."""

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


class UnitreeSimExecutor:
    def __init__(self, safety: SafetyController, max_speed: float = 0.5, dt: float = 0.1) -> None:
        self.safety = safety
        self.max_speed = min(max_speed, safety.max_speed)
        self.dt = dt
        self._state = Pose3D()
        self._collision = False
        self._stopped = True
        self._frame_index = 0

    def reset(self) -> None:
        self._state = Pose3D()
        self._collision = False
        self._stopped = True
        self._frame_index = 0

    def get_state(self) -> Pose3D:
        return self._state

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

    def stop(self) -> None:
        self._stopped = True

    def emergency_stop(self) -> None:
        self.safety.emergency_stop()
        self.stop()

    def is_collision(self) -> bool:
        return self._collision
