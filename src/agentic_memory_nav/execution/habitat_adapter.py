"""Optional Habitat-Sim executor boundary."""

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


class HabitatAdapter:
    def __init__(self, scene: str) -> None:
        self.scene = scene
        self.available = importlib.util.find_spec("habitat_sim") is not None

    def start(self) -> None:
        if not self.available:
            raise RuntimeError("Habitat-Sim is not installed; select execution.backend=unitree_sim")
        raise NotImplementedError(
            "Habitat runtime requires a validated scene and matched Habitat installation"
        )


class HabitatSimExecutor:
    """Thin RobotBackend-compatible wrapper over habitat_sim for MVP integration."""

    def __init__(
        self,
        scene: str,
        safety: SafetyController,
        max_speed: float = 0.5,
        dt: float = 0.1,
    ) -> None:
        if importlib.util.find_spec("habitat_sim") is None:
            raise RuntimeError("habitat_sim is not importable in this Python environment")
        if not Path(scene).expanduser().exists():
            raise FileNotFoundError(f"Habitat scene not found: {scene}")

        # Import lazily so mock-only workflows never require Habitat dependencies.
        import habitat_sim  # type: ignore[import-not-found]

        self.scene = str(Path(scene).expanduser().resolve())
        self.safety = safety
        self.max_speed = min(max_speed, safety.max_speed)
        self.dt = dt
        self._frame_index = 0
        self._collision = False
        self._stopped = True
        self._yaw = 0.0

        self._habitat_sim = habitat_sim
        self._sim = self._build_simulator(habitat_sim)
        self._agent = self._sim.initialize_agent(0)
        self._set_agent_position((0.0, 0.0, 0.0))

    def _build_simulator(self, habitat_sim: Any) -> Any:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = self.scene

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color_sensor"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [64, 96]
        rgb_spec.position = [0.0, 1.5, 0.0]

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [64, 96]
        depth_spec.position = [0.0, 1.5, 0.0]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        return habitat_sim.Simulator(cfg)

    def _set_agent_position(self, position: Vector3) -> None:
        state = self._agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        self._agent.set_state(state)

    def _state_from_agent(self) -> Pose3D:
        state = self._agent.get_state()
        x, y, z = [float(value) for value in state.position]
        return Pose3D(position=(x, y, z), yaw=self._yaw)

    def reset(self) -> None:
        self._frame_index = 0
        self._collision = False
        self._stopped = True
        self._yaw = 0.0
        self._set_agent_position((0.0, 0.0, 0.0))

    def get_state(self) -> Pose3D:
        return self._state_from_agent()

    def get_observation(self) -> FrameObservation:
        observations = self._sim.get_sensor_observations()
        rgb = np.asarray(observations["color_sensor"])[:, :, :3].astype(np.uint8)
        depth = np.asarray(observations["depth_sensor"]).astype(np.float32)
        height, width = rgb.shape[:2]

        frame = FrameObservation(
            frame_id=f"frame_{self._frame_index:04d}",
            timestamp=float(self._frame_index),
            rgb=rgb,
            depth=depth,
            camera_intrinsics=CameraIntrinsics(80.0, 80.0, width / 2, height / 2, width, height),
            camera_pose=self.get_state(),
            robot_pose=self.get_state(),
            provenance=["habitat_sim"],
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

        state = self._agent.get_state()
        position = np.asarray(state.position, dtype=np.float32)
        position += np.asarray([vx * self.dt, vy * self.dt, 0.0], dtype=np.float32)
        state.position = position
        self._agent.set_state(state)

        self._yaw += wz * self.dt
        self._stopped = False
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

        current = np.asarray(self.get_state().position, dtype=np.float32)
        target = np.asarray(waypoint, dtype=np.float32)
        delta = target - current
        distance = float(np.linalg.norm(delta[[0, 2]]))
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
        destination = current + direction * travel
        self._set_agent_position(
            (float(destination[0]), float(destination[1]), float(destination[2]))
        )
        self._yaw = math.atan2(float(delta[0]), float(delta[2]))
        self._stopped = False
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
        self._sim.close()
