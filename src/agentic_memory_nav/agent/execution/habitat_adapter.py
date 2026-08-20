"""Optional Habitat-Sim executor boundary."""

# 【模块】可选的 Habitat-Sim 执行器边界。
# 【作用】提供 RobotBackend 兼容的 Habitat-Sim 包装，用于 MVP 阶段接入真实室内仿真；
#         依赖惰性导入，纯 mock 流程不会要求安装 Habitat。

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
from agentic_memory_nav.agent.execution.safety_controller import SafetyController, SafetyError


# 【类】Habitat 可用性边界（镜像 IsaacSimAdapter）。
# 【原因】available 检测 habitat_sim 是否可导入；未安装则 start() 抛错，
#        提示改用 execution.backend=unitree_sim。
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


# 【类】Habitat-Sim 执行器（RobotBackend 实现）。
# 【原因】薄封装 Habitat，提供 RGB+深度相机、运动学积分与航点导航；
#        坐标约定：Pose3D 第 2 个分量是高度(对应 Habitat 的 z)。
# 【状态】_sim/_agent 仿真与智能体；_yaw 偏航；_collision/_stopped 状态标志。
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

    # 【方法】构建 Habitat 仿真器：RGB+深度两个 64×96 相机(位于 z=1.5m)。
    # 【原因】MVP 用低分辨率相机以节省算力，位置 1.5m 模拟头相机高度。
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

    # 【方法】把智能体位置设到给定坐标（写回 Habitat 状态）。
    def _set_agent_position(self, position: Vector3) -> None:
        state = self._agent.get_state()
        state.position = np.asarray(position, dtype=np.float32)
        self._agent.set_state(state)

    # 【方法】从智能体状态读出位姿，重映射为 Pose3D(x, height, north)。
    def _state_from_agent(self) -> Pose3D:
        state = self._agent.get_state()
        x, y, z = [float(value) for value in state.position]
        return Pose3D(position=(x, y, z), yaw=self._yaw)

    # 【方法】重置：位置归零、清碰撞/停状态、偏航归零、帧计数归零。
    def reset(self) -> None:
        self._frame_index = 0
        self._collision = False
        self._stopped = True
        self._yaw = 0.0
        self._set_agent_position((0.0, 0.0, 0.0))

    # 【方法】返回当前位姿。
    def get_state(self) -> Pose3D:
        return self._state_from_agent()

    # 【方法】取一帧观察：RGB 取 color_sensor 的 RGB 通道，深度取 depth_sensor；
    #        内参固定 f=80、主点在画面中心。
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

    # 【方法】速度指令（运动学积分）：速度/角速度超限时急停；否则按 dt 积分位置与偏航。
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

    # 【方法】航点指令：安全校验 → 计算 X/Z 平面距离 → 限速 travel → 沿方向移动 → 偏航指向目标 → 到达判定。
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

    # 【方法】关闭 Habitat 仿真器，释放资源。
    def close(self) -> None:
        self._sim.close()
