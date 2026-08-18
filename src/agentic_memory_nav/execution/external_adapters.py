"""Explicitly disabled real robot and high-fidelity simulator boundaries."""

# 【模块】显式禁用的真实机器人 / 高保真仿真后端边界。
# 【原因】这些后端默认关闭以保证安全；需平台相关实现时才启用。

from __future__ import annotations


# 【类】被禁用的外部机器人适配器基类。
# 【原因】默认 enabled=False，start() 直接抛错，防止误接入真机。
class DisabledExternalRobotAdapter:
    def __init__(self, backend: str, enabled: bool = False) -> None:
        self.backend = backend
        self.enabled = enabled

    def start(self) -> None:
        if not self.enabled:
            raise RuntimeError(f"{self.backend} integration is disabled by default for safety")
        raise NotImplementedError(
            f"{self.backend} adapter requires platform-specific implementation"
        )


# 【类】ROS2/Unitree 后端占位（继承禁用基类，默认关闭）。
class ROS2Adapter(DisabledExternalRobotAdapter):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__("ROS2/Unitree", enabled)


# 【类】Gazebo 后端占位（继承禁用基类，默认关闭）。
class GazeboAdapter(DisabledExternalRobotAdapter):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__("Gazebo", enabled)


# 【类】Isaac Sim 后端占位（继承禁用基类，默认关闭）。
class IsaacSimAdapter(DisabledExternalRobotAdapter):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__("Isaac Sim", enabled)
