"""Explicitly disabled real robot and high-fidelity simulator boundaries."""

from __future__ import annotations


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


class ROS2Adapter(DisabledExternalRobotAdapter):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__("ROS2/Unitree", enabled)


class GazeboAdapter(DisabledExternalRobotAdapter):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__("Gazebo", enabled)


class IsaacSimAdapter(DisabledExternalRobotAdapter):
    def __init__(self, enabled: bool = False) -> None:
        super().__init__("Isaac Sim", enabled)
