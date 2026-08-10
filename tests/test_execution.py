from agentic_memory_nav.common.types import ActionIntent, ActionType, new_id
from agentic_memory_nav.execution.safety_controller import SafetyController
from agentic_memory_nav.execution.unitree_sim import UnitreeSimExecutor


def test_waypoint_execution_and_emergency_stop():
    safety = SafetyController()
    executor = UnitreeSimExecutor(safety)
    intent = ActionIntent(
        new_id("action"),
        ActionType.NAVIGATE,
        "cup",
        (1.0, 0.0, 0.0),
        4.0,
        ["collision_free"],
        0.9,
        "test",
    )
    assert executor.send_waypoint(intent.waypoint, intent).success
    executor.emergency_stop()
    assert not executor.send_waypoint(intent.waypoint, intent).success
