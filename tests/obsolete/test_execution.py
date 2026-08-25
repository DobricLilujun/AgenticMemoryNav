import numpy as np

from agentic_memory_nav.common.types import ActionIntent, ActionType, new_id
from agentic_memory_nav.agent.execution.isaacsim_adapter import _euler_xyz_deg_to_quat_wxyz
from agentic_memory_nav.agent.execution.safety_controller import SafetyController
from agentic_memory_nav.agent.execution.unitree_sim import UnitreeSimExecutor


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


def test_camera_orient_accepts_xyz_euler_degrees():
    orientation = _euler_xyz_deg_to_quat_wxyz(0.0, 90.0, 0.0)

    np.testing.assert_allclose(
        orientation,
        np.array([0.70710677, 0.0, 0.70710677, 0.0]),
        atol=1e-6,
    )


def test_isaacsim_executor_fails_closed_when_unavailable():
    import importlib.util

    if importlib.util.find_spec("isaacsim") is not None:
        import pytest

        pytest.skip("isaacsim is importable in this environment; fail-closed path not exercised")

    from agentic_memory_nav.agent.execution.isaacsim_adapter import IsaacSimExecutor

    try:
        IsaacSimExecutor(scene=None, safety=SafetyController())
    except RuntimeError as error:
        assert "isaacsim is not importable" in str(error)
    else:
        raise AssertionError("expected RuntimeError when isaacsim is unavailable")
