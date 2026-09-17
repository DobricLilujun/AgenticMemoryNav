"""Unit tests for the MCP infinite-canvas simulation engine.

These exercise the engine directly (no MCP layer, no network) — the engine is the
dependency-free core that the MCP tools and HTTP routes both build on.
"""

from __future__ import annotations

import pytest

from agentic_memory_nav.mcp.core.engine import SimulationEngine


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    """Magnitude-aware closeness (``math.isclose`` defaults ``abs_tol=0``)."""
    return abs(a - b) <= tol


@pytest.fixture()
def engine() -> SimulationEngine:
    return SimulationEngine()


# --------------------------------------------------------------------------- #
# moves
# --------------------------------------------------------------------------- #
def test_three_forward_moves(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    for _ in range(3):
        result = engine.execute_action("r", action="move_forward")
    assert _close(result.pose_after["x"], 3.0)
    assert _close(result.pose_after["y"], 0.0)
    assert engine.get_canvas()["trajectory_count"] == 3


def test_forward_then_backward_returns_to_origin(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    engine.execute_action("r", action="move_forward")
    result = engine.execute_action("r", action="move_backward")
    assert _close(result.pose_after["x"], 0.0)
    assert _close(result.pose_after["y"], 0.0)
    assert engine.get_canvas()["trajectory_count"] == 2


def test_turn_left_big_then_forward_moves_along_y(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    engine.execute_action("r", action_id=3)  # turn_left_big == +90 deg
    result = engine.execute_action("r", action="move_forward")
    assert _close(result.pose_after["theta"], 90.0)
    assert _close(result.pose_after["x"], 0.0)
    assert _close(result.pose_after["y"], 1.0)
    assert engine.get_canvas()["trajectory_count"] == 1  # rotation adds no segment


# --------------------------------------------------------------------------- #
# rotation and no-op actions
# --------------------------------------------------------------------------- #
def test_rotation_changes_heading_only(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    engine.execute_action("r", action="move_forward")  # to (1.0, 0)
    result = engine.execute_action("r", action="turn_left")
    assert not result.trajectory_added
    assert _close(result.pose_after["x"], 1.0)  # position unchanged
    assert _close(result.pose_after["theta"], 15.0)


@pytest.mark.parametrize("action", ["stop", "look_up", "look_down"])
def test_noop_actions_change_nothing(engine: SimulationEngine, action: str) -> None:
    engine.create_robot("r")
    engine.execute_action("r", action="move_forward")  # to (1.0, 0)
    result = engine.execute_action("r", action=action)
    assert not result.trajectory_added
    assert _close(result.pose_after["x"], 1.0)
    assert _close(result.pose_after["y"], 0.0)
    assert result.pose_after["theta"] == 0.0
    assert result.success  # a no-op still succeeds


# --------------------------------------------------------------------------- #
# obstacles and collision
# --------------------------------------------------------------------------- #
def test_add_point_obstacle_and_block(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    engine.add_obstacle("r", {"obstacle_type": "point", "x": 0.2, "y": 0.0, "radius": 0.08})
    result = engine.execute_action("r", action="move_forward")
    assert not result.success  # blocked by the obstacle at x=0.2
    assert _close(result.pose_after["x"], 0.0)  # stays put
    assert result.trajectory_added  # a blocked move still records a (red) segment


def test_add_line_obstacle_records_geometry(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    record = engine.add_obstacle("r", {"obstacle_type": "line", "start": [-0.3, -0.3],
                                        "end": [0.3, 0.3], "width": 0.05})
    assert _close(record["width"], 0.05)
    assert record["geometry"] == [[-0.3, -0.3], [0.3, 0.3]]


def test_update_and_remove_obstacle(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    record = engine.add_obstacle("r", {"obstacle_type": "point", "x": 0.5, "radius": 0.1})
    oid = record["obstacle_id"]
    engine.update_obstacle(oid, {"radius": 0.3})
    canvas = engine.get_canvas()
    assert _close(next(o["radius"] for o in canvas["obstacles"] if o["obstacle_id"] == oid), 0.3)
    engine.remove_obstacle(oid)
    assert all(o["obstacle_id"] != oid for o in engine.get_canvas()["obstacles"])


# --------------------------------------------------------------------------- #
# events, version, reset
# --------------------------------------------------------------------------- #
def test_actions_bump_canvas_version_and_emit_events(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    v0 = engine.version
    engine.execute_action("r", action="move_forward")
    engine.execute_action("r", action="turn_left")
    assert engine.version > v0
    events = engine.get_canvas()["events"]
    assert "trajectory_added" in {e["event"] for e in events}
    assert "robot_updated" in {e["event"] for e in events}


def test_get_canvas_marks_unit_length(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    canvas = engine.get_canvas()
    assert canvas["coordinate_unit"] == "unit_length"
    assert canvas["robots"][0]["position"] == [0.0, 0.0]
    assert canvas["robots"][0]["theta"] == 0.0


def test_reset_restores_pose_and_clears_state(engine: SimulationEngine) -> None:
    engine.create_robot("r")
    engine.execute_action("r", action="move_forward")
    engine.add_obstacle("r", {"obstacle_type": "point", "x": 0.5, "radius": 0.1})
    engine.reset_simulation("r")
    canvas = engine.get_canvas()
    assert canvas["robots"][0]["position"] == [0.0, 0.0]
    assert canvas["trajectory_count"] == 0
    assert canvas["obstacles"] == []


def test_execute_action_auto_creates_robot(engine: SimulationEngine) -> None:
    # no explicit create; the robot is created on first action
    result = engine.execute_action("robot_x", action="move_forward")
    assert result.success
    assert "robot_x" in engine.robots