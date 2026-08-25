import math

import pytest

from agentic_memory_nav.agent.execution.discrete_actions import (
    ACTION_BY_ID,
    LOOK_PITCH_LIMIT_RAD,
    LOOK_STEP_RAD,
    MOVE_STEP_M,
    TURN_STEP_RAD,
    DiscreteAction,
    parse_discrete_action,
)
from agentic_memory_nav.agent.execution.safety_controller import SafetyController
from agentic_memory_nav.agent.execution.unitree_sim import UnitreeSimExecutor


def _executor() -> UnitreeSimExecutor:
    return UnitreeSimExecutor(SafetyController())


def test_action_by_id_matches_spec_order():
    assert [ACTION_BY_ID[i] for i in range(6)] == [
        DiscreteAction.TURN_LEFT,
        DiscreteAction.TURN_RIGHT,
        DiscreteAction.MOVE_FORWARD,
        DiscreteAction.STOP,
        DiscreteAction.LOOK_UP,
        DiscreteAction.LOOK_DOWN,
    ]


def test_parse_discrete_action_accepts_id_or_name():
    assert parse_discrete_action("2") is DiscreteAction.MOVE_FORWARD
    assert parse_discrete_action("move_forward") is DiscreteAction.MOVE_FORWARD
    with pytest.raises(ValueError):
        parse_discrete_action("6")
    with pytest.raises(ValueError):
        parse_discrete_action("nonsense")


def test_turn_left_and_right_change_yaw_by_15_degrees():
    executor = _executor()
    executor.apply_discrete_action(DiscreteAction.TURN_LEFT)
    assert executor.get_state().yaw == pytest.approx(TURN_STEP_RAD)
    executor.apply_discrete_action(DiscreteAction.TURN_RIGHT)
    assert executor.get_state().yaw == pytest.approx(0.0)


def test_move_forward_advances_along_current_yaw():
    executor = _executor()
    executor.apply_discrete_action(DiscreteAction.TURN_LEFT)
    yaw = executor.get_state().yaw
    executor.apply_discrete_action(DiscreteAction.MOVE_FORWARD)
    position = executor.get_state().position
    assert position[0] == pytest.approx(MOVE_STEP_M * math.cos(yaw))
    assert position[1] == pytest.approx(MOVE_STEP_M * math.sin(yaw))


def test_stop_halts_and_reports_success():
    executor = _executor()
    feedback = executor.apply_discrete_action(DiscreteAction.STOP)
    assert feedback.success
    assert executor._stopped is True


def test_look_up_and_down_clamp_to_pitch_limit():
    executor = _executor()
    steps = int(LOOK_PITCH_LIMIT_RAD / LOOK_STEP_RAD) + 3
    for _ in range(steps):
        executor.apply_discrete_action(DiscreteAction.LOOK_UP)
    assert executor._manual_pitch_offset_rad == pytest.approx(LOOK_PITCH_LIMIT_RAD)
    for _ in range(2 * steps):
        executor.apply_discrete_action(DiscreteAction.LOOK_DOWN)
    assert executor._manual_pitch_offset_rad == pytest.approx(-LOOK_PITCH_LIMIT_RAD)
