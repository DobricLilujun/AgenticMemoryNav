"""Standard discrete navigation action set (ObjectNav-style, 9 actions)."""

from __future__ import annotations

import math
from enum import StrEnum


class DiscreteAction(StrEnum):
    TURN_LEFT = "turn_left"  # small turn (default 15°)
    TURN_RIGHT = "turn_right"  # small turn (default 15°)
    TURN_LEFT_BIG = "turn_left_big"  # big turn (default 90°)
    TURN_RIGHT_BIG = "turn_right_big"  # big turn (default 90°)
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    STOP = "stop"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"


# Canonical action IDs, matching the expanded ObjectNav-style action set.
ACTION_BY_ID: dict[int, DiscreteAction] = {
    0: DiscreteAction.TURN_LEFT,
    1: DiscreteAction.TURN_RIGHT,
    2: DiscreteAction.MOVE_FORWARD,
    3: DiscreteAction.STOP,
    4: DiscreteAction.LOOK_UP,
    5: DiscreteAction.LOOK_DOWN,
    6: DiscreteAction.TURN_LEFT_BIG,
    7: DiscreteAction.TURN_RIGHT_BIG,
    8: DiscreteAction.MOVE_BACKWARD,
}
ID_BY_ACTION: dict[DiscreteAction, int] = {action: id_ for id_, action in ACTION_BY_ID.items()}

TURN_STEP_RAD = math.radians(15.0)
TURN_BIG_STEP_RAD = math.radians(90.0)
MOVE_STEP_M = 0.25
LOOK_STEP_RAD = math.radians(30.0)
LOOK_PITCH_LIMIT_RAD = math.radians(60.0)


def parse_discrete_action(value: str) -> DiscreteAction:
    """Resolve a numeric id (e.g. "2") or action name (e.g. "move_forward")."""
    if value.isdigit():
        action = ACTION_BY_ID.get(int(value))
        if action is None:
            raise ValueError(f"Unknown discrete action id: {value!r} (expected 0-8)")
        return action
    try:
        return DiscreteAction(value)
    except ValueError as error:
        raise ValueError(f"Unknown discrete action: {value!r}") from error
