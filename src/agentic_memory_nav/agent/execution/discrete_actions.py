"""Standard discrete navigation action set (Habitat-style, 6 actions)."""

from __future__ import annotations

import math
from enum import StrEnum


class DiscreteAction(StrEnum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE_FORWARD = "move_forward"
    STOP = "stop"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"


# Canonical action IDs, matching the standard PointNav/ObjectNav-style action set.
ACTION_BY_ID: dict[int, DiscreteAction] = {
    0: DiscreteAction.TURN_LEFT,
    1: DiscreteAction.TURN_RIGHT,
    2: DiscreteAction.MOVE_FORWARD,
    3: DiscreteAction.STOP,
    4: DiscreteAction.LOOK_UP,
    5: DiscreteAction.LOOK_DOWN,
}
ID_BY_ACTION: dict[DiscreteAction, int] = {action: id_ for id_, action in ACTION_BY_ID.items()}

TURN_STEP_RAD = math.radians(15.0)
MOVE_STEP_M = 0.25
LOOK_STEP_RAD = math.radians(30.0)
LOOK_PITCH_LIMIT_RAD = math.radians(60.0)


def parse_discrete_action(value: str) -> DiscreteAction:
    """Resolve a numeric id (e.g. "2") or action name (e.g. "move_forward")."""
    if value.isdigit():
        action = ACTION_BY_ID.get(int(value))
        if action is None:
            raise ValueError(f"Unknown discrete action id: {value!r} (expected 0-5)")
        return action
    try:
        return DiscreteAction(value)
    except ValueError as error:
        raise ValueError(f"Unknown discrete action: {value!r}") from error
