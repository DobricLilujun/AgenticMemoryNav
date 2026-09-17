"""Action catalogue and state-transition definitions for the fake robot.

The canonical action mapping from the specification is:

    ACTION_BY_ID = {
        0: "turn_left",
        1: "turn_right",
        2: "move_forward",
        3: "turn_left_big",
        4: "turn_right_big",
        5: "move_backward",
    }

The specification's full-action evaluation (scene 4) additionally exercises
``stop``, ``look_up`` and ``look_down``; those are appended as ids 6-8 so the
name/id mapping stays complete and unambiguous while keeping the original
0-5 ordering intact.
"""

from __future__ import annotations

from enum import Enum

# The canonical id -> name mapping required by the specification.
ACTION_BY_ID: dict[int, str] = {
    0: "turn_left",
    1: "turn_right",
    2: "move_forward",
    3: "turn_left_big",
    4: "turn_right_big",
    5: "move_backward",
    # Extended actions exercised by scene 4 (all actions) of the evaluation.
    6: "stop",
    7: "look_up",
    8: "look_down",
}

ACTION_ID_BY_NAME: dict[str, int] = {name: action_id for action_id, name in ACTION_BY_ID.items()}

# The default action parameters from the specification.
MOVE_STEP = 1.0  # unit lengths per move
SMALL_TURN_DEGREES = 15.0
BIG_TURN_DEGREES = 90.0


class ActionKind(str, Enum):
    """High-level classification used to decide whether an action moves the robot."""

    ROTATE = "rotate"
    MOVE = "move"
    NOOP = "noop"


# Which actions change 2D position (they generate a trajectory segment), which only
# rotate the robot, and which change neither (they only generate an event).
ACTION_KIND: dict[str, ActionKind] = {
    "turn_left": ActionKind.ROTATE,
    "turn_right": ActionKind.ROTATE,
    "turn_left_big": ActionKind.ROTATE,
    "turn_right_big": ActionKind.ROTATE,
    "move_forward": ActionKind.MOVE,
    "move_backward": ActionKind.MOVE,
    "stop": ActionKind.NOOP,
    "look_up": ActionKind.NOOP,
    "look_down": ActionKind.NOOP,
}

# Rotation delta (degrees, counter-clockwise positive) applied per action.
TURN_DELTA_DEGREES: dict[str, float] = {
    "turn_left": SMALL_TURN_DEGREES,
    "turn_right": -SMALL_TURN_DEGREES,
    "turn_left_big": BIG_TURN_DEGREES,
    "turn_right_big": -BIG_TURN_DEGREES,
}

# Forward sign of a move along the heading (+ forward, - backward).
MOVE_SIGN: dict[str, float] = {
    "move_forward": 1.0,
    "move_backward": -1.0,
}


def resolve_action(action: str | None = None, action_id: int | None = None) -> str:
    """Resolve an action from either its name or its id.

    Exactly one of ``action`` / ``action_id`` must be provided. Raises ``ValueError``
    for an unknown action name or id, or when neither / both are supplied.
    """

    if (action is None) == (action_id is None):
        raise ValueError("Provide exactly one of 'action' or 'action_id'.")
    if action is not None:
        if action not in ACTION_ID_BY_NAME:
            raise ValueError(f"Unknown action name: {action!r}")
        return action
    assert action_id is not None
    name = ACTION_BY_ID.get(action_id)
    if name is None:
        raise ValueError(f"Unknown action id: {action_id}")
    return name


def kind(name: str) -> ActionKind:
    """Return the :class:`ActionKind` of an action name."""

    return ACTION_KIND[name]


def is_move(name: str) -> bool:
    return ACTION_KIND[name] is ActionKind.MOVE


def is_rotate(name: str) -> bool:
    return ACTION_KIND[name] is ActionKind.ROTATE


def is_noop(name: str) -> bool:
    return ACTION_KIND[name] is ActionKind.NOOP


def resolve_action_ids() -> dict[int, str]:
    """Return a copy of the canonical id -> name mapping."""

    return dict(ACTION_BY_ID)