"""Automated evaluation scenarios for the MCP infinite-canvas system.

Each scenario drives the robot *through the MCP tools* via the fake robot (an
independent MCP client) and asserts the resulting state. The fake-robot methods
return the plain dict a tool produced, so every assertion reads them with
bracket access. ``trajectory_count`` and ``events`` live on the canvas and are
read with ``get_canvas``.
"""

from __future__ import annotations

from typing import Any, cast

TOL = 1e-9


# --------------------------------------------------------------------------- #
# assertion helpers
# --------------------------------------------------------------------------- #
class Check:
    """A single named assertion with a pass/fail result."""

    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = bool(passed)
        self.detail = detail

    @property
    def ok(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f"  ({self.detail})" if self.detail else "")


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def _near(pos: tuple[float, float], target: tuple[float, float]) -> bool:
    return _close(pos[0], target[0]) and _close(pos[1], target[1])


def _pos(result: dict[str, Any]) -> tuple[float, float]:
    pa = result.get("pose_after", {})
    return (pa.get("x", 0.0), pa.get("y", 0.0))


def _theta(result: dict[str, Any]) -> float:
    return float(result.get("pose_after", {}).get("theta", 0.0))


def _actions_mapping(raw: dict[str, Any]) -> dict[str, str]:
    """Normalise the list_actions payload to a flat id -> name mapping."""

    data = raw.get("actions", raw)
    return {str(k): str(v) for k, v in data.items()}


async def _canvas(robot: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", await robot.get_canvas())


# --------------------------------------------------------------------------- #
# Scenario 1: three forward moves
# --------------------------------------------------------------------------- #
async def scenario_1(robot: Any) -> list[Check]:
    checks: list[Check] = []
    await robot.reset()
    result: dict[str, Any] = {}
    for _ in range(3):
        result = await robot.execute(action="move_forward")
    canvas = await _canvas(robot)
    checks.append(Check("3x move_forward: position (3.0, 0)", _near(_pos(result), (3.0, 0.0))))
    checks.append(Check("theta unchanged at 0", _theta(result) == 0.0))
    checks.append(Check("trajectory_count == 3", canvas["trajectory_count"] == 3))
    return checks


# --------------------------------------------------------------------------- #
# Scenario 2: big left turn, then one forward move (now faces +y)
# --------------------------------------------------------------------------- #
async def scenario_2(robot: Any) -> list[Check]:
    checks: list[Check] = []
    await robot.reset()
    await robot.execute(action_id=3)  # turn_left_big == +90 degrees
    result = await robot.execute(action="move_forward")
    canvas = await _canvas(robot)
    checks.append(Check("theta == 90 after turn_left_big", _close(_theta(result), 90.0)))
    checks.append(Check("position (0, 1.0) after forward", _near(_pos(result), (0.0, 1.0))))
    checks.append(Check("trajectory_count == 1 (rotation adds no segment)", canvas["trajectory_count"] == 1))
    return checks


# --------------------------------------------------------------------------- #
# Scenario 3: forward then backward returns to origin
# --------------------------------------------------------------------------- #
async def scenario_3(robot: Any) -> list[Check]:
    checks: list[Check] = []
    await robot.reset()
    await robot.execute(action="move_forward")
    result = await robot.execute(action="move_backward")
    canvas = await _canvas(robot)
    checks.append(Check("net position (0, 0)", _near(_pos(result), (0.0, 0.0))))
    checks.append(Check("theta unchanged at 0", _theta(result) == 0.0))
    checks.append(Check("trajectory_count == 2", canvas["trajectory_count"] == 2))
    return checks


# --------------------------------------------------------------------------- #
# Scenario 4: every action is callable and has the right side effects
# --------------------------------------------------------------------------- #
async def scenario_4(robot: Any) -> list[Check]:
    checks: list[Check] = []
    mapping = _actions_mapping(await robot.list_actions())
    checks.append(Check("list_actions returns 9 actions", len(mapping) == 9))
    checks.append(Check("canonical mapping 0=turn_left, 2=move_forward, 5=move_backward",
                         mapping["0"] == "turn_left" and mapping["2"] == "move_forward"
                         and mapping["5"] == "move_backward"))
    checks.append(Check("extended ids 6=stop, 7=look_up, 8=look_down",
                         mapping["6"] == "stop" and mapping["7"] == "look_up" and mapping["8"] == "look_down"))

    await robot.reset()
    move = await robot.execute(action="move_forward")
    checks.append(Check("move_forward: success + trajectory added",
                         move["success"] and move["trajectory_added"]))
    checks.append(Check("move_forward: position advanced to (1.0, 0)", _near(_pos(move), (1.0, 0.0))))

    await robot.reset()
    rot = await robot.execute(action="turn_left")
    checks.append(Check("turn_left: no trajectory, no position change",
                         not rot["trajectory_added"] and _near(_pos(rot), (0.0, 0.0))))
    checks.append(Check("turn_left: heading changed to 15", _close(_theta(rot), 15.0)))

    await robot.reset()
    await robot.execute(action="move_forward")  # to (1.0, 0)
    stop = await robot.execute(action="stop")
    checks.append(Check("stop: no 2D change, no trajectory",
                         not stop["trajectory_added"] and _near(_pos(stop), (1.0, 0.0))))
    up = await robot.execute(action="look_up")
    checks.append(Check("look_up: no 2D change, no trajectory",
                         not up["trajectory_added"] and _near(_pos(up), (1.0, 0.0))))
    down = await robot.execute(action="look_down")
    checks.append(Check("look_down: no 2D change, no trajectory",
                         not down["trajectory_added"] and _near(_pos(down), (1.0, 0.0))))
    return checks


# --------------------------------------------------------------------------- #
# Scenario 5: point + line obstacle management, perception and blocking
# --------------------------------------------------------------------------- #
async def scenario_5(robot: Any) -> list[Check]:
    checks: list[Check] = []
    await robot.reset()
    point = await robot.add_obstacle({"obstacle_type": "point", "x": 0.2, "y": 0.0, "radius": 0.08,
                                       "confidence": 0.9, "label": "wall", "source": "camera"})
    checks.append(Check("point obstacle: position (0.2, 0)", _near(tuple(point["position"]), (0.2, 0.0))))
    checks.append(Check("point obstacle: radius 0.08", _close(point["radius"], 0.08)))
    checks.append(Check("point obstacle: label 'wall'", point["label"] == "wall"))
    checks.append(Check("point obstacle: confidence 0.9", _close(point["confidence"], 0.9)))

    line = await robot.add_obstacle({"obstacle_type": "line", "start": [-0.3, -0.3], "end": [0.3, 0.3],
                                      "width": 0.05, "label": "barrier", "confidence": 0.95})
    checks.append(Check("line obstacle: geometry endpoints",
                        tuple(line["geometry"][0]) == (-0.3, -0.3) and tuple(line["geometry"][1]) == (0.3, 0.3)))
    checks.append(Check("line obstacle: width 0.05", _close(line["width"], 0.05)))
    checks.append(Check("line obstacle: label 'barrier'", line["label"] == "barrier"))

    canvas = await _canvas(robot)
    checks.append(Check("both obstacles appear in get_canvas", len(canvas["obstacles"]) == 2))

    blocked = await robot.execute(action="move_forward")
    checks.append(Check("move into obstacle: blocked (success=False)", not blocked["success"]))
    checks.append(Check("move into obstacle: robot stays at (0, 0)", _near(_pos(blocked), (0.0, 0.0))))
    checks.append(Check("move into obstacle: red trajectory recorded", blocked["trajectory_added"]))
    checks.append(Check("move into obstacle: obstacle perceived", point["obstacle_id"] in blocked["perceived"]))
    return checks


ALL_SCENARIOS: tuple[tuple[str, Any], ...] = (
    ("Scenario 1: three forward moves", scenario_1),
    ("Scenario 2: big left turn then forward", scenario_2),
    ("Scenario 3: forward then backward", scenario_3),
    ("Scenario 4: every action is callable", scenario_4),
    ("Scenario 5: obstacle management and blocking", scenario_5),
)