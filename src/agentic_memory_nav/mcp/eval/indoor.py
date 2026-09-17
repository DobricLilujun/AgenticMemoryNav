"""Indoor floor-plan scenarios for the MCP infinite-canvas system.

These extend the five canonical specification scenarios with **indoor** layouts:
rooms, walls, corridors and furniture, all expressed in the abstract *unit
length*. Each scenario drives the robot through the engine (the fake robot would
produce identical results through the MCP tools), records the canvas at key
stages for rendering, and checks the resulting state (position, trajectory and
blocking behaviour).

Because the robot advances in discrete ``move_step`` (1.0 unit) increments, a
move that would penetrate an obstacle is *blocked*: a red trajectory segment is
recorded and the robot stays in place -- exactly the behaviour these indoor
scenarios exercise (a robot pressing into a wall, a table or a room boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.engine import SimulationEngine

# ---------------------------------------------------------------------------
# obstacle builders (unit length)
# ---------------------------------------------------------------------------


def _line(x0: float, y0: float, x1: float, y1: float,
         width: float = 0.1, label: str = "") -> dict[str, Any]:
    """A line obstacle (a wall / corridor side)."""

    return {"obstacle_type": "line", "start": (x0, y0), "end": (x1, y1),
           "width": width, "label": label, "confidence": 1.0, "source": "floorplan"}


def _point(x: float, y: float, radius: float = 0.25, label: str = "point") -> dict[str, Any]:
    """A point obstacle (a table / pillar / block)."""

    return {"obstacle_type": "point", "position": (x, y), "radius": radius,
           "label": label, "confidence": 1.0, "source": "floorplan"}


def room_walls(x0: float, y0: float, x1: float, y1: float) -> list[dict[str, Any]]:
    """The four line obstacles forming the boundary of a room."""

    return [
        _line(x0, y0, x1, y0, label="wall-S"),
        _line(x0, y1, x1, y1, label="wall-N"),
        _line(x0, y0, x0, y1, label="wall-W"),
        _line(x1, y0, x1, y1, label="wall-E"),
    ]


# ---------------------------------------------------------------------------
# a single indoor scenario
# ---------------------------------------------------------------------------


@dataclass
class IndoorScenario:
    """An indoor layout plus the navigation the robot runs through it."""

    name: str
    start: dict[str, Any]
    floorplan: list[dict[str, Any]]
    sequence: list[str]
    notes: str = ""
    expect_blocked: int = 0
    robot_id: str = "indoor_0"

    def run(self) -> dict[str, Any]:
        """Drive the scenario and return the results plus canvas snapshots."""

        engine = SimulationEngine()
        for obstacle in self.floorplan:
            engine.add_obstacle(self.robot_id, obstacle)
        engine.create_robot(self.robot_id, self.start)

        # the "entry" frame: the floor plan before any motion
        frames = [engine.get_canvas()]
        blocked = 0
        results: list[dict[str, Any]] = []
        for action in self.sequence:
            result = engine.execute_action(self.robot_id, action).to_dict()
            results.append(result)
            if result["success"] is False and result["kind"] == "move":
                blocked += 1
        frames.append(engine.get_canvas())
        canvas = engine.get_canvas()
        pose = canvas["robots"][0]
        return {
            "name": self.name,
            "notes": self.notes,
            "robot_id": self.robot_id,
            "start": self.start,
            "final_pose": pose,
            "trajectory_count": canvas["trajectory_count"],
            "obstacle_count": len(canvas["obstacles"]),
            "blocked_moves": blocked,
            "expect_blocked": self.expect_blocked,
            "results": results,
            "frames": frames,
            "canvas": canvas,
        }


# ---------------------------------------------------------------------------
# the indoor scenarios
# ---------------------------------------------------------------------------


def scenario_room_blocking() -> IndoorScenario:
    """A room with a wall ahead; the robot presses into it, then turns and moves."""

    return IndoorScenario(
        "indoor: room + wall blocking",
        start={"x": 0.0, "y": 0.0, "theta": 0.0},
        floorplan=[_line(3.5, -0.8, 3.5, 0.8, label="wall-E")],
        sequence=[
            "move_forward", "move_forward", "move_forward",  # reach x=3.0
            "move_forward",  # 4th move would cross x=3.5 into the wall -> blocked
            "turn_right_big",  # face -y
            "move_forward", "move_forward",  # move south along the wall
        ],
        notes="3 successful forward moves, then the 4th is blocked by wall-E; the robot "
        "turns right and moves along the wall.",
        expect_blocked=1,
    )


def scenario_corridor() -> IndoorScenario:
    """A corridor between two walls; the robot walks down it and turns out."""

    return IndoorScenario(
        "indoor: corridor",
        start={"x": 0.0, "y": 0.0, "theta": 0.0},
        floorplan=[
            _line(-0.5, 0.4, 4.0, 0.4, label="corridor-N"),
            _line(-0.5, -1.4, 4.0, -1.4, label="corridor-S"),
            _line(3.4, -1.4, 3.4, 0.4, label="corridor-E"),
        ],
        sequence=[
            "move_forward", "move_forward", "move_forward",  # walk down the corridor
            "move_forward",  # hit the east end
            "turn_left_big",  # face +y
            "move_backward",  # step back, away from the east wall
            "turn_right", "turn_right", "turn_right",  # face -30deg
        ],
        notes="The robot moves down a corridor bounded by two walls, reaches the east "
        "end, turns north and steps out.",
        expect_blocked=1,
    )


def scenario_furniture_avoidance() -> IndoorScenario:
    """A table in a room: the robot is blocked by it and steers around."""

    return IndoorScenario(
        "indoor: avoid a table (furniture)",
        start={"x": 0.0, "y": 0.0, "theta": 0.0},
        floorplan=[_point(3.0, 0.0, radius=0.3, label="table")],
        sequence=[
            "move_forward", "move_forward", "move_forward",  # approach the table
            "turn_left_big",  # face +y
            "move_forward", "move_forward",  # go north of the table
            "turn_right_big",  # face +x again
            "move_forward", "move_forward",  # come back past the table
        ],
        notes="The table sits on the +x axis at x=3.0. The robot advances until the "
        "table blocks it, turns north to clear it, then continues east.",
        expect_blocked=1,
    )


def scenario_doorway() -> IndoorScenario:
    """Two rooms joined by a doorway (a gap in a wall)."""

    return IndoorScenario(
        "indoor: doorway between two rooms",
        start={"x": 0.0, "y": -0.3, "theta": 90.0},
        floorplan=[
            _line(0.2, -2.0, 0.2, -0.4, label="door-wall-low"),
            _line(-0.2, 1.8, -0.2, 3.0, label="door-wall-high"),
            _line(-2.0, -2.0, 0.2, -2.0, label="room-A-S"),
            _line(-0.2, 3.0, 2.0, 3.0, label="room-B-N"),
        ],
        sequence=["move_forward", "move_forward", "turn_left_big", "move_forward"],
        notes="The robot faces north and walks up through the doorway (the gap between "
        "the two wall segments at x=0.0), then turns west into the far room.",
        expect_blocked=0,
    )


def scenario_multi_room() -> IndoorScenario:
    """Two rooms with a pillar in the first; the robot clears it and passes through."""

    return IndoorScenario(
        "indoor: pillar + doorway",
        start={"x": -0.3, "y": 0.0, "theta": 0.0},
        floorplan=[
            _line(2.5, -0.5, 2.5, -0.25, label="door-wall-low"),
            _line(2.5, 0.25, 2.5, 0.5, label="door-wall-high"),
            _point(0.5, 0.0, radius=0.2, label="pillar"),
        ],
        sequence=[
            "move_forward",  # press into the pillar (blocked)
            "turn_left_big",  # face +y
            "move_forward",  # go north of the pillar
            "turn_right_big",  # face +x again
            "move_forward", "move_forward",  # walk through the doorway
        ],
        notes="The first move hits the pillar (blocked); the robot goes around it "
        "(north, then east) and walks through the doorway into the far room.",
        expect_blocked=1,
    )


ALL_INDOOR: list[IndoorScenario] = [
    scenario_room_blocking(),
    scenario_corridor(),
    scenario_furniture_avoidance(),
    scenario_doorway(),
    scenario_multi_room(),
]


def run_all() -> list[dict[str, Any]]:
    """Run every indoor scenario and return their results."""

    return [scenario.run() for scenario in ALL_INDOOR]