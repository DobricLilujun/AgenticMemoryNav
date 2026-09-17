"""Generate a 100-step manual-validation path and render the final canvas.

The path simulates an operator driving the robot with the same rules as the
/manual console: every move is either clear (success) or blocked (recorded as a
red failed segment, and a 2x2 meta-obstacle is reported with its near edge at
the blocked target point). The JSONL records one line per action:

    {"step": i, "action": "move_forward", "blocked": false}

The final canvas is rendered to a PNG next to the JSONL file.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

from agentic_memory_nav.mcp.core.engine import SimulationEngine
from agentic_memory_nav.mcp.rendering import render_canvas

OUT_DIR = Path("/home/yiqun/project_lujun/AgenticMemoryNav/notebooks")
JSONL_PATH = OUT_DIR / "manual_path_100.jsonl"
PNG_PATH = OUT_DIR / "manual_path_100.png"

# 2x2 meta-obstacle, same standard as the /manual console.
META_SIZE = 2.0
MOVE_STEP = 1.0


def add_meta_obstacle(engine: SimulationEngine, robot_id: str) -> None:
    """Report a 2x2 square whose near edge sits at the blocked target point."""

    robot = engine.get_robot(robot_id)
    pose = robot.pose
    rad = math.radians(pose.theta)
    # centre = one move step ahead + half the square size along the heading
    fx = pose.x + (MOVE_STEP + META_SIZE / 2) * math.cos(rad)
    fy = pose.y + (MOVE_STEP + META_SIZE / 2) * math.sin(rad)
    half = META_SIZE / 2
    corners = [
        (fx - half, fy - half),
        (fx + half, fy - half),
        (fx + half, fy + half),
        (fx - half, fy + half),
    ]
    group = f"meta_{len(engine.obstacles)}"
    for i in range(4):
        engine.add_obstacle(robot_id, {
            "obstacle_type": "line",
            "geometry": {"start": corners[i], "end": corners[(i + 1) % 4]},
            "width": 0.02,
            "label": "",
            "properties": {"group_id": group, "confirmed": False},
        })


def blocked_move(engine: SimulationEngine, robot_id: str, action: str) -> None:
    """Record a failed move: temporarily place a hidden blocker at the target."""

    robot = engine.get_robot(robot_id)
    pose = robot.pose
    rad = math.radians(pose.theta)
    sign = -1.0 if action == "move_backward" else 1.0
    tx = pose.x + sign * MOVE_STEP * math.cos(rad)
    ty = pose.y + sign * MOVE_STEP * math.sin(rad)
    blocker = engine.add_obstacle(robot_id, {
        "obstacle_type": "point", "x": tx, "y": ty, "radius": 0.05,
        "properties": {"visible": False, "tag": "virtual_blocker"},
    })
    engine.execute_action(robot_id, action)  # fails, records the red segment
    engine.remove_obstacle(blocker["obstacle_id"])
    add_meta_obstacle(engine, robot_id)


async def main() -> None:
    engine = SimulationEngine()
    robot_id = "robot_0"
    engine.create_robot(robot_id, {"x": 0.0, "y": 0.0, "theta": 0.0})

    # Reactive 100-step path: drive forward, and whenever a move is blocked by a
    # hidden ground-truth block, report a meta-obstacle and turn 30 degrees to
    # look for a way around. Once a block has been reported it is treated as
    # known and removed, so the robot does not keep pushing into the same wall.
    hidden_blocks = [
        (6.0, 0.0),      # first wall straight ahead
        (8.5, 1.5),      # after the first left detour
        (11.0, 0.0),     # back toward the axis
        (13.5, -1.5),    # lower corridor
        (16.0, 0.0),     # return to the axis
        (18.5, 1.5),     # upper corridor
        (21.0, 0.0),     # straight again
        (23.5, -1.5),    # final lower stretch
    ]
    hidden_ids: dict[tuple[float, float], list[str]] = {}
    for hx, hy in hidden_blocks:
        half = 0.5  # hidden blocks are larger so they genuinely stop the robot
        corners = [
            (hx - half, hy - half),
            (hx + half, hy - half),
            (hx + half, hy + half),
            (hx - half, hy + half),
        ]
        group = f"hidden_{hx}_{hy}"
        hidden_ids[(hx, hy)] = []
        for i in range(4):
            record = engine.add_obstacle(robot_id, {
                "obstacle_type": "line",
                "geometry": {"start": corners[i], "end": corners[(i + 1) % 4]},
                "width": 0.02,
                "label": "",
                "properties": {"group_id": group, "visible": False},
            })
            hidden_ids[(hx, hy)].append(record["obstacle_id"])

    turn_cycle = ["turn_left", "turn_left", "turn_right", "turn_right"]
    turn_index = 0

    with JSONL_PATH.open("w") as fh:
        step = 0
        while step < 100:
            # Try to move forward; if blocked, report the obstacle and turn away.
            result = engine.execute_action(robot_id, "move_forward")
            if result.success:
                fh.write(json.dumps({"step": step, "action": "move_forward", "blocked": False}) + "\n")
                step += 1
                continue

            # Blocked by a hidden world block: record the failed segment, report
            # the meta-obstacle, then remove the hidden block so the robot can
            # continue past it (the obstacle is now part of the known map).
            blocked_move(engine, robot_id, "move_forward")
            fh.write(json.dumps({"step": step, "action": "move_forward", "blocked": True}) + "\n")
            step += 1

            # Remove the hidden block that caused the blockage.
            robot = engine.get_robot(robot_id)
            pose = robot.pose
            for (hx, hy), ids in list(hidden_ids.items()):
                if math.hypot(pose.x - hx, pose.y - hy) < 1.6:
                    for oid in ids:
                        engine.remove_obstacle(oid)
                    del hidden_ids[(hx, hy)]

            turn = turn_cycle[turn_index % len(turn_cycle)]
            turn_index += 1
            engine.execute_action(robot_id, turn)
            fh.write(json.dumps({"step": step, "action": turn, "blocked": False}) + "\n")
            step += 1

    canvas = engine.get_canvas()
    render_canvas(canvas, str(PNG_PATH), title="Manual 100-step path")
    print(f"Wrote {JSONL_PATH}")
    print(f"Wrote {PNG_PATH}")
    print(f"trajectory segments: {canvas['trajectory_count']}, obstacles: {len(canvas['obstacles'])}")


if __name__ == "__main__":
    asyncio.run(main())
