"""MCP robot simulation agent: 100-step obstacle-aware run.

This agent is a *robot simulator*. It owns the world model (hidden blockers) and
all collision decisions. The MCP server has no prior obstacle knowledge; the
robot only calls ``execute_action`` to move and ``add_obstacle`` to report what
it perceives. MCP is a passive renderer and state recorder.

Usage:

    python -m agentic_memory_nav.agent.mcp_robot_simu_agent

Output:

    outputs/mcp_robot_simu_agent/<timestamp>/simulation_log.jsonl
    outputs/mcp_robot_simu_agent/<timestamp>/simulation_final.png
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_memory_nav.mcp.client import FakeRobot
from agentic_memory_nav.mcp.core.collision import check_move
from agentic_memory_nav.mcp.core.geometry import move as move_point
from agentic_memory_nav.mcp.core.models import Obstacle, Pose
from agentic_memory_nav.mcp.rendering import render_canvas
from agentic_memory_nav.mcp.server.app import build_app

# Same physical constants used by the /manual console and the renderer.
META_SIZE = 1.0
MOVE_STEP = 1.0

# Real world blockers placed every 3 units along the x-axis. They live ONLY in
# the robot simulator; MCP never sees them until the robot reports a perceived
# meta-obstacle at encounter time.
HIDDEN_BLOCK_CENTERS = [(6.0 + 3.0 * i, 0.0) for i in range(8)]

# After a blockage the robot performs a fixed left-hand detour that goes up to
# y=2, over the obstacle, and returns to the corridor line two units ahead.
DETOUR = [
    "turn_left_big",
    "move_forward",
    "move_forward",
    "turn_right_big",
    "move_forward",
    "move_forward",
    "turn_right_big",
    "move_forward",
    "move_forward",
    "turn_left_big",
]


def _now_stamp() -> str:
    """Return an ISO-like UTC timestamp suitable for folder names."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_square_corners(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    """Return the four corners of an axis-aligned square centered at (cx, cy)."""

    half = size / 2.0
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def _build_hidden_obstacles() -> list[Obstacle]:
    """Build the simulator's private 1x1 square blockers."""

    obstacles: list[Obstacle] = []
    for center in HIDDEN_BLOCK_CENTERS:
        corners = _make_square_corners(center[0], center[1], 1.0)
        group = f"hidden_{center[0]}_{center[1]}"
        for i in range(4):
            record = Obstacle.from_dict(
                "robot_0",
                {
                    "type": "line",
                    "geometry": {"start": corners[i], "end": corners[(i + 1) % 4]},
                    "width": 0.02,
                    "properties": {"group_id": group},
                },
            )
            record.obstacle_id = f"{group}_edge_{i}"
            obstacles.append(record)
    return obstacles


def _predict_blocked(pose: Pose, hidden_obstacles: list[Obstacle]) -> bool:
    """Return whether ``move_forward`` from ``pose`` would hit a real blocker."""

    target = move_point(pose.to_point(), pose.theta, MOVE_STEP)
    swept: tuple[tuple[float, float], tuple[float, float]] = (pose.to_point(), target)
    result = check_move(hidden_obstacles, swept)
    return result.blocked


def _report_meta_obstacle_request(pose: Pose, group_id: str) -> dict[str, Any]:
    """Build a 1x1 square meta-obstacle request centered on the blocked target."""

    rad = math.radians(pose.theta)
    cx = pose.x + MOVE_STEP * math.cos(rad)
    cy = pose.y + MOVE_STEP * math.sin(rad)
    corners = _make_square_corners(cx, cy, META_SIZE)
    return {
        "type": "line",
        "geometry": {"start": corners[0], "end": corners[1]},
        "width": 0.02,
        "label": "",
        "properties": {"group_id": group_id, "perceived": True},
    }, {
        "type": "line",
        "geometry": {"start": corners[1], "end": corners[2]},
        "width": 0.02,
        "label": "",
        "properties": {"group_id": group_id, "perceived": True},
    }, {
        "type": "line",
        "geometry": {"start": corners[2], "end": corners[3]},
        "width": 0.02,
        "label": "",
        "properties": {"group_id": group_id, "perceived": True},
    }, {
        "type": "line",
        "geometry": {"start": corners[3], "end": corners[0]},
        "width": 0.02,
        "label": "",
        "properties": {"group_id": group_id, "perceived": True},
    }


def _format_step_log(
    step: int,
    action: str,
    result: dict[str, Any],
    *,
    reported_meta: list[str] | None = None,
    predicted_blocked: bool = False,
) -> dict[str, Any]:
    """Build a single JSONL line describing one action and its encounter state."""

    return {
        "step": step,
        "action": action,
        "success": result.get("success", True),
        "pose": result.get("pose_after", {}),
        "blocked_by": result.get("blocked_by", []),
        "nearest_clearance": round(result.get("nearest_clearance", 0.0), 4),
        "perceived_obstacles": len(result.get("perceived", [])),
        "reported_meta_obstacles": reported_meta or [],
        "predicted_blocked": predicted_blocked,
    }


async def run_simulation(
    *,
    steps: int,
    output_dir: Path,
    url: str | None,
) -> dict[str, Any]:
    """Run the 100-step simulation and write the log + final render."""

    if url:
        robot = FakeRobot.from_url("robot_0", url)
    else:
        _app, _engine, _hub, mcp = build_app()
        robot = FakeRobot.in_process("robot_0", mcp)

    await robot.connect()
    await robot.reset(clear_events=True)
    await robot.create({"x": 0.0, "y": 0.0, "theta": 0.0})

    # The simulator owns the real obstacles; MCP starts with none of them.
    hidden_obstacles = _build_hidden_obstacles()
    pose = Pose(x=0.0, y=0.0, theta=0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "simulation_log.jsonl"
    png_path = output_dir / "simulation_final.png"

    with jsonl_path.open("w") as fh:
        step = 0
        while step < steps:
            predicted = _predict_blocked(pose, hidden_obstacles)

            if not predicted:
                # Free path: just move.
                result = await robot.execute("move_forward")
                pose = Pose.from_dict(result.get("pose_after", pose.to_dict()))
                fh.write(
                    json.dumps(_format_step_log(step, "move_forward", result))
                    + "\n"
                )
                step += 1
                continue

            # Blocked: report the perceived obstacle to MCP first, then execute
            # the blocked move so MCP renders the red failure segment.
            group_id = f"meta_{step}"
            meta_requests = _report_meta_obstacle_request(pose, group_id)
            meta_ids: list[str] = []
            for req in meta_requests:
                record = await robot.add_obstacle(req)
                meta_ids.append(record["obstacle_id"])

            result = await robot.execute("move_forward")
            pose = Pose.from_dict(result.get("pose_after", pose.to_dict()))
            fh.write(
                json.dumps(
                    _format_step_log(
                        step,
                        "move_forward",
                        result,
                        reported_meta=meta_ids,
                        predicted_blocked=True,
                    )
                )
                + "\n"
            )
            step += 1

            if step >= steps:
                break

            # Detour around the perceived obstacle.
            for detour_action in DETOUR:
                if step >= steps:
                    break
                result = await robot.execute(detour_action)
                pose = Pose.from_dict(result.get("pose_after", pose.to_dict()))
                fh.write(
                    json.dumps(_format_step_log(step, detour_action, result))
                    + "\n"
                )
                step += 1

    canvas = await robot.get_canvas()
    render_canvas(canvas, str(png_path), title=f"Robot simulation ({steps} steps)")
    await robot.close()

    blocked_steps = sum(
        1 for line in jsonl_path.read_text().splitlines() if '"success": false' in line
    )
    summary = {
        "steps": steps,
        "jsonl": str(jsonl_path),
        "render": str(png_path),
        "trajectory_count": canvas.get("trajectory_count", 0),
        "obstacle_count": len(canvas.get("obstacles", [])),
        "blocked_steps": blocked_steps,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate 100 steps of robot motion with obstacle encounters."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of actions to simulate (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "mcp_robot_simu_agent",
        help="Base output directory (a timestamped sub-folder is created).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="MCP server URL (omit for in-process simulation).",
    )
    args = parser.parse_args()

    run_dir = args.output_dir / _now_stamp()
    summary = asyncio.run(
        run_simulation(steps=args.steps, output_dir=run_dir, url=args.url)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
