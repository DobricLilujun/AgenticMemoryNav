#!/usr/bin/env python3
"""Run the Isaac Sim PointNav harness and report SPL/success metrics.

Must be launched with Isaac Sim's bundled Python, e.g.:
    ~/isaacsim/python.sh scripts/run_isaacsim_pointnav.py --config configs/isaacsim_pointnav.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.common.logging import configure_logging  # noqa: E402
from agentic_memory_nav.common.types import ActionIntent, ActionType, new_id  # noqa: E402
from agentic_memory_nav.datasets.pointnav import PointNavDataset  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.evaluation.metrics import (  # noqa: E402
    goal_reached,
    success_weighted_path_length,
)
from agentic_memory_nav.execution.isaacsim_adapter import IsaacSimExecutor  # noqa: E402
from agentic_memory_nav.execution.safety_controller import SafetyController  # noqa: E402


def _planar_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist((a[0], a[2]), (b[0], b[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isaacsim_pointnav.yaml"))
    return parser.parse_args()


def run_episode(
    executor: IsaacSimExecutor,
    run: ExperimentRun,
    episode,
    max_steps: int,
    waypoint_duration_s: float,
) -> dict:
    executor.teleport(episode.start)
    path_length = 0.0
    previous_position = executor.get_state().position
    success = False
    collided = False
    steps_taken = 0
    for step in range(max_steps):
        state = executor.get_state()
        distance_to_goal = _planar_distance(state.position, episode.goal)
        run.append_trajectory(
            {
                "episode_id": episode.episode_id,
                "step": step,
                "position": state.position,
                "distance_to_goal": distance_to_goal,
            }
        )
        if goal_reached(distance_to_goal, episode.success_threshold_m):
            success = True
            break
        intent = ActionIntent(
            new_id("action"),
            ActionType.NAVIGATE,
            None,
            episode.goal,
            waypoint_duration_s,
            ["collision_free"],
            1.0,
            "pointnav_benchmark",
        )
        feedback = executor.send_waypoint(episode.goal, intent)
        path_length += _planar_distance(previous_position, feedback.state.position)
        previous_position = feedback.state.position
        steps_taken = step + 1
        if feedback.collision:
            collided = True
            break
    final_distance = _planar_distance(executor.get_state().position, episode.goal)
    if not success:
        success = goal_reached(final_distance, episode.success_threshold_m)
    spl = success_weighted_path_length(success, episode.shortest_path_m, path_length)
    return {
        "episode_id": episode.episode_id,
        "success": success,
        "collided": collided,
        "steps": steps_taken,
        "shortest_path_m": episode.shortest_path_m,
        "path_length_m": path_length,
        "final_distance_to_goal_m": final_distance,
        "spl": spl,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    execution = config.section("execution")
    pointnav = config.section("pointnav")
    output_root = ROOT / str(config.section("runtime").get("output_root", "outputs"))

    run = ExperimentRun(output_root, config.raw)
    logger = configure_logging(run.path / "logs.jsonl", verbose=False)

    dataset = PointNavDataset(
        num_episodes=int(pointnav.get("num_episodes", 10)),
        seed=int(pointnav.get("seed", 0)),
        resolution=float(pointnav.get("resolution", 0.1)),
        success_threshold_m=float(pointnav.get("success_threshold_m", 0.36)),
        min_geodesic_m=float(pointnav.get("min_geodesic_m", 1.0)),
    )
    episodes = dataset.generate()

    safety = SafetyController(
        max_speed=float(execution.get("max_speed", 0.5)),
        max_angular_speed=float(execution.get("max_angular_speed", 1.0)),
        max_timeout=float(execution.get("max_action_timeout", 15.0)),
    )
    executor = IsaacSimExecutor(
        scene=None,
        safety=safety,
        max_speed=float(execution.get("max_speed", 0.5)),
        headless=bool(execution.get("headless", True)),
    )

    results = []
    try:
        for episode in episodes:
            result = run_episode(
                executor,
                run,
                episode,
                max_steps=int(pointnav.get("max_steps_per_episode", 40)),
                waypoint_duration_s=float(pointnav.get("waypoint_duration_s", 4.0)),
            )
            results.append(result)
            logger.info(
                "pointnav episode completed",
                extra={"fields": {"episode": result["episode_id"], "success": result["success"]}},
            )
    finally:
        # Write results before closing: SimulationApp.close() terminates the process.
        success_rate = sum(item["success"] for item in results) / max(1, len(results))
        mean_spl = sum(item["spl"] for item in results) / max(1, len(results))
        summary = {
            "run_id": run.run_id,
            "num_episodes": len(results),
            "success_rate": success_rate,
            "spl": mean_spl,
        }
        run.write_json("episodes.json", results)
        run.write_json("metrics.json", summary)
        print(json.dumps(summary, indent=2))
        print(f"Run artifacts: {run.path}")
        run.close()
        executor.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
