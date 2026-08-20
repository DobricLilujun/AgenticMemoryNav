#!/usr/bin/env python3
"""Run one Isaac Sim ObjectNav experiment (InteriorAgent-style scene) and report metrics.

Must be launched with Isaac Sim's bundled Python, e.g.:
    ~/isaacsim/python.sh scripts/run_isaacsim_objectnav.py \\
        --config configs/isaacsim_objectnav.yaml \\
        --scene-root /path/to/InteriorAgent \\
        --experiments-json /path/to/InteriorAgent/experiments.json \\
        --experiment kujiale_0020_bottle_moved

Requires the InteriorAgent dataset (huggingface.co/datasets/spatialverse/InteriorAgent),
which is not bundled with this repository. See notebooks/isaacsim_benchmarks.ipynb for
download instructions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.common.logging import configure_logging  # noqa: E402
from agentic_memory_nav.agent.datasets.objectnav import load_experiments  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.evaluation.metrics import goal_reached  # noqa: E402
from agentic_memory_nav.agent.execution.isaacsim_objectnav_adapter import (  # noqa: E402
    IsaacSimObjectNavExecutor,
)
from agentic_memory_nav.agent.execution.safety_controller import SafetyController  # noqa: E402
from agentic_memory_nav.agent.orchestration.pipeline import NavigationPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isaacsim_objectnav.yaml"))
    parser.add_argument("--scene-root", default=None, help="InteriorAgent dataset root directory")
    parser.add_argument("--experiments-json", default=None, help="Path to experiments.json")
    parser.add_argument("--experiment", default=None, help="Experiment name from experiments.json")
    return parser.parse_args()


def instruction_for_goal(goal) -> str:
    if goal.task == "search" and goal.label:
        return f"Find the {goal.label}"
    return "Explore the room"


async def run() -> int:
    args = parse_args()
    config = load_config(args.config)
    objectnav = config.section("objectnav")
    execution = config.section("execution")

    scene_root = args.scene_root or objectnav.get("scene_root")
    experiments_json = args.experiments_json or objectnav.get("experiments_json")
    experiment_name = args.experiment or objectnav.get("experiment_name")
    if not scene_root or not experiments_json or not experiment_name:
        raise ValueError(
            "scene_root, experiments_json, and experiment_name are all required "
            "(via --scene-root/--experiments-json/--experiment or the config file)"
        )

    experiments = load_experiments(experiments_json)
    experiment = experiments[experiment_name]

    output_root = ROOT / str(config.section("runtime").get("output_root", "outputs"))
    run = ExperimentRun(output_root, config.raw)
    logger = configure_logging(run.path / "logs.jsonl", verbose=False)

    safety = SafetyController(
        max_speed=float(execution.get("max_speed", 0.5)),
        max_angular_speed=float(execution.get("max_angular_speed", 1.0)),
        max_timeout=float(execution.get("max_action_timeout", 15.0)),
    )
    executor = IsaacSimObjectNavExecutor(
        scene_root=str(scene_root),
        experiment=experiment,
        safety=safety,
        max_speed=float(execution.get("max_speed", 0.5)),
        headless=bool(execution.get("headless", True)),
    )
    pipeline = NavigationPipeline(config.raw, run, logger, executor_override=executor)

    pipeline_metrics = None
    summary = None
    try:
        pipeline_metrics = await pipeline.run_task(instruction_for_goal(experiment.goal))
        distance_to_goal = executor.shortest_distance_to_goal()
        success_threshold = float(objectnav.get("success_threshold_m", 1.0))
        success = distance_to_goal is not None and goal_reached(distance_to_goal, success_threshold)
        summary = {
            "run_id": run.run_id,
            "experiment": experiment.name,
            "goal_task": experiment.goal.task,
            "goal_label": experiment.goal.label,
            "distance_to_goal_m": distance_to_goal,
            "success_threshold_m": success_threshold,
            "success": success,
            "pipeline_success_rate": pipeline_metrics["success_rate"],
            "pipeline_spl": pipeline_metrics["spl"],
        }
    finally:
        # Write results before close(): Isaac Sim's executor.close() terminates the process.
        if summary is not None:
            run.write_json("objectnav_metrics.json", summary)
            print(json.dumps(summary, indent=2))
            print(f"Run artifacts: {run.path}")
        pipeline.close()

    return 0 if summary is not None and summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
