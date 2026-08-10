#!/usr/bin/env python3
"""Run the fully local mock end-to-end navigation demo."""

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
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.orchestration.pipeline import NavigationPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/dev.yaml"))
    parser.add_argument(
        "--execution-backend",
        default=None,
        choices=("unitree_sim", "habitat"),
        help="Optional execution backend override",
    )
    parser.add_argument("--scene", default=None, help="Optional scene path for habitat backend")
    parser.add_argument(
        "--instruction",
        default="找到厨房里的红色杯子并移动到附近",
        help="Natural-language navigation instruction",
    )
    return parser.parse_args()


async def run() -> int:
    args = parse_args()
    config = load_config(args.config)
    execution = config.section("execution")
    if args.execution_backend is not None:
        execution["backend"] = args.execution_backend
    if args.scene is not None:
        execution["scene"] = args.scene
    output_root = ROOT / str(config.section("runtime").get("output_root", "outputs"))
    experiment = ExperimentRun(output_root, config.raw)
    logger = configure_logging(
        experiment.path / "logs.jsonl", bool(config.section("runtime").get("verbose", False))
    )
    pipeline = NavigationPipeline(config.raw, experiment, logger)
    try:
        metrics = await pipeline.run_task(args.instruction)
    finally:
        pipeline.close()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Run artifacts: {experiment.path}")
    return 0 if metrics["success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
