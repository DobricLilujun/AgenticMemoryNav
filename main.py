"""CLI entry point for the mock end-to-end navigation demo."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agentic_memory_nav.agent.orchestration.pipeline import NavigationPipeline
from agentic_memory_nav.common.config import load_config
from agentic_memory_nav.common.logging import configure_logging
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun


def _redact(config: dict) -> dict:
    """Return a copy with sensitive keys hidden for logging."""

    def walk(value, key: str = ""):
        lowered = key.lower()
        if any(token in lowered for token in ("api_key", "token", "secret", "password")):
            return "<redacted>"
        if isinstance(value, dict):
            return {k: walk(v, str(k)) for k, v in value.items()}
        return value

    return walk(config)


async def run(config_path: str | Path, instruction: str) -> None:
    app_config = load_config(config_path)
    config = app_config.raw
    output_root = Path(config.get("runtime", {}).get("output_root", "outputs")).resolve()
    run = ExperimentRun(output_root, config)
    verbose = config.get("runtime", {}).get("verbose", False)
    logger = configure_logging(run.path / "logs.jsonl", verbose=verbose)
    logger.info(
        "starting demo",
        extra={"fields": {"instruction": instruction, "config": _redact(config)}},
    )

    pipeline = NavigationPipeline(config, run, logger)
    try:
        metrics = await pipeline.run_task(instruction)
        logger.info("demo finished", extra={"fields": metrics})
        print(f"Run: {run.run_id}")
        print(f"Success: {metrics['success_rate']:.2f}")
        print(f"Path length: {metrics['path_length_m']:.2f} m")
    finally:
        pipeline.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock navigation demo.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--instruction",
        default="find the red cup in the kitchen",
        help="Natural-language navigation instruction.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.config, args.instruction))


if __name__ == "__main__":
    main()
