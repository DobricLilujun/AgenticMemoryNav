#!/usr/bin/env python3
"""Run navigation with Habitat when available, otherwise use the safe mock simulator."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_demo import run as run_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/simulation.yaml"))
    args = parser.parse_args()
    backend = "habitat"
    if importlib.util.find_spec("habitat_sim") is None:
        backend = "unitree_sim"
        print("Habitat-Sim is unavailable; falling back to Unitree-like mock execution.")
    sys.argv = [
        sys.argv[0],
        "--config",
        args.config,
        "--instruction",
        args.instruction,
        "--execution-backend",
        backend,
        "--scene",
        args.scene,
    ]
    return asyncio.run(run_demo())


if __name__ == "__main__":
    raise SystemExit(main())