#!/usr/bin/env python3
"""Build and print the deterministic demo scene graph."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.common.types import FrameObservation  # noqa: E402
from agentic_memory_nav.agent.mapping.mock_mapper import MockMapper  # noqa: E402
from agentic_memory_nav.agent.perception.mock_perception import MockPerception  # noqa: E402
from agentic_memory_nav.scene_graph.graph import SceneGraph  # noqa: E402
from agentic_memory_nav.scene_graph.updater import SceneGraphUpdater  # noqa: E402


def main() -> int:
    graph, mapper, detector = SceneGraph(), MockMapper(), MockPerception()
    updater = SceneGraphUpdater(graph)
    mapper.start()
    for index in range(3):
        frame = FrameObservation(f"frame_{index}", float(index), np.zeros((64, 96, 3), np.uint8))
        updater.update(detector.detect(frame, mapper.update(frame)))
    print(json.dumps(graph.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
