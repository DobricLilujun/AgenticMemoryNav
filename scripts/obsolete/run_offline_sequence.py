#!/usr/bin/env python3
"""Run mapping over a NumPy RGB-D folder or a video file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.common.types import FrameObservation  # noqa: E402
from agentic_memory_nav.agent.datasets.base import NumpyRGBDSequence  # noqa: E402
from agentic_memory_nav.agent.mapping.mock_mapper import MockMapper  # noqa: E402


def video_frames(path: Path) -> list[FrameObservation]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Video input requires the optional 'opencv-python' package") from error
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(
            FrameObservation(
                f"frame_{index:06d}",
                index / fps,
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                source="video",
            )
        )
        index += 1
    capture.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--output", default=str(ROOT / "outputs/offline_map"))
    args = parser.parse_args()
    config = load_config(args.config)
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(source)
    frames = list(NumpyRGBDSequence(source).frames()) if source.is_dir() else video_frames(source)
    if not frames:
        raise RuntimeError(f"No frames found in {source}")
    mapper = MockMapper(int(config.section("mapping").get("keyframe_interval", 2)))
    mapper.start()
    for frame in frames:
        mapper.update(frame)
    output = Path(args.output)
    mapper.save_state(output)
    print(
        json.dumps(
            {
                "frames": len(frames),
                "points": len(mapper.get_global_pointcloud()),
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
