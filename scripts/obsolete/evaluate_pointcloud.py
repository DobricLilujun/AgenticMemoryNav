#!/usr/bin/env python3
"""Compare a predicted point cloud against a paired ground-truth cloud.

Both artifacts must be NPZ files containing a float array named `points` with shape
``(N, 3)`` in meters. The default `--alignment none` evaluates the actual shared
world coordinate frame. `--alignment centroid` is a diagnostic translation-only
comparison, never a substitute for trajectory or coordinate-frame calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.evaluation.pointcloud import (  # noqa: E402
    evaluate_pointclouds,
    load_npz_pointcloud,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="LingBot prediction NPZ path")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth NPZ path")
    parser.add_argument("--threshold-m", type=float, default=0.05)
    parser.add_argument("--alignment", choices=("none", "centroid"), default="none")
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument("--output", help="Optional metrics JSON output path")
    args = parser.parse_args()

    metrics = evaluate_pointclouds(
        load_npz_pointcloud(args.prediction),
        load_npz_pointcloud(args.ground_truth),
        threshold_m=args.threshold_m,
        alignment=args.alignment,
        max_points=args.max_points,
    )
    metrics["prediction_path"] = str(Path(args.prediction).resolve())
    metrics["ground_truth_path"] = str(Path(args.ground_truth).resolve())
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
