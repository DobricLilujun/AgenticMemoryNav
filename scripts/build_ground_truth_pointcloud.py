#!/usr/bin/env python3
"""Build a standard NPZ ground-truth cloud from calibrated metric depth frames.

Required inputs:
- ``depth_*.npy`` files in ``--depth-dir`` with metric depth in meters;
- ``--intrinsics-json`` containing a $3\times3$ camera matrix or ``{"matrix": [...]}``;
- ``--poses-json`` mapping each depth filename to a camera-to-world $4\times4$ matrix.

The generated NPZ uses ``points: float32 (N, 3)`` and can be compared against a
LingBot export using ``scripts/evaluate_pointcloud.py``. Both clouds must be in
the same world coordinate convention for the default, valid ``--alignment none``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.geometry.ground_truth import backproject_depth_to_world  # noqa: E402


def _matrix(payload: object, name: str, expected_shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(payload, dtype=np.float32)
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--intrinsics-json", required=True)
    parser.add_argument("--poses-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=50.0)
    args = parser.parse_args()

    intrinsics_payload = json.loads(Path(args.intrinsics_json).read_text(encoding="utf-8"))
    if isinstance(intrinsics_payload, dict):
        intrinsics_payload = intrinsics_payload.get("matrix")
    intrinsics = _matrix(intrinsics_payload, "intrinsics", (3, 3))
    poses = json.loads(Path(args.poses_json).read_text(encoding="utf-8"))
    if not isinstance(poses, dict):
        raise ValueError("poses JSON must map depth filenames to camera_to_world matrices")

    clouds = []
    used_frames = []
    for depth_path in sorted(Path(args.depth_dir).glob("depth_*.npy")):
        pose_payload = poses.get(depth_path.name)
        if pose_payload is None:
            raise ValueError(f"No camera_to_world pose for {depth_path.name}")
        cloud = backproject_depth_to_world(
            np.load(depth_path),
            intrinsics,
            _matrix(pose_payload, f"pose for {depth_path.name}", (4, 4)),
            stride=args.stride,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        if len(cloud):
            clouds.append(cloud)
            used_frames.append(depth_path.name)
    if not clouds:
        raise RuntimeError("No valid ground-truth points were reconstructed")
    points = np.concatenate(clouds, axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, points=points)
    manifest = {
        "artifact": str(output),
        "coordinate_frame": "camera_to_world_from_poses_json",
        "frames": used_frames,
        "points": len(points),
        "stride": args.stride,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
