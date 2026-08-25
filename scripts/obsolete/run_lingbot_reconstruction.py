#!/usr/bin/env python3
"""Export LingBot-Map world points to the project's standard NPZ artifact.

Run this script with the isolated Python 3.10 LingBot environment, for example:
``.lingbot-venv/bin/python scripts/run_lingbot_reconstruction.py ...``.
The output uses the same ``points: float32 (N, 3)`` NPZ contract as
``scripts/evaluate_pointcloud.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINGBOT_ROOT = ROOT / "external-lib" / "lingbot-map"
sys.path.insert(0, str(LINGBOT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--first-k", type=int)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--num-scale-frames", type=int, default=1)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    args = parser.parse_args()

    import torch
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    if not torch.cuda.is_available():
        raise RuntimeError("LingBot reconstruction requires a CUDA-capable PyTorch runtime")
    image_paths = sorted(Path(args.image_folder).glob("*.png"))
    image_paths.extend(sorted(Path(args.image_folder).glob("*.jpg")))
    if args.first_k is not None:
        image_paths = image_paths[: args.first_k]
    if not image_paths:
        raise FileNotFoundError(f"No PNG/JPG frames found under {args.image_folder}")

    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode="crop",
        image_size=args.image_size,
        patch_size=14,
    )
    model = GCTStream(
        img_size=args.image_size,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=args.camera_num_iterations,
        enable_point=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model = model.to("cuda").eval()
    model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = model.inference_streaming(
            images.to("cuda"),
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=torch.device("cpu"),
        )

    world_points = predictions["world_points"].squeeze(0).numpy().reshape(-1, 3)
    confidence = predictions["world_points_conf"].squeeze(0).numpy().reshape(-1)
    valid = (confidence >= args.confidence_threshold) & torch.isfinite(
        torch.from_numpy(world_points)
    ).all(dim=1).numpy()
    points = world_points[valid].astype("float32", copy=False)
    if len(points) == 0:
        raise RuntimeError("No world points survived the configured confidence threshold")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(output, points=points, confidence=confidence[valid].astype("float32"))
    manifest = {
        "artifact": str(output),
        "coordinate_frame": "lingbot_world",
        "frames": [path.name for path in image_paths],
        "input_frames": len(image_paths),
        "points": len(points),
        "confidence_threshold": args.confidence_threshold,
        "missing_checkpoint_keys": len(missing),
        "unexpected_checkpoint_keys": len(unexpected),
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
