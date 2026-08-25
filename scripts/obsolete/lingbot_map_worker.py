#!/usr/bin/env python3
"""Persistent LingBot-Map streaming worker for the Isaac Sim preview."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch  # type: ignore[import-not-found]
from PIL import Image, ImageOps
from torchvision import transforms as TF  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external-lib" / "lingbot-map"))

from lingbot_map.models.gct_stream import GCTStream  # type: ignore[import-not-found]  # noqa: E402
from lingbot_map.utils.geometry import closed_form_inverse_se3_general  # type: ignore[import-not-found]  # noqa: E402
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore[import-not-found]  # noqa: E402


_ISAAC_TO_OPENCV = np.eye(4, dtype=np.float32)
_ISAAC_TO_OPENCV[:3, :3] = np.diag([1.0, -1.0, -1.0])


def _encode_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return base64.b64encode(zlib.compress(buffer.getvalue(), level=1)).decode("ascii")


def _decode_image(value: str) -> Image.Image:
    return Image.open(io.BytesIO(zlib.decompress(base64.b64decode(value)))).convert("RGB")


def _preprocess(image: Image.Image, image_size: int, patch_size: int) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    tensor = TF.ToTensor()(image)
    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        tensor = tensor[:, start_y : start_y + image_size, :]
    return tensor


class LingBotRuntime:
    """One GCTStream instance with the same streaming calls as LingBot's demo."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.image_size = args.image_size
        self.patch_size = args.patch_size
        self.num_scale_frames = max(1, args.num_scale_frames)
        self.keyframe_interval = max(1, args.keyframe_interval)
        self.frame_index = 0
        self.alignment: np.ndarray | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float32
        )
        with contextlib.redirect_stdout(sys.stderr):
            self.model = GCTStream(
                img_size=self.image_size,
                patch_size=self.patch_size,
                enable_3d_rope=True,
                max_frame_num=args.max_frame_num,
                kv_cache_sliding_window=args.kv_cache_sliding_window,
                kv_cache_scale_frames=self.num_scale_frames,
                kv_cache_cross_frame_special=True,
                kv_cache_include_scale_frames=True,
                use_sdpa=True,
                camera_num_iterations=args.camera_num_iterations,
                enable_point=True,
            )
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                f"LingBot checkpoint: missing={len(missing)} unexpected={len(unexpected)}",
                file=sys.stderr,
                flush=True,
            )
        self.model = self.model.to(self.device).eval()
        if self.dtype != torch.float32 and getattr(self.model, "aggregator", None) is not None:
            self.model.aggregator = self.model.aggregator.to(dtype=self.dtype)

    def predict(self, request: dict[str, object]) -> dict[str, str | int | float]:
        image = _decode_image(str(request["image"]))
        tensor = _preprocess(image, self.image_size, self.patch_size).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=self.device.type == "cuda"
        ):
            if self.frame_index == 0:
                output = self.model.forward(
                    tensor,
                    num_frame_for_scale=1,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            else:
                is_keyframe = (self.frame_index - 1) % self.keyframe_interval == 0
                if not is_keyframe:
                    self.model._set_skip_append(True)
                output = self.model.forward(
                    tensor,
                    num_frame_for_scale=self.num_scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
                if not is_keyframe:
                    self.model._set_skip_append(False)

        pose_enc = output["pose_enc"].detach().float()
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw=(int(tensor.shape[-2]), int(tensor.shape[-1])),
            build_intrinsics=True,
        )
        model_c2w = closed_form_inverse_se3_general(extrinsics)[0, 0].detach().cpu().numpy()
        isaac_c2w = np.asarray(request["isaac_c2w"], dtype=np.float32)
        if self.alignment is None:
            self.alignment = isaac_c2w @ _ISAAC_TO_OPENCV @ np.linalg.inv(model_c2w)
        aligned_c2w = self.alignment @ model_c2w

        depth = output["depth"][0, 0].detach().float().cpu().numpy().squeeze(-1)
        confidence = output["depth_conf"][0, 0].detach().float().cpu().numpy()
        world_points = output["world_points"][0, 0].detach().float().cpu().numpy()
        world_points = world_points.reshape(-1, 3) @ self.alignment[:3, :3].T
        world_points += self.alignment[:3, 3]
        point_confidence = output["world_points_conf"][0, 0].detach().float().cpu().numpy().reshape(-1)
        valid = np.isfinite(world_points).all(axis=1) & (point_confidence >= 1.5)
        points = world_points[valid].astype(np.float32, copy=False)
        self.frame_index += 1
        return {
            "frame_id": str(request["frame_id"]),
            "camera_pose": json.dumps({
                "position": request["robot_position"],
                "yaw": request["robot_yaw"],
                "camera_to_world": aligned_c2w.tolist(),
            }),
            "depth": _encode_array(depth),
            "confidence": _encode_array(confidence),
            "intrinsics": _encode_array(intrinsics[0, 0].detach().float().cpu().numpy()),
            "global_pointcloud": _encode_array(points),
            "point_count": len(points),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--num-scale-frames", type=int, default=1)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    runtime = LingBotRuntime(_parse_args())
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = runtime.predict(request)
            print(json.dumps({"ok": True, "result": response}), flush=True)
        except Exception as error:
            print(json.dumps({"ok": False, "error": str(error)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())