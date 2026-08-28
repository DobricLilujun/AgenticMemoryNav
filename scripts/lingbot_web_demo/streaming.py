"""Live, frame-by-frame streaming 3D reconstruction using LingBot-Map (GCTStream).

Standalone (no Isaac Sim, no ground-truth pose) variant of the sub-agent runtime in
``agentic_memory_nav.agent.lingbot.agent``: it feeds a video into GCTStream one frame
at a time (after an initial batch of "scale frames"), using the model's own KV-cache
causal streaming path, and yields a :class:`FrameResult` as soon as each frame is
computed -- suitable for pushing over a WebSocket in real time.

Note: unlike ``agentic_memory_nav.agent.lingbot.agent.LingBotMapAgent``, this module
does **not** enable the model's optional point head (``enable_point=True``). The
public ``lingbot-map.pt`` checkpoint does not contain point-head weights (verified by
loading it with ``enable_point=True``: 62 ``point_head.*`` parameters come back
missing/uninitialized), so that path would silently produce garbage geometry. Points
here are instead computed by unprojecting the (checkpoint-trained) *depth* head
through the estimated camera pose -- the same path the reference ``demo.py`` viewer
uses when ``enable_point`` is left at its default.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Must be set before `import torch`; see external-lib/lingbot-map/demo.py for rationale.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from torchvision import transforms as TF  # noqa: E402

_LINGBOT_MAP_ROOT = Path(__file__).resolve().parents[2] / "external-lib" / "lingbot-map"
if str(_LINGBOT_MAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_LINGBOT_MAP_ROOT))

from lingbot_map.models.gct_stream import GCTStream  # type: ignore[import-not-found]  # noqa: E402
from lingbot_map.utils.geometry import (  # type: ignore[import-not-found]  # noqa: E402
    closed_form_inverse_se3_general,
    unproject_depth_map_to_point_map,
)
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore[import-not-found]  # noqa: E402


@dataclass
class FrameResult:
    frame_index: int
    timestamp: float
    camera_to_world: np.ndarray  # (4, 4) float32, row-major
    points: np.ndarray  # (N, 3) float32
    colors: np.ndarray  # (N, 3) uint8


class LiveGCTStreamer:
    """Loads GCTStream once and replays a video through it, frame by frame."""

    def __init__(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        image_size: int = 518,
        patch_size: int = 14,
        num_scale_frames: int = 8,
        keyframe_interval: int = 1,
        max_frame_num: int = 1024,
        kv_cache_sliding_window: int = 64,
        camera_num_iterations: int = 1,
        conf_threshold: float = 1.5,
        point_stride: int = 8,
        device: str | None = None,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_scale_frames = max(1, num_scale_frames)
        self.keyframe_interval = max(1, keyframe_interval)
        self.conf_threshold = conf_threshold
        self.point_stride = max(1, point_stride)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float32
        )

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"LingBot-Map checkpoint not found: {checkpoint_path}")

        print("Loading LingBot-Map checkpoint...", flush=True)
        self.model = GCTStream(
            img_size=image_size,
            patch_size=patch_size,
            enable_3d_rope=True,
            max_frame_num=max_frame_num,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=self.num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=True,
            camera_num_iterations=camera_num_iterations,
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        self.model = self.model.to(self.device).eval()
        aggregator = getattr(self.model, "aggregator", None)
        if self.dtype != torch.float32 and aggregator is not None:
            self.model.aggregator = aggregator.to(dtype=self.dtype)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Resize/crop one BGR frame to the model's input grid.

        Returns the preprocessed tensor *and* the RGB array at the exact same
        (H, W) as the model's depth/point output grid, so pixel (i, j) in the
        returned array is the color of point (i, j) in the reconstructed cloud.
        """
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        new_width = self.image_size
        new_height = round(height * (new_width / width) / self.patch_size) * self.patch_size
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        if new_height > self.image_size:
            top = (new_height - self.image_size) // 2
            image = image.crop((0, top, new_width, top + self.image_size))
        rgb = np.asarray(image, dtype=np.uint8)
        tensor = TF.ToTensor()(image)
        return tensor, rgb

    def _load_video(
        self, video_path: str | os.PathLike[str], target_fps: float | None
    ) -> tuple[torch.Tensor, np.ndarray, list[float]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = target_fps or src_fps
        interval = max(1, round(src_fps / fps))
        tensors: list[torch.Tensor] = []
        rgbs: list[np.ndarray] = []
        timestamps: list[float] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                tensor, rgb = self._preprocess(frame)
                tensors.append(tensor)
                rgbs.append(rgb)
                timestamps.append(idx / src_fps)
            idx += 1
        cap.release()
        if not tensors:
            raise RuntimeError(f"No frames decoded from video: {video_path}")
        return torch.stack(tensors), np.stack(rgbs), timestamps

    def _frame_result(
        self,
        output: dict[str, torch.Tensor],
        slot: int,
        rgb: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> FrameResult:
        pose_enc = output["pose_enc"][:, slot : slot + 1].detach().float()
        depth = output["depth"][:, slot : slot + 1].detach().float()
        depth_conf = output["depth_conf"][:, slot : slot + 1].detach().float()
        height, width = int(depth.shape[2]), int(depth.shape[3])
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(height, width), build_intrinsics=True
        )
        camera_to_world = closed_form_inverse_se3_general(extrinsics)[0, 0].cpu().numpy()
        depth_np = depth[0, 0, ..., 0].cpu().numpy()
        conf_np = depth_conf[0, 0].cpu().numpy()
        world_points = unproject_depth_map_to_point_map(
            depth_np[None, ..., None],
            extrinsics[0].cpu().numpy(),
            intrinsics[0].cpu().numpy(),
        )[0]

        stride = self.point_stride
        sampled_depth = depth_np[::stride, ::stride]
        sampled_conf = conf_np[::stride, ::stride]
        mask = (sampled_conf > self.conf_threshold) & (sampled_depth > 1e-4)
        points = world_points[::stride, ::stride][mask].astype(np.float32, copy=False)
        colors = rgb[::stride, ::stride][mask].astype(np.uint8, copy=False)
        return FrameResult(
            frame_index=frame_index,
            timestamp=timestamp,
            camera_to_world=np.ascontiguousarray(camera_to_world, dtype=np.float32),
            points=points,
            colors=colors,
        )

    def stream(
        self, video_path: str | os.PathLike[str], target_fps: float | None = None
    ) -> Iterator[FrameResult]:
        """Yield one :class:`FrameResult` per video frame, as soon as it is computed."""
        tensors, rgbs, timestamps = self._load_video(video_path, target_fps)
        num_frames = tensors.shape[0]
        scale_frames = min(self.num_scale_frames, num_frames)
        self.model.clean_kv_cache()
        autocast_enabled = self.device.type == "cuda"

        scale_block = tensors[:scale_frames].unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=autocast_enabled
        ):
            output = self.model.forward(
                scale_block,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
        for slot in range(scale_frames):
            yield self._frame_result(output, slot, rgbs[slot], timestamps[slot], slot)
        del output

        for i in range(scale_frames, num_frames):
            frame_tensor = tensors[i : i + 1].unsqueeze(0).to(self.device)
            is_keyframe = (i - scale_frames) % self.keyframe_interval == 0
            if not is_keyframe:
                self.model._set_skip_append(True)
            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=self.dtype, enabled=autocast_enabled
            ):
                output = self.model.forward(
                    frame_tensor,
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            if not is_keyframe:
                self.model._set_skip_append(False)
            yield self._frame_result(output, 0, rgbs[i], timestamps[i], i)
            del output

        if self.device.type == "cuda":
            torch.cuda.synchronize()
