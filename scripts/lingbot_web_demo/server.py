"""Local web demo: real-time LingBot-Map streaming reconstruction from a video.

Serves a two-pane page (left: source video, right: live 3D point cloud rendered
with Three.js) from a single aiohttp app. A WebSocket pushes one binary message
per video frame as soon as LingBot-Map computes it -- see ``static/app.js`` for
the wire format.

Usage (must run inside the isolated LingBot venv -- this repo's ``.lingbot-venv``,
which has the matching torch/scipy/lingbot-map install; the base env lacks scipy):

    .lingbot-venv/bin/python3 scripts/lingbot_web_demo/server.py

Then open http://localhost:8000/ in a browser. Defaults point at the checkpoint and
example video already present in this repo; override with --checkpoint/--video.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import struct
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
from streaming import FrameResult, LiveGCTStreamer  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "external-lib/lingbot-map/models/lingbot-map/lingbot-map.pt"
_DEFAULT_VIDEO = _REPO_ROOT / "outputs/lingbot_demo/courthouse_demo.mp4"
_STATIC_DIR = Path(__file__).resolve().parent / "static"

# One reconstruction session (and its GPU-resident KV cache) at a time.
_session_lock = threading.Lock()


def _encode_frame(result: FrameResult) -> bytes:
    """Pack one frame's result into a compact binary WebSocket message.

    Layout: uint32 frame_index, float32 timestamp, uint32 point_count,
    16x float32 camera_to_world (row-major), then point_count x (3x float32
    xyz), then point_count x (3x uint8 rgb).
    """
    header = struct.pack("<IfI", result.frame_index, float(result.timestamp), len(result.points))
    c2w_bytes = np.ascontiguousarray(result.camera_to_world, dtype="<f4").tobytes()
    points_bytes = np.ascontiguousarray(result.points, dtype="<f4").tobytes()
    colors_bytes = np.ascontiguousarray(result.colors, dtype="<u1").tobytes()
    return header + c2w_bytes + points_bytes + colors_bytes


def _produce(
    streamer: LiveGCTStreamer,
    video_path: Path,
    fps: float | None,
    result_queue: "queue.Queue[tuple[str, Any]]",
    stop_event: threading.Event,
) -> None:
    """Run in a background thread: feed the model and enqueue results as they land."""
    try:
        for result in streamer.stream(video_path, target_fps=fps):
            if stop_event.is_set():
                return
            result_queue.put(("frame", result))
    except Exception as exc:  # noqa: BLE001 - surface to the client instead of crashing
        result_queue.put(("error", str(exc)))
        return
    result_queue.put(("done", None))


async def websocket_handler(request: web.Request) -> web.StreamResponse:
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)

    if not _session_lock.acquire(blocking=False):
        await ws.send_json(
            {"type": "busy", "message": "A reconstruction session is already running; try again shortly."}
        )
        await ws.close()
        return ws

    streamer: LiveGCTStreamer = request.app["streamer"]
    video_path: Path = request.app["video_path"]
    fps: float | None = request.app["fps"]
    result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_produce, args=(streamer, video_path, fps, result_queue, stop_event), daemon=True
    )
    worker.start()

    await ws.send_json({"type": "start", "video_url": "/video/source.mp4"})
    try:
        while True:
            kind, payload = await asyncio.to_thread(result_queue.get)
            if kind == "frame":
                await ws.send_bytes(_encode_frame(payload))
            elif kind == "error":
                await ws.send_json({"type": "error", "message": payload})
                break
            else:
                await ws.send_json({"type": "end"})
                break
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        _session_lock.release()
        if not ws.closed:
            await ws.close()
    return ws


def build_app(streamer: LiveGCTStreamer, video_path: Path, fps: float | None) -> web.Application:
    app = web.Application()
    app["streamer"] = streamer
    app["video_path"] = video_path
    app["fps"] = fps

    async def index(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(_STATIC_DIR / "index.html")

    async def serve_video(_request: web.Request) -> web.FileResponse:
        return web.FileResponse(video_path)

    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/video/source.mp4", serve_video)
    app.router.add_static("/", _STATIC_DIR)
    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--video", type=Path, default=_DEFAULT_VIDEO)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=10.0, help="Frame sampling rate for streaming")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--conf-threshold", type=float, default=1.5)
    parser.add_argument("--point-stride", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    streamer = LiveGCTStreamer(
        args.checkpoint,
        image_size=args.image_size,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        conf_threshold=args.conf_threshold,
        point_stride=args.point_stride,
    )
    app = build_app(streamer, args.video, args.fps)
    print(f"LingBot-Map web demo ready: http://{args.host}:{args.port}/", flush=True)
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
