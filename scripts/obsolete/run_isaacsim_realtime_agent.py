#!/usr/bin/env python3
"""Live closed loop: real Isaac Sim RGB+pose -> MemoryAgent -> NavigationAgent -> Isaac Sim.

Every step is real-time and in-process:
  1. `IsaacSimExecutor.get_observation()` renders one real Isaac Sim frame (RGB, depth,
     camera/robot pose) — no recorded or synthetic data.
  2. `MemoryAgent.ingest_frame()` extracts/builds the scene graph and knowledge memory
     from that single frame.
  3. `NavigationAgent.decide()` reads the updated graph/memory and returns the next
     navigation action.
  4. `IsaacSimExecutor.send_waypoint()` executes that action back in the running
     Isaac Sim world, and the loop continues with the resulting new frame.

Must be launched with Isaac Sim's bundled Python, e.g.:
    ~/isaacsim/python.sh scripts/run_isaacsim_realtime_agent.py \\
        --config configs/isaacsim_realtime_agent.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.agent.realtime.memory_agent import MemoryAgent  # noqa: E402
from agentic_memory_nav.agent.realtime.navigation_agent import NavigationAgent  # noqa: E402
from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.common.logging import configure_logging  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.agent.execution.isaacsim_adapter import IsaacSimExecutor  # noqa: E402
from agentic_memory_nav.agent.execution.safety_controller import SafetyController  # noqa: E402
from agentic_memory_nav.agent.orchestration.pipeline import NavigationPipeline  # noqa: E402


def _as_vector3(value: object, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence of 3 numbers, got {type(value).__name__}: {value!r}")
    vector = tuple(float(item) for item in value)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {vector!r}")
    return vector


def _parse_robot_start_pose(value: object, name: str) -> tuple[tuple[float, float, float] | None, float]:
    if value is None:
        return None, 0.0
    if isinstance(value, dict):
        position = value.get("position")
        if position is None:
            raise ValueError(f"{name} must define a 'position' list/tuple when using a mapping")
        yaw_deg = float(value.get("yaw_deg", value.get("yaw", 0.0)))
        return _as_vector3(position, f"{name}.position"), yaw_deg
    if isinstance(value, (list, tuple, np.ndarray)):
        sequence = tuple(float(item) for item in value)
        if len(sequence) == 3:
            return sequence, 0.0
        if len(sequence) == 4:
            return (sequence[0], sequence[1], sequence[2]), float(sequence[3])
        raise ValueError(f"{name} must contain either 3 values or 4 values [x, y, z, yaw_deg]")
    raise ValueError(f"{name} must be a 3D position, a 4D pose [x, y, z, yaw_deg], or a mapping")


class LiveFrameServer:
    """Plain-HTTP live viewer for the rendered frames, refreshed every simulation step.

    Isaac Sim's WebRTC livestream needs the proprietary Omniverse Streaming Client; this
    serves the same real-time frames as plain JPEG over HTTP so any browser can watch.
    """

    def __init__(self, host: str, port: int) -> None:
        self._lock = threading.Lock()
        self._jpeg = b""
        holder = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/latest.jpg":
                    with holder._lock:
                        payload = holder._jpeg
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    body = (
                        b"<html><body style='margin:0;background:#111'>"
                        b"<img src='/latest.jpg' style='width:100%' "
                        b"onload=\"setTimeout(()=>{this.src='/latest.jpg?'+Date.now()}, 200)\">"
                        b"</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update(self, rgb: np.ndarray) -> None:
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)
        with self._lock:
            self._jpeg = buffer.getvalue()

    def close(self) -> None:
        self._server.shutdown()


def _render_video(rgb_dir: Path, video_path: Path, fps: int = 4) -> Path | None:
    """Assemble the per-step rendered frames into an MP4 so they can be viewed without a display."""
    if not any(rgb_dir.glob("frame_*.png")):
        return None
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(rgb_dir / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isaacsim_realtime_agent.yaml"))
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--view-port", type=int, default=8090)
    parser.add_argument("--no-view", action="store_true", help="disable the live HTTP viewer")
    parser.add_argument(
        "--livestream", action="store_true", help="serve the run over Omniverse WebRTC livestream"
    )
    parser.add_argument("--public-ip", default=None, help="override auto-detected public IP")
    return parser.parse_args()


def _resolve_public_ip(override: str | None) -> str:
    if override:
        return override
    from urllib.request import urlopen

    # Force IPv4: dual-stack DNS for ifconfig.me-style services often returns an
    # IPv6 address, which most WebRTC clients aren't configured to dial.
    with urlopen("https://api4.ipify.org", timeout=10) as response:
        return response.read().decode().strip()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    runtime = config.section("runtime")
    mapping_config = config.section("mapping")
    perception_config = config.section("perception")
    execution = config.section("execution")
    robot_start, robot_yaw_deg = _parse_robot_start_pose(execution.get("robot_start"), "robot_start")

    instruction = args.instruction or str(runtime.get("instruction", "Find the red cup"))
    max_steps = int(args.steps or runtime.get("max_steps", 40))
    output_root = ROOT / str(runtime.get("output_root", "outputs"))

    run = ExperimentRun(output_root, config.raw)
    logger = configure_logging(run.path / "logs.jsonl", verbose=False)
    rgb_dir = run.artifacts / "rgb"
    rgb_dir.mkdir(exist_ok=True)

    live_view = None
    if not args.no_view:
        live_view = LiveFrameServer("0.0.0.0", args.view_port)
        print(f"Live view: http://<this-host>:{args.view_port}/ (updates every step)")

    # Reuse the pipeline's mapper/perception factories so this script and the offline
    # pipeline stay consistent about which mock/real backends `mapping.backend` and
    # `perception.backend` select.
    mapper = NavigationPipeline._build_mapper(mapping_config)
    perception = NavigationPipeline._build_perception(perception_config)
    memory_agent = MemoryAgent(run.path, mapper=mapper, perception=perception)
    navigation_agent = NavigationAgent(instruction)

    safety = SafetyController(
        max_speed=float(execution.get("max_speed", 0.5)),
        max_angular_speed=float(execution.get("max_angular_speed", 1.0)),
        max_timeout=float(execution.get("max_action_timeout", 15.0)),
    )
    livestream_args = None
    window_resolution = None
    camera_fps = int(execution.get("camera_fps", 30))
    if not 30 <= camera_fps <= 60:
        raise ValueError(f"execution.camera_fps must be between 30 and 60, got {camera_fps}")
    if args.livestream:
        public_ip = _resolve_public_ip(args.public_ip)
        livestream_args = [
            f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={public_ip}",
            "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100",
            "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998",
            f"--/exts/omni.kit.livestream.app/primaryStream/targetFps={camera_fps}",
        ]
        # The WebRTC client negotiates a max frame size on connect; a mismatched
        # window/render resolution makes the plugin drop every frame.
        window_resolution = (
            int(execution.get("stream_width", 1280)),
            int(execution.get("stream_height", 720)),
        )
        print(f"Livestream: WebRTC signal at {public_ip}:49100 (stream port 47998)")
        print(f"Livestream resolution: {window_resolution[0]}x{window_resolution[1]}")
        print(f"Camera and livestream target FPS: {camera_fps}")

    executor = IsaacSimExecutor(
        scene=execution.get("scene"),
        safety=safety,
        max_speed=float(execution.get("max_speed", 0.5)),
        camera_resolution=(
            int(execution.get("camera_height", 192)),
            int(execution.get("camera_width", 256)),
        ),
        headless=bool(execution.get("headless", True)),
        livestream_args=livestream_args,
        window_resolution=window_resolution,
        robot_usd=execution.get("robot_usd"),
        bind_viewport_to_camera=bool(execution.get("bind_viewport_to_camera", False)),
        scene_up_axis=str(execution.get("scene_up_axis", "z")),
        go2_camera_orient=_as_vector3(execution.get("go2_camera_orient"), "go2_camera_orient"),
        head_scan_yaw_deg=float(execution.get("head_scan_yaw_deg", 0.0)),
        head_scan_pitch_deg=float(execution.get("head_scan_pitch_deg", 0.0)),
        head_scan_period_frames=int(execution.get("head_scan_period_frames", 1)),
        validate_initial_placement=bool(execution.get("validate_initial_placement", False)),
        initial_robot_position=robot_start,
        initial_robot_yaw_deg=robot_yaw_deg,
        camera_fps=camera_fps,
        camera_focal_length=float(execution.get("camera_focal_length", 12.0)),
        environment_planes=dict(execution.get("environment_planes", {})),
        robot_motion_mode=str(execution.get("robot_motion_mode", "kinematic")),
        light_rig=str(execution.get("light_rig", "gray_studio")),
    )

    objectnav = config.section("objectnav")
    if objectnav.get("target_name"):
        # A real, physically-simulated target the VLM must actually observe to
        # succeed — not the InteriorAgent dataset, which is not downloaded here.
        executor.spawn_object(
            name=str(objectnav["target_name"]),
            position=tuple(float(v) for v in objectnav.get("target_position", (1.5, 0.0, 0.15))),
            color=tuple(float(v) for v in objectnav.get("target_color", (1.0, 0.0, 0.0))),
            scale=float(objectnav.get("target_scale", 0.15)),
        )

    success = False
    step_results = []
    try:
        executor.reset()
        # Head lens is static by default; arm a short look-around only when the
        # scene graph gains new nodes (i.e. the VLM actually detected something new).
        previous_graph_nodes = 0
        for step in range(max_steps):
            step_start = time.perf_counter()
            frame = executor.get_observation()  # live Isaac Sim render: real RGB + pose
            # This host has no display/X server; persist each rendered frame so the
            # real-time render can be watched afterwards as a video.
            Image.fromarray(frame.rgb).save(rgb_dir / f"frame_{step:04d}.png")
            if live_view is not None:
                live_view.update(frame.rgb)
            snapshot = memory_agent.ingest_frame(frame)
            robot_pose = executor.get_state()
            # Arm a look-around only when this frame added new scene-graph nodes,
            # so the camera stays still in steady state and turns on demand.
            if snapshot.graph_nodes > previous_graph_nodes:
                executor.trigger_head_scan(
                    int(execution.get("head_scan_period_frames", 24))
                )
            previous_graph_nodes = snapshot.graph_nodes
            plan = navigation_agent.decide(
                robot_pose,
                memory_agent.graph,
                memory_agent.memory,
                replan_reason=f"new_rgb_frame:{frame.frame_id}",
            )
            feedback = executor.send_waypoint(plan.action.waypoint, plan.action)
            step_latency_s = time.perf_counter() - step_start

            record = {
                "step": step,
                "frame_id": frame.frame_id,
                "action_type": plan.action.action_type.value,
                "waypoint": plan.action.waypoint,
                "confidence": plan.confidence,
                "graph_nodes": snapshot.graph_nodes,
                "graph_edges": snapshot.graph_edges,
                "knowledge_facts_created": snapshot.knowledge_facts_created,
                "feedback_success": feedback.success,
                "collision": feedback.collision,
                "robot_position": feedback.state.position,
                "latency_s": step_latency_s,
            }
            step_results.append(record)
            run.append_trajectory(record)
            logger.info(
                "realtime agent step",
                extra={"fields": record},
            )
            if feedback.collision:
                break
            if plan.action.action_type.value == "navigate" and feedback.success:
                success = True
                break
    finally:
        summary = {
            "run_id": run.run_id,
            "instruction": instruction,
            "steps_taken": len(step_results),
            "success": success,
            "final_graph_nodes": step_results[-1]["graph_nodes"] if step_results else 0,
            "final_graph_edges": step_results[-1]["graph_edges"] if step_results else 0,
        }
        run.write_json("realtime_summary.json", summary)
        print(json.dumps(summary, indent=2))
        print(f"Run artifacts: {run.path}")
        video_path = _render_video(rgb_dir, run.path / "realtime_render.mp4")
        if video_path is not None:
            print(f"Real-time render video: {video_path}")
        if live_view is not None:
            live_view.close()
        memory_agent.close()
        run.close()
        if args.livestream and bool(execution.get("keep_streaming", False)):
            print(
                "Experiment complete; WebRTC stream remains active. Press Ctrl+C to stop.",
                flush=True,
            )
            executor.stream_until_interrupted()
        # Isaac Sim's executor.close() terminates the SimulationApp/process; do it last.
        executor.close()

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
