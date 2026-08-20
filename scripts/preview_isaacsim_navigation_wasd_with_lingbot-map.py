#!/usr/bin/env python3
"""WASD navigation preview with LingBot-Map video->3D point-cloud reconstruction.

This is the dedicated LingBot-Map variant of ``preview_isaacsim_navigation_wasd.py``.
It drives the Go2 robot with WASD/QE keys inside an Isaac Sim scene and, for every
rendered head-camera frame, feeds that video stream into the LingBot-Map agent, which
reconstructs an accumulated world-frame 3D point cloud from the video. The point cloud
is streamed in real time to an in-browser 3D viewer (open the URL printed at startup),
so you can walk the scene with WASD and watch LingBot-Map build the 3D map live.

Pipeline (per frame):
    Isaac Sim renders head-camera RGB  ->  LingBot-Map agent (isolated CUDA venv)
    ->  accumulated global 3D point cloud  ->  RealtimePointCloudViewer (HTTP)  ->  browser.

Run with Isaac Sim's Python (the LingBot-Map model itself runs in its own venv):
    ~/isaacsim/python.sh scripts/preview_isaacsim_navigation_wasd_with_lingbot-map.py \
        --config configs/isaacsim_realtime_agent_internscenes.yaml --livestream

Controls:
    W: Forward   S: Backward   A: Left   D: Right
    Q: Turn Left  E: Turn Right   Space: Stop   P: Exit
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.common.types import FrameObservation  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.agent.execution.isaacsim_adapter import (  # noqa: E402
    IsaacSimExecutor,
    _GO2_CAMERA_OFFSET_M,
)
from agentic_memory_nav.agent.execution.safety_controller import SafetyController  # noqa: E402
from agentic_memory_nav.agent.lingbot.adapter import LingBotMapAdapter  # noqa: E402
from agentic_memory_nav.agent.lingbot.client import SubAgentDispatcher  # noqa: E402
from agentic_memory_nav.visualization.realtime_pointcloud_viewer import (  # noqa: E402
    RealtimePointCloudViewer,
)


# ---------------------------------------------------------------------------
# Keyboard control (same WASD model as the base preview; raw-mode terminal).
# ---------------------------------------------------------------------------
_key_state: Dict[str, bool] = {
    'w': False, 'a': False, 's': False, 'd': False,
    'q': False, 'e': False, ' ': False, 'p': False
}
_key_lock = threading.Lock()
_key_timestamps: Dict[str, float] = {}
_key_hold_duration = 0.15
_running = True


def _get_keyboard_input() -> tuple[float | None, float | None, float | None]:
    """Read current keyboard state and return a velocity command (vx, vy, wz)."""
    current_time = time.time()
    with _key_lock:
        for key in _key_state:
            if _key_state[key] and key in _key_timestamps:
                if current_time - _key_timestamps[key] > _key_hold_duration:
                    _key_state[key] = False
        keys = _key_state.copy()
    if keys['p']:
        return None, None, None  # exit
    vx, vy, wz = 0.0, 0.0, 0.0
    speed = 0.3  # m/s
    turn_speed = 0.5  # rad/s
    if keys['w']:
        vx += speed
    if keys['s']:
        vx -= speed
    if keys['a']:
        vy += speed
    if keys['d']:
        vy -= speed
    if keys['q']:
        wz += turn_speed
    if keys['e']:
        wz -= turn_speed
    if keys[' ']:
        vx, vy, wz = 0.0, 0.0, 0.0
    return vx, vy, wz


def _keyboard_listener() -> None:
    """Background thread that listens for keyboard input on a raw-mode terminal."""
    global _running
    try:
        import termios
        import tty
        import select

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while _running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not ready:
                    continue
                char = sys.stdin.read(1)
                if not char:
                    continue
                key = char.lower()
                now = time.time()
                with _key_lock:
                    if key in _key_state:
                        _key_state[key] = True
                        _key_timestamps[key] = now
                if key == 'p':
                    print("\n[P] pressed - exiting...")
                    _running = False
                    break
            with _key_lock:
                for key in _key_state:
                    _key_state[key] = False
                _key_timestamps.clear()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception as e:
        print(f"Keyboard listener error: {e}", file=sys.stderr)
        _running = False


def _as_vector3(value: object, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence of 3 numbers, got {type(value).__name__}")
    vector = tuple(float(item) for item in value)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {vector!r}")
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isaacsim_realtime_agent_internscenes.yaml"))
    parser.add_argument("--livestream", action="store_true", help="Enable Isaac Sim WebRTC livestream of the head camera.")
    parser.add_argument("--public-ip", default="127.0.0.1", help="Public IP advertised for the WebRTC livestream.")
    parser.add_argument("--lingbot-python", default=str(ROOT / ".lingbot-venv/bin/python"),
                        help="Python executable for the isolated real LingBot-Map worker.")
    parser.add_argument("--lingbot-checkpoint", default=str(ROOT / "external-lib/lingbot-map/models/lingbot-map/lingbot-map.pt"))
    parser.add_argument("--lingbot-image-size", type=int, default=518)
    parser.add_argument("--lingbot-keyframe-interval", type=int, default=1)
    parser.add_argument("--map-port", type=int, default=8091, help="Port for the in-browser 3D point-cloud viewer.")
    parser.add_argument("--no-map-view", action="store_true", help="Disable the in-browser 3D point-cloud viewer.")
    parser.add_argument("--demo-move", action="store_true",
                        help="Auto-drive the robot with a scripted velocity schedule (no keyboard), "
                             "logging the pose each phase, then exit. For non-interactive verification.")
    parser.add_argument("--demo-duration", type=float, default=18.0,
                        help="Seconds to run the --demo-move schedule (default 18s).")
    return parser.parse_args()


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


def _demo_velocity(
    elapsed: float,
    *,
    speed: float = 0.3,
    turn: float = 0.5,
) -> tuple[float, float, float, str]:
    """Scripted velocity schedule for --demo-move: a short, pose-revealing tour.

    Returns (vx, vy, wz, phase). The robot moves forward, strafes, turns, and
    stops in a repeating 12s pattern so the logged pose changes in a way that is
    unambiguous (forward / lateral / rotation / stopped).
    """
    t = elapsed % 12.0
    if t < 4.0:
        return speed, 0.0, 0.0, "forward"
    if t < 6.0:
        return 0.0, speed, 0.0, "strafe_right"
    if t < 8.0:
        return 0.0, 0.0, turn, "turn_left"
    if t < 10.0:
        return -speed, 0.0, 0.0, "backward"
    return 0.0, 0.0, 0.0, "stop"


def main() -> int:
    global _running

    args = parse_args()
    config = load_config(args.config)
    execution = config.section("execution")
    mapping_config = config.section("mapping")

    robot_start, robot_yaw_deg = _parse_robot_start_pose(execution.get("robot_start"), "robot_start")
    camera_fps = int(execution.get("camera_fps", 30))
    if not 30 <= camera_fps <= 60:
        raise ValueError(f"execution.camera_fps must be between 30 and 60, got {camera_fps}")

    livestream_args = None
    if args.livestream:
        livestream_args = [
            f"--/exts/omni.kit.livestream.app/primaryStream/publicIp={args.public_ip}",
            "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49100",
            "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47998",
            f"--/exts/omni.kit.livestream.app/primaryStream/targetFps={camera_fps}",
        ]
        print(f"Livestream: WebRTC signal at {args.public_ip}:49100 (stream port 47998)")

    run = ExperimentRun(ROOT / str(config.section("runtime").get("output_root", "outputs")), config.raw)
    head_dir = run.artifacts / "head_rgb"
    head_dir.mkdir(exist_ok=True)

    # --- Load the LingBot-Map agent (guaranteed open; fails fast otherwise) ---
    # The dispatcher launches the LingBot-Map model in the isolated venv and waits for
    # the "ready" signal before the simulation starts. Its predictor is the single
    # entry-point handed to the mapping adapter (video stream -> 3D point cloud).
    dispatcher = SubAgentDispatcher(
        lingbot_python=args.lingbot_python,
        lingbot_checkpoint=args.lingbot_checkpoint,
        lingbot_image_size=args.lingbot_image_size,
        lingbot_keyframe_interval=args.lingbot_keyframe_interval,
        enable_lingbot_map=True,
    )
    dispatcher.start()
    mapper = LingBotMapAdapter(
        checkpoint=Path(str(mapping_config.get("checkpoint", "lingbot-map.pt"))).expanduser(),
        device=str(mapping_config.get("device", "cuda")),
        predictor=dispatcher.map_predictor(),
        keyframe_interval=int(mapping_config.get("keyframe_interval", 1)),
        local_submap_frames=int(mapping_config.get("local_submap_frames", 30)),
        local_submap_stride=int(mapping_config.get("local_submap_stride", 1)),
        local_submap_stability_threshold_m=float(mapping_config.get("local_submap_stability_threshold_m", 0.50)),
        local_submap_max_points_per_frame=int(mapping_config.get("local_submap_max_points_per_frame", 20_000)),
    )
    
    map_viewer = None if args.no_map_view else RealtimePointCloudViewer(port=args.map_port)
    print(f"LingBot-Map 3D viewer: http://<this-host>:{args.map_port}/", flush=True)

    safety = SafetyController(
        max_speed=float(execution.get("max_speed", 0.5)),
        max_angular_speed=float(execution.get("max_angular_speed", 1.0)),
        max_timeout=float(execution.get("max_action_timeout", 15.0)),
    )
    executor = IsaacSimExecutor(
        scene=execution.get("scene"),
        safety=safety,
        max_speed=float(execution.get("max_speed", 0.5)),
        camera_resolution=(int(execution.get("camera_height", 192)), int(execution.get("camera_width", 256))),
        headless=bool(execution.get("headless", True)),
        livestream_args=livestream_args,
        window_resolution=(int(execution.get("stream_width", 1280)), int(execution.get("stream_height", 720))),
        robot_usd=execution.get("robot_usd"),
        bind_viewport_to_camera=bool(execution.get("bind_viewport_to_camera", False)),
        scene_up_axis=str(execution.get("scene_up_axis", "z")),
        go2_camera_orient=_as_vector3(execution.get("go2_camera_orient"), "go2_camera_orient"),
        head_scan_yaw_deg=float(execution.get("head_scan_yaw_deg", 0.0)),
        head_scan_pitch_deg=float(execution.get("head_scan_pitch_deg", 0.0)),
        head_scan_period_frames=int(execution.get("head_scan_period_frames", 1)),
        validate_initial_placement=bool(execution.get("validate_initial_placement", True)),
        initial_robot_position=robot_start,
        initial_robot_yaw_deg=robot_yaw_deg,
        camera_fps=camera_fps,
        camera_focal_length=float(execution.get("camera_focal_length", 12.0)),
        environment_planes=dict(execution.get("environment_planes", {})),
        robot_motion_mode=str(execution.get("robot_motion_mode", "kinematic")),
        light_rig=str(execution.get("light_rig", "gray_studio")),
    )

    collision = False
    frame_index = 0
    demo_poses: list[dict] = []
    demo_start = time.monotonic()
    demo_mode = getattr(args, "demo_move", False)
    print("\n=== WASD Control + LingBot-Map 3D Reconstruction ===")
    print("W: Forward  | S: Backward")
    print("A: Left     | D: Right")
    print("Q: Turn Left  | E: Turn Right")
    print("Space: Stop")
    print("P: Exit")
    if map_viewer is not None:
        print(f"\nViewer (camera + 3D map): http://<this-host>:{args.map_port}/  (left=live head camera, right=3D map)")
    if demo_mode:
        print(f"\n[DEMO] auto-driving for {args.demo_duration}s (no keyboard needed)...\n")
    else:
        print("\nRunning... press P to exit\n")

    try:
        executor.reset()
        mapper.start()
        if demo_mode:
            print("Demo move: auto-driving scripted velocity schedule.", flush=True)
        else:
            listener_thread = threading.Thread(target=_keyboard_listener, daemon=True)
            listener_thread.start()

        while _running:
            frame = executor.get_observation()
            Image.fromarray(frame.rgb).save(head_dir / f"frame_{frame_index:04d}.png")
            mapping = mapper.update(frame)  # video frame -> 3D point cloud (accumulated)
            state = executor.get_state()
            if map_viewer is not None:
                map_viewer.update(
                    mapping.global_pointcloud,
                    frame_id=frame.frame_id,
                    robot=state.position,
                    yaw=state.yaw,
                    rgb=frame.rgb,
                )
                if frame_index % 15 == 0:
                    print(f"[frame {frame_index:03d}] LingBot-Map points: {len(mapping.global_pointcloud)}", flush=True)

            if demo_mode:
                elapsed = time.monotonic() - demo_start
                vx, vy, wz, phase = _demo_velocity(
                    elapsed,
                    speed=float(execution.get("max_speed", 0.5)),
                    turn=float(execution.get("max_angular_speed", 1.0)),
                )
                if vx != 0.0 or vy != 0.0 or wz != 0.0:
                    feedback = executor.send_velocity_command(vx, vy, wz)
                    collision = collision or feedback.collision
                    if feedback.collision:
                        print(f"Collision blocked at frame {frame_index}: {feedback.reason}")
                # Log pose once per second for an unambiguous movement record.
                if int(elapsed) > len(demo_poses) - 1 or demo_poses and demo_poses[-1].get("phase") != phase:
                    pose = {"t": round(elapsed, 1), "phase": phase,
                            "pos": [round(v, 3) for v in state.position], "yaw": round(state.yaw, 3)}
                    demo_poses.append(pose)
                    print(f"[DEMO t={pose['t']:5.1f}s] {phase:12s} pos=[{pose['pos'][0]:.3f},{pose['pos'][1]:.3f},{pose['pos'][2]:.3f}] yaw={pose['yaw']:.3f}", flush=True)
                if elapsed >= args.demo_duration:
                    print(f"[DEMO] finished after {args.demo_duration}s", flush=True)
                    _running = False
            else:
                vx, vy, wz = _get_keyboard_input()
                if vx is None or vy is None or wz is None:
                    break
                if vx != 0.0 or vy != 0.0 or wz != 0.0:
                    feedback = executor.send_velocity_command(vx, vy, wz)
                    collision = collision or feedback.collision
                    if feedback.collision:
                        print(f"Collision blocked at frame {frame_index}: {feedback.reason}")

            time.sleep(1.0 / camera_fps)
            frame_index += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Summarize the demo movement (how far / how much the pose changed).
        movement_report = None
        if demo_mode and demo_poses:
            p0 = demo_poses[0]["pos"]
            p1 = demo_poses[-1]["pos"]
            moved = round(sum((a - b) ** 2 for a, b in zip(p0, p1)) ** 0.5, 3)
            yaw_delta = round(demo_poses[-1]["yaw"] - demo_poses[0]["yaw"], 3)
            movement_report = {
                "start": p0, "end": p1, "moved_m": moved, "yaw_delta": yaw_delta,
                "phases": [p["phase"] for p in demo_poses],
            }
            print(f"[DEMO] moved {moved} m (start={p0} end={p1}, yaw_delta={yaw_delta})", flush=True)
        run.write_json(
            "preview_summary.json",
            {
                "frames_rendered": frame_index,
                "collision_blocked": collision,
                "mode": "lingbot_map_wasd_demo" if demo_mode else "lingbot_map_wasd",
                "movement": movement_report,
            },
        )
        print(f"Preview artifacts: {run.path}", flush=True)
        if mapper.get_global_pointcloud().size:
            mapper.save_state(run.path / "lingbot_map")
        if map_viewer is not None:
            map_viewer.close()
        dispatcher.close()
        run.close()
        executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())