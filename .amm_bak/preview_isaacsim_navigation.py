#!/usr/bin/env python3
"""Render the configured Isaac Sim navigation scene without running an agent.

The preview creates the scene, dynamic-bound ground/ceiling planes, and kinematic
Go2, then optionally applies a short translation/turn motion with PhysX pre-checks.
Run with Isaac Sim's Python:
    ~/isaacsim/python.sh scripts/preview_isaacsim_navigation.py \
        --config configs/isaacsim_realtime_agent_internscenes.yaml --livestream
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.execution.isaacsim_adapter import IsaacSimExecutor  # noqa: E402
from agentic_memory_nav.execution.safety_controller import SafetyController  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isaacsim_realtime_agent_internscenes.yaml"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--motion", choices=("stationary", "translate_turn"), default="translate_turn")
    parser.add_argument("--livestream", action="store_true")
    parser.add_argument("--public-ip", default="127.0.0.1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("frames must be positive")
    config = load_config(args.config)
    execution = config.section("execution")
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
    overhead_dir = run.artifacts / "overhead_rgb"
    head_dir.mkdir(exist_ok=True)
    overhead_dir.mkdir(exist_ok=True)
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
        overhead_camera_position=(
            tuple(float(value) for value in execution["overhead_camera_position"])
            if execution.get("overhead_camera_position")
            else None
        ),
        overhead_camera_orient=(
            tuple(float(value) for value in execution["overhead_camera_orient"])
            if execution.get("overhead_camera_orient") is not None
            else None
        ),
        go2_camera_orient=(
            tuple(float(value) for value in execution["go2_camera_orient"])
            if execution.get("go2_camera_orient") is not None
            else None
        ),
        head_scan_yaw_deg=float(execution.get("head_scan_yaw_deg", 0.0)),
        head_scan_pitch_deg=float(execution.get("head_scan_pitch_deg", 0.0)),
        head_scan_period_frames=int(execution.get("head_scan_period_frames", 1)),
        validate_initial_placement=bool(execution.get("validate_initial_placement", True)),
        initial_robot_position=(
            tuple(float(value) for value in execution["robot_start"])
            if execution.get("robot_start")
            else None
        ),
        camera_fps=camera_fps,
        livestream_camera=str(execution.get("livestream_camera", "overhead")),
        environment_planes=dict(execution.get("environment_planes", {})),
        robot_motion_mode=str(execution.get("robot_motion_mode", "kinematic")),
        light_rig=str(execution.get("light_rig", "gray_studio")),
    )

    collision = False
    try:
        executor.reset()
        for frame_index in range(args.frames):
            frame = executor.get_observation()
            Image.fromarray(frame.rgb).save(head_dir / f"frame_{frame_index:04d}.png")
            overhead = executor.get_overhead_rgb()
            if overhead is not None:
                Image.fromarray(overhead).save(overhead_dir / f"frame_{frame_index:04d}.png")
            if args.motion == "translate_turn" and frame_index % 3 == 0:
                feedback = executor.send_velocity_command(0.12, 0.0, 0.35)
                collision = collision or feedback.collision
                if feedback.collision:
                    print(f"Collision blocked at frame {frame_index}: {feedback.message}")
                    break
            time.sleep(1.0 / camera_fps)
    finally:
        run.write_json(
            "preview_summary.json",
            {"frames_requested": args.frames, "motion": args.motion, "collision_blocked": collision},
        )
        print(f"Preview artifacts: {run.path}", flush=True)
        run.close()
        if args.livestream and bool(execution.get("keep_streaming", False)):
            print(
                "Preview complete; WebRTC stream remains active. Press Ctrl+C to stop.",
                flush=True,
            )
            executor.stream_until_interrupted()
        executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
