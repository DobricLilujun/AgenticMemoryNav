#!/usr/bin/env python3
"""Open an Isaac Sim scene and render frames without interactive actions.

This is a lightweight preview script: it loads the configured scene, spawns the
robot at the configured (or scene.json-derived) pose, and continuously renders
head-camera frames until the user presses Ctrl+C. No keyboard action control is
performed, so it is useful for quickly verifying lighting, initial placement,
and camera framing.

Run with Isaac Sim's Python:
    ~/isaacsim/python.sh scripts/preview_isaacsim_scene_only.py \
        --config configs/isaacsim_realtime_agent_internscenes.yaml --livestream
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.agent.execution.isaacsim_adapter import IsaacSimExecutor  # noqa: E402
from agentic_memory_nav.agent.execution.safety_controller import SafetyController  # noqa: E402
from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402


def _as_vector3(value: object, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(
            f"{name} must be a sequence of 3 numbers, got {type(value).__name__}: {value!r}"
        )
    vector = tuple(float(item) for item in value)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {vector!r}")
    return vector


def _parse_robot_start_pose(
    value: object, name: str
) -> tuple[tuple[float, float, float] | None, float]:
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


def _load_scene_go2_placement(scene_path: str | Path) -> tuple[tuple[float, float, float] | None, float]:
    """Fallback robot start pose from the InternScenes scene.json go2_placement field."""
    scene_file = Path(scene_path).expanduser()
    if not scene_file.exists():
        return None, 0.0
    json_path = scene_file.with_suffix(".json")
    if not json_path.exists():
        return None, 0.0
    try:
        payload = json.loads(json_path.read_text())
    except Exception:
        return None, 0.0
    placement = payload.get("go2_placement") or {}
    if not placement.get("valid", False):
        return None, 0.0
    pos = placement.get("position_m") or {}
    x = pos.get("x")
    y = pos.get("y")
    if x is None or y is None:
        return None, 0.0
    return (float(x), float(y), 0.35), 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs/isaacsim_realtime_agent_internscenes.yaml")
    )
    parser.add_argument("--livestream", action="store_true")
    parser.add_argument("--public-ip", default="127.0.0.1")
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Render for this many seconds then exit (0 = run until Ctrl+C)."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    execution = config.section("execution")
    scene_path = execution.get("scene")
    robot_start, robot_yaw_deg = _parse_robot_start_pose(
        execution.get("robot_start"), "robot_start"
    )
    if robot_start is None and scene_path:
        robot_start_from_scene, scene_yaw_deg = _load_scene_go2_placement(scene_path)
        if robot_start_from_scene is not None:
            robot_start, robot_yaw_deg = robot_start_from_scene, scene_yaw_deg

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

    run = ExperimentRun(
        ROOT / str(config.section("runtime").get("output_root", "outputs")), config.raw
    )
    head_dir = run.artifacts / "head_rgb"
    head_dir.mkdir(exist_ok=True)
    safety = SafetyController(
        max_speed=float(execution.get("max_speed", 0.5)),
        max_angular_speed=float(execution.get("max_angular_speed", 1.0)),
        max_timeout=float(execution.get("max_action_timeout", 15.0)),
    )
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
        window_resolution=(
            int(execution.get("stream_width", 1280)),
            int(execution.get("stream_height", 720)),
        ),
        robot_usd=execution.get("robot_usd"),
        bind_viewport_to_camera=bool(execution.get("bind_viewport_to_camera", False)),
        scene_up_axis=str(execution.get("scene_up_axis", "z")),
        go2_camera_orient=_as_vector3(execution.get("go2_camera_orient"), "go2_camera_orient"),
        validate_initial_placement=bool(execution.get("validate_initial_placement", True)),
        initial_robot_position=robot_start,
        initial_robot_yaw_deg=robot_yaw_deg,
        camera_fps=camera_fps,
        camera_focal_length=float(execution.get("camera_focal_length", 12.0)),
        environment_planes=dict(execution.get("environment_planes", {})),
        robot_motion_mode=str(execution.get("robot_motion_mode", "kinematic")),
        light_rig=str(execution.get("light_rig", "gray_studio")),
    )

    print("Scene-only preview. Rendering frames...")
    print("Press Ctrl+C to stop.\n")

    frame_index = 0
    start_time = time.time()
    try:
        executor.reset()
        initial_state = executor.get_state()
        print(
            f"Initial robot pose: x={initial_state.position[0]:.3f}, "
            f"y={initial_state.position[1]:.3f}, z={initial_state.position[2]:.3f}, "
            f"yaw={initial_state.yaw:.3f}"
        )
        while True:
            frame = executor.get_observation()
            Image.fromarray(frame.rgb).save(head_dir / f"frame_{frame_index:04d}.png")
            frame_index += 1
            if args.duration > 0 and time.time() - start_time >= args.duration:
                print(f"Reached --duration={args.duration}s; exiting.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        run.write_json(
            "preview_summary.json",
            {
                "frames_rendered": frame_index,
                "mode": "scene_only_preview",
            },
        )
        print(f"Preview artifacts: {run.path}", flush=True)
        run.close()
        executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
