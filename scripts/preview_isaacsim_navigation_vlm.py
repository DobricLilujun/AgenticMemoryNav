#!/usr/bin/env python3
"""VLM self-deciding navigation baseline (direct reactive control).

A minimal, runnable baseline that drives the robot in the configured Isaac Sim
scene with a *direct* VLM decision each step, instead of keyboard input or a
scene-graph planner:

  1. `IsaacSimExecutor.get_observation()` renders one real Isaac Sim frame
     (RGB + robot pose).
  2. `VLMSelfDecidingNavigationAgent.decide()` sends the RGB + instruction to the
     vLLM server (default Inferact/Qwen3.8-27B-NVFP4 @ 10.6.32.16:8000/v1) and
     returns the next motion primitive (forward / back / left / right /
     turn_left / turn_right / stop).
  3. The primitive is converted to a base-link velocity and executed via
     `IsaacSimExecutor.send_velocity_command()`, and the loop continues with the
     resulting new frame.

This is intentionally a baseline: no scene graph, no long-term memory, no
replanning — just the live camera + instruction + VLM. It isolates "how far can a
single vision-language model go with direct reactive control?".

Run with Isaac Sim's bundled Python:
    ~/isaacsim/python.sh scripts/preview_isaacsim_navigation_vlm.py \\\
        --config configs/isaacsim_realtime_agent_internscenes.yaml --livestream
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from agentic_memory_nav.agent.vlm_navigation_agent import (  # noqa: E402
    VLMSelfDecidingNavigationAgent,
)
from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.common.types import Pose3D  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.execution.isaacsim_adapter import IsaacSimExecutor  # noqa: E402
from agentic_memory_nav.execution.safety_controller import SafetyController  # noqa: E402
from agentic_memory_nav.perception.subagent import SubAgentDispatcher  # noqa: E402


def _parse_robot_start_pose(
    value: object, name: str
) -> tuple[tuple[float, float, float] | None, float]:
    if value is None:
        return None, 0.0
    if isinstance(value, dict):
        position = value.get("position")
        yaw_deg = float(value.get("yaw_deg", value.get("yaw", 0.0)))
        sequence = tuple(float(item) for item in position) if position else None
        return sequence, yaw_deg
    if isinstance(value, (list, tuple, np.ndarray)):
        sequence = tuple(float(item) for item in value)
        if len(sequence) == 3:
            return sequence, 0.0
        if len(sequence) == 4:
            return (sequence[0], sequence[1], sequence[2]), float(sequence[3])
        raise ValueError(f"{name} must contain 3 or 4 values [x, y, z[, yaw_deg]]")
    raise ValueError(f"{name} must be a position or a mapping")


def _as_vector3(value: object, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence of 3 numbers")
    vector = tuple(float(item) for item in value)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {vector!r}")
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isaacsim_realtime_agent_internscenes.yaml"))
    parser.add_argument("--instruction", default=None, help="navigation instruction (overrides config)")
    parser.add_argument("--steps", type=int, default=30, help="number of VLM decision steps")
    parser.add_argument("--livestream", action="store_true")
    parser.add_argument("--public-ip", default="127.0.0.0.1")
    parser.add_argument("--no-map", action="store_true", help="do not start the LingBot map agent")
    parser.add_argument("--lingbot-python", default=str(ROOT / ".lingbot-venv/bin/python"))
    parser.add_argument("--lingbot-checkpoint", default=str(ROOT / "external-lib/lingbot-map/models/lingbot-map/lingbot-map.pt"))
    parser.add_argument("--lingbot-image-size", type=int, default=518)
    parser.add_argument("--lingbot-keyframe-interval", type=int, default=1)
    return parser.parse_args()


def _build_agent(config, instruction: str) -> VLMSelfDecidingNavigationAgent:
    execution = config.section("execution")
    perception = config.section("perception")
    model_id = perception.get("model_id")
    if not model_id:
        raise ValueError("perception.model_id is required for the VLM navigation baseline")
    return VLMSelfDecidingNavigationAgent(
        instruction,
        model_id=str(model_id),
        base_url=str(perception.get("base_url", "http://10.6.32.16:8000/v1")),
        api_key=str(perception.get("api_key", "dummy")),
        api=str(perception.get("api", "openai-completions")),
        timeout=float(perception.get("timeout", 120.0)),
        max_speed=float(execution.get("max_speed", 0.35)),
        max_angular_speed=float(execution.get("max_angular_speed", 0.8)),
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    execution = config.section("execution")
    robot_start, robot_yaw_deg = _parse_robot_start_pose(execution.get("robot_start"), "robot_start")
    camera_fps = int(execution.get("camera_fps", 30))
    if not 30 <= camera_fps <= 60:
        raise ValueError(f"execution.camera_fps must be between 30 and 60, got {camera_fps}")

    instruction = args.instruction or str(config.section("runtime").get("instruction", "Navigate the room"))

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

    # Unified sub-agent dispatch: load the LingBot map agent at sim start and
    # guarantee it is open. For a pure VLM baseline --no-map skips it.
    dispatcher = SubAgentDispatcher(
        lingbot_python=args.lingbot_python,
        lingbot_checkpoint=args.lingbot_checkpoint,
        lingbot_image_size=args.lingbot_image_size,
        lingbot_keyframe_interval=args.lingbot_keyframe_interval,
        enable_lingbot_map=(not args.no_map),
    )
    dispatcher.start()

    safety = SafetyController(
        max_speed=float(execution.get("max_speed", 0.35)),
        max_angular_speed=float(execution.get("max_angular_speed", 0.8)),
        max_timeout=float(execution.get("max_action_timeout", 20.0)),
    )
    executor = IsaacSimExecutor(
        scene=execution.get("scene"),
        safety=safety,
        max_speed=float(execution.get("max_speed", 0.35)),
        camera_resolution=(int(execution.get("camera_height", 720)), int(execution.get("camera_width", 1280))),
        headless=bool(execution.get("headless", True)),
        livestream_args=livestream_args,
        window_resolution=(int(execution.get("stream_width", 1280)), int(execution.get("stream_height", 720))),
        robot_usd=execution.get("robot_usd"),
        bind_viewport_to_camera=bool(execution.get("bind_viewport_to_camera", True)),
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

    agent = _build_agent(config, instruction)
    print(f"VLM navigation baseline: instruction={instruction!r}, model={agent.model_id}")
    print(f"action set: {['forward','back','left','right','turn_left','turn_right','stop']}")
    print(f"running {args.steps} VLM decision steps...\n")

    trajectory: list[dict] = []
    collision = False
    try:
        executor.reset()
        for step in range(args.steps):
            frame = executor.get_observation()
            Image.fromarray(frame.rgb).save(head_dir / f"frame_{step:04d}.png")
            # One VLM call per step: decide the action, then convert to a velocity.
            action_name, reason, confidence = agent.decide_action(frame)
            vx, vy, wz = agent.velocity_for(action_name)
            feedback = executor.send_velocity_command(vx, vy, wz)
            collision = collision or feedback.collision
            record = {
                "step": step,
                "frame_id": frame.frame_id,
                "action": action_name,
                "reason": reason,
                "confidence": confidence,
                "velocity": [vx, vy, wz],
                "collision": feedback.collision,
                "robot_position": list(feedback.state.position),
            }
            trajectory.append(record)
            print(
                f"[{step:03d}] {action_name:>11s} "
                f"(c={confidence:.2f}) v=({vx:+.2f},{vy:+.2f},{wz:+.2f}) "
                f"{reason}"
            )
            if feedback.collision:
                print(f"Collision blocked at step {step}: {feedback.reason}")
                break
            time.sleep(1.0 / camera_fps)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        run.write_json(
            "vlm_baseline_summary.json",
            {
                "instruction": instruction,
                "model": agent.model_id,
                "steps_taken": len(trajectory),
                "collision_blocked": collision,
                "trajectory": trajectory,
            },
        )
        print(f"VLM baseline artifacts: {run.path}", flush=True)
        dispatcher.close()
        run.close()
        executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())