#!/usr/bin/env python3
"""VLM-driven discrete-action navigation in Isaac Sim — find the green shoe.

No LingBot-Map, no scene graph, no long-term memory. A VLM (Qwen3.8 via the
configured vLLM endpoint) looks at each RGB frame and picks one of the expanded
standard discrete actions:

  turn_left | turn_right | turn_left_big | turn_right_big |
  move_forward | move_backward | stop | look_up | look_down

The robot searches the room until the green shoe is visible and centered, then
the VLM should emit ``stop``. The loop terminates on ``stop`` or after a configurable
maximum number of steps -- a blocked ``move_forward`` never ends the episode: the
robot turns away from the obstacle and keeps deciding (see
:class:`agentic_memory_nav.agent.vlm.discrete_navigation.VLMDiscreteNavigationAgent`).

Run with Isaac Sim's bundled Python:

    conda deactivate  # avoid conda's Python shadowing Isaac Sim
    ~/isaacsim/python.sh scripts/preview_isaacsim_navigation_vlm_discrete.py \
        --config configs/isaacsim_realtime_agent_internscenes.yaml \
        --livestream \
        --public-ip 127.0.0.1

The default instruction is "Find the green shoe in the room".
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

from agentic_memory_nav.agent.execution.discrete_actions import DiscreteAction  # noqa: E402
from agentic_memory_nav.agent.execution.isaacsim_adapter import IsaacSimExecutor  # noqa: E402
from agentic_memory_nav.agent.execution.safety_controller import SafetyController  # noqa: E402
from agentic_memory_nav.agent.vlm.discrete_navigation import (  # noqa: E402
    VLMDiscreteNavigationAgent,
)
from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs/isaacsim_realtime_agent_internscenes.yaml")
    )
    parser.add_argument(
        "--instruction",
        default="Find the green shoe in the room",
        help="navigation instruction (overrides config)",
    )
    parser.add_argument("--steps", type=int, default=60, help="maximum VLM decision steps")
    parser.add_argument(
        "--max-look-count",
        type=int,
        default=1,
        help="maximum number of times the VLM may call look_up and look_down each",
    )
    parser.add_argument(
        "--turn-step-deg", type=float, default=15.0, help="degrees rotated per turn_left/turn_right"
    )
    parser.add_argument(
        "--turn-big-step-deg",
        type=float,
        default=90.0,
        help="degrees rotated per turn_left_big/turn_right_big",
    )
    parser.add_argument(
        "--move-step-m",
        type=float,
        default=0.25,
        help="meters advanced per move_forward/move_backward",
    )
    parser.add_argument(
        "--look-step-deg", type=float, default=30.0, help="degrees tilted per look_up/look_down"
    )
    parser.add_argument("--livestream", action="store_true")
    parser.add_argument("--public-ip", default="127.0.0.1")
    return parser.parse_args()


def _build_agent(config, instruction: str, max_look_count: int) -> VLMDiscreteNavigationAgent:
    perception = config.section("perception")
    model_id = perception.get("model_id")
    if not model_id:
        raise ValueError("perception.model_id is required for the VLM navigation baseline")
    return VLMDiscreteNavigationAgent(
        instruction,
        model_id=str(model_id),
        base_url=str(perception.get("base_url", "http://10.6.32.16:8000/v1")),
        api_key=str(perception.get("api_key", "dummy")),
        api=str(perception.get("api", "openai-completions")),
        timeout=float(perception.get("timeout", 120.0)),
        max_look_count=max_look_count,
    )


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

    instruction = args.instruction or str(
        config.section("runtime").get("instruction", "Navigate the room")
    )

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
        max_speed=float(execution.get("max_speed", 0.35)),
        max_angular_speed=float(execution.get("max_angular_speed", 0.8)),
        max_timeout=float(execution.get("max_action_timeout", 20.0)),
    )
    executor = IsaacSimExecutor(
        scene=execution.get("scene"),
        safety=safety,
        max_speed=float(execution.get("max_speed", 0.35)),
        camera_resolution=(
            int(execution.get("camera_height", 720)),
            int(execution.get("camera_width", 1280)),
        ),
        headless=bool(execution.get("headless", True)),
        livestream_args=livestream_args,
        window_resolution=(
            int(execution.get("stream_width", 1280)),
            int(execution.get("stream_height", 720)),
        ),
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
        turn_step_deg=args.turn_step_deg,
        turn_big_step_deg=args.turn_big_step_deg,
        move_step_m=args.move_step_m,
        look_step_deg=args.look_step_deg,
    )

    agent = _build_agent(config, instruction, args.max_look_count)
    print(f"VLM discrete navigation: instruction={instruction!r}, model={agent.model_id}")
    print(f"action set: {[a.value for a in DiscreteAction]}")
    print(
        f"turn_step={args.turn_step_deg:.1f}deg "
        f"turn_big_step={args.turn_big_step_deg:.1f}deg "
        f"move_step={args.move_step_m:.2f}m "
        f"look_step={args.look_step_deg:.1f}deg"
    )
    print(f"running up to {args.steps} VLM decision steps...\n")

    trajectory: list[dict] = []
    collision = False
    last_turn: DiscreteAction | None = None
    try:
        executor.reset()
        for step in range(args.steps):
            frame = executor.get_observation()
            Image.fromarray(frame.rgb).save(head_dir / f"frame_{step:04d}.png")

            action, reason, confidence = agent.decide_action(frame)
            feedback = executor.apply_discrete_action(action)
            # Only move_* can fail due to a collision; feedback.collision is a
            # sticky whole-episode flag, so use feedback.success for this-step detection.
            step_collided = (
                action in (DiscreteAction.MOVE_FORWARD, DiscreteAction.MOVE_BACKWARD)
                and not feedback.success
            )
            collision = collision or step_collided

            trajectory.append(
                {
                    "step": step,
                    "frame_id": frame.frame_id,
                    "action": action.value,
                    "reason": reason,
                    "confidence": confidence,
                    "collision": step_collided,
                    "robot_position": list(feedback.state.position),
                    "robot_yaw": feedback.state.yaw,
                }
            )
            print(
                f"[{step:03d}] {action.value:>12s} "
                f"(c={confidence:.2f}) "
                f"pos=({feedback.state.position[0]:+.2f},{feedback.state.position[1]:+.2f}) "
                f"yaw={feedback.state.yaw:+.2f}  {reason}"
            )
            if action in (
                DiscreteAction.TURN_LEFT,
                DiscreteAction.TURN_RIGHT,
                DiscreteAction.TURN_LEFT_BIG,
                DiscreteAction.TURN_RIGHT_BIG,
            ):
                last_turn = action

            if step_collided:
                # Requirement: never stop the episode on collision. Turn away from the
                # obstacle immediately (deterministic recovery, not another VLM call),
                # then let the VLM keep deciding on the next frame with a warning that
                # its last heading was blocked (dynamic obstacle avoidance).
                if action is DiscreteAction.MOVE_BACKWARD:
                    recovery_action = DiscreteAction.TURN_LEFT
                elif (
                    last_turn is DiscreteAction.TURN_LEFT
                    or last_turn is DiscreteAction.TURN_LEFT_BIG
                ):
                    recovery_action = DiscreteAction.TURN_RIGHT
                else:
                    recovery_action = DiscreteAction.TURN_LEFT
                recovery_feedback = executor.apply_discrete_action(recovery_action)
                last_turn = recovery_action
                agent.note_collision(recovery_action)
                trajectory.append(
                    {
                        "step": step,
                        "frame_id": frame.frame_id,
                        "action": f"recovery_{recovery_action.value}",
                        "reason": f"collision recovery after {feedback.reason}",
                        "confidence": 1.0,
                        "collision": False,
                        "robot_position": list(recovery_feedback.state.position),
                        "robot_yaw": recovery_feedback.state.yaw,
                    }
                )
                print(
                    f"      collision blocked ({feedback.reason}); "
                    f"recovering with {recovery_action.value}"
                )
                time.sleep(1.0 / camera_fps)
                continue
            if action is DiscreteAction.STOP:
                print("VLM emitted stop.")
                break
            time.sleep(1.0 / camera_fps)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        run.write_json(
            "vlm_discrete_summary.json",
            {
                "instruction": instruction,
                "model": agent.model_id,
                "steps_taken": len(trajectory),
                "collision_blocked": collision,
                "trajectory": trajectory,
            },
        )
        print(f"VLM discrete navigation artifacts: {run.path}", flush=True)
        run.close()
        executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
