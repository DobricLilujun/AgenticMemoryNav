#!/usr/bin/env python3
"""Render the configured Isaac Sim navigation scene with WASD keyboard control.


The preview creates the scene, dynamic-bound ground/ceiling planes, and kinematic
Go2, then allows interactive control via WASD keys:
- W: Move forward
- S: Move backward
- A: Move left
- D: Move right
- Q: Turn left (counter-clockwise)
- E: Turn right (clockwise)
- Space: Stop / Brake
- P: Pause / Exit
Run with Isaac Sim's Python:
    ~/isaacsim/python.sh scripts/preview_isaacsim_keyboard.py \
        --config configs/isaacsim_realtime_agent_internscenes.yaml --livestream
"""


from __future__ import annotations


import argparse
import base64
import io
import json
import os
import sys
import subprocess
import time
import threading
import zlib
from pathlib import Path
from typing import Dict


import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.common.types import FrameObservation  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.execution.isaacsim_adapter import (  # noqa: E402
    IsaacSimExecutor,
    _GO2_CAMERA_OFFSET_M,
)
from agentic_memory_nav.execution.safety_controller import SafetyController  # noqa: E402
from agentic_memory_nav.geometry.ground_truth import backproject_depth_to_world  # noqa: E402
from agentic_memory_nav.mapping.lingbot_map_adapter import (  # noqa: E402
    LingBotMapAdapter,
    LingBotPredictor,
)
from agentic_memory_nav.perception.subagent import SubAgentDispatcher  # noqa: E402
from agentic_memory_nav.visualization.realtime_pointcloud_viewer import (  # noqa: E402
    RealtimePointCloudViewer,
)


# Global state for keyboard control
_key_state: Dict[str, bool] = {
    'w': False, 'a': False, 's': False, 'd': False,
    'q': False, 'e': False, ' ': False, 'p': False
}
_key_lock = threading.Lock()
_key_timestamps: Dict[str, float] = {}  # 记录每个键最后按下的时间
_key_hold_duration = 0.15  # 按键保持时间（秒）
_running = True


def _get_keyboard_input() -> tuple[float | None, float | None, float | None]:
    """Read current keyboard state and return velocity command (vx, vy, wz)."""
    current_time = time.time()
    
    with _key_lock:
        # 检查每个键是否超时，超时则释放
        for key in _key_state:
            if _key_state[key] and key in _key_timestamps:
                if current_time - _key_timestamps[key] > _key_hold_duration:
                    _key_state[key] = False
        
        keys = _key_state.copy()
    
    # Check for exit key
    if keys['p']:
        return None, None, None  # Signal to exit
    
    vx, vy, wz = 0.0, 0.0, 0.0
    speed = 0.3  # m/s
    turn_speed = 0.5  # rad/s
    
    # Forward/backward
    if keys['w']:
        vx += speed
    if keys['s']:
        vx -= speed
    
    # Left/right strafe
    if keys['a']:
        vy += speed
    if keys['d']:
        vy -= speed
    
    # Turn left/right
    if keys['q']:
        wz += turn_speed
    if keys['e']:
        wz -= turn_speed
    
    # Space to stop
    if keys[' ']:
        vx, vy, wz = 0.0, 0.0, 0.0
    
    return vx, vy, wz


def _keyboard_listener() -> None:
    """Background thread to listen for keyboard input."""
    global _running
    
    try:
        import sys
        import termios
        import tty
        import select
        
        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            # Set terminal to raw mode
            tty.setraw(fd)
            
            while _running:
                # Non-blocking check for input
                ready, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not ready:
                    continue
                
                char = sys.stdin.read(1)
                if not char:
                    continue
                
                key = char.lower()
                current_time = time.time()
                
                # Update key state and timestamp
                with _key_lock:
                    if key in _key_state:
                        _key_state[key] = True
                        _key_timestamps[key] = current_time
                
                # Check for exit
                if key == 'p':
                    print("\n[P] pressed - exiting...")
                    _running = False
                    break
                    
            # Reset all keys when thread exits
            with _key_lock:
                for key in _key_state:
                    _key_state[key] = False
                _key_timestamps.clear()
                    
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            
    except Exception as e:
        print(f"Keyboard listener error: {e}", file=sys.stderr)
        _running = False


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


def _as_vector3(value: object, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a sequence of 3 numbers, got {type(value).__name__}: {value!r}")
    vector = tuple(float(item) for item in value)
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {vector!r}")
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/isaacsim_realtime_agent_internscenes.yaml"))
    parser.add_argument("--livestream", action="store_true")
    parser.add_argument("--public-ip", default="127.0.0.0.1")
    parser.add_argument(
        "--map-backend",
        choices=("head_depth", "lingbot"),
        default="lingbot",
        help="Use real LingBot-Map by default; head_depth is the explicit fallback.",
    )
    parser.add_argument(
        "--lingbot-python",
        default=str(ROOT / ".lingbot-venv/bin/python"),
        help="Python executable for the isolated real LingBot-Map worker.",
    )
    parser.add_argument(
        "--lingbot-checkpoint",
        default=str(ROOT / "external-lib/lingbot-map/models/lingbot-map/lingbot-map.pt"),
    )
    parser.add_argument("--lingbot-image-size", type=int, default=518)
    parser.add_argument("--lingbot-keyframe-interval", type=int, default=1)
    parser.add_argument("--map-port", type=int, default=8091)
    parser.add_argument("--no-map-view", action="store_true")
    return parser.parse_args()


def _camera_to_world(frame: FrameObservation) -> np.ndarray:
    """Build head optical-camera c2w from the robot pose and fixed head extrinsics."""
    robot_pose = frame.robot_pose
    if robot_pose is None:
        raise ValueError("Head-camera mapping requires FrameObservation.robot_pose")
    x, y, z = robot_pose.position
    cos_yaw, sin_yaw = np.cos(robot_pose.yaw), np.sin(robot_pose.yaw)
    forward, left, up = _GO2_CAMERA_OFFSET_M
    camera_position = np.array(
        [x + cos_yaw * forward - sin_yaw * left, y + sin_yaw * forward + cos_yaw * left, z + up],
        dtype=np.float32,
    )
    # Isaac optical axes map to base_link (-Y, +Z, -X).
    optical_to_base = np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    base_to_world = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = base_to_world @ optical_to_base
    transform[:3, 3] = camera_position
    return transform


def _head_depth_predictor() -> LingBotPredictor:
    global_cloud = np.empty((0, 3), dtype=np.float32)

    def predict(frame):
        nonlocal global_cloud
        if frame.depth is None or frame.camera_intrinsics is None:
            raise ValueError("Head camera must provide depth and camera intrinsics for mapping")
        intrinsics = frame.camera_intrinsics
        matrix = np.array(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        c2w = _camera_to_world(frame)
        confidence = np.isfinite(frame.depth).astype(np.float32)
        local = backproject_depth_to_world(
            np.asarray(frame.depth, dtype=np.float32), matrix, c2w, stride=4
        )
        global_cloud = np.concatenate((global_cloud, local), axis=0)
        if len(global_cloud) > 60_000:
            step = max(1, len(global_cloud) // 60_000)
            global_cloud = global_cloud[::step][:60_000]
        return {
            "camera_pose": {
                "position": list(frame.robot_pose.position),
                "yaw": frame.robot_pose.yaw,
                "camera_to_world": c2w,
            },
            "depth": np.asarray(frame.depth, dtype=np.float32),
            "confidence": confidence,
            "intrinsics": matrix,
            "global_pointcloud": global_cloud,
        }

    return predict


def main() -> int:
    global _running
    
    args = parse_args()
    config = load_config(args.config)
    execution = config.section("execution")
    mapping_config = config.section("mapping")
    map_backend = args.map_backend
    if map_backend not in {"head_depth", "lingbot"}:
        raise ValueError(f"Unsupported WASD map backend: {map_backend}")
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
    # Unified sub-agent dispatch: load the LingBot map agent once when the simulation
    # starts and GUARANTEE it is open (SubAgentDispatcher.start() fails fast if the
    # agent cannot launch or does not become ready). The dispatcher's map predictor is
    # the single entry-point handed to the mapping adapter.
    dispatcher = SubAgentDispatcher(
        lingbot_python=args.lingbot_python,
        lingbot_checkpoint=args.lingbot_checkpoint,
        lingbot_image_size=args.lingbot_image_size,
        lingbot_keyframe_interval=args.lingbot_keyframe_interval,
        enable_lingbot_map=(map_backend == "lingbot"),
    )
    dispatcher.start()
    mapper_predictor = (
        dispatcher.map_predictor() if map_backend == "lingbot" else _head_depth_predictor()
    )
    mapper = LingBotMapAdapter(
        checkpoint=Path(str(mapping_config.get("checkpoint", "lingbot-map.pt"))).expanduser(),
        device=str(mapping_config.get("device", "cuda")),
        predictor=mapper_predictor,
        keyframe_interval=int(mapping_config.get("keyframe_interval", 1)),
        local_submap_frames=int(mapping_config.get("local_submap_frames", 30)),
        local_submap_stride=int(mapping_config.get("local_submap_stride", 1)),
        local_submap_stability_threshold_m=float(
            mapping_config.get("local_submap_stability_threshold_m", 0.50)
        ),
        local_submap_max_points_per_frame=int(
            mapping_config.get("local_submap_max_points_per_frame", 20_000)
        ),
    )
    map_viewer = None if args.no_map_view else RealtimePointCloudViewer(port=args.map_port)
    print(f"3D map viewer: http://<this-host>:{args.map_port}/", flush=True)
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
    print("\n=== WASD Control ===")
    print("W: Forward  | S: Backward")
    print("A: Left     | D: Right")
    print("Q: Turn Left  | E: Turn Right")
    print("Space: Stop")
    print("P: Exit")
    print("\nRunning... press P to exit\n")
    
    try:
        executor.reset()
        mapper.start()
        
        # Start keyboard listener in background
        listener_thread = threading.Thread(target=_keyboard_listener, daemon=True)
        listener_thread.start()
        
        while _running:
            frame = executor.get_observation()
            Image.fromarray(frame.rgb).save(head_dir / f"frame_{frame_index:04d}.png")
            mapping = mapper.update(frame)
            if map_viewer is not None:
                map_viewer.update(
                    mapping.global_pointcloud,
                    frame_id=frame.frame_id,
                    robot=executor.get_state().position,
                    yaw=executor.get_state().yaw,
                )
            
            # Get keyboard input and send velocity command
            vx, vy, wz = _get_keyboard_input()
            
            # Check for exit signal
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
        run.write_json(
            "preview_summary.json",
            {"frames_rendered": frame_index, "collision_blocked": collision, "mode": "keyboard_control"},
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