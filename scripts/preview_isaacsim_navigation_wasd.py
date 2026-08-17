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
import sys
import time
import threading
from pathlib import Path
from typing import Dict


import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from agentic_memory_nav.common.config import load_config  # noqa: E402
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun  # noqa: E402
from agentic_memory_nav.execution.isaacsim_adapter import IsaacSimExecutor  # noqa: E402
from agentic_memory_nav.execution.safety_controller import SafetyController  # noqa: E402


# Global state for keyboard control
_key_state: Dict[str, bool] = {
    'w': False, 'a': False, 's': False, 'd': False,
    'q': False, 'e': False, ' ': False, 'p': False
}
_key_lock = threading.Lock()
_key_timestamps: Dict[str, float] = {}  # 记录每个键最后按下的时间
_key_hold_duration = 0.15  # 按键保持时间（秒）
_running = True


def _get_keyboard_input() -> tuple[float, float, float]:
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
    return parser.parse_args()


def main() -> int:
    global _running
    
    args = parse_args()
    config = load_config(args.config)
    execution = config.section("execution")
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
        overhead_camera_position=_as_vector3(
            execution.get("overhead_camera_position"), "overhead_camera_position"
        ),
        overhead_camera_orient=_as_vector3(
            execution.get("overhead_camera_orient"), "overhead_camera_orient"
        ),
        go2_camera_orient=_as_vector3(execution.get("go2_camera_orient"), "go2_camera_orient"),
        head_scan_yaw_deg=float(execution.get("head_scan_yaw_deg", 0.0)),
        head_scan_pitch_deg=float(execution.get("head_scan_pitch_deg", 0.0)),
        head_scan_period_frames=int(execution.get("head_scan_period_frames", 1)),
        validate_initial_placement=bool(execution.get("validate_initial_placement", True)),
        initial_robot_position=robot_start,
        initial_robot_yaw_deg=robot_yaw_deg,
        camera_fps=camera_fps,
        camera_focal_length=float(execution.get("camera_focal_length", 12.0)),
        livestream_camera=str(execution.get("livestream_camera", "overhead")),
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
        
        # Start keyboard listener in background
        listener_thread = threading.Thread(target=_keyboard_listener, daemon=True)
        listener_thread.start()
        
        while _running:
            frame = executor.get_observation()
            Image.fromarray(frame.rgb).save(head_dir / f"frame_{frame_index:04d}.png")
            overhead = executor.get_overhead_rgb()
            if overhead is not None:
                Image.fromarray(overhead).save(overhead_dir / f"frame_{frame_index:04d}.png")
            
            # Get keyboard input and send velocity command
            vx, vy, wz = _get_keyboard_input()
            
            # Check for exit signal
            if vx is None:
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
        run.close()
        executor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())