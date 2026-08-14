#!/usr/bin/env python3
"""Record RGB, metric depth, c2w poses, and RTX LiDAR from a quadruped proxy.

Run only with Isaac Sim's bundled Python:
``~/isaacsim/python.sh scripts/record_isaacsim_sequence.py --output ...``.

The generated robot is a procedural *quadruped proxy* (body plus four legs) moved
kinematically. It is deliberately not presented as a physically simulated Unitree Go2.
Its value is a reproducible PointNav/ObjectNav-style sensor rig that emits paired depth
and camera-to-world ground truth for LingBot-Map reconstruction evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORWARD_CAMERA_ORIENTATION_WXYZ = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float32)


def _quaternion_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return a $4\times4$ USD-world transform from a scalar-first quaternion."""
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("Camera quaternion must be non-zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _camera_c2w_cv(position: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    """Convert Isaac USD camera axes to the OpenCV RGB-D convention.

    USD camera axes are right/up/back; the RGB-D back-projector expects right/down/
    forward. The fixed axis conversion preserves the physically rendered camera pose.
    """
    transform = _quaternion_matrix(orientation_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float32)
    usd_camera_from_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return transform @ usd_camera_from_cv


def _add_quadruped_proxy(world):
    """Create a visual kinematic quadruped proxy and return its movable pieces."""
    from isaacsim.core.api.objects import FixedCuboid

    pieces = [
        world.scene.add(
            FixedCuboid(
                prim_path="/World/quadruped_proxy/body",
                name="quadruped_body",
                position=np.array([0.0, 0.0, 0.42], dtype=np.float32),
                scale=np.array([0.75, 0.34, 0.24], dtype=np.float32),
            )
        )
    ]
    for index, (offset_x, offset_y) in enumerate(
        ((0.27, 0.18), (0.27, -0.18), (-0.27, 0.18), (-0.27, -0.18))
    ):
        pieces.append(
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/quadruped_proxy/leg_{index}",
                    name=f"quadruped_leg_{index}",
                    position=np.array([offset_x, offset_y, 0.2], dtype=np.float32),
                    scale=np.array([0.12, 0.12, 0.4], dtype=np.float32),
                )
            )
        )
    return pieces


def _move_proxy(pieces, center_xy: tuple[float, float]) -> None:
    offsets = (
        (0.0, 0.0, 0.42),
        (0.27, 0.18, 0.2),
        (0.27, -0.18, 0.2),
        (-0.27, 0.18, 0.2),
        (-0.27, -0.18, 0.2),
    )
    for piece, (offset_x, offset_y, height) in zip(pieces, offsets, strict=True):
        piece.set_world_pose(
            position=np.array(
                [center_xy[0] + offset_x, center_xy[1] + offset_y, height],
                dtype=np.float32,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=96)
    parser.add_argument("--step-m", type=float, default=0.12)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0 or args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("frames and camera dimensions must be positive")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    try:
        import matplotlib.pyplot as plt
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.sensors.camera import Camera
        from isaacsim.sensors.rtx import LidarRtx

        output = Path(args.output)
        rgb_dir, depth_dir, lidar_dir = output / "rgb", output / "depth", output / "lidar"
        for directory in (rgb_dir, depth_dir, lidar_dir):
            directory.mkdir(parents=True, exist_ok=True)

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        for index, (x, y) in enumerate(((1.5, 0.0), (-1.5, 1.0), (0.5, -1.5))):
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/obstacle_{index}",
                    name=f"obstacle_{index}",
                    position=np.array([x, y, 0.35], dtype=np.float32),
                    scale=np.array([0.5, 0.5, 0.7], dtype=np.float32),
                )
            )
        pieces = _add_quadruped_proxy(world)
        camera = Camera(
            prim_path="/World/quadruped_proxy/camera",
            position=np.array([0.0, 0.0, 0.62], dtype=np.float32),
            orientation=FORWARD_CAMERA_ORIENTATION_WXYZ,
            resolution=(args.camera_width, args.camera_height),
        )
        lidar = LidarRtx(
            prim_path="/World/quadruped_proxy/lidar",
            name="quadruped_lidar",
            position=np.array([0.0, 0.0, 0.55], dtype=np.float32),
            config_file_name="Example_Rotary_2D",
        )
        world.reset()
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
        lidar.initialize()
        lidar.attach_annotator("IsaacComputeRTXLidarFlatScan")
        from pxr import UsdLux

        dome = UsdLux.DomeLight.Define(world.stage, "/World/semantic_dome_light")
        dome.CreateIntensityAttr(1000.0)
        center_pixel = np.array(
            [[args.camera_width / 2.0, args.camera_height / 2.0]], dtype=np.float32
        )
        target_position = np.asarray(
            camera.get_world_points_from_image_coords(
                center_pixel, np.array([2.5], dtype=np.float32)
            )
        )[0]
        target_position[2] = max(float(target_position[2]), 0.35)
        world.scene.add(
            FixedCuboid(
                prim_path="/World/objectnav_target_red_cube",
                name="objectnav_target_red_cube",
                position=target_position,
                scale=np.array([0.55, 0.55, 0.7], dtype=np.float32),
                color=np.array([0.9, 0.02, 0.02], dtype=np.float32),
            )
        )
        target_pixel = camera.get_image_coords_from_world_points(target_position[None])[0]
        for _ in range(12):
            world.step(render=True)

        intrinsics = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float32)
        poses: dict[str, list[list[float]]] = {}
        lidar_counts = []
        for index in range(args.frames):
            center = (index * args.step_m, 0.0)
            _move_proxy(pieces, center)
            camera.set_world_pose(
                position=np.array([center[0], center[1], 0.62], dtype=np.float32),
                orientation=FORWARD_CAMERA_ORIENTATION_WXYZ,
            )
            lidar.set_world_pose(position=np.array([center[0], center[1], 0.55], dtype=np.float32))
            world.step(render=True)
            frame_name = f"frame_{index:06d}"
            rgba = camera.get_rgba()
            depth = camera.get_current_frame().get("distance_to_image_plane")
            if rgba is None or depth is None:
                raise RuntimeError(f"Camera did not produce RGB-D for {frame_name}")
            plt.imsave(rgb_dir / f"{frame_name}.png", np.asarray(rgba)[..., :3])
            np.save(depth_dir / f"depth_{index:06d}.npy", np.asarray(depth, dtype=np.float32))
            camera_position, camera_orientation = camera.get_world_pose()
            poses[f"depth_{index:06d}.npy"] = _camera_c2w_cv(
                camera_position, camera_orientation
            ).tolist()

            lidar_frame = lidar.get_current_frame()
            scan = np.asarray(lidar_frame.get("linear_depth_data", []), dtype=np.float32)
            azimuth = np.asarray(lidar_frame.get("azimuth_range", []), dtype=np.float32)
            lidar_position, lidar_orientation = lidar.get_world_pose()
            np.savez_compressed(
                lidar_dir / f"{frame_name}.npz",
                linear_depth_m=scan,
                azimuth_range=azimuth,
                sensor_position_usd=np.asarray(lidar_position, dtype=np.float32),
                sensor_orientation_wxyz=np.asarray(lidar_orientation, dtype=np.float32),
            )
            lidar_counts.append(int(len(scan)))

        (output / "intrinsics.json").write_text(
            json.dumps(intrinsics.tolist(), indent=2) + "\n", encoding="utf-8"
        )
        (output / "camera_to_world.json").write_text(
            json.dumps(poses, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "robot": "procedural_quadruped_proxy_kinematic",
            "frames": args.frames,
            "camera_resolution": [args.camera_width, args.camera_height],
            "coordinate_frame": "OpenCV camera-to-Isaac-USD world c2w",
            "lidar": "RTX Example_Rotary_2D flat scan",
            "lidar_scan_counts": lidar_counts,
            "objectnav_target_usd": target_position.tolist(),
            "objectnav_target_initial_pixel": target_pixel.tolist(),
            "limitations": "Proxy has no articulated joints, contact physics, or Go2 controller.",
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
