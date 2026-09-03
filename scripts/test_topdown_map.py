"""Quick test / demo for the 2D top-down spatial memory map.

Run with:
    python scripts/test_topdown_map.py

It exercises the full API and saves a demo image to outputs/topdown_demo.png.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from agentic_memory_nav.visualization.topdown_map import create_topdown_map


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # 1. Initialize a 10x10 m room, robot at (1, 1), facing +X.
    room = create_topdown_map(width=10.0, height=10.0, resolution=0.05, robot_x=1.0, robot_y=1.0)

    # 2. Move forward 2 m -> (3, 1).
    room.move_forward(2.0)

    # 3. VLM sees a sofa at +30 deg (left/front in Z-up convention), 1.5 m away.
    room.add_landmark_from_relative(
        label="sofa",
        rel_distance=1.5,
        rel_angle=math.radians(30.0),
        category="sofa",
        color="green",
    )

    # 4. Turn left 30 deg and move another 1.5 m.
    room.rotate(math.radians(30.0))
    room.move_forward(1.5)

    # 5. Absolute landmark (e.g. from a global detector).
    room.add_landmark(label="table", x=6.0, y=4.0, category="table", color="orange")

    # 6. VLM reports an obstacle straight ahead 2 m.
    room.add_landmark_from_relative(
        label="obstacle",
        rel_distance=2.0,
        rel_angle=0.0,
        category="obstacle",
        color="red",
    )

    # 7. Render to file.
    save_path = out_dir / "topdown_demo.png"
    room.render_map(save_path=save_path, show=False)
    print(f"Saved static map to {save_path}")

    # 8. Get numpy image and print shape.
    img = room.get_map_image()
    print(f"Map image shape: {img.shape} (H, W, C)")

    # 9. Tiny real-time loop (just matplotlib non-blocking updates).
    print("Running 20 real-time steps...")
    for step in range(20):
        room.rotate(math.radians(10.0))
        room.move_forward(0.1)
        room.render_map(show=True)
        time.sleep(0.1)

    final_path = out_dir / "topdown_demo_final.png"
    room.render_map(save_path=final_path, show=False)
    print(f"Saved final map to {final_path}")


if __name__ == "__main__":
    main()
