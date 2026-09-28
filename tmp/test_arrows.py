"""Quick visual check that trajectory midpoint arrows are visible."""

from pathlib import Path

from agentic_memory_nav.mcp.rendering import render_canvas


canvas = {
    "version": 1,
    "coordinate_unit": "unit_length",
    "robots": [
        {"start": [0, 0], "pose": {"x": 5.0, "y": 1.0, "theta": 0}},
    ],
    "trajectory": [
        {"start": [0, 0], "end": [2, 0], "success": True},
        {"start": [2, 0], "end": [3, 2], "success": True},
        {"start": [3, 2], "end": [5, 2], "success": True},
        {"start": [5, 2], "end": [6, 0], "success": True},
        {"start": [6, 0], "end": [8, 0], "success": True},
        {"start": [8, 0], "end": [8, 3], "success": False},  # blocked segment
    ],
    "obstacles": [],
}

out = Path("/home/yiqun/project_lujun/AgenticMemoryNav/tmp/test_arrows.png")
render_canvas(canvas, str(out), title="Trajectory arrow visibility test", dpi=150)
print(f"Saved: {out}")
