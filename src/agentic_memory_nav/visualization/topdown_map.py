"""Lightweight 2D top-down spatial memory map for indoor navigation.

This module implements a tiny "mental map" agent: it keeps track of the robot pose,
trajectory, and semantic landmarks observed in ego-centric coordinates, and renders
a top-down view with matplotlib.

Coordinates
-----------
- World frame: Z is up. The 2D map uses the (x, y) plane.
- Robot pose: (x, y, theta), where theta is the yaw around +Z.
  theta = 0 means the robot looks along +X.
- Relative observation: (distance, angle) in the robot frame.
  angle = 0 means straight ahead (+X_body); positive = counter-clockwise
  (i.e. left turn), matching the right-handed Z-up convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray


@dataclass
class Landmark:
    """A semantic landmark placed on the map."""

    label: str
    x: float
    y: float
    category: str
    color: str


@dataclass
class TopDownMap:
    """In-memory 2D spatial memory map."""

    width: float = 10.0
    height: float = 10.0
    resolution: float = 0.05
    robot_x: float = 0.0
    robot_y: float = 0.0
    robot_theta: float = 0.0
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    landmarks: list[Landmark] = field(default_factory=list)
    figure_size: tuple[float, float] = (8.0, 8.0)
    _fig: Any | None = field(default=None, repr=False)
    _ax: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        self.trajectory.append((self.robot_x, self.robot_y))

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def update_robot_pose(self, x: float, y: float, theta: float) -> None:
        """Set the robot pose in world coordinates."""
        self.robot_x = float(x)
        self.robot_y = float(y)
        self.robot_theta = float(theta)
        self.trajectory.append((self.robot_x, self.robot_y))

    def move_forward(self, distance: float) -> None:
        """Move the robot forward along its current heading."""
        self.robot_x += distance * math.cos(self.robot_theta)
        self.robot_y += distance * math.sin(self.robot_theta)
        self.trajectory.append((self.robot_x, self.robot_y))

    def rotate(self, angle_rad: float) -> None:
        """Rotate the robot; positive angle = counter-clockwise."""
        self.robot_theta += angle_rad

    def add_landmark(
        self,
        label: str,
        x: float,
        y: float,
        category: str = "obstacle",
        color: str = "red",
    ) -> Landmark:
        """Add a landmark given absolute world coordinates."""
        lm = Landmark(label=label, x=x, y=y, category=category, color=color)
        self.landmarks.append(lm)
        return lm

    def add_landmark_from_relative(
        self,
        label: str,
        rel_distance: float,
        rel_angle: float,
        category: str = "obstacle",
        color: str = "red",
    ) -> Landmark:
        """Add a landmark from an ego-centric observation.

        rel_angle is measured in the robot body frame:
            0       -> straight ahead (+X_body)
            +pi/2   -> left (+Y_body in Z-up right-handed frame)
        """
        global_angle = self.robot_theta + rel_angle
        obj_x = self.robot_x + rel_distance * math.cos(global_angle)
        obj_y = self.robot_y + rel_distance * math.sin(global_angle)
        return self.add_landmark(label, obj_x, obj_y, category, color)

    def get_map_image(self) -> NDArray[np.uint8]:
        """Render the current map and return it as an RGB numpy array."""
        self._ensure_figure()
        self._draw()
        self._fig.canvas.draw()
        canvas = FigureCanvasAgg(self._fig)
        canvas.draw()
        width, height = canvas.get_width_height()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        rgb = buf.reshape(height, width, 4)[:, :, :3].copy()
        return rgb

    def render_map(
        self,
        save_path: str | Path | None = None,
        show: bool = True,
        block: bool = False,
    ) -> None:
        """Render the map to screen and/or save to disk."""
        self._ensure_figure()
        self._draw()
        if save_path is not None:
            self._fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
        if show:
            plt.pause(0.001)  # non-blocking update when block=False
            if block:
                plt.show(block=True)

    def clear(self) -> None:
        """Reset map contents but keep dimensions."""
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.landmarks.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_figure(self) -> None:
        if self._fig is None or self._ax is None:
            # Use a non-interactive backend when no display is available.
            try:
                plt.figure()
                plt.close()
            except Exception:  # pragma: no cover
                matplotlib.use("Agg")
            self._fig, self._ax = plt.subplots(figsize=self.figure_size)

    def _draw(self) -> None:
        ax = self._ax
        ax.clear()

        # Room bounds.
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(0.0, self.height)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("2D Spatial Memory Map")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Trajectory.
        if len(self.trajectory) > 1:
            xs, ys = zip(*self.trajectory)
            ax.plot(xs, ys, "b-", linewidth=1.5, alpha=0.7, label="Trajectory")

        # Landmarks.
        category_marker = {
            "sofa": ("o", 10),
            "chair": ("s", 8),
            "table": ("^", 8),
            "obstacle": ("X", 8),
            "default": ("o", 8),
        }
        for lm in self.landmarks:
            marker, size = category_marker.get(lm.category, category_marker["default"])
            ax.scatter(
                lm.x,
                lm.y,
                marker=marker,
                s=size * 12,
                c=lm.color,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
            ax.annotate(
                lm.label,
                (lm.x, lm.y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                zorder=4,
            )

        # Robot arrow.
        arrow_len = min(self.width, self.height) * 0.06
        dx = arrow_len * math.cos(self.robot_theta)
        dy = arrow_len * math.sin(self.robot_theta)
        ax.arrow(
            self.robot_x,
            self.robot_y,
            dx,
            dy,
            head_width=arrow_len * 0.35,
            head_length=arrow_len * 0.35,
            fc="cyan",
            ec="black",
            linewidth=1.2,
            zorder=5,
        )
        ax.scatter(
            self.robot_x,
            self.robot_y,
            marker="*",
            s=200,
            c="cyan",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
            label="Robot",
        )

        # Legend once.
        ax.legend(loc="upper right", fontsize=8)


# Convenience alias matching the task description.
init_map = TopDownMap


def create_topdown_map(
    width: float = 10.0,
    height: float = 10.0,
    resolution: float = 0.05,
    robot_x: float = 0.0,
    robot_y: float = 0.0,
    robot_theta: float = 0.0,
) -> TopDownMap:
    """Factory for a fresh top-down map."""
    return TopDownMap(
        width=width,
        height=height,
        resolution=resolution,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_theta=robot_theta,
    )
