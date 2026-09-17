"""Static renderer for the MCP infinite canvas.

Turns a canvas snapshot (the dict returned by ``get_canvas``) into a PNG image so
the 2D world can be inspected without a browser. The visual style is inspired by
the clean, right-handed Z-up top-down map used elsewhere in the project: light
background, labelled axes, friendly markers for obstacles, and a cyan robot
arrow that clearly shows heading.

All geometry is expressed in the abstract **unit length**; the renderer never
assumes metres / centimetres / millimetres. ``theta`` is in degrees,
counter-clockwise positive, ``theta = 0`` faces ``+x`` and ``y`` grows upward.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# palette (clean, light theme matching the project's top-down map)
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_AXES = "#333333"
C_GRID = "#e0e0e0"
C_GRID_MAJOR = "#bdbdbd"
C_TRAJ = "#1f77b4"  # latest successful move
C_TRAJ_OLD = "#9e9e9e"  # past moves
C_BLOCKED = "#d62728"  # blocked move
C_START = "#2ca02c"  # trajectory starting point
C_ROBOT = "#17becf"
C_ROBOT_INACTIVE = "#7f8c8d"
C_OBSTACLE = "#e74c3c"
C_LABEL = "#2c3e50"
C_TEXT = "#333333"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _add_point(xs: list[float], ys: list[float], pt: Any) -> None:
    """Append a 2-D point to the bounding-box accumulator."""

    if isinstance(pt, (list, tuple)) and len(pt) == 2:
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))


def _bounds(canvas: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) enclosing all content plus a margin."""

    xs: list[float] = []
    ys: list[float] = []

    for seg in canvas.get("trajectory", []) or []:
        _add_point(xs, ys, seg.get("start"))
        _add_point(xs, ys, seg.get("end"))
    for rob in canvas.get("robots", []) or []:
        _add_point(xs, ys, rob.get("position"))
    for obs in canvas.get("obstacles", []) or []:
        _add_point(xs, ys, obs.get("position"))
        for pt in obs.get("geometry", []) or []:
            _add_point(xs, ys, pt)

    if not xs or not ys:
        return (-1.5, 1.5, -1.5, 1.5)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max(xmax - xmin, ymax - ymin, 0.5) * 0.25 + 0.5
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)


def _plot_trajectory(ax: plt.Axes, trajectory: list[dict[str, Any]]) -> None:
    """Draw trajectory with step markers and clearly visible blocked attempts."""

    # The engine does not tag "current" explicitly, so treat the last segment as
    # the current one and all earlier segments as historical.
    n = len(trajectory)
    for index, seg in enumerate(trajectory):
        start = seg.get("start")
        end = seg.get("end")
        if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple))):
            continue

        blocked = seg.get("success") is False
        current = seg.get("current", index == n - 1)

        if blocked:
            color = C_BLOCKED
            width = 3.0
            linestyle = "solid"
        elif current:
            color = C_TRAJ
            width = 2.4
            linestyle = "solid"
        else:
            color = C_TRAJ_OLD
            width = 1.6
            linestyle = (0, (5, 4))

        # Draw the segment line.
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=width,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=3,
        )

        # Mark the trajectory origin with a distinct green square on the first segment.
        if index == 0:
            ax.scatter(
                [start[0]],
                [start[1]],
                marker="s",
                s=100,
                c=C_START,
                edgecolors="black",
                linewidths=1.0,
                zorder=4,
            )

        # Step marker at the end of every segment so each discrete move is visible.
        ax.scatter(
            [end[0]],
            [end[1]],
            marker="o",
            s=40 if blocked else 55,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )

        # Highlight blocked attempts with a cross on the attempted endpoint.
        if blocked:
            ax.scatter(
                [end[0]],
                [end[1]],
                marker="X",
                s=70,
                c=C_BLOCKED,
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )


def _plot_obstacles(ax: plt.Axes, obstacles: list[dict[str, Any]]) -> None:
    """Draw point and line obstacles with labels.

    Line obstacles are grouped by a shared ``group_id`` in ``properties`` so
    squares made of four segments can be rendered as a single polygon.
    Unconfirmed obstacles (``confirmed=False`` in ``properties``) are drawn
    as translucent dashed outlines to indicate perceived-but-not-verified state.
    """

    from collections import defaultdict

    # Group line segments that belong to the same logical obstacle.
    groups: dict[str, list[tuple[dict[str, Any], list[float], list[float]]]] = defaultdict(list)
    singles: list[dict[str, Any]] = []

    for obs in obstacles:
        props = obs.get("properties") or {}
        # Invisible obstacles belong to the hidden world: they still block
        # movement but are not drawn until the robot reports them.
        if props.get("visible") is False:
            continue
        otype = obs.get("obstacle_type") or obs.get("type") or "point"
        if otype == "line":
            group_id = props.get("group_id")
            if group_id:
                groups[group_id].append(obs)
            else:
                singles.append(obs)
        else:
            singles.append(obs)

    # Render grouped squares (four sides) as polygons.
    for group_id, members in groups.items():
        first = members[0]
        confirmed = bool((first.get("properties") or {}).get("confirmed", True))
        # Collect corner points from all segments.
        pts = []
        for m in members:
            geom = m.get("geometry") or []
            if len(geom) >= 2:
                pts.extend(geom[:2])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2.0
        if confirmed:
            # Confirmed obstacle: solid red square.
            ax.add_patch(
                plt.Rectangle(
                    (cx - half, cy - half),
                    2 * half,
                    2 * half,
                    facecolor=C_OBSTACLE,
                    alpha=0.2,
                    edgecolor=C_OBSTACLE,
                    linewidth=1.5,
                    zorder=2,
                )
            )
        else:
            # Perceived-only obstacle: translucent dashed outline.
            ax.add_patch(
                plt.Rectangle(
                    (cx - half, cy - half),
                    2 * half,
                    2 * half,
                    facecolor="none",
                    edgecolor=C_OBSTACLE,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.4,
                    zorder=2,
                )
            )

    # Render single (non-grouped) obstacles.
    for obs in singles:
        otype = obs.get("obstacle_type") or obs.get("type") or "point"
        geometry = obs.get("geometry") or []
        label = obs.get("label") or ""
        confirmed = bool((obs.get("properties") or {}).get("confirmed", True))

        if otype == "line":
            if len(geometry) >= 2:
                (x1, y1), (x2, y2) = geometry[0], geometry[1]
                width = float(obs.get("width") or 0.1)
                if confirmed:
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=C_OBSTACLE,
                        linewidth=max(2.5, width * 40.0),
                        solid_capstyle="round",
                        zorder=2,
                        alpha=0.9,
                    )
                else:
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=C_OBSTACLE,
                        linewidth=max(1.5, width * 40.0),
                        linestyle="--",
                        solid_capstyle="round",
                        zorder=2,
                        alpha=0.4,
                    )
                mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            else:
                continue
        else:
            pos = obs.get("position")
            if not pos:
                continue
            radius = float(obs.get("radius") or 0.2)
            mid_x, mid_y = pos[0], pos[1]
            if confirmed:
                ax.add_patch(
                    plt.Circle(
                        (mid_x, mid_y),
                        radius,
                        facecolor=C_OBSTACLE,
                        alpha=0.18,
                        edgecolor=C_OBSTACLE,
                        linewidth=1.5,
                        zorder=2,
                    )
                )
            else:
                ax.add_patch(
                    plt.Circle(
                        (mid_x, mid_y),
                        radius,
                        facecolor="none",
                        edgecolor=C_OBSTACLE,
                        linewidth=1.0,
                        linestyle="--",
                        alpha=0.4,
                        zorder=2,
                    )
                )

        if label and confirmed:
            ax.annotate(
                label,
                xy=(mid_x, mid_y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color=C_LABEL,
                fontsize=8,
                fontweight="bold",
                zorder=4,
            )


def _plot_robot(ax: plt.Axes, robot: dict[str, Any], scale: float | None = None) -> None:
    """Draw the robot as a cyan arrow plus a star marker at its position."""

    pos = robot.get("position")
    if not pos:
        return

    x, y = float(pos[0]), float(pos[1])
    theta = math.radians(float(robot.get("theta", 0.0)))
    color = C_ROBOT if robot.get("active", True) is not False else C_ROBOT_INACTIVE

    # Auto-scale arrow length from the current axis limits.
    if scale is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scale = min(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.045

    dx = scale * math.cos(theta)
    dy = scale * math.sin(theta)
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=scale * 0.45,
        head_length=scale * 0.45,
        fc=color,
        ec="black",
        linewidth=1.0,
        zorder=5,
    )
    ax.scatter(
        [x],
        [y],
        marker="*",
        s=220,
        c=color,
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def render_canvas(
    canvas: dict[str, Any],
    path: str,
    *,
    title: str = "",
    dpi: int = 120,
    figsize: tuple[float, float] | None = None,
) -> str:
    """Render a ``get_canvas`` snapshot to a PNG file and return its path."""

    xmin, xmax, ymin, ymax = _bounds(canvas)
    width = xmax - xmin
    height = ymax - ymin
    if figsize is None:
        figsize = (max(7.0, width * 1.1), max(5.5, height * 1.1))

    fig, ax = plt.subplots(figsize=figsize, facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    # Hide the coordinate system for an infinite-canvas feel.
    ax.axis("off")

    _plot_obstacles(ax, canvas.get("obstacles", []) or [])
    _plot_trajectory(ax, canvas.get("trajectory", []) or [])
    for robot in canvas.get("robots", []) or []:
        _plot_robot(ax, robot)

    unit = canvas.get("coordinate_unit", "unit_length")
    version = canvas.get("version", 0)
    heading = title or f"MCP Infinite Canvas  ·  {unit}"
    ax.set_title(heading, color=C_TEXT, fontsize=12, pad=10, fontweight="bold")

    info = f"version {version}"
    ax.annotate(
        info,
        xy=(1.0, -0.04),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color=C_TEXT,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=C_BG, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def render_frames(
    frames: list[dict[str, Any]],
    directory: str,
    *,
    prefix: str = "frame",
    **kwargs: Any,
) -> list[str]:
    """Render a list of canvas snapshots to numbered PNG files in ``directory``."""

    import os

    os.makedirs(directory, exist_ok=True)
    paths: list[str] = []
    for index, canvas in enumerate(frames):
        path = os.path.join(directory, f"{prefix}_{index:02d}.png")
        render_canvas(canvas, path, **kwargs)
        paths.append(path)
    return paths