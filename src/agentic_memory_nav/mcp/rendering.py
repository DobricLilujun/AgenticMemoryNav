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
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# palette (clean, light theme matching the project's top-down map)
# ---------------------------------------------------------------------------
C_BG = "#ffffff"
C_AXES = "#333333"
C_GRID = "#e0e0e0"
C_GRID_MAJOR = "#bdbdbd"
C_TRAJ = "#7f8c8d"  # all trajectory: dashed grey
C_BLOCKED = "#d62728"  # blocked move
C_START = "#2ca02c"  # trajectory starting point
C_ROBOT = "#1f77b4"  # current robot pose (blue)
C_ROBOT_NEXT = "#1f77b4"  # predicted next-pose ghost
C_ROBOT_INACTIVE = "#7f8c8d"
C_OBSTACLE = "#e74c3c"
C_LABEL = "#2c3e50"
C_TEXT = "#333333"

# Visual size constants in abstract unit lengths.
START_MARKER_RADIUS = 0.1
ROBOT_MARKER_SIZE = 0.2
TRAJ_LINE_WIDTH = 0.05
NEXT_ARROW_WIDTH = 0.04


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


def _line_width_units(ax: Axes, units: float = 0.3) -> float:
    """Convert a line width in *unit lengths* to matplotlib points.

    Matplotlib line widths are in points (1/72 inch); this helper converts a
    geometric width in canvas units into the equivalent point size for the
    current axis scaling so lines keep a constant real-world thickness.
    """

    xlim = ax.get_xlim()
    fig = ax.figure
    bbox = ax.get_window_extent()
    width_units = xlim[1] - xlim[0]
    if width_units <= 0 or bbox.width <= 0:
        return 1.0
    points_per_unit = bbox.width / width_units  # pixels per unit
    # 1 pixel = 72/96 points at the standard 96-dpi display assumption.
    return max(0.5, units * points_per_unit * 72.0 / 96.0)


def _plot_start_marker(ax: Axes, robots: list[dict[str, Any]]) -> None:
    """Draw each robot start as a green disk with a fixed world radius."""

    for robot in robots:
        start = robot.get("start")
        if not (
            isinstance(start, (list, tuple))
            and len(start) == 2
        ):
            continue

        x, y = float(start[0]), float(start[1])

        ax.add_patch(
            Circle(
                (x, y),
                radius=START_MARKER_RADIUS,
                facecolor=C_START,
                edgecolor="black",
                linewidth=_line_width_units(ax, 0.025),
                zorder=7,
            )
        )


def _plot_trajectory(ax: Axes, trajectory: list[dict[str, Any]]) -> None:
    """Draw trajectory as dashed grey lines, with red only for blocked moves."""

    line_width = _line_width_units(ax, TRAJ_LINE_WIDTH)
    for seg in trajectory:
        start = seg.get("start")
        end = seg.get("end")
        if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple))):
            continue

        blocked = seg.get("success") is False
        color = C_BLOCKED if blocked else C_TRAJ

        # Draw the segment line.
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=line_width * (1.4 if blocked else 1.0),
            linestyle="--" if not blocked else "solid",
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=3,
        )

        # Direction arrow in the middle of the segment so the VLM can see
        # the travel direction at a glance.
        seg_dx = float(end[0]) - float(start[0])
        seg_dy = float(end[1]) - float(start[1])
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len > 1e-6:
            angle = math.atan2(seg_dy, seg_dx)
            arrow_len = min(seg_len * 0.35, 0.4)
            mid_x = (float(start[0]) + float(end[0])) / 2.0
            mid_y = (float(start[1]) + float(end[1])) / 2.0
            adx = arrow_len * math.cos(angle)
            ady = arrow_len * math.sin(angle)
            ax.arrow(
                mid_x - adx / 2.0,
                mid_y - ady / 2.0,
                adx,
                ady,
                head_width=arrow_len * 0.6,
                head_length=arrow_len * 0.55,
                fc=color,
                ec=color,
                linewidth=_line_width_units(ax, 0.025),
                alpha=0.9,
                zorder=4,
            )


def _plot_obstacles(ax: Axes, obstacles: list[dict[str, Any]]) -> None:
    """Draw point and line obstacles with labels.

    Line obstacles are grouped by a shared ``group_id`` in ``properties`` so
    squares made of four segments can be rendered as a single polygon.
    Unconfirmed obstacles (``confirmed=False`` in ``properties``) are drawn
    as translucent dashed outlines to indicate perceived-but-not-verified state.
    """

    from collections import defaultdict

    # Group line segments that belong to the same logical obstacle.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
    edge_width = _line_width_units(ax, 0.1)
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
                Rectangle(
                    (cx - half, cy - half),
                    2 * half,
                    2 * half,
                    facecolor=C_OBSTACLE,
                    alpha=0.2,
                    edgecolor=C_OBSTACLE,
                    linewidth=edge_width,
                    zorder=2,
                )
            )
        else:
            # Perceived-only obstacle: translucent dashed outline.
            ax.add_patch(
                Rectangle(
                    (cx - half, cy - half),
                    2 * half,
                    2 * half,
                    facecolor="none",
                    edgecolor=C_OBSTACLE,
                    linewidth=edge_width,
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
                if confirmed:
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=C_OBSTACLE,
                        linewidth=edge_width,
                        solid_capstyle="round",
                        zorder=2,
                        alpha=0.9,
                    )
                else:
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=C_OBSTACLE,
                        linewidth=edge_width,
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
                    Circle(
                        (mid_x, mid_y),
                        radius,
                        facecolor=C_OBSTACLE,
                        alpha=0.18,
                        edgecolor=C_OBSTACLE,
                        linewidth=edge_width,
                        zorder=2,
                    )
                )
            else:
                ax.add_patch(
                    Circle(
                        (mid_x, mid_y),
                        radius,
                        facecolor="none",
                        edgecolor=C_OBSTACLE,
                        linewidth=edge_width,
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


def _draw_robot_icon(
    ax: Axes,
    x: float,
    y: float,
    theta: float,
    *,
    size: float,
    facecolor: str,
    edgecolor: str = "black",
    alpha: float = 1.0,
    zorder: int = 5,
) -> None:
    """Draw a robot-shaped icon (rounded body + two wheels + heading arrow) at (x, y)."""

    rad = math.radians(theta)
    half = size / 2.0
    wheel_r = half * 0.35
    wheel_offset = half * 0.55

    # Wheels (two small circles on the sides).
    wx = wheel_offset * math.sin(rad)
    wy = -wheel_offset * math.cos(rad)
    for sign in (-1, 1):
        ax.add_patch(
            Circle(
                (x + sign * wx, y + sign * wy),
                radius=wheel_r,
                facecolor="#2c3e50",
                edgecolor=edgecolor,
                linewidth=_line_width_units(ax, 0.02),
                alpha=alpha,
                zorder=zorder,
            )
        )

    # Body: rounded main chassis.
    ax.add_patch(
        Circle(
            (x, y),
            radius=half,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=_line_width_units(ax, 0.025),
            alpha=alpha,
            zorder=zorder + 1,
        )
    )

    # Heading arrow on top of the body.
    arrow_len = size * 0.9
    dx = arrow_len * math.cos(rad)
    dy = arrow_len * math.sin(rad)
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=size * 0.45,
        head_length=size * 0.45,
        fc=facecolor,
        ec=edgecolor,
        linewidth=_line_width_units(ax, 0.02),
        alpha=alpha,
        zorder=zorder + 2,
    )


def _plot_robot(ax: Axes, robot: dict[str, Any]) -> None:
    """Draw the current robot as a blue icon and a dashed arrow to its predicted next pose."""

    pos = robot.get("position")
    if not pos:
        return

    x, y = float(pos[0]), float(pos[1])
    theta = float(robot.get("theta", 0.0))
    color = C_ROBOT if robot.get("active", True) is not False else C_ROBOT_INACTIVE

    _draw_robot_icon(ax, x, y, theta, size=ROBOT_MARKER_SIZE, facecolor=color, zorder=5)

    # Predicted next pose: dashed arrow to one move-step ahead, ending in a
    # translucent ghost robot icon.
    rad = math.radians(theta)
    nx = x + 1.0 * math.cos(rad)
    ny = y + 1.0 * math.sin(rad)
    ax.annotate(
        "",
        xy=(nx, ny),
        xytext=(x, y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_ROBOT_NEXT,
            lw=_line_width_units(ax, NEXT_ARROW_WIDTH),
            ls="--",
            alpha=0.55,
        ),
        zorder=4,
    )
    _draw_robot_icon(
        ax,
        nx,
        ny,
        theta,
        size=ROBOT_MARKER_SIZE,
        facecolor=C_ROBOT_NEXT,
        alpha=0.35,
        zorder=4,
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
    _plot_start_marker(ax, canvas.get("robots", []) or [])
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