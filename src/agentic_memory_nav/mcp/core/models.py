"""Data models for the MCP infinite-canvas simulation.

These are dependency-free dataclasses that form the single source of truth for the
simulation engine. The MCP server, WebSocket layer and the fake-robot client all
interact through the engine and these models; no consumer reads engine internals
directly.

Every length is an abstract **unit length**; nothing is tied to metres, centimetres
or millimetres.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .geometry import Point, Segment
from .actions import ACTION_ID_BY_NAME

# ---------------------------------------------------------------------------
# ids / time
# ---------------------------------------------------------------------------


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_point(value: Any) -> Point:
    """Coerce an ``{x, y}`` dict or a ``[x, y]`` pair into a :data:`Point`."""

    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise ValueError(f"Point requires 'x' and 'y', got {value!r}")
        return (float(value["x"]), float(value["y"]))
    if hasattr(value, "__len__") and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Cannot interpret {value!r} as a point")


def _as_segment(value: Any) -> Segment:
    """Coerce a geometry dict ``{"start": [...], "end": [...]}`` into a segment."""

    if isinstance(value, dict):
        return (_as_point(value["start"]), _as_point(value["end"]))
    raise ValueError(f"Line geometry must be a dict with 'start' and 'end', got {value!r}")


# ---------------------------------------------------------------------------
# robot pose
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """Robot pose in the abstract 2D world.

    ``x`` and ``y`` are unit lengths and ``theta`` is a heading in degrees
    (counter-clockwise positive, ``0`` faces ``+x``).
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def to_point(self) -> Point:
        return (self.x, self.y)

    def copy(self) -> "Pose":
        return Pose(x=self.x, y=self.y, theta=self.theta)

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "theta": self.theta}

    @classmethod
    def from_dict(cls, value: Any) -> "Pose":
        if value is None:
            return cls()
        if isinstance(value, Pose):
            return value.copy()
        return cls(
            x=float(value.get("x", 0.0)),
            y=float(value.get("y", 0.0)),
            theta=float(value.get("theta", 0.0)),
        )


# ---------------------------------------------------------------------------
# robot
# ---------------------------------------------------------------------------


@dataclass
class Robot:
    """A robot managed by the engine.

    Only the fake robot drives the canvas through MCP, but the engine keeps a
    registry so multiple robots could coexist. ``start`` is the pose restored on a
    simulation reset. ``vertical`` is an optional out-of-plane state used by
    ``look_up`` / ``look_down``; it never affects the 2D position or trajectory.
    """

    robot_id: str
    pose: Pose = field(default_factory=Pose)
    start: Pose = field(default_factory=Pose)
    active: bool = True
    created_at: float = field(default_factory=_now)
    vertical: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "position": [round(self.pose.x, 6), round(self.pose.y, 6)],
            "theta": round(self.pose.theta, 6),
            "active": self.active,
        }


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------


@dataclass
class TrajectorySegment:
    """A single movement trajectory segment.

    ``success`` is ``True`` for a completed move (drawn black) and ``False`` for a
    move blocked by an obstacle (drawn red). Past segments are drawn as dashed
    lines by the viewer while the latest segment is drawn solid.
    """

    segment_id: str
    robot_id: str
    start: Point
    end: Point
    action: str
    success: bool
    created_at: float = field(default_factory=_now)
    # Obstacle ids that blocked this move (empty when successful).
    blocked_by: list[str] = field(default_factory=list)
    # Obstacle ids perceived while executing the action.
    perceived: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "robot_id": self.robot_id,
            "start": [round(self.start[0], 6), round(self.start[1], 6)],
            "end": [round(self.end[0], 6), round(self.end[1], 6)],
            "action": self.action,
            "action_id": ACTION_ID_BY_NAME.get(self.action, -1),
            "success": self.success,
            "blocked_by": list(self.blocked_by),
            "perceived": list(self.perceived),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# obstacles
# ---------------------------------------------------------------------------


@dataclass
class Obstacle:
    """A point or line obstacle stored in unit lengths.

    * ``point`` obstacles carry a ``position`` (Point) and a ``radius``.
    * ``line`` obstacles carry a ``geometry`` (start/end Segment) and a ``width``.

    ``label``, ``confidence``, ``source``, ``timestamp`` and ``properties`` are
    metadata carried alongside the geometry.
    """

    obstacle_id: str
    robot_id: str
    type: str  # "point" or "line"
    position: Point
    geometry: Segment
    radius: float
    width: float
    label: str
    confidence: float
    source: str = "robot"
    timestamp: float = field(default_factory=_now)
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def is_point(self) -> bool:
        return self.type == "point"

    @property
    def is_line(self) -> bool:
        return self.type == "line"

    @classmethod
    def from_dict(cls, robot_id: str, value: dict[str, Any]) -> "Obstacle":
        """Build an :class:`Obstacle` from a request ``obstacle`` mapping.

        Accepts several naming conventions so the client is forgiving: the type may
        be given as ``obstacle_type`` or ``type``; a point position as
        ``position``/``[x, y]``/flat ``x``+``y``; a line as ``geometry``/``start``+``end``;
        and the size as ``size`` (``{radius, width}``) or flat ``radius``/``width``.
        """

        raw_id = value.get("obstacle_id")
        obstacle_id = str(raw_id) if raw_id else _new_id("obs")
        otype = value.get("obstacle_type") or value.get("type") or "point"
        if otype not in ("point", "line"):
            raise ValueError(f"Unknown obstacle type {otype!r}; expected 'point' or 'line'")

        size = value.get("size") or {}
        radius = float(value["radius"]) if "radius" in value else float(size.get("radius", 0.2))
        width = float(value["width"]) if "width" in value else float(size.get("width", 0.1))

        if otype == "point":
            if "position" in value:
                position = _as_point(value["position"])
            elif "x" in value or "y" in value:
                position = (float(value.get("x", 0.0)), float(value.get("y", 0.0)))
            elif "geometry" in value:
                geometry = _as_segment(value["geometry"])
                position = (
                    (geometry[0][0] + geometry[1][0]) / 2.0,
                    (geometry[0][1] + geometry[1][1]) / 2.0,
                )
            else:
                position = (0.0, 0.0)
            geometry = (position, position)
            width = 0.0
        else:
            if "geometry" in value:
                geometry = _as_segment(value["geometry"])
            elif "start" in value and "end" in value:
                geometry = (_as_point(value["start"]), _as_point(value["end"]))
            else:
                position = _as_point(value.get("position", [0.0, 0.0]))
                geometry = (position, position)
            position = (
                (geometry[0][0] + geometry[1][0]) / 2.0,
                (geometry[0][1] + geometry[1][1]) / 2.0,
            )
            radius = 0.0

        return cls(
            obstacle_id=obstacle_id,
            robot_id=robot_id,
            type=otype,
            position=position,
            geometry=geometry,
            radius=radius,
            width=width,
            label=str(value.get("label", "")),
            confidence=float(value.get("confidence", 1.0)),
            source=str(value.get("source", "robot")),
            timestamp=float(value.get("timestamp", _now())),
            properties=dict(value.get("properties", {}) or {}),
        )

    def apply_patch(self, patch: dict[str, Any]) -> None:
        """Apply a partial update to the obstacle in place."""

        if "type" in patch and patch["type"] in ("point", "line"):
            self.type = patch["type"]
        if "obstacle_type" in patch and patch["obstacle_type"] in ("point", "line"):
            self.type = patch["obstacle_type"]
        if "position" in patch:
            self.position = _as_point(patch["position"])
        if "x" in patch or "y" in patch:
            self.position = (float(patch.get("x", self.position[0])), float(patch.get("y", self.position[1])))
        if "geometry" in patch:
            self.geometry = _as_segment(patch["geometry"])
        if "label" in patch:
            self.label = str(patch["label"])
        if "confidence" in patch:
            self.confidence = float(patch["confidence"])
        if "source" in patch:
            self.source = str(patch["source"])
        if "timestamp" in patch:
            self.timestamp = float(patch["timestamp"])
        if "properties" in patch:
            self.properties = dict(patch["properties"])
        if "radius" in patch:
            self.radius = float(patch["radius"])
        if "width" in patch:
            self.width = float(patch["width"])
        if "size" in patch:
            size = patch["size"] or {}
            if "radius" in size:
                self.radius = float(size["radius"])
            if "width" in size:
                self.width = float(size["width"])
        if self.is_line:
            self.position = (
                (self.geometry[0][0] + self.geometry[1][0]) / 2.0,
                (self.geometry[0][1] + self.geometry[1][1]) / 2.0,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "obstacle_type": self.type,
            "type": self.type,
            "position": [round(self.position[0], 6), round(self.position[1], 6)],
            "geometry": [
                [round(self.geometry[0][0], 6), round(self.geometry[0][1], 6)],
                [round(self.geometry[1][0], 6), round(self.geometry[1][1], 6)],
            ],
            "radius": round(self.radius, 6) if self.is_point else None,
            "width": round(self.width, 6) if self.is_line else None,
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "source": self.source,
            "timestamp": self.timestamp,
            "properties": dict(self.properties),
            "robot_id": self.robot_id,
        }


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


# The event types pushed over the WebSocket, matching the specification.
EVENT_TYPES = (
    "robot_updated",
    "trajectory_added",
    "obstacle_added",
    "obstacle_updated",
    "obstacle_removed",
    "simulation_reset",
)


@dataclass
class Event:
    """A single canvas event, broadcast over the WebSocket and kept in history."""

    event: str
    canvas_version: int
    data: dict[str, Any]
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "canvas_version": self.canvas_version,
            "data": self.data,
            "created_at": self.created_at,
        }