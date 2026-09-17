"""Simulation engine: the single source of truth for the infinite canvas.

The engine owns the robot poses, trajectories, obstacles, event history and the
canvas version, and implements the action state-transition rules from the
specification. The MCP server, the WebSocket layer and the fake-robot client all
talk to this engine; no consumer reads engine internals directly.

Transition rules
----------------
* **Rotation** (turn_left / turn_right / turn_left_big / turn_right_big): changes
  only the heading, keeps the 2D position, generates no trajectory, emits a
  ``robot_updated`` event.
* **Move** (move_forward / move_backward): advances the robot along its heading by
  ``move_step`` unit lengths. A move is *blocked* when its swept segment penetrates
  an obstacle. Whether it succeeds or is blocked, a trajectory segment is recorded
  (success -> black, blocked -> red). A blocked move leaves the robot in place.
* **No-op** (stop / look_up / look_down): changes neither 2D position nor
  trajectory, but still emits a ``robot_updated`` event so every action is recorded.

Every action increments the canvas version and publishes an event over the
registered listeners (the WebSocket hub subscribes to these).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .actions import (
    MOVE_SIGN,
    TURN_DELTA_DEGREES,
    ACTION_BY_ID,
    ACTION_ID_BY_NAME,
    is_move,
    is_noop,
    is_rotate,
    resolve_action,
)
from .collision import check_move
from .geometry import Segment, move as move_point, normalize_theta
from .models import Event, Obstacle, Pose, Robot, TrajectorySegment, _new_id

# Default action parameters from the specification (abstract unit lengths / degrees).
DEFAULT_MOVE_STEP = 1.0
DEFAULT_SMALL_TURN_DEGREES = 15.0
DEFAULT_BIG_TURN_DEGREES = 90.0
DEFAULT_CANVAS_ID = "main"
DEFAULT_FRAME = "world"
COORDINATE_UNIT = "unit_length"
PERCEPTION_RADIUS = 1.0e6  # by default the robot perceives every obstacle


def resolve_action_id(name: str) -> int:
    """Return the canonical action id for a name."""

    return ACTION_ID_BY_NAME[name]


def kind_name(name: str) -> str:
    """Human-readable action category for an action name."""

    if is_move(name):
        return "move"
    if is_rotate(name):
        return "rotate"
    if is_noop(name):
        return "noop"
    return "unknown"


@dataclass
class ActionResult:
    """The structured result returned by :meth:`SimulationEngine.execute_action`."""

    robot_id: str
    action: str
    action_id: int
    kind: str
    pose_before: dict[str, Any]
    pose_after: dict[str, Any]
    success: bool
    trajectory_added: bool
    blocked_by: list[str] = field(default_factory=list)
    perceived: list[str] = field(default_factory=list)
    nearest_clearance: float = 0.0
    canvas_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "action": self.action,
            "action_id": self.action_id,
            "kind": self.kind,
            "pose_before": self.pose_before,
            "pose_after": self.pose_after,
            "success": self.success,
            "trajectory_added": self.trajectory_added,
            "blocked_by": self.blocked_by,
            "perceived": self.perceived,
            "nearest_clearance": self.nearest_clearance,
            "canvas_version": self.canvas_version,
        }


class SimulationEngine:
    """In-memory simulation of the 2D robot canvas."""

    def __init__(
        self,
        *,
        canvas_id: str = DEFAULT_CANVAS_ID,
        frame: str = DEFAULT_FRAME,
        move_step: float = DEFAULT_MOVE_STEP,
        small_turn_degrees: float = DEFAULT_SMALL_TURN_DEGREES,
        big_turn_degrees: float = DEFAULT_BIG_TURN_DEGREES,
    ) -> None:
        self.canvas_id = canvas_id
        self.frame = frame
        self.move_step = move_step
        self.small_turn_degrees = small_turn_degrees
        self.big_turn_degrees = big_turn_degrees
        self.version = 0

        self.robots: dict[str, Robot] = {}
        self.trajectories: list[TrajectorySegment] = []
        self.obstacles: dict[str, Obstacle] = {}
        self.events: list[Event] = []
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    # ------------------------------------------------------------------
    # pub/sub (the WebSocket hub subscribes here)
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback invoked with each published event dict."""

        self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _publish(self, event_type: str, data: dict[str, Any]) -> Event:
        """Record an event in history and push it to all subscribers."""

        self.version += 1
        event = Event(event=event_type, canvas_version=self.version, data=data)
        self.events.append(event)
        payload = event.to_dict()
        for callback in list(self._listeners):
            callback(payload)
        return event

    # ------------------------------------------------------------------
    # robots
    # ------------------------------------------------------------------

    def create_robot(self, robot_id: str, start_pose: Any = None) -> dict[str, Any]:
        """Create (or reset to the given start pose) a robot and return its state."""

        start = Pose.from_dict(start_pose)
        existing = self.robots.get(robot_id)
        robot = Robot(
            robot_id=robot_id,
            pose=start.copy(),
            start=start.copy(),
            active=True,
            vertical=existing.vertical if existing is not None else 0.0,
        )
        self.robots[robot_id] = robot
        event = self._publish("robot_updated", {"robot_id": robot_id, "robot": robot.to_dict()})
        return {"robot_id": robot_id, "pose": start.to_dict(), "canvas_version": event.canvas_version}

    def get_robot(self, robot_id: str) -> Robot:
        robot = self.robots.get(robot_id)
        if robot is None:
            raise KeyError(f"Unknown robot_id: {robot_id!r}")
        return robot

    def _ensure(self, robot_id: str) -> Robot:
        """Create the robot if it does not exist yet, then return it.

        Makes every robot-facing operation idempotent: a robot can drive the
        server (execute / add / reset) without a separate create call, matching
        how an HMI control endpoint would auto-create on first action.
        """
        if robot_id not in self.robots:
            self.create_robot(robot_id)
        return self.get_robot(robot_id)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def execute_action(self, robot_id: str, action: str | None = None, action_id: int | None = None) -> ActionResult:
        """Execute an action for a robot through the state-transition rules."""

        name = resolve_action(action, action_id)
        robot = self._ensure(robot_id)
        pose_before = robot.pose.copy()
        action_id = resolve_action_id(name)
        kind = kind_name(name)

        if is_rotate(name):
            delta = TURN_DELTA_DEGREES[name]
            robot.pose.theta = normalize_theta(robot.pose.theta + delta)
            event = self._publish(
                "robot_updated",
                {"robot_id": robot_id, "action": name, "action_id": action_id, "robot": robot.to_dict()},
            )
            return ActionResult(
                robot_id=robot_id,
                action=name,
                action_id=action_id,
                kind=kind,
                pose_before=pose_before.to_dict(),
                pose_after=robot.pose.to_dict(),
                success=True,
                trajectory_added=False,
                canvas_version=event.canvas_version,
            )

        if is_noop(name):
            # stop / look_up / look_down: no 2D change, but still an event.
            event = self._publish(
                "robot_updated",
                {"robot_id": robot_id, "action": name, "action_id": action_id, "robot": robot.to_dict()},
            )
            return ActionResult(
                robot_id=robot_id,
                action=name,
                action_id=action_id,
                kind=kind,
                pose_before=pose_before.to_dict(),
                pose_after=robot.pose.to_dict(),
                success=True,
                trajectory_added=False,
                canvas_version=event.canvas_version,
            )

        # A move action: compute target, test for collision, then record.
        sign = MOVE_SIGN[name]
        target = move_point(pose_before.to_point(), pose_before.theta, sign * self.move_step)
        swept: Segment = (pose_before.to_point(), target)
        result = check_move(list(self.obstacles.values()), swept, perception_radius=PERCEPTION_RADIUS)
        # A move that runs into an obstacle the robot already reported as a
        # meta-obstacle (confirmed=False) is a *repeat* of a known blockage; do
        # not record duplicate red segments for it.
        success = not result.blocked
        if not success:
            known = {
                obs.obstacle_id
                for obs in self.obstacles.values()
                if (obs.properties or {}).get("confirmed") is False
            }
            if result.blocked_by and all(b in known for b in result.blocked_by):
                event = self._publish(
                    "robot_updated",
                    {"robot_id": robot_id, "action": name, "action_id": action_id, "robot": robot.to_dict()},
                )
                return ActionResult(
                    robot_id=robot_id,
                    action=name,
                    action_id=action_id,
                    kind=kind,
                    pose_before=pose_before.to_dict(),
                    pose_after=robot.pose.to_dict(),
                    success=False,
                    trajectory_added=False,
                    blocked_by=result.blocked_by,
                    perceived=result.perceived,
                    nearest_clearance=result.nearest_clearance,
                    canvas_version=event.canvas_version,
                )

        if success:
            robot.pose = Pose(x=target[0], y=target[1], theta=pose_before.theta)
        # On a blocked move the robot stays put (pose unchanged).

        segment = TrajectorySegment(
            segment_id=_new_id("seg"),
            robot_id=robot_id,
            start=pose_before.to_point(),
            end=target,
            action=name,
            success=success,
            blocked_by=result.blocked_by,
            perceived=result.perceived,
        )
        self.trajectories.append(segment)
        event = self._publish(
            "trajectory_added",
            {
                "robot_id": robot_id,
                "action": name,
                "action_id": action_id,
                "success": success,
                "segment": segment.to_dict(),
                "robot": robot.to_dict(),
                "blocked_by": result.blocked_by,
                "perceived": result.perceived,
            },
        )
        return ActionResult(
            robot_id=robot_id,
            action=name,
            action_id=action_id,
            kind=kind,
            pose_before=pose_before.to_dict(),
            pose_after=robot.pose.to_dict(),
            success=success,
            trajectory_added=True,
            blocked_by=result.blocked_by,
            perceived=result.perceived,
            nearest_clearance=result.nearest_clearance,
            canvas_version=event.canvas_version,
        )

    # ------------------------------------------------------------------
    # obstacles
    # ------------------------------------------------------------------

    def add_obstacle(self, robot_id: str, obstacle: dict[str, Any]) -> dict[str, Any]:
        """Add a point or line obstacle and record an ``obstacle_added`` event."""

        obstacle_id = str(obstacle.get("obstacle_id") or _new_id("obs"))
        record = Obstacle.from_dict(robot_id, obstacle)
        record.obstacle_id = obstacle_id
        if record.robot_id == "":
            record.robot_id = robot_id
        self.obstacles[record.obstacle_id] = record
        event = self._publish("obstacle_added", {"robot_id": robot_id, "obstacle": record.to_dict()})
        result = record.to_dict()
        result["canvas_version"] = event.canvas_version
        return result

    def update_obstacle(self, obstacle_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial update to an existing obstacle."""

        record = self.obstacles.get(obstacle_id)
        if record is None:
            raise KeyError(f"Unknown obstacle_id: {obstacle_id!r}")
        record.apply_patch(patch)
        event = self._publish("obstacle_updated", {"obstacle_id": obstacle_id, "obstacle": record.to_dict()})
        result = record.to_dict()
        result["canvas_version"] = event.canvas_version
        return result

    def remove_obstacle(self, obstacle_id: str) -> dict[str, Any]:
        """Remove an obstacle by id."""

        record = self.obstacles.pop(obstacle_id, None)
        if record is None:
            raise KeyError(f"Unknown obstacle_id: {obstacle_id!r}")
        event = self._publish(
            "obstacle_removed",
            {"obstacle_id": obstacle_id, "robot_id": record.robot_id},
        )
        return {"obstacle_id": obstacle_id, "removed": True, "canvas_version": event.canvas_version}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset_simulation(self, robot_id: str | None = None, clear_events: bool = False) -> dict[str, Any]:
        """Reset robot poses, trajectories and obstacle state; optionally clear events."""

        for robot in self.robots.values():
            robot.pose = robot.start.copy()
            robot.vertical = 0.0
        if robot_id is not None:
            self._ensure(robot_id)
        self.trajectories.clear()
        self.obstacles.clear()
        if clear_events:
            self.events.clear()
        event = self._publish(
            "simulation_reset",
            {"robot_id": robot_id or "*", "clear_events": clear_events},
        )
        return {"reset": True, "robot_id": robot_id or "*", "canvas_version": event.canvas_version}

    # ------------------------------------------------------------------
    # canvas snapshot
    # ------------------------------------------------------------------

    def get_canvas(self) -> dict[str, Any]:
        """Return the full 2D canvas snapshot (``get_canvas``)."""

        robots = [robot.to_dict() for robot in self.robots.values()]
        trajectory = _trajectory_view(self.trajectories)
        obstacles = [record.to_dict() for record in self.obstacles.values()]
        events = [event.to_dict() for event in self.events]
        return {
            "canvas_id": self.canvas_id,
            "frame": self.frame,
            "coordinate_unit": COORDINATE_UNIT,
            "version": self.version,
            "robots": robots,
            "trajectory": trajectory,
            "trajectory_count": len(self.trajectories),
            "obstacles": obstacles,
            "events": events,
        }

    def action_ids(self) -> dict[int, str]:
        """Return the canonical action id -> name mapping."""

        return dict(ACTION_BY_ID)


def _trajectory_view(segments: list[TrajectorySegment]) -> list[dict[str, Any]]:
    """Render trajectory segments; the latest per robot is drawn solid, older dashed."""

    last_index_by_robot: dict[str, int] = {}
    for index, segment in enumerate(segments):
        last_index_by_robot[segment.robot_id] = index

    view: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        entry = segment.to_dict()
        entry["current"] = last_index_by_robot.get(segment.robot_id) == index
        view.append(entry)
    return view