"""Collision detection between a robot's move and the canvas obstacles.

A move is a segment from the robot's current position to its would-be target
position (both in unit lengths). The move *succeeds* when that segment stays clear
of every obstacle; it is *blocked* when the segment penetrates an obstacle.

* A **point** obstacle is a disc of ``radius``; the move is blocked when the swept
  segment comes within ``radius`` of the centre.
* A **line** obstacle is a wall of ``width``; the move is blocked when the swept
  segment comes within ``width / 2`` of the wall segment.

All distances are abstract unit lengths; nothing here assumes a physical unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import EPS, Segment, distance_point_to_segment, distance_segment_to_segment
from .models import Obstacle


@dataclass
class CollisionResult:
    """Outcome of testing a move against the current obstacle set."""

    blocked: bool
    blocked_by: list[str] = field(default_factory=list)
    # Every obstacle the robot "perceived" (visual perception) while executing.
    perceived: list[str] = field(default_factory=list)
    # Smallest clearance to any obstacle, for diagnostics and perception.
    nearest_clearance: float = 0.0


def clearance_to_obstacle(obstacle: Obstacle, robot: Segment) -> float:
    """Clearance (unit length) between the robot's swept segment and one obstacle.

    Negative clearance means the swept segment penetrates the obstacle.
    """

    if obstacle.is_point:
        return distance_point_to_segment(obstacle.position, robot[0], robot[1]) - obstacle.radius
    return distance_segment_to_segment(robot, obstacle.geometry) - (obstacle.width / 2.0)


def check_move(obstacles: list[Obstacle], robot: Segment, *, perception_radius: float | None = None) -> CollisionResult:
    """Test the robot's swept ``robot`` segment against all obstacles.

    ``perception_radius`` optionally limits which obstacles are reported as
    *perceived*; when ``None`` every obstacle is perceived (no finite field of
    view is assumed). An obstacle blocks the move when its clearance is
    non-positive.
    """

    result = CollisionResult(blocked=False, nearest_clearance=0.0)
    for obstacle in obstacles:
        distance_to_robot = distance_point_to_segment(obstacle.position, robot[0], robot[1])
        if perception_radius is not None and distance_to_robot > perception_radius:
            continue
        result.perceived.append(obstacle.obstacle_id)
        clearance = clearance_to_obstacle(obstacle, robot)
        if clearance < result.nearest_clearance:
            result.nearest_clearance = clearance
        if clearance < -EPS:
            result.blocked = True
            result.blocked_by.append(obstacle.obstacle_id)
    if not result.perceived:
        result.nearest_clearance = 0.0
    return result