"""Pure geometry primitives for the 2D canvas.

All coordinates and lengths are expressed in the abstract **unit length** defined by
the MCP infinite-canvas specification. No value here is tied to meters, centimeters,
or any other physical unit; the numbers only describe relative geometry for collision
detection and rendering.

A :data:`Point` is an ``(x, y)`` pair in unit lengths and a :data:`Segment` is a
pair of :data:`Point` values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

# A point is an (x, y) pair expressed in the abstract unit length.
Point = tuple[float, float]
# A line segment is a pair of points: (start, end).
Segment = tuple[Point, Point]

# Tolerance for treating geometric quantities as equal to zero.
EPS = 1.0e-9


def deg2rad(degrees: float) -> float:
    """Convert degrees to radians."""

    return math.radians(degrees)


def heading_vector(theta_deg: float) -> Point:
    """Return the unit heading vector for a heading in degrees.

    ``theta = 0`` faces ``+x`` and positive rotation is counter-clockwise, so the
    vector is ``(cos, sin)`` with ``y`` growing upward.
    """

    rad = deg2rad(theta_deg)
    return (math.cos(rad), math.sin(rad))


def normalize_theta(degrees: float) -> float:
    """Wrap a heading into the half-open interval ``(-180, 180]``."""

    value = degrees % 360.0
    if value > 180.0:
        value -= 360.0
    return value


def add(a: Point, b: Point) -> Point:
    """Add two points component-wise."""

    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    """Subtract point ``b`` from point ``a`` component-wise."""

    return (a[0] - b[0], a[1] - b[1])


def scale(a: Point, s: float) -> Point:
    """Scale a point by a scalar ``s``."""

    return (a[0] * s, a[1] * s)


def dot(a: Point, b: Point) -> float:
    """Dot product of two points treated as vectors."""

    return a[0] * b[0] + a[1] * b[1]


def length(a: Point) -> float:
    """Euclidean length of a point treated as a vector from the origin."""

    return math.hypot(a[0], a[1])


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""

    return math.hypot(a[0] - b[0], a[1] - b[1])


def move(along: Point, heading_deg: float, steps: float) -> Point:
    """Advance ``along`` by ``steps`` unit lengths in the ``heading_deg`` direction."""

    vx, vy = heading_vector(heading_deg)
    return (along[0] + vx * steps, along[1] + vy * steps)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def project_onto_segment(p: Point, a: Point, b: Point) -> Point:
    """Return the closest point on segment ``[a, b]`` to point ``p``."""

    ab = sub(b, a)
    denom = dot(ab, ab)
    if denom <= EPS:
        return a
    t = _clip(dot(sub(p, a), ab) / denom, 0.0, 1.0)
    return (a[0] + ab[0] * t, a[1] + ab[1] * t)


def distance_point_to_segment(p: Point, a: Point, b: Point) -> float:
    """Minimum Euclidean distance from point ``p`` to segment ``[a, b]``."""

    return distance(p, project_onto_segment(p, a, b))


def _segments_intersect(s1: Segment, s2: Segment) -> bool:
    """Return whether two segments share at least one point."""

    def orient(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    a, b = s1
    c, d = s2
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if (o1 > EPS and o2 > EPS) or (o1 < -EPS and o2 < -EPS):
        return False
    if (o3 > EPS and o4 > EPS) or (o3 < -EPS and o4 < -EPS):
        return False
    return True


def distance_segment_to_segment(s1: Segment, s2: Segment) -> float:
    """Minimum Euclidean distance between two segments.

    If the segments intersect the distance is ``0.0``; otherwise it is the smallest
    of the four endpoint-to-segment distances, which is the geometric minimum
    distance between two line segments in the plane.
    """

    if _segments_intersect(s1, s2):
        return 0.0
    a, b = s1
    c, d = s2
    candidates = (
        distance_point_to_segment(a, c, d),
        distance_point_to_segment(b, c, d),
        distance_point_to_segment(c, a, b),
        distance_point_to_segment(d, a, b),
    )
    return min(candidates)


def midpoint(a: Point, b: Point) -> Point:
    """Midpoint of two points."""

    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def as_pair(value: object) -> Point:
    """Coerce a length-2 sequence into a :data:`Point`."""

    if not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError(f"Expected a length-2 coordinate, got {value!r}")
    pair = cast(Sequence[Any], value)
    return (float(pair[0]), float(pair[1]))