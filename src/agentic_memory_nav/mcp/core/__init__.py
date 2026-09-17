"""Dependency-free simulation core for the MCP infinite canvas.

Everything in this package is plain Python (no third-party dependencies) so the
state-transition logic can be tested in isolation and is fully deterministic.
"""

from __future__ import annotations

from .actions import ACTION_BY_ID, resolve_action
from .engine import SimulationEngine, ActionResult
from .models import Event, Obstacle, Pose, Robot, TrajectorySegment

__all__ = [
    "ACTION_BY_ID",
    "ActionResult",
    "Event",
    "Obstacle",
    "Pose",
    "Robot",
    "SimulationEngine",
    "TrajectorySegment",
    "resolve_action",
]