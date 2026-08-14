"""ObjectNav experiment loader for the InteriorAgent-style `experiments.json` schema.

This module independently reimplements only the (uncopyrightable) data schema
documented by github.com/learnsyslab/isaac-objnav-semistatic-eval's README; no
code from that repository is used (see docs/dependency-decisions.md for the
license check that led to this decision).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agentic_memory_nav.common.types import Vector3


@dataclass(slots=True)
class ObjectNavGoal:
    task: Literal["explore", "search"]
    label: str | None = None
    asset: str | list[str] | None = None
    prior_map_object: str | None = None


@dataclass(slots=True)
class ObjectNavExperiment:
    name: str
    scene: str
    goal: ObjectNavGoal
    max_runtime: float
    robot_start: Vector3 = (0.0, 0.0, 0.0)
    remove_assets: list[str] = field(default_factory=list)
    exclude_remove_assets: list[str] = field(default_factory=list)
    initialmap_experiment: str | None = None


def _parse_goal(payload: dict[str, Any]) -> ObjectNavGoal:
    return ObjectNavGoal(
        task=payload["task"],
        label=payload.get("label"),
        asset=payload.get("asset"),
        prior_map_object=payload.get("prior_map_object"),
    )


def _parse_experiment(payload: dict[str, Any]) -> ObjectNavExperiment:
    start = payload.get("robot_start", {}).get("position", [0.0, 0.0, 0.0])
    return ObjectNavExperiment(
        name=payload["name"],
        scene=payload["scene"],
        goal=_parse_goal(payload["goal"]),
        max_runtime=float(payload.get("max_runtime", 300.0)),
        robot_start=(float(start[0]), float(start[1]), float(start[2])),
        remove_assets=list(payload.get("remove_assets", [])),
        exclude_remove_assets=list(payload.get("exclude_remove_assets", [])),
        initialmap_experiment=payload.get("initialmap_experiment"),
    )


def load_experiments(path: str | Path) -> dict[str, ObjectNavExperiment]:
    """Parse an `experiments.json` file into a name-keyed mapping of experiments."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"experiments.json not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    experiments = [_parse_experiment(item) for item in payload["experiments"]]
    return {experiment.name: experiment for experiment in experiments}
