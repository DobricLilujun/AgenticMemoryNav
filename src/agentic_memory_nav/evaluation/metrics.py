"""Lightweight geometry, semantic, memory, and navigation metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from agentic_memory_nav.common.types import Pose3D


def absolute_trajectory_error(reference: Sequence[Pose3D], estimate: Sequence[Pose3D]) -> float:
    if not reference or len(reference) != len(estimate):
        return 0.0
    errors = [
        math.dist(left.position, right.position)
        for left, right in zip(reference, estimate, strict=True)
    ]
    return float(np.sqrt(np.mean(np.square(errors))))


def relative_pose_error(reference: Sequence[Pose3D], estimate: Sequence[Pose3D]) -> float:
    if len(reference) < 2 or len(reference) != len(estimate):
        return 0.0
    errors = []
    for index in range(1, len(reference)):
        reference_step = np.asarray(reference[index].position) - reference[index - 1].position
        estimate_step = np.asarray(estimate[index].position) - estimate[index - 1].position
        errors.append(float(np.linalg.norm(reference_step - estimate_step)))
    return float(np.mean(errors))


def success_weighted_path_length(success: bool, shortest: float, actual: float) -> float:
    return float(success) * shortest / max(shortest, actual, 1e-9)


def precision_recall(true_positive: int, predicted: int, expected: int) -> dict[str, float]:
    return {
        "precision": true_positive / max(1, predicted),
        "recall": true_positive / max(1, expected),
    }


def provenance_completeness(provenance_lists: Sequence[Sequence[str]]) -> float:
    return sum(bool(items) for items in provenance_lists) / max(1, len(provenance_lists))
