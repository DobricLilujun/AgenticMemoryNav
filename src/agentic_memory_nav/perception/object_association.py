"""Explainable geometry and appearance based object association."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from agentic_memory_nav.common.types import ObjectObservation, SceneNode


@dataclass(frozen=True, slots=True)
class AssociationDecision:
    node_id: str | None
    score: float
    accepted: bool
    components: dict[str, float]


class ObjectAssociator:
    def __init__(self, threshold: float = 0.65, max_distance: float = 1.0) -> None:
        self.threshold = threshold
        self.max_distance = max_distance

    def associate(
        self, observation: ObjectObservation, candidates: list[SceneNode]
    ) -> AssociationDecision:
        best = AssociationDecision(None, 0.0, False, {})
        for node in candidates:
            category = float(node.label == observation.category)
            distance = float(np.linalg.norm(np.asarray(node.position_3d) - observation.center_3d))
            geometry = exp(-distance / max(self.max_distance, 1e-6))
            overlap = self._iou(node.bbox_3d, observation.bbox_3d)
            appearance = self._appearance(node.embedding, observation.embedding)
            recency = exp(-max(0.0, observation.timestamp - node.last_seen) / 10.0)
            confidence = min(node.confidence, observation.confidence)
            score = (
                0.30 * category
                + 0.25 * geometry
                + 0.15 * overlap
                + 0.15 * appearance
                + 0.10 * recency
                + 0.05 * confidence
            )
            components = {
                "category": category,
                "geometry": geometry,
                "iou_3d": overlap,
                "appearance": appearance,
                "recency": recency,
                "confidence": confidence,
                "distance_m": distance,
            }
            if score > best.score:
                best = AssociationDecision(node.node_id, score, score >= self.threshold, components)
        return best

    @staticmethod
    def _appearance(stored: list[float] | None, observed: np.ndarray | None) -> float:
        if stored is None or observed is None:
            return 0.5
        left, right = np.asarray(stored), np.asarray(observed)
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        return float(np.dot(left, right) / denominator) if denominator else 0.0

    @staticmethod
    def _iou(left: tuple[float, ...] | None, right: tuple[float, ...]) -> float:
        if left is None:
            return 0.0
        low = np.maximum(left[:3], right[:3])
        high = np.minimum(left[3:], right[3:])
        intersection = float(np.prod(np.maximum(0.0, high - low)))
        left_volume = float(np.prod(np.maximum(0.0, np.asarray(left[3:]) - left[:3])))
        right_volume = float(np.prod(np.maximum(0.0, np.asarray(right[3:]) - right[:3])))
        union = left_volume + right_volume - intersection
        return intersection / union if union > 0 else 0.0
