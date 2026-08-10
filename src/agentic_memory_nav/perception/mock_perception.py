"""Deterministic perception for the red-cup-in-kitchen demo."""

from __future__ import annotations

import numpy as np

from agentic_memory_nav.common.types import (
    FrameObservation,
    MappingUpdate,
    ObjectObservation,
    new_id,
)


class MockPerception:
    def __init__(self) -> None:
        self._calls = 0

    def detect(self, frame: FrameObservation, mapping: MappingUpdate) -> list[ObjectObservation]:
        self._calls += 1
        observations = [
            ObjectObservation(
                observation_id=new_id("obs"),
                category="kitchen",
                attributes={"kind": "room"},
                bbox_2d=(0, 0, frame.rgb.shape[1], frame.rgb.shape[0]),
                center_3d=(2.5, 0.0, 2.5),
                dimensions_3d=(6.0, 3.0, 6.0),
                confidence=0.99,
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                provenance=[frame.frame_id, "mock_perception"],
            )
        ]
        if self._calls >= 2:
            jitter = 0.03 * (self._calls - 2)
            observations.append(
                ObjectObservation(
                    observation_id=new_id("obs"),
                    category="cup",
                    attributes={"color": "red", "material": "ceramic"},
                    bbox_2d=(42, 24, 58, 48),
                    center_3d=(2.0 + jitter, 0.85, 2.2),
                    dimensions_3d=(0.12, 0.16, 0.12),
                    confidence=0.94,
                    timestamp=frame.timestamp,
                    frame_id=frame.frame_id,
                    embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    provenance=[frame.frame_id, "mock_perception"],
                )
            )
        return observations
