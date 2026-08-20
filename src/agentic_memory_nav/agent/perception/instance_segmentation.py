"""Instance-mask contracts and geometry enrichment for object observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from agentic_memory_nav.common.types import FrameObservation, MappingUpdate, ObjectObservation
from agentic_memory_nav.agent.geometry.backprojection import backproject_mask
from agentic_memory_nav.agent.geometry.pointcloud_store import PointCloudStore


@dataclass(slots=True)
class InstanceMask:
    observation_id: str
    mask: np.ndarray
    confidence: float
    provenance: list[str]


class InstanceSegmenter(Protocol):
    def segment(
        self,
        frame: FrameObservation,
        candidates: list[ObjectObservation],
    ) -> list[InstanceMask]: ...


class BoundingBoxSegmenter:
    """Deterministic fallback that converts VLM/detector boxes into binary masks."""

    def segment(
        self,
        frame: FrameObservation,
        candidates: list[ObjectObservation],
    ) -> list[InstanceMask]:
        height, width = frame.rgb.shape[:2]
        masks: list[InstanceMask] = []
        for candidate in candidates:
            left, top, right, bottom = candidate.bbox_2d
            mask = np.zeros((height, width), dtype=bool)
            left, right = sorted((max(0, left), min(width, right)))
            top, bottom = sorted((max(0, top), min(height, bottom)))
            if left < right and top < bottom:
                mask[top:bottom, left:right] = True
            masks.append(
                InstanceMask(
                    observation_id=candidate.observation_id,
                    mask=mask,
                    confidence=candidate.confidence,
                    provenance=[candidate.observation_id, "bounding_box_segmenter"],
                )
            )
        return masks


class InstanceGeometryEnricher:
    """Attaches a point-cloud artifact to observations with valid instance masks."""

    def __init__(self, segmenter: InstanceSegmenter, store: PointCloudStore) -> None:
        self.segmenter = segmenter
        self.store = store

    def enrich(
        self,
        frame: FrameObservation,
        mapping: MappingUpdate,
        observations: list[ObjectObservation],
    ) -> list[ObjectObservation]:
        if frame.camera_intrinsics is None:
            return observations
        masks = {item.observation_id: item for item in self.segmenter.segment(frame, observations)}
        for observation in observations:
            instance_mask = masks.get(observation.observation_id)
            if instance_mask is None:
                continue
            points = backproject_mask(instance_mask.mask, mapping, frame.camera_intrinsics)
            if len(points) == 0:
                continue
            geometry = self.store.put(
                observation.observation_id,
                points,
                coordinate_frame="world",
                confidence=min(observation.confidence, instance_mask.confidence),
                mask_provenance=instance_mask.provenance,
            )
            observation.geometry = geometry
            observation.center_3d = geometry.centroid_3d
            observation.dimensions_3d = geometry.dimensions_3d
            observation.provenance.extend(geometry.mask_provenance)
        return observations
