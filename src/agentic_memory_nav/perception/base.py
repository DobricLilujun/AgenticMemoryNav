"""Perception backend protocol."""

from __future__ import annotations

from typing import Protocol

from agentic_memory_nav.common.types import FrameObservation, MappingUpdate, ObjectObservation


class PerceptionBackend(Protocol):
    def detect(
        self, frame: FrameObservation, mapping: MappingUpdate
    ) -> list[ObjectObservation]: ...
