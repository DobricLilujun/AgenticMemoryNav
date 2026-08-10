"""Lazy optional VLM backend boundary."""

from __future__ import annotations

from agentic_memory_nav.common.types import FrameObservation, MappingUpdate, ObjectObservation


class VLMBackend:
    def __init__(self, model_id: str, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device

    def detect(self, frame: FrameObservation, mapping: MappingUpdate) -> list[ObjectObservation]:
        raise RuntimeError(
            f"VLM backend {self.model_id!r} is optional and not enabled. "
            "Select perception.backend=mock."
        )
