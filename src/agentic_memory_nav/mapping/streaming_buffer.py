"""Bounded frame buffer that preserves keyframes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from agentic_memory_nav.common.types import FrameObservation


@dataclass(slots=True)
class BufferedFrame:
    frame: FrameObservation
    is_keyframe: bool


class StreamingBuffer:
    def __init__(self, max_recent: int = 16, max_keyframes: int = 64) -> None:
        self._recent: deque[BufferedFrame] = deque(maxlen=max_recent)
        self._keyframes: deque[FrameObservation] = deque(maxlen=max_keyframes)

    def add(self, frame: FrameObservation, is_keyframe: bool) -> None:
        self._recent.append(BufferedFrame(frame, is_keyframe))
        if is_keyframe:
            self._keyframes.append(frame)

    @property
    def recent(self) -> list[FrameObservation]:
        return [item.frame for item in self._recent]

    @property
    def keyframes(self) -> list[FrameObservation]:
        return list(self._keyframes)

    def clear(self) -> None:
        self._recent.clear()
        self._keyframes.clear()
