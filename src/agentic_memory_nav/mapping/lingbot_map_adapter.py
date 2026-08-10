"""Optional LingBot-Map integration boundary.

The adapter deliberately fails with installation guidance when the optional backend is
unavailable. MockMapper remains the CPU fallback.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from agentic_memory_nav.common.types import FrameObservation, MappingUpdate


class LingBotMapAdapter:
    def __init__(self, checkpoint: Path, device: str = "cuda", **_: object) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._available = importlib.util.find_spec("lingbot_map") is not None

    def start(self) -> None:
        if not self._available:
            raise RuntimeError(
                "LingBot-Map is not installed. Install the optional backend "
                "or select mapping.backend=mock."
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"LingBot-Map checkpoint not found: {self.checkpoint}")
        raise NotImplementedError(
            "LingBot-Map runtime is gated pending pose-convention validation; "
            "use MockMapper for the MVP."
        )

    def update(self, frame: FrameObservation) -> MappingUpdate:
        raise RuntimeError(f"LingBot-Map adapter is not started for frame {frame.frame_id}")
