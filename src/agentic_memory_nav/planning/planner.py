"""Optional LLM planner boundary with deterministic fallback."""

from __future__ import annotations


class LLMPlanner:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    def plan(self, *_: object, **__: object) -> None:
        raise RuntimeError(
            f"LLM planner {self.provider}/{self.model} is not configured; "
            "select planning.backend=rule_based."
        )
