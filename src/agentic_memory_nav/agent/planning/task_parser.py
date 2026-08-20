"""Deterministic parser for the MVP instruction family."""

from __future__ import annotations

import re

from agentic_memory_nav.common.types import NavigationTask, new_id


class RuleBasedTaskParser:
    def parse(self, instruction: str) -> NavigationTask:
        lowered = instruction.lower()
        color = "red" if "red" in lowered or "红" in instruction else None
        category = self._parse_category(lowered, instruction)
        room = "kitchen" if "kitchen" in lowered or "厨房" in instruction else None
        goal = {"action": "navigate_near", "object": category, "color": color, "room": room}
        return NavigationTask(
            task_id=new_id("task"),
            natural_language_instruction=instruction,
            parsed_goal=goal,
            subgoals=[
                {"action": "locate", "room": room, "object": category, "color": color},
                {"action": "navigate_to", "object": category},
                {"action": "verify", "object": category, "color": color},
            ],
            constraints=["collision_free", "known_or_exploration_space", "bounded_velocity"],
        )

    def _parse_category(self, lowered: str, instruction: str) -> str:
        if "cup" in lowered or "杯" in instruction:
            return "cup"
        # Fallback for other targets (e.g. ObjectNav goal labels): "find the <noun>".
        match = re.search(r"\bfind (?:the )?(\w+)", lowered)
        return match.group(1) if match else "object"
