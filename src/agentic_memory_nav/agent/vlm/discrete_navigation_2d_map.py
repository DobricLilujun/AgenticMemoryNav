"""VLM discrete-action agent augmented with a 2D top-down spatial memory map.

This is an **ablation-isolated** variant of :class:`VLMDiscreteNavigationAgent`.
It keeps the same discrete action set and safety rules, but adds a dedicated
system prompt and request builder that present the live top-down map to the VLM
as an extra image.  The map shows the robot pose, trajectory, and any semantic
obstacles/landmarks observed so far.

Because the prompt and map handling live in this class, the baseline
:class:`VLMDiscreteNavigationAgent` remains untouched — enabling clean A/B
comparisons between "camera-only" and "camera + 2D map" navigation.
"""

from __future__ import annotations

import base64
import math
import re
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from agentic_memory_nav.agent.execution.discrete_actions import DiscreteAction
from agentic_memory_nav.agent.vlm.discrete_navigation import VLMDiscreteNavigationAgent
from agentic_memory_nav.common.types import FrameObservation


class VLMDiscreteNavigation2DMapAgent(VLMDiscreteNavigationAgent):
    """Discrete VLM agent that also receives a 2D top-down memory map image."""

    def __init__(
        self,
        instruction: str,
        *,
        model_id: str,
        base_url: str = "http://10.6.32.16:8000/v1",
        api_key: str = "dummy",
        api: str = "openai-completions",
        timeout: float = 120.0,
        max_look_count: int = 1,
        map_image_interval: int = 1,
    ) -> None:
        super().__init__(
            instruction,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            api=api,
            timeout=timeout,
            max_look_count=max_look_count,
        )
        # How often (in decision steps) to include the top-down map image.
        # 1 = every step; larger values reduce token/latency cost.
        self.map_image_interval = max(1, map_image_interval)
        self._2d_map_system_prompt = self._build_2d_map_system_prompt()
        self._last_map_image: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    def _build_2d_map_system_prompt(self) -> str:
        return (
            "You are the vision controller of a quadruped robot (Unitree Go2) searching an "
            "indoor scene. You receive TWO images at every step:\n"
            "1. The robot's current forward-facing head-camera image "
            "(bottom = floor, top = ceiling, left/right = robot's left/right).\n"
            "2. A 2D top-down spatial memory map built from the robot's odometry and prior "
            "observations.\n\n"
            "How to read the top-down map:\n"
            "- The cyan star is the robot; the cyan arrow shows its heading.\n"
            "- The blue line is the robot's trajectory so far.\n"
            "- Red markers with text labels are obstacles or objects the robot has observed.\n"
            "- X increases to the right, Y increases upward. The map is in metric meters.\n"
            "- Use the map to detect dead ends, revisited areas, and open directions.\n\n"
            "Decision strategy:\n"
            "1. Prefer moving toward the goal while avoiding regions already explored unless "
            "   the goal may be there.\n"
            "2. Use the map to choose the open side around an obstacle: if the map shows free "
            "   space on the left, turn left; if free space on the right, turn right.\n"
            "3. If the camera view and the map disagree (e.g., the camera sees open space but "
            "   the map shows a previously recorded obstacle), trust the camera for the "
            "   immediate next step but still record the conflict in your reasoning.\n"
            "4. When the forward path is blocked, do not alternate left/right in place; commit "
            "   to one clear side and move forward around the obstacle.\n"
            "5. Use look_up/look_down ONLY when the goal could be above or below eye level; "
            "   the camera is auto-leveled afterward.\n"
            "6. Your objective is to keep approaching the goal. Do NOT emit stop unless the "
            "   goal is directly in front of you, clearly visible, and you are already as close "
            "   as the discrete actions allow. In all other cases, choose an action that reduces "
            "   the distance to the goal or re-positions the robot for a clearer approach.\n\n"
            "When describing objects in your reason, include a rough distance in meters if "
            "possible (e.g., \"sofa 1.5m ahead\" or \"obstacle about 2m to the left\"). "
            "This helps maintain the spatial memory map.\n\n"
            f"You may use look_up at most {self.max_look_count} time(s) and look_down at most "
            f"{self.max_look_count} time(s) during the episode.\n"
            "You may choose exactly one action from this set: turn_left, turn_right, "
            "turn_left_big, turn_right_big, move_forward, move_backward, look_up, look_down. "
            "Avoid using stop; only use stop as a last resort if the goal is directly ahead "
            "and you cannot get any closer with the available actions.\n"
            "Respond with ONLY a single JSON object and nothing else. No markdown, no code "
            "fences, no prose. The object must have exactly these fields:\n"
            '  - "action": one of [turn_left, turn_right, turn_left_big, turn_right_big, '
            "move_forward, move_backward, look_up, look_down]\n"
            '  - "reason": a short (<= 30 word) justification\n'
            '  - "confidence": a number between 0 and 1\n'
            "Do not include any other keys or any text before/after the JSON."
        )

    # ------------------------------------------------------------------
    # Public helpers for the execution loop
    # ------------------------------------------------------------------
    def set_map_image(self, map_image: np.ndarray) -> None:
        """Update the top-down map image to be sent with the next VLM request.

        The execution loop should call this after rendering the map each step.
        """
        self._last_map_image = np.asarray(map_image, dtype=np.uint8)

    @staticmethod
    def extract_landmarks_from_reason(reason: str) -> list[dict[str, float | str]]:
        """Best-effort parse of landmarks mentioned in the VLM reason string.

        All detected landmarks are drawn with the same obstacle style (red) while
        preserving the original semantic label from the VLM reason. The parser does
        not rely on a predefined object list; it looks for any noun followed by a
        distance.
        """
        results: list[dict[str, float | str]] = []
        text = reason.lower()

        # Match: <word> [optional direction words] <distance> m/meter/meters/metres
        pattern = re.compile(
            r"(\b[a-z]+\b)\s+(?:\w+\s+){0,4}?(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)"
        )

        for match in pattern.finditer(text):
            keyword = match.group(1)
            dist = float(match.group(2))

            window = text[max(0, match.start() - 30) : match.end() + 30]
            angle = 0.0
            if "left" in window:
                angle = math.radians(30.0)
            elif "right" in window:
                angle = math.radians(-30.0)
            elif "behind" in window:
                angle = math.radians(180.0)

            if dist > 0.0:
                results.append(
                    {
                        "label": keyword,
                        "category": "obstacle",
                        "distance_m": dist,
                        "angle_rad": angle,
                        "color": "red",
                    }
                )

        return results

    # ------------------------------------------------------------------
    # VLM request builder
    # ------------------------------------------------------------------
    def _build_request(self, frame: FrameObservation) -> dict[str, Any]:
        payload = super()._build_request(frame)

        # Replace the system prompt with the 2D-map-aware version.
        for message in payload["messages"]:
            if message.get("role") == "system":
                message["content"] = self._2d_map_system_prompt

        # Optionally append the top-down map image to the user message.
        if (
            self._last_map_image is not None
            and (self._step % self.map_image_interval) == 0
        ):
            map_url = self._rgb_data_url(self._last_map_image)
            for message in payload["messages"]:
                if message.get("role") == "user" and isinstance(
                    message.get("content"), list
                ):
                    message["content"].append(
                        {
                            "type": "text",
                            "text": (
                                "Top-down spatial memory map (current). The cyan star/arrow "
                                "is the robot, blue line is trajectory, red markers are "
                                "observed obstacles/objects."
                            ),
                        }
                    )
                    message["content"].append(
                        {"type": "image_url", "image_url": {"url": map_url}}
                    )
                    break

        return payload

    @staticmethod
    def _rgb_data_url(rgb: np.ndarray) -> str:
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
