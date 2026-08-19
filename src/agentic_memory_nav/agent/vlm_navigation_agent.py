"""VLM navigation agent — a direct-decision baseline (dynamic, self-deciding).

Unlike :class:`NavigationAgent` (which plans from a pre-built scene graph and a
deterministic rule-based planner), this agent lets the VLM **directly decide the
next motion primitive from the current RGB frame**. It is a *baseline*: no scene
graph, no long-term memory, no replanning — just the live camera and a task
instruction, so it isolates "how far can a single vision-language model go with
direct reactive control?".

Design
------
* **Prompt** — a system prompt that constrains the model to emit *only* a JSON
  action, with a fixed action vocabulary and a short reasoning field. This keeps
  latency low and parsing deterministic.
* **Action set** — six motion primitives, one-per-step:

    +------------+--------------------------------------------------+
    | action     | base_link velocity (forward=+X, left=+Y, yaw CCW)|
    +------------+--------------------------------------------------+
    | ``forward``| ``vx`` forward                                 |
    | ``back``   | ``-vx`` forward                                |
    | ``left``   | ``+vy`` (strafe left)                         |
    | ``right``  | ``-vy`` (strafe right)                        |
    | ``turn_left``  | ``+wz`` (counter-clockwise)              |
    | ``turn_right`` | ``-wz`` (clockwise)                    |
    +------------+--------------------------------------------------+

  Speeds are clamped by the config ``max_speed`` / ``max_angular_speed`` so the
  VLM can never ask for an unsafe velocity.
* **Decision** — one chat/completions call per step (OpenAI-compatible vLLM),
  temperature 0, ``enable_thinking=False``. If the model's output cannot be
  parsed into a valid action, the agent **fails closed to STOP** rather than
  guessing a motion.

The agent reuses the project's OpenAI-compatible client contract (same
``base_url`` / ``api_key`` / model as the perception VLM backend) so a single
vLLM server can drive both perception and navigation.

Run directly against the live vLLM server:

    ~/isaacsim/python.sh scripts/preview_isaacsim_navigation_vlm.py \\\
        --config configs/isaacsim_realtime_agent_internscenes.yaml --livestream
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any
from urllib import request

import numpy as np
from PIL import Image

from agentic_memory_nav.common.types import ActionType, ActionIntent, FrameObservation


# ---------------------------------------------------------------------------
# Action vocabulary (the baseline motion set).
# ---------------------------------------------------------------------------
# Base-link convention (ROS REP 105): +X forward, +Y left, +Z up; yaw is CCW about
# +Z. The executor rotates (vx, vy) from base_link to world by the robot yaw, so a
# positive ``vy`` here strafes the robot to its LEFT in base_link.
ACTION_FORWARD = "forward"
ACTION_BACK = "back"
ACTION_LEFT = "left"
ACTION_RIGHT = "right"
ACTION_TURN_LEFT = "turn_left"
ACTION_TURN_RIGHT = "turn_right"
ACTION_STOP = "stop"

# Canonical name -> (linear vx, linear vy, angular wz) in base_link units; the
# magnitudes are scaled by the config speeds at execution time.
ACTION_SIGNS: dict[str, tuple[float, float, float]] = {
    ACTION_FORWARD: (1.0, 0.0, 0.0),
    ACTION_BACK: (-1.0, 0.0, 0.0),
    ACTION_LEFT: (0.0, 1.0, 0.0),
    ACTION_RIGHT: (0.0, -1.0, 0.0),
    ACTION_TURN_LEFT: (0.0, 0.0, 1.0),
    ACTION_TURN_RIGHT: (0.0, 0.0, -1.0),
    ACTION_STOP: (0.0, 0.0, 0.0),
}
ACTION_NAMES = tuple(ACTION_SIGNS)


def _action_type(action: str) -> ActionType:
    """Map a primitive to the coarse ActionType used by the rest of the pipeline."""
    if action in (ACTION_STOP,):
        return ActionType.STOP
    if action in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
        return ActionType.EXPLORE
    return ActionType.NAVIGATE


_SYSTEM_PROMPT = (
    "You are the vision controller of a quadruped robot (Unitree Go2) navigating an "
    "indoor scene. You receive the robot's current head-camera image (a forward-facing "
    "view: the bottom of the image is the floor, the top is the ceiling, left/right are "
    "the robot's left/right) and a natural-language navigation instruction.\n"
    "Decide the SINGLE motion primitive to execute next, then immediately continue. "
    "Prefer moving/turning toward the instruction's goal; stop or turn when you are "
    "facing a wall, an obstacle, or the goal is already in front of you.\n"
    "You may choose exactly one action from this set: forward, back, left, right, "
    "turn_left, turn_right, stop.\n"
    "Respond with ONLY a single JSON object and nothing else. No markdown, no code "
    "fences, no prose. The object must have exactly these fields:\n"
    "  - \"action\": one of [forward, back, left, right, turn_left, turn_right, stop]\n"
    "  - \"reason\": a short (<= 20 word) justification\n"
    "  - \"confidence\": a number between 0 and 1\n"
    "Do not include any other keys or any text before/after the JSON."
)


class VLMSelfDecidingNavigationAgent:
    """Direct-decision navigation baseline: one VLM call per step -> one action."""

    def __init__(
        self,
        instruction: str,
        *,
        model_id: str,
        base_url: str = "http://10.6.32.16:8000/v1",
        api_key: str = "dummy",
        api: str = "openai-completions",
        timeout: float = 120.0,
        max_speed: float = 0.35,
        max_angular_speed: float = 0.8,
    ) -> None:
        self.instruction = instruction
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api = api
        self.timeout = timeout
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self._step = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def decide(
        self, frame: FrameObservation
    ) -> ActionIntent:
        """Return the next :class:`ActionIntent` for the given RGB frame."""
        self._step += 1
        action_name, reason, confidence = self._decide(frame)
        sign = ACTION_SIGNS[action_name]
        return ActionIntent(
            action_id=f"vlm_step_{self._step}_{action_name}",
            action_type=_action_type(action_name),
            target=self.instruction,
            waypoint=None,
            duration=1.0 / 30.0,
            safety_constraints=[
                f"vx<=±{self.max_speed}",
                f"vy<=±{self.max_speed}",
                f"wz<=±{self.max_angular_speed}",
            ],
            confidence=confidence,
            reason=reason,
            expected_observation=f"after {action_name}: {self.instruction}",
        )

    def decide_velocity(
        self, frame: FrameObservation
    ) -> tuple[float, float, float]:
        """Return (vx, vy, wz) in base_link units for the current frame (baseline)."""
        action_name, _reason, _confidence = self._decide(frame)
        return self.velocity_for(action_name)

    def decide_action(
        self, frame: FrameObservation
    ) -> tuple[str, str, float]:
        """One VLM call -> (action_name, reason, confidence). Use with ``velocity_for``
        to convert to a velocity without a second model call."""
        return self._decide(frame)

    def velocity_for(self, action: str) -> tuple[float, float, float]:
        """Convert an action name to a base-link (vx, vy, wz), scaled by config speeds."""
        vx, vy, wz = ACTION_SIGNS[action]
        return vx * self.max_speed, vy * self.max_speed, wz * self.max_angular_speed

    # ------------------------------------------------------------------
    # VLM decision
    # ------------------------------------------------------------------
    def _decide(self, frame: FrameObservation) -> tuple[str, str, float]:
        try:
            response = self._post(self._build_request(frame))
            parsed = self._parse_action(response)
        except Exception as error:  # noqa: BLE001 — any failure => fail closed
            return ACTION_STOP, f"decision failed: {error}", 0.0
        if parsed is None:
            # No valid action in the model output: fail closed to STOP.
            return ACTION_STOP, "unparseable VLM output; stopping", 0.0
        return parsed

    def _build_request(self, frame: FrameObservation) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Instruction: {self.instruction}"},
                        {
                            "type": "text",
                            "text": (
                                "Current robot pose (odom): "
                                f"x={frame.robot_pose.position[0]:.2f} "
                                f"y={frame.robot_pose.position[1]:.2f} "
                                f"yaw={frame.robot_pose.yaw:.2f}"
                                if frame.robot_pose is not None
                                else "Current robot pose: unknown"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _rgb_data_url(frame.rgb)},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
        if self.api == "openai-completions":
            # vLLM keeps emitting reasoning text unless thinking is disabled; send it
            # at the top level (the OpenAI client's extra_body is flattened for us).
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)

    def _parse_action(
        self, response: dict[str, Any]
    ) -> tuple[str, str, float] | None:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(content, str):
            return None
        parsed = _extract_json(content)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("action"), str):
            return None
        action = parsed["action"].strip().lower()
        if action not in ACTION_SIGNS:
            return None
        confidence = float(parsed.get("confidence", 0.5))
        reason = str(parsed.get("reason", "")).strip()[:120]
        return action, reason, confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rgb_data_url(rgb: np.ndarray) -> str:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None