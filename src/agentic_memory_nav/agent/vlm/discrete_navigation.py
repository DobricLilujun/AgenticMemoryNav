"""VLM navigation agent using the standard 6 discrete actions.

This is a variant of :class:`VLMSelfDecidingNavigationAgent` that constrains the
model to the project's standard ObjectNav-style action set rather than velocity
primitives. It is intended for use with :meth:`IsaacSimExecutor.apply_discrete_action`,
so no LingBot-Map, no scene graph, and no long-term memory are required.

Action set
----------
* ``turn_left``   — rotate left in place (angle configurable via the executor)
* ``turn_right``  — rotate right in place (angle configurable via the executor)
* ``move_forward``— step forward (distance configurable via the executor)
* ``look_up``     — tilt head camera up one step (angle configurable via the executor)
* ``look_down``   — tilt head camera down one step (angle configurable via the executor)
* ``stop``        — declare the goal is found/finished

The prompt asks the VLM to actively search the room, use look_up/look_down when
the floor or shelves might contain the target, and emit ``stop`` once the green
shoe is clearly visible and centered in the frame. On top of the raw VLM output this
agent enforces three safety/behavior rules: (1) a per-direction budget on
look_up/look_down, (2) automatic re-leveling of the camera if a look did not find the
target, and (3) promoting turns to ``move_forward`` after a full rotation without
moving, so the robot keeps exploring instead of spinning in place.
"""

from __future__ import annotations

import math
from typing import Any

from agentic_memory_nav.agent.execution.discrete_actions import DiscreteAction
from agentic_memory_nav.agent.vlm.navigation import (
    VLMSelfDecidingNavigationAgent,
    _extract_json,
)
from agentic_memory_nav.common.types import ActionIntent, ActionType, FrameObservation


class VLMDiscreteNavigationAgent(VLMSelfDecidingNavigationAgent):
    """Direct-decision agent constrained to the standard 6 discrete actions.

    look_up/look_down are each limited to ``max_look_count`` executions (default 1) to
    prevent the VLM from wasting steps oscillating up and down instead of exploring.
    """

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
    ) -> None:
        super().__init__(
            instruction,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            api=api,
            timeout=timeout,
            max_speed=0.0,
            max_angular_speed=0.0,
        )
        self.max_look_count = max(0, max_look_count)
        self._look_up_used = 0
        self._look_down_used = 0
        self._total_yaw_turned = 0.0
        self._last_yaw: float | None = None
        # +1 = camera tilted up one step, -1 = tilted down one step, 0 = level.
        self._camera_pitch_state = 0
        self._pending_note: str | None = None
        self._system_prompt = (
            "You are the vision controller of a quadruped robot (Unitree Go2) searching an "
            "indoor scene. Your goal is to find the object described in the instruction.\n"
            "You receive the robot's current forward-facing head-camera image "
            "(bottom of the image is the floor, top is the ceiling, left/right are the robot's "
            "left/right) and the robot's current pose.\n"
            "Strategy:\n"
            "1. DEFAULT to horizontal exploration: keep the camera at eye level and prefer "
            "   moving forward or turning only small amounts to scan the room.\n"
            "2. Use look_up or look_down ONLY when the goal could be above or below eye level "
            "   (e.g., on a shelf/table or on the floor very close to the robot). If the goal "
            "   is not there, the camera is automatically leveled again before your next turn "
            "   — you do not need to manually undo look_up/look_down.\n"
            "3. Avoid spinning in place: if you have already turned roughly 360° without moving, "
            "   choose move_forward to explore a new location instead of another turn.\n"
            "4. Finding a free path is key: when facing open space, move_forward; when blocked, "
            "   turn to find open space, then move_forward. If move_forward is blocked by an "
            "   obstacle, the robot automatically turns to recover — on your next decision, "
            "   pick a genuinely clear direction instead of repeating the same move.\n"
            "Stop only when the goal object is clearly visible and centered in the image.\n"
            f"You may use look_up at most {self.max_look_count} time(s) and look_down at most "
            f"{self.max_look_count} time(s) during the episode.\n"
            "You may choose exactly one action from this set: turn_left, turn_right, "
            "move_forward, look_up, look_down, stop.\n"
            "Respond with ONLY a single JSON object and nothing else. No markdown, no code "
            "fences, no prose. The object must have exactly these fields:\n"
            '  - "action": one of [turn_left, turn_right, move_forward, look_up, look_down, stop]\n'
            '  - "reason": a short (<= 20 word) justification\n'
            '  - "confidence": a number between 0 and 1\n'
            "Do not include any other keys or any text before/after the JSON."
        )

    def note_collision(self, recovery_action: DiscreteAction | None = None) -> None:
        """Record that the last move_forward was blocked so the next prompt warns about it.

        Called by the execution loop right after a collision-recovery turn, so the VLM's
        next decision is made with the knowledge that the previous heading was blocked
        (dynamic obstacle avoidance) instead of blindly repeating the same move.
        """
        if recovery_action is not None:
            self._pending_note = (
                "Your last move_forward was blocked by an obstacle directly ahead; the robot "
                f"automatically executed {recovery_action.value} to recover. Look at the new "
                "view and choose a genuinely clear direction — do not immediately repeat "
                "move_forward toward the same heading."
            )
        else:
            self._pending_note = (
                "Your last move_forward was blocked by an obstacle directly ahead. Avoid "
                "repeating move_forward toward the same heading; find a clear path first."
            )

    def decide_action(self, frame: FrameObservation) -> tuple[DiscreteAction, str, float]:
        """One VLM call -> (DiscreteAction, reason, confidence).

        Enforces the look budget, the "don't spin forever" rule, and prompt-level
        auto-leveling: after roughly one full rotation (~360°) without meaningful
        forward motion, further turn actions are promoted to ``move_forward``; and if
        the camera is tilted from a previous look_up/look_down that did not find the
        target, it is leveled again before any other action proceeds.
        """
        self._update_yaw_history(frame)
        try:
            response = self._post(self._build_request(frame))
            parsed = self._parse_discrete_action(response)
        except Exception as error:  # noqa: BLE001
            return DiscreteAction.STOP, f"decision failed: {error}", 0.0
        if parsed is None:
            return DiscreteAction.STOP, "unparseable VLM output; stopping", 0.0
        action, reason, confidence = parsed
        action, reason = self._enforce_look_budget(action, reason)
        action, reason = self._enforce_return_to_level(action, reason)
        action, reason = self._enforce_rotation_limit(action, reason)
        self._update_camera_pitch_state(action)
        return action, reason, confidence

    def _enforce_return_to_level(
        self, action: DiscreteAction, reason: str
    ) -> tuple[DiscreteAction, str]:
        """If the camera is tilted and the target wasn't found, level it before anything else."""
        if self._camera_pitch_state == 0 or action is DiscreteAction.STOP:
            return action, reason
        leveling_action = (
            DiscreteAction.LOOK_DOWN if self._camera_pitch_state > 0 else DiscreteAction.LOOK_UP
        )
        if action is leveling_action:
            return action, reason
        return leveling_action, f"leveling camera before continuing ({reason})"

    def _update_camera_pitch_state(self, action: DiscreteAction) -> None:
        if action is DiscreteAction.LOOK_UP:
            self._camera_pitch_state += 1
        elif action is DiscreteAction.LOOK_DOWN:
            self._camera_pitch_state -= 1

    def _update_yaw_history(self, frame: FrameObservation) -> None:
        yaw = frame.robot_pose.yaw if frame.robot_pose is not None else None
        if yaw is None:
            return
        if self._last_yaw is None:
            self._last_yaw = yaw
            return
        delta = self._angle_diff(yaw, self._last_yaw)
        self._total_yaw_turned += abs(delta)
        self._last_yaw = yaw

    @staticmethod
    def _angle_diff(current: float, previous: float) -> float:
        """Signed smallest angle difference in radians."""
        diff = (current - previous + math.pi) % (2 * math.pi) - math.pi
        return diff

    def _enforce_rotation_limit(
        self, action: DiscreteAction, reason: str
    ) -> tuple[DiscreteAction, str]:
        if action not in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT):
            return action, reason
        if self._total_yaw_turned < 2 * math.pi - 1e-6:
            return action, reason
        return (
            DiscreteAction.MOVE_FORWARD,
            f"already turned ~360°; moving forward to explore instead ({reason})",
        )

    def _enforce_look_budget(
        self, action: DiscreteAction, reason: str
    ) -> tuple[DiscreteAction, str]:
        if action is DiscreteAction.LOOK_UP:
            if self._look_up_used >= self.max_look_count:
                return (
                    DiscreteAction.TURN_LEFT,
                    f"look_up budget exhausted; turned left instead ({reason})",
                )
            self._look_up_used += 1
        elif action is DiscreteAction.LOOK_DOWN:
            if self._look_down_used >= self.max_look_count:
                return (
                    DiscreteAction.TURN_LEFT,
                    f"look_down budget exhausted; turned left instead ({reason})",
                )
            self._look_down_used += 1
        return action, reason

    def decide(self, frame: FrameObservation) -> ActionIntent:
        self._step += 1
        action, reason, confidence = self.decide_action(frame)
        if action is DiscreteAction.MOVE_FORWARD:
            # Reaching a new location resets the scan budget so the robot can look around again.
            self._total_yaw_turned = 0.0
        return ActionIntent(
            action_id=f"vlm_step_{self._step}_{action.value}",
            action_type=ActionType.STOP if action is DiscreteAction.STOP else ActionType.NAVIGATE,
            target=self.instruction,
            waypoint=None,
            duration=1.0 / 30.0,
            safety_constraints=["standard discrete action set"],
            confidence=confidence,
            reason=reason,
            expected_observation=f"after {action.value}: {self.instruction}",
        )

    def _build_request(self, frame: FrameObservation) -> dict[str, Any]:
        # Reuse the base request builder but swap the system prompt.
        payload = super()._build_request(frame)
        for message in payload["messages"]:
            if message.get("role") == "system":
                message["content"] = self._system_prompt
        if self._pending_note is not None:
            for message in payload["messages"]:
                if message.get("role") == "user" and isinstance(message.get("content"), list):
                    message["content"].append({"type": "text", "text": self._pending_note})
                    break
            self._pending_note = None
        return payload

    def _parse_discrete_action(
        self, response: dict[str, Any]
    ) -> tuple[DiscreteAction, str, float] | None:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(content, str):
            return None
        parsed = _extract_json(content)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("action"), str):
            return None
        try:
            action = DiscreteAction(parsed["action"].strip().lower())
        except ValueError:
            return None
        confidence = float(parsed.get("confidence", 0.5))
        reason = str(parsed.get("reason", "")).strip()[:120]
        return action, reason, confidence
