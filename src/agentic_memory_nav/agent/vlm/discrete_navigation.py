"""VLM navigation agent using the expanded standard discrete actions.

This is a variant of :class:`VLMSelfDecidingNavigationAgent` that constrains the
model to the project's expanded ObjectNav-style action set rather than velocity
primitives. It is intended for use with :meth:`IsaacSimExecutor.apply_discrete_action`,
so no LingBot-Map, no scene graph, and no long-term memory are required.

Action set
----------
* ``turn_left``       — rotate left in place, small angle (default 15°)
* ``turn_right``      — rotate right in place, small angle (default 15°)
* ``turn_left_big``   — rotate left in place, big angle (default 90°)
* ``turn_right_big``  — rotate right in place, big angle (default 90°)
* ``move_forward``    — step forward (distance configurable via the executor)
* ``move_backward``   — step backward (same distance as move_forward)
* ``look_up``         — tilt head camera up one step (angle configurable via the executor)
* ``look_down``       — tilt head camera down one step (angle configurable via the executor)
* ``stop``            — declare the goal is found/finished

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
import re
from typing import Any

from agentic_memory_nav.agent.execution.discrete_actions import DiscreteAction
from agentic_memory_nav.agent.vlm.navigation import (
    VLMSelfDecidingNavigationAgent,
    _extract_json,
)
from agentic_memory_nav.common.types import ActionIntent, ActionType, FrameObservation


class VLMDiscreteNavigationAgent(VLMSelfDecidingNavigationAgent):
    """Direct-decision agent constrained to the expanded standard discrete actions."""

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
        self._circumnavigation_mode = False
        self._circumnavigation_steps = 0
        self._circumnavigation_side: str | None = None
        self._collision_count = 0
        self._circumnavigation_start_pos: tuple[float, float] | None = None
        self._circumnavigation_start_yaw: float | None = None
        self._clear_path_count = 0
        # +1 = camera tilted up one step, -1 = tilted down one step, 0 = level.
        self._camera_pitch_state = 0
        self._pending_note: str | None = None
        self._last_frame: FrameObservation | None = None
        self._last_action: DiscreteAction | None = None
        self._last_action_reason: str | None = None
        self._last_action_outcome: str | None = None
        self._last_obstacle_feedback: str | None = None
        self._recent_action_history: list[dict[str, str]] = []
        self._recent_obstacle_history: list[str] = []
        self._recent_view_history: list[dict[str, Any]] = []
        self._history_window = 5
        self._obstacle_history_limit = 5
        self._system_prompt = (
            "You are the vision controller of a quadruped robot (Unitree Go2) searching an "
            "indoor scene. Your goal is to find the object described in the instruction.\n"
            "You receive the robot's current forward-facing head-camera image "
            "from the robot's own point of view. In this image, the left side corresponds to "
            "the robot's left, the right side corresponds to the robot's right, the bottom of "
            "the image is the floor, and the top is the ceiling. The camera is looking ahead "
            "in the robot's forward direction.\n"
            "Visual obstacle rule:\n"
            "- If the right side of the current view is blocked by a wall, obstacle, shelf, or "
            "  clutter, prefer turning left to open a clearer route.\n"
            "- If the left side of the current view is blocked by a wall, obstacle, shelf, or "
            "  clutter, prefer turning right to open a clearer route.\n"
            "- When the forward path is blocked, do not keep alternating left/right in place; "
            "  choose a clear side and turn toward the open space.\n"
            "Spatial reasoning rule (CRITICAL):\n"
            "- Use the target's relative direction in the current image to decide how to approach "
            "  it: if the target is ahead-right, prefer a route that opens on the left side; if "
            "  it is ahead-left, prefer a route that opens on the right side.\n"
            "- When the target is visible but the direct path is blocked, do NOT repeatedly try "
            "  move_forward toward the target. Instead, first choose a clear detour by turning "
            "  left_big or right_big to circumnavigate the obstacle from an open side.\n"
            "- If a detour is not available immediately, move_backward to create space, then turn "
            "  to a new angle. Only consider move_forward again once the direct path is clearly "
            "  open.\n"
            "History memory rule:\n"
            "- The recent action history, recent obstacle history, and recent camera-view "
            "  sequence are all part of the decision context. Prioritize them over a "
            "  single-frame guess.\n"
            "- Treat the recent camera history as time-ordered visual memory: compare the "
            "  current image with the previous camera views to detect whether the robot is "
            "  re-scanning the same area or moving into a new region.\n"
            "- If the recent history shows repeated turns, repeated look actions, repeated "
            "  blocked paths, or a very similar camera orientation with little change, avoid "
            "  doing the same thing again unless the new image clearly justifies it.\n"
            "- Repeating the same action several times with little new information is low-value "
            "  exploration and should be avoided.\n"
            "Strategy:\n"
            "1. Prioritize exploring unseen areas: prefer actions that point the camera toward "
            "   regions you have not yet scanned. If the current view looks familiar or you have "
            "   already turned multiple times in this direction, choose a turn or move that faces "
            "   a new part of the room instead of repeating the same motion.\n"
            "2. DEFAULT to horizontal exploration: keep the camera at eye level and prefer "
            "   moving forward or turning only small amounts to scan the room. Use "
            "   turn_left_big / turn_right_big only when you want to quickly face a very "
            "   different direction (e.g., after deciding the goal is behind you or when the "
            "   forward direction is blocked).\n"
            "3. In open space, move forward as far as safely possible: when the view shows a "
            "   clear, unobstructed path ahead (no nearby obstacles, walls, or drop-offs), "
            "   prefer move_forward to maximize exploration. Only turn when the forward path is "
            "   blocked or you have already scanned this direction thoroughly.\n"
            "4. Obstacle handling — do not push against obstacles: if move_forward is blocked or "
            "   the view shows an obstacle directly ahead, first make a small reorientation to "
            "   shift the obstacle away from center, then move forward a short distance to "
            "   create lateral clearance. Use a large turn only when the open side is clearly "
            "   on one flank and the obstacle still blocks the center of view. If the right side "
            "   is blocked, choose a left turn; if the left side is blocked, choose a right turn. "
            "   Do not repeatedly try move_forward toward the same blocked heading, and do not "
            "   alternate left/right turns just to 'search' when the frontal path is blocked.\n"
            "5. Use the previous action history to decide the next move: if the previous action "
            "   was ineffective, blocked, or repeated in the same direction, prefer a different "
            "   route or a larger turn. If the previous action succeeded and opened a new view, "
            "   continue from that line of exploration instead of restarting.\n"
            "6. Use look_up or look_down ONLY when the goal could be above or below eye level "
            "   (e.g., on a shelf/table or on the floor very close to the robot). If the goal "
            "   is not there, the camera is automatically leveled again before your next turn "
            "   — you do not need to manually undo look_up/look_down.\n"
            "7. Rotation limits apply only in open space. If the direct route is blocked, if the "
            "   previous move_forward was ineffective, or if the robot has not made meaningful "
            "   progress, never force move_forward merely because the robot has turned about 360°. "
            "   In that case, treat the obstacle as a detour problem and commit to a clear bypass.\n"
            "8. A visible target is NOT a valid reason to push straight ahead through a blocked path. "
            "   If the direct route is obstructed, turn slightly away from the obstacle, create "
            "   lateral displacement, and re-acquire the target from a new angle. Once the path is "
            "   visibly clear, resume move_forward immediately instead of continuing to turn.\n"
            "9. Stop immediately when the goal object is clearly visible and centered in the image. "
            "   Once you can confidently say the instruction is satisfied, emit stop and do not "
            "   take any further exploration actions.\n"
            "You may use look_up at most {self.max_look_count} time(s) and look_down at most "
            "{self.max_look_count} time(s) during the episode.\n"
            "You may choose exactly one action from this set: turn_left, turn_right, "
            "turn_left_big, turn_right_big, move_forward, move_backward, look_up, look_down, "
            "stop.\n"
            "Respond with ONLY a single JSON object and nothing else. No markdown, no code "
            "fences, no prose. The object must have exactly these fields:\n"
            '  - "action": one of [turn_left, turn_right, turn_left_big, turn_right_big, '
            "move_forward, move_backward, look_up, look_down, stop]\n"
            '  - "reason": a short (<= 20 word) justification\n'
            '  - "confidence": a number between 0 and 1\n'
            "Do not include any other keys or any text before/after the JSON."
        )

    def note_collision(
        self, recovery_action: DiscreteAction | None = None, target_bearing: float | None = None
    ) -> None:
        """Record that the last move_forward was blocked so the next prompt warns about it.

        Called by the execution loop right after a collision-recovery turn, so the VLM's
        next decision is made with the knowledge that the previous heading was blocked
        (dynamic obstacle avoidance) instead of blindly repeating the same move.
        """
        self._collision_count += 1
        self._circumnavigation_mode = True
        self._circumnavigation_steps = 0
        self._clear_path_count = 0
        self._total_yaw_turned = 0.0

        if self._last_frame is not None:
            last_pose = getattr(self._last_frame, "robot_pose", None)
            if last_pose is not None:
                self._circumnavigation_start_pos = (last_pose.x, last_pose.y)
                self._circumnavigation_start_yaw = last_pose.yaw

        if target_bearing is not None:
            if target_bearing > 15.0:
                target_side = "left"
                target_hint = f" (target was ahead-right at +{target_bearing:.0f}°)"
            elif target_bearing < -15.0:
                target_side = "right"
                target_hint = f" (target was ahead-left at {target_bearing:.0f}°)"
            else:
                target_side = "left"
                target_hint = " (target was directly ahead)"
        else:
            target_side = "left"
            target_hint = ""

        if self._circumnavigation_side is None:
            self._circumnavigation_side = "left" if target_side == "left" else "right"
            if target_bearing is not None and target_bearing > 0:
                self._circumnavigation_side = "left"
            elif target_bearing is not None and target_bearing < 0:
                self._circumnavigation_side = "right"

        bypass_side_action = (
            DiscreteAction.TURN_LEFT_BIG
            if self._circumnavigation_side == "left"
            else DiscreteAction.TURN_RIGHT_BIG
        )

        bearing_text = target_hint
        if recovery_action is not None:
            self._last_obstacle_feedback = (
                f"Forward path blocked by obstacle{bearing_text}: do not oscillate left/right "
                f"or repeat the same heading. The robot recovered with {recovery_action.value}; "
                "choose a new clear route by turning away from the obstacle or stepping back "
                "to create space."
            )
            self._pending_note = (
                "CIRCUMNAVIGATION MODE is active. The direct path to the target is blocked. "
                f"Use {bypass_side_action.value} and then move forward to go around the obstacle. "
                "Do not choose move_forward toward the blocked heading. "
                "Do not switch the bypass side unless the selected side is visibly blocked."
            )
        else:
            self._last_obstacle_feedback = (
                f"Forward path blocked by obstacle{bearing_text}: do not oscillate left/right "
                "or repeat the same heading. Choose a new clear route by turning away from "
                "the obstacle, stepping back, or taking a wider detour."
            )
            self._pending_note = (
                "CIRCUMNAVIGATION MODE is active. The direct path to the target is blocked. "
                f"Use {bypass_side_action.value} and then move forward to go around the obstacle. "
                "Do not choose move_forward toward the blocked heading. "
                "Do not switch the bypass side unless the selected side is visibly blocked."
            )
        self._recent_obstacle_history.append(self._last_obstacle_feedback)
        if len(self._recent_obstacle_history) > self._obstacle_history_limit:
            self._recent_obstacle_history.pop(0)

    def _check_clear_path(self, frame: FrameObservation) -> None:
        if not self._circumnavigation_mode:
            return

        pose = getattr(frame, "robot_pose", None)
        if self._circumnavigation_start_pos is not None and pose is not None:
            dx = pose.x - self._circumnavigation_start_pos[0]
            dy = pose.y - self._circumnavigation_start_pos[1]
            displacement = math.hypot(dx, dy)
            if displacement > 0.5:
                self._clear_path_count = max(self._clear_path_count, 2)

        if self._last_action_reason is not None:
            last_reason = self._last_action_reason.lower()
            if not any(
                keyword in last_reason
                for keyword in ("obstacle", "blocked", "collision", "barrier", "cabinet", "wall")
            ):
                self._clear_path_count += 1
            else:
                self._clear_path_count = max(0, self._clear_path_count - 1)

        if self._clear_path_count >= 2:
            self._circumnavigation_mode = False
            self._circumnavigation_side = None
            self._circumnavigation_steps = 0
            self._circumnavigation_start_pos = None
            self._circumnavigation_start_yaw = None
            self._clear_path_count = 0

    def decide_action(self, frame: FrameObservation) -> tuple[DiscreteAction, str, float]:
        """One VLM call -> (DiscreteAction, reason, confidence).

        Enforces the look budget, the "don't spin forever" rule, and prompt-level
        auto-leveling: after roughly one full rotation (~360°) without meaningful
        forward motion, further turn actions are promoted to ``move_forward``; and if
        the camera is tilted from a previous look_up/look_down that did not find the
        target, it is leveled again before any other action proceeds.
        """
        self._last_frame = frame
        self._update_yaw_history(frame)
        if self._circumnavigation_mode:
            self._check_clear_path(frame)
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
        action, reason = self._enforce_circumnavigation(action, reason, frame)
        action, reason = self._enforce_rotation_limit(action, reason)
        previous_pitch_state = self._camera_pitch_state
        self._update_camera_pitch_state(action)
        self._last_action = action
        self._last_action_reason = reason
        self._last_action_outcome = (
            "goal found and stop selected"
            if action is DiscreteAction.STOP
            else "exploration chosen based on current view and prior action history"
        )
        self._record_action_history(action, reason)
        self._record_view_history(
            action=action,
            reason=reason,
            pitch_before=previous_pitch_state,
            pitch_after=self._camera_pitch_state,
        )
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

    def _record_action_history(self, action: DiscreteAction, reason: str) -> None:
        entry = {
            "action": action.value,
            "reason": reason,
        }
        self._recent_action_history.append(entry)
        if len(self._recent_action_history) > self._history_window:
            self._recent_action_history.pop(0)

    def _record_view_history(
        self, *, action: DiscreteAction, reason: str, pitch_before: int, pitch_after: int
    ) -> None:
        pitch_label_before = self._describe_pitch_state(pitch_before)
        pitch_label_after = self._describe_pitch_state(pitch_after)
        self._recent_view_history.append(
            {
                "action": action.value,
                "reason": reason,
                "pitch_before": pitch_label_before,
                "pitch_after": pitch_label_after,
            }
        )
        if len(self._recent_view_history) > self._history_window:
            self._recent_view_history.pop(0)

    @staticmethod
    def _describe_pitch_state(pitch_state: int) -> str:
        if pitch_state > 0:
            return "up"
        if pitch_state < 0:
            return "down"
        return "level"

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
        if self._circumnavigation_mode:
            return action, reason
        if self._recent_obstacle_history:
            return action, reason
        if action not in (
            DiscreteAction.TURN_LEFT,
            DiscreteAction.TURN_RIGHT,
            DiscreteAction.TURN_LEFT_BIG,
            DiscreteAction.TURN_RIGHT_BIG,
        ):
            return action, reason
        if self._total_yaw_turned < 2 * math.pi - 1e-6:
            return action, reason
        return (
            DiscreteAction.MOVE_FORWARD,
            f"open-space scan completed; moving forward ({reason})",
        )

    def _compute_target_bearing(self, frame: FrameObservation) -> str | None:
        """Best-effort target bearing from a target detection object or image center."""
        target_info = getattr(frame, "target_detection", None)
        if target_info is not None:
            if isinstance(target_info, dict):
                center_x = target_info.get("center_x")
                bbox = target_info.get("bbox_2d")
                if center_x is None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    center_x = (bbox[0] + bbox[2]) / 2.0
                if center_x is not None:
                    width = getattr(frame.rgb, "shape", (0, 0))[1] if getattr(frame, "rgb", None) is not None else 0
                    if width > 0:
                        normalized = (float(center_x) / float(width)) * 2.0 - 1.0
                        bearing_deg = normalized * 45.0
                        if abs(bearing_deg) < 15.0:
                            return "directly ahead (bearing ≈ 0°)"
                        if bearing_deg > 0:
                            return f"ahead-right (bearing ≈ +{bearing_deg:.0f}°)"
                        return f"ahead-left (bearing ≈ {bearing_deg:.0f}°)"
            elif hasattr(target_info, "center_x"):
                center_x = float(target_info.center_x)
                width = getattr(frame.rgb, "shape", (0, 0))[1] if getattr(frame, "rgb", None) is not None else 0
                if width > 0:
                    normalized = (center_x / float(width)) * 2.0 - 1.0
                    bearing_deg = normalized * 45.0
                    if abs(bearing_deg) < 15.0:
                        return "directly ahead (bearing ≈ 0°)"
                    if bearing_deg > 0:
                        return f"ahead-right (bearing ≈ +{bearing_deg:.0f}°)"
                    return f"ahead-left (bearing ≈ {bearing_deg:.0f}°)"

        rgb = getattr(frame, "rgb", None)
        if rgb is not None and getattr(rgb, "shape", None) is not None:
            width = rgb.shape[1]
            if width > 0:
                center = width / 2.0
                normalized = 0.0
                return f"directly ahead (bearing ≈ {normalized:.0f}°)"
        return None

    @staticmethod
    def _parse_bearing_label(bearing_text: str | None) -> float | None:
        if bearing_text is None:
            return None
        match = re.search(r"[-+]?\d+(?:\.\d+)?", bearing_text)
        if match is None:
            return None
        return float(match.group(0))

    def _enforce_circumnavigation(
        self, action: DiscreteAction, reason: str, frame: FrameObservation
    ) -> tuple[DiscreteAction, str]:
        if not self._circumnavigation_mode and not self._recent_obstacle_history:
            return action, reason

        if self._circumnavigation_mode:
            self._circumnavigation_steps += 1

            if self._clear_path_count >= 2:
                self._circumnavigation_mode = False
                self._circumnavigation_side = None
                self._circumnavigation_steps = 0
                self._circumnavigation_start_pos = None
                self._circumnavigation_start_yaw = None
                self._clear_path_count = 0
                return DiscreteAction.MOVE_FORWARD, f"path clear; exit avoidance mode ({reason})"

            if self._circumnavigation_steps >= 10:
                self._circumnavigation_mode = False
                self._circumnavigation_side = None
                self._circumnavigation_steps = 0
                self._circumnavigation_start_pos = None
                self._circumnavigation_start_yaw = None
                self._clear_path_count = 0
                return action, f"avoidance timeout reached; return control ({reason})"

            preferred_turn = (
                DiscreteAction.TURN_LEFT if self._circumnavigation_side == "left" else DiscreteAction.TURN_RIGHT
            )
            if self._circumnavigation_steps <= 2:
                if action is preferred_turn:
                    return action, f"avoidance step {self._circumnavigation_steps}: small turn ({reason})"
                return preferred_turn, f"avoidance step {self._circumnavigation_steps}: forced small turn ({reason})"

            if action is DiscreteAction.MOVE_FORWARD:
                return action, f"avoidance forward step ({reason})"
            if action in (
                DiscreteAction.TURN_LEFT_BIG,
                DiscreteAction.TURN_RIGHT_BIG,
            ):
                if self._circumnavigation_steps <= 6:
                    return preferred_turn, f"avoidance step {self._circumnavigation_steps}: replaced big turn with small turn ({reason})"
            return action, reason

        if action is not DiscreteAction.MOVE_FORWARD:
            return action, reason

        bearing_text = self._compute_target_bearing(frame)
        if bearing_text is None:
            return action, reason

        bearing_deg = self._parse_bearing_label(bearing_text)
        if bearing_deg is None:
            return action, reason

        if bearing_deg > 15.0:
            return (
                DiscreteAction.TURN_LEFT,
                f"target ahead-right but path blocked; small detour to left side ({reason})",
            )
        if bearing_deg < -15.0:
            return (
                DiscreteAction.TURN_RIGHT,
                f"target ahead-left but path blocked; small detour to right side ({reason})",
            )
        return (
            DiscreteAction.TURN_LEFT,
            f"target directly ahead but blocked; small detour to left side ({reason})",
        )

    def _enforce_look_budget(
        self, action: DiscreteAction, reason: str
    ) -> tuple[DiscreteAction, str]:
        if action is DiscreteAction.LOOK_UP:
            if self._camera_pitch_state > 0:
                return (
                    DiscreteAction.TURN_LEFT,
                    f"camera already pitched up; look_up budget exhausted; turned left instead ({reason})",
                )
            self._look_up_used += 1
        elif action is DiscreteAction.LOOK_DOWN:
            if self._camera_pitch_state < 0:
                return (
                    DiscreteAction.TURN_RIGHT,
                    f"camera already pitched down; look_down budget exhausted; turned right instead ({reason})",
                )
            self._look_down_used += 1
        return action, reason

    def decide(self, frame: FrameObservation) -> ActionIntent:
        self._step += 1
        action, reason, confidence = self.decide_action(frame)
        if action in (DiscreteAction.MOVE_FORWARD, DiscreteAction.MOVE_BACKWARD):
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

    def _format_recent_action_history(self) -> str:
        if not self._recent_action_history:
            return "Recent action history: no prior action yet."
        formatted = "Recent action history:\n"
        for index, entry in enumerate(self._recent_action_history, start=1):
            action = entry.get("action", "unknown")
            reason = entry.get("reason", "no recorded reason")
            formatted += f"- step {index}: action={action}; reason={reason}\n"
        return formatted.strip()

    def _format_recent_obstacle_history(self) -> str:
        if not self._recent_obstacle_history:
            return "Recent obstacle history: no blocked path recorded."
        formatted = "Recent obstacle history:\n"
        for index, event in enumerate(self._recent_obstacle_history, start=1):
            formatted += f"- obstacle {index}: {event}\n"
        if self._circumnavigation_mode:
            formatted += (
                "Circumnavigation status:\n"
                f"- mode: active\n"
                f"- steps: {self._circumnavigation_steps}\n"
                f"- clear_path_count: {self._clear_path_count}\n"
                f"- avoidance_direction: {self._circumnavigation_side or 'unset'}\n"
            )
        return formatted.strip()

    def _format_recent_view_history(self) -> str:
        if not self._recent_view_history:
            return "Recent camera-view history: no prior camera observations."
        formatted = "Recent camera-view history (time ordered oldest -> newest):\n"
        for index, entry in enumerate(self._recent_view_history, start=1):
            action = entry.get("action", "unknown")
            reason = entry.get("reason", "no recorded reason")
            pitch_before = entry.get("pitch_before", "level")
            pitch_after = entry.get("pitch_after", "level")
            formatted += (
                f"- view {index}: action={action}; camera={pitch_before} -> {pitch_after}; "
                f"reason={reason}\n"
            )
        return formatted.strip()

    def _format_last_action_history(self) -> str:
        if self._last_action is None:
            return "Previous action history: no prior action yet."
        action = self._last_action.value
        reason = self._last_action_reason or "no recorded reason"
        outcome = self._last_action_outcome or "outcome unknown"
        obstacle_feedback = self._last_obstacle_feedback or "no obstacle feedback"
        return (
            "Previous action history: "
            f"last_action={action}; last_reason={reason}; outcome={outcome}; "
            f"obstacle_feedback={obstacle_feedback}. This means the robot should avoid "
            "repeating the same blocked route and choose a new heading or detour instead "
            "of oscillating left/right."
        )

    def _build_request(self, frame: FrameObservation) -> dict[str, Any]:
        # Reuse the base request builder but swap the system prompt.
        payload = super()._build_request(frame)
        for message in payload["messages"]:
            if message.get("role") == "system":
                message["content"] = self._system_prompt

        bearing_desc = self._compute_target_bearing(frame)
        bearing_text = (
            f"Target spatial relation: {bearing_desc}\n"
            if bearing_desc is not None
            else "Target spatial relation: not detected in current view\n"
        )
        history_text = self._format_last_action_history()
        recent_history_text = self._format_recent_action_history()
        obstacle_history_text = self._format_recent_obstacle_history()
        view_history_text = self._format_recent_view_history()
        circumvention_example = (
            "\n\nExample of good obstacle avoidance:\n"
            "Situation: target is ahead-right (bearing +30°), but forward path is blocked by a "
            "large obstacle.\n"
            "Good action: turn_left_big.\n"
            "Reason: circumnavigate from the open side, then re-approach the target from a new angle.\n"
            "Bad action: move_forward toward the blocked path or turn_right toward the same obstacle.\n"
        )
        prompt_lines = [
            bearing_text,
            history_text,
            recent_history_text,
            obstacle_history_text,
            view_history_text,
            circumvention_example,
            "Use the recent history as memory and prefer a changed direction or valid detour "
            "over repeated, low-value actions. Compare the current image to the time-ordered "
            "camera history to determine whether the robot is revisiting the same area or "
            "making progress into new space. Avoid repeating turns or head tilts that match "
            "recent patterns unless the current image clearly supports a new plan.",
        ]

        for message in payload["messages"]:
            if message.get("role") == "user" and isinstance(message.get("content"), list):
                for line in prompt_lines:
                    message["content"].append({"type": "text", "text": line})
                break

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
