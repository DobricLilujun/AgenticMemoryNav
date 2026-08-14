"""Graph and memory aware deterministic planner."""

from __future__ import annotations

from agentic_memory_nav.common.types import (
    ActionIntent,
    ActionType,
    NavigationPlan,
    NavigationTask,
    Pose3D,
    new_id,
)
from agentic_memory_nav.memory.knowledge_memory import KnowledgeMemory
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.reasoning.native_reasoner import NativeReasoner
from agentic_memory_nav.scene_graph.graph import SceneGraph


class RuleBasedPlanner:
    def __init__(self, approach_distance: float = 0.6) -> None:
        self.apach_distance = approach_distance

    def plan(
        self,
        task: NavigationTask,
        robot_pose: Pose3D,
        graph: SceneGraph,
        memory: SQLiteMemory,
        replan_reason: str | None = None,
    ) -> NavigationPlan:
        goal = task.parsed_goal
        reasoning = NativeReasoner(KnowledgeMemory(memory)).resolve(graph, goal)
        memory_hits = memory.retrieve_for_task(task.natural_language_instruction)
        assumptions = ["plan uses deterministic geometric relations", reasoning.reason]
        if replan_reason:
            assumptions.append(f"replanned because: {replan_reason}")

        if reasoning.target_id is not None:
            target = graph.get_node(reasoning.target_id)
            room_ok = not reasoning.requires_verification
            waypoint = (
                target.position_3d[0] - self.apach_distance,
                robot_pose.position[1],
                target.position_3d[2],
            )
            action = ActionIntent(
                action_id=new_id("action"),
                action_type=ActionType.NAVIGATE,
                target=target.node_id,
                waypoint=waypoint,
                duration=8.0,
                safety_constraints=list(task.constraints),
                confidence=reasoning.confidence,
                reason=reasoning.reason,
                expected_observation="red cup visible at close range",
            )
            return NavigationPlan(
                task_id=task.task_id,
                goal=task.natural_language_instruction,
                subgoals=task.subgoals,
                required_entities=[str(goal["room"]), str(goal["object"])],
                waypoints=[waypoint],
                information_gaps=[] if room_ok else ["target room relation requires verification"],
                assumptions=assumptions,
                confidence=reasoning.confidence,
                replan_required=not room_ok,
                action=action,
            )

        exploration = (
            robot_pose.position[0] + 0.75,
            robot_pose.position[1],
            robot_pose.position[2],
        )
        action = ActionIntent(
            action_id=new_id("action"),
            action_type=ActionType.EXPLORE,
            target=str(goal.get("room")),
            waypoint=exploration,
            duration=4.0,
            safety_constraints=[*task.constraints, "reduced_exploration_speed"],
            confidence=0.45 if memory_hits else 0.35,
            reason="target has not been observed",
            expected_observation="new view of the kitchen",
        )
        return NavigationPlan(
            task_id=task.task_id,
            goal=task.natural_language_instruction,
            subgoals=task.subgoals,
            required_entities=[str(goal["room"]), str(goal["object"])],
            waypoints=[exploration],
            information_gaps=["red cup location is unknown"],
            assumptions=assumptions,
            confidence=action.confidence,
            replan_required=True,
            action=action,
        )
