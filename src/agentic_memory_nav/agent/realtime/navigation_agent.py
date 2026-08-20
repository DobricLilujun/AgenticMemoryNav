"""Agent responsible for turning current graph/knowledge memory into a navigation action.

This is the "navigation instruction" half of the realtime loop: it owns task parsing and
the deterministic planner. It never touches sensors or the scene graph directly; it only
reads the `SceneGraph`/`SQLiteMemory` state produced by `MemoryAgent`.
"""

from __future__ import annotations

from agentic_memory_nav.common.types import NavigationPlan, NavigationTask, Pose3D
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.agent.planning.rule_based_fallback import RuleBasedPlanner
from agentic_memory_nav.agent.planning.task_parser import RuleBasedTaskParser
from agentic_memory_nav.scene_graph.graph import SceneGraph


class NavigationAgent:
    """Decide the next navigation action from the current memory graph state."""

    def __init__(
        self,
        instruction: str,
        *,
        planner: RuleBasedPlanner | None = None,
    ) -> None:
        self.task: NavigationTask = RuleBasedTaskParser().parse(instruction)
        self.planner = planner or RuleBasedPlanner()

    def decide(
        self,
        robot_pose: Pose3D,
        graph: SceneGraph,
        memory: SQLiteMemory,
        *,
        replan_reason: str | None = None,
    ) -> NavigationPlan:
        return self.planner.plan(
            self.task,
            robot_pose,
            graph,
            memory,
            replan_reason=replan_reason,
        )
