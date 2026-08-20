"""Frame-by-frame RGB agent that turns updated memory graphs into navigation actions.

Composes two independent agents so the realtime loop matches a two-role split when
driven live against Isaac Sim: `MemoryAgent` extracts/builds graph + knowledge memory
from each RGB(+pose) frame, and `NavigationAgent` turns that memory state into the next
navigation instruction. `RealtimeAgent` itself owns no sensing or planning logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_memory_nav.agent.realtime.memory_agent import MemoryAgent
from agentic_memory_nav.agent.realtime.navigation_agent import NavigationAgent
from agentic_memory_nav.common.types import FrameObservation, MappingUpdate, NavigationPlan, Pose3D
from agentic_memory_nav.agent.planning.rule_based_fallback import RuleBasedPlanner


@dataclass(frozen=True, slots=True)
class RealtimeDecision:
    frame_id: str
    mapping: MappingUpdate
    plan: NavigationPlan
    graph_nodes: int
    graph_edges: int
    knowledge_facts_created: int


class RealtimeAgent:
    """Maintain graph memory and emit a navigation action after every RGB frame.

    The caller owns camera capture and robot control. `ingest_frame` is deliberately
    synchronous and bounded: it updates state from one observation, then returns the
    next high-level action for the controller to execute.
    """

    def __init__(
        self,
        scratch_dir: Path | str,
        instruction: str,
        *,
        mapper: Any | None = None,
        perception: Any | None = None,
        planner: RuleBasedPlanner | None = None,
    ) -> None:
        self.memory_agent = MemoryAgent(Path(scratch_dir), mapper=mapper, perception=perception)
        self.navigation_agent = NavigationAgent(instruction, planner=planner)

    @property
    def task(self):
        return self.navigation_agent.task

    @property
    def graph(self):
        return self.memory_agent.graph

    @property
    def memory(self):
        return self.memory_agent.memory

    def start(self) -> None:
        self.memory_agent.start()

    def ingest_frame(
        self,
        frame: FrameObservation,
        robot_pose: Pose3D | None = None,
    ) -> RealtimeDecision:
        """Update memory graph from one frame and return the immediate next action."""
        snapshot = self.memory_agent.ingest_frame(frame)
        current_pose = robot_pose or frame.robot_pose or frame.camera_pose or Pose3D()
        plan = self.navigation_agent.decide(
            current_pose,
            self.memory_agent.graph,
            self.memory_agent.memory,
            replan_reason=f"new_rgb_frame:{frame.frame_id}",
        )
        return RealtimeDecision(
            frame_id=snapshot.frame_id,
            mapping=snapshot.mapping,
            plan=plan,
            graph_nodes=snapshot.graph_nodes,
            graph_edges=snapshot.graph_edges,
            knowledge_facts_created=snapshot.knowledge_facts_created,
        )

    def close(self) -> None:
        self.memory_agent.close()
