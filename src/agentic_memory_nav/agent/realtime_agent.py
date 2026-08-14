"""Frame-by-frame RGB agent that turns updated memory graphs into navigation actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_memory_nav.common.types import (
    FrameObservation,
    MappingUpdate,
    MemoryItem,
    MemoryType,
    NavigationPlan,
    Pose3D,
    new_id,
)
from agentic_memory_nav.mapping.mock_mapper import MockMapper
from agentic_memory_nav.memory.knowledge_memory import KnowledgeMemory
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.perception.mock_perception import MockPerception
from agentic_memory_nav.perception.vlm_backend import VLMBackend
from agentic_memory_nav.planning.rule_based_fallback import RuleBasedPlanner
from agentic_memory_nav.planning.task_parser import RuleBasedTaskParser
from agentic_memory_nav.scene_graph.graph import SceneGraph
from agentic_memory_nav.scene_graph.updater import SceneGraphUpdater


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
        root = Path(scratch_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.task = RuleBasedTaskParser().parse(instruction)
        self.mapper = mapper or MockMapper()
        self.perception = perception or MockPerception()
        self.graph = SceneGraph()
        self.updater = SceneGraphUpdater(self.graph)
        self.memory = SQLiteMemory(root / "memory.sqlite3")
        self.knowledge = KnowledgeMemory(self.memory)
        self.planner = planner or RuleBasedPlanner()
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.mapper.start()
            self._started = True

    def ingest_frame(
        self,
        frame: FrameObservation,
        robot_pose: Pose3D | None = None,
    ) -> RealtimeDecision:
        """Update memory graph from one frame and return the immediate next action."""
        self.start()
        mapping = self.mapper.update(frame)
        triples = []
        if isinstance(self.perception, VLMBackend):
            observations, triples = self.perception.analyze(frame, mapping)
        else:
            observations = self.perception.detect(frame, mapping)
        self.updater.update(observations)
        if triples:
            self.updater.add_vlm_triples(triples, observations)
        knowledge_facts_created = self.knowledge.materialize(self.graph)
        for observation in observations:
            self.memory.add_observation(
                MemoryItem(
                    memory_id=new_id("mem"),
                    memory_type=MemoryType.EPISODIC,
                    content=f"Observation: {observation.category} at {observation.center_3d}",
                    structured_payload={
                        "entity_id": observation.track_id,
                        "category": observation.category,
                        "attributes": observation.attributes,
                        "frame_id": observation.frame_id,
                    },
                    timestamp=observation.timestamp,
                    location=observation.center_3d,
                    confidence=observation.confidence,
                    provenance=[observation.observation_id, *observation.provenance],
                )
            )
        current_pose = robot_pose or frame.robot_pose or frame.camera_pose or Pose3D()
        plan = self.planner.plan(
            self.task,
            current_pose,
            self.graph,
            self.memory,
            replan_reason=f"new_rgb_frame:{frame.frame_id}",
        )
        return RealtimeDecision(
            frame_id=frame.frame_id,
            mapping=mapping,
            plan=plan,
            graph_nodes=len(self.graph.nodes()),
            graph_edges=len(self.graph.edges()),
            knowledge_facts_created=knowledge_facts_created,
        )

    def close(self) -> None:
        self.memory.close()
