"""Minimal single-agent wrapper around the project’s core modules.

This intentionally keeps the existing object-centric graph / memory / planner stack,
while exposing a single central agent API that can be used as the primary runtime
surface for a light-weight world-model agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_memory_nav.common.types import MemoryItem, MemoryType, new_id
from agentic_memory_nav.execution.unitree_sim import UnitreeSimExecutor
from agentic_memory_nav.mapping.mock_mapper import MockMapper
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.perception.mock_perception import MockPerception
from agentic_memory_nav.planning.rule_based_fallback import RuleBasedPlanner
from agentic_memory_nav.planning.task_parser import RuleBasedTaskParser
from agentic_memory_nav.scene_graph.graph import SceneGraph
from agentic_memory_nav.scene_graph.updater import SceneGraphUpdater


class SingleAgent:
    """A single central agent with internal specialist responsibilities.

    The design is intentionally not a full multi-process agent framework. Instead,
    it packages the existing modules behind a single interface with a shared state:

    - perception: object detection
    - mapping: point-cloud update
    - graph: semantic scene graph
    - memory: persistent memory
    - planner: task-aware action selection
    - execution: bounded robot step
    """

    def __init__(
        self,
        scratch_dir: Path | str,
        instruction: str,
        max_frames: int = 4,
    ) -> None:
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.instruction = instruction
        self.max_frames = max_frames

        self.mapper = MockMapper(keyframe_interval=2, depth_m=2.0)
        self.perception = MockPerception()
        self.graph = SceneGraph()
        self.updater = SceneGraphUpdater(self.graph)
        self.memory = SQLiteMemory(self.scratch_dir / "memory.sqlite3")
        self.parser = RuleBasedTaskParser()
        self.planner = RuleBasedPlanner(approach_distance=0.6)
        self.executor = UnitreeSimExecutor(safety=self._build_safety(), max_speed=0.5)

    @staticmethod
    def _build_safety() -> Any:
        from agentic_memory_nav.execution.safety_controller import SafetyController

        return SafetyController(max_speed=0.5, max_angular_speed=1.0, max_timeout=15.0)

    def run_task(self) -> dict[str, Any]:
        task = self.parser.parse(self.instruction)
        self.mapper.start()
        self.executor.reset()

        last_action = "explore"
        graph_nodes = 0
        memory_items = 0

        for _ in range(self.max_frames):
            frame = self.executor.get_observation()
            mapping = self.mapper.update(frame)
            observations = self.perception.detect(frame, mapping)
            self.updater.update(observations)

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

            plan = self.planner.plan(
                task,
                self.executor.get_state(),
                self.graph,
                self.memory,
                replan_reason="new_observation",
            )
            last_action = plan.action.action_type.value
            feedback = self.executor.send_waypoint(plan.action.waypoint, plan.action)

            if plan.action.action_type.value == "navigate" and feedback.success:
                return {
                    "task_status": "succeeded",
                    "graph_nodes": len(self.graph.nodes()),
                    "memory_items": len(self.memory.all_items()),
                    "plan_action": last_action,
                    "final_pose": self.executor.get_state(),
                }

            graph_nodes = len(self.graph.nodes())
            memory_items = len(self.memory.all_items())

        return {
            "task_status": "failed",
            "graph_nodes": graph_nodes or len(self.graph.nodes()),
            "memory_items": memory_items or len(self.memory.all_items()),
            "plan_action": last_action,
            "final_pose": self.executor.get_state(),
        }
