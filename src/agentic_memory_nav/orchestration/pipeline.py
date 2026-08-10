"""Mock-first event-driven navigation pipeline."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt

from agentic_memory_nav.common.types import MemoryItem, MemoryType, TaskStatus, jsonable, new_id
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun
from agentic_memory_nav.evaluation.metrics import (
    provenance_completeness,
    success_weighted_path_length,
)
from agentic_memory_nav.evaluation.visualization import render_run_artifacts
from agentic_memory_nav.execution.habitat_adapter import HabitatSimExecutor
from agentic_memory_nav.execution.safety_controller import SafetyController
from agentic_memory_nav.execution.unitree_sim import UnitreeSimExecutor
from agentic_memory_nav.mapping.mock_mapper import MockMapper
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.orchestration.event_bus import Event, EventBus, EventType
from agentic_memory_nav.perception.mock_perception import MockPerception
from agentic_memory_nav.planning.rule_based_fallback import RuleBasedPlanner
from agentic_memory_nav.planning.task_parser import RuleBasedTaskParser
from agentic_memory_nav.scene_graph.graph import SceneGraph
from agentic_memory_nav.scene_graph.updater import SceneGraphUpdater


class NavigationPipeline:
    def __init__(self, config: dict[str, Any], run: ExperimentRun, logger: logging.Logger) -> None:
        self.config = config
        self.run = run
        self.logger = logger
        self.bus = EventBus(int(config.get("runtime", {}).get("queue_size", 32)))
        self.events = self.bus.subscribe()
        mapping = config.get("mapping", {})
        execution = config.get("execution", {})
        self.mapper = MockMapper(
            keyframe_interval=int(mapping.get("keyframe_interval", 2)),
            depth_m=float(mapping.get("mock_depth_m", 2.0)),
        )
        self.perception = MockPerception()
        self.graph = SceneGraph()
        self.updater = SceneGraphUpdater(self.graph)
        self.memory = SQLiteMemory(run.path / "memory.sqlite3")
        self.parser = RuleBasedTaskParser()
        self.planner = RuleBasedPlanner(
            float(config.get("planning", {}).get("approach_distance", 0.6))
        )
        safety = SafetyController(
            max_speed=float(execution.get("max_speed", 0.5)),
            max_angular_speed=float(execution.get("max_angular_speed", 1.0)),
            max_timeout=float(execution.get("max_action_timeout", 15.0)),
        )
        backend = str(execution.get("backend", "unitree_sim")).lower()
        if backend == "habitat":
            scene = execution.get("scene")
            if not scene:
                raise ValueError("execution.scene is required when execution.backend=habitat")
            self.executor = HabitatSimExecutor(
                scene=str(scene),
                safety=safety,
                max_speed=float(execution.get("max_speed", 0.5)),
            )
        else:
            self.executor = UnitreeSimExecutor(
                safety, max_speed=float(execution.get("max_speed", 0.5))
            )
        self.sequence = 0
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.replans = 0
        self.collisions = 0

    async def run_task(self, instruction: str) -> dict[str, Any]:
        task = self.parser.parse(instruction)
        task.status = TaskStatus.ACTIVE
        self.mapper.start()
        self.executor.reset()
        max_frames = int(self.config.get("runtime", {}).get("max_frames", 6))
        start = time.perf_counter()
        success = False
        path_length = 0.0
        previous_position = self.executor.get_state().position
        last_plan: dict[str, Any] | None = None

        for frame_index in range(max_frames):
            cycle_start = time.perf_counter()
            frame = self.executor.get_observation()
            await self._event(
                EventType.FRAME_RECEIVED,
                "executor",
                {"frame_id": frame.frame_id},
                frame.provenance,
                False,
            )

            mapping = self._timed("mapping", self.mapper.update, frame)
            await self._event(
                EventType.MAP_UPDATED,
                "mapper",
                {
                    "frame_id": frame.frame_id,
                    "map_version": mapping.map_version,
                    "keyframe": mapping.is_keyframe,
                },
                mapping.provenance,
            )

            observations = self._timed("perception", self.perception.detect, frame, mapping)
            await self._event(
                EventType.OBJECTS_DETECTED,
                "perception",
                {"frame_id": frame.frame_id, "count": len(observations)},
                [observation.observation_id for observation in observations],
                False,
            )
            decisions = self._timed("scene_graph", self.updater.update, observations)
            await self._event(
                EventType.SCENE_GRAPH_UPDATED,
                "scene_graph",
                {
                    "version": self.graph.version,
                    "nodes": len(self.graph.nodes()),
                    "edges": len(self.graph.edges()),
                },
                [observation.observation_id for observation in observations],
            )

            memory_start = time.perf_counter()
            for observation in observations:
                self.memory.add_observation(
                    MemoryItem(
                        memory_id=new_id("mem"),
                        memory_type=MemoryType.EPISODIC,
                        content=self._memory_content(observation),
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
            self.latencies["memory"].append(time.perf_counter() - memory_start)
            await self._event(
                EventType.MEMORY_UPDATED,
                "memory",
                {"items": len(self.memory.all_items())},
                [observation.observation_id for observation in observations],
            )

            if frame_index:
                self.replans += 1
                await self._event(
                    EventType.REPLAN_REQUESTED,
                    "coordinator",
                    {"reason": "new_observation", "map_version": mapping.map_version},
                    [frame.frame_id],
                )
            plan = self._timed(
                "planning",
                self.planner.plan,
                task,
                self.executor.get_state(),
                self.graph,
                self.memory,
                "new observation" if frame_index else None,
            )
            last_plan = jsonable(plan)
            await self._event(EventType.PLAN_GENERATED, "planner", last_plan, [frame.frame_id])
            await self._event(
                EventType.ACTION_ISSUED, "coordinator", jsonable(plan.action), [frame.frame_id]
            )

            feedback = self._timed(
                "execution", self.executor.send_waypoint, plan.action.waypoint, plan.action
            )
            if feedback.collision:
                self.collisions += 1
                await self._event(
                    EventType.COLLISION_DETECTED,
                    "executor",
                    jsonable(feedback),
                    [plan.action.action_id],
                )
            await self._event(
                EventType.ACTION_FEEDBACK, "executor", jsonable(feedback), [plan.action.action_id]
            )

            path_length += math.dist(previous_position, feedback.state.position)
            previous_position = feedback.state.position
            cycle_latency = time.perf_counter() - cycle_start
            self.latencies["end_to_end"].append(cycle_latency)
            self._save_frame(frame.frame_id, frame.rgb)
            self.run.append_trajectory(
                {
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "pose": feedback.state,
                    "action": plan.action,
                    "feedback": feedback,
                    "map_version": mapping.map_version,
                    "graph_version": self.graph.version,
                    "latency_s": cycle_latency,
                    "association": [jsonable(decision) for decision in decisions],
                }
            )
            self.logger.info(
                "navigation cycle completed",
                extra={
                    "fields": {
                        "frame": frame.frame_id,
                        "action": plan.action.action_type,
                        "latency_s": cycle_latency,
                    }
                },
            )
            if plan.action.action_type.value == "navigate" and feedback.success:
                success = True
                task.status = TaskStatus.SUCCEEDED
                await self._event(
                    EventType.GOAL_REACHED,
                    "coordinator",
                    {"task_id": task.task_id},
                    [plan.action.action_id],
                )
                break

        elapsed = time.perf_counter() - start
        if not success:
            task.status = TaskStatus.FAILED
        shortest = math.dist((0.0, 0.0, 0.0), self.executor.get_state().position)
        memories = self.memory.all_items()
        metrics = {
            "run_id": self.run.run_id,
            "success_rate": float(success),
            "spl": success_weighted_path_length(success, shortest, path_length),
            "path_length_m": path_length,
            "goal_distance_m": 0.0 if success else None,
            "collision_count": self.collisions,
            "replanning_count": self.replans,
            "frames": self.executor._frame_index,
            "elapsed_s": elapsed,
            "fps": self.executor._frame_index / max(elapsed, 1e-9),
            "queue_size": self.events.qsize(),
            "dropped_events": self.bus.dropped,
            "graph_nodes": len(self.graph.nodes()),
            "graph_edges": len(self.graph.edges()),
            "memory_items": len(memories),
            "provenance_completeness": provenance_completeness(
                [item.provenance for item in memories]
            ),
            "latency_ms": {
                key: {"mean": 1000 * sum(values) / len(values), "max": 1000 * max(values)}
                for key, values in self.latencies.items()
                if values
            },
            "last_plan": last_plan,
        }
        self._save_artifacts(metrics)
        return metrics

    async def _event(
        self,
        event_type: EventType,
        producer: str,
        payload: dict[str, Any],
        provenance: list[str],
        lossless: bool = True,
    ) -> None:
        self.sequence += 1
        await self.bus.publish(
            Event(event_type, self.run.run_id, producer, self.sequence, payload, provenance),
            lossless=lossless,
        )

    def _timed(self, name: str, function: Any, *args: Any) -> Any:
        started = time.perf_counter()
        result = function(*args)
        self.latencies[name].append(time.perf_counter() - started)
        return result

    @staticmethod
    def _memory_content(observation: Any) -> str:
        attributes = " ".join(observation.attributes.values())
        return f"Observed {attributes} {observation.category}".strip()

    def _save_artifacts(self, metrics: dict[str, Any]) -> None:
        self.run.write_json("metrics.json", metrics)
        self.graph.save(self.run.path / "scene_graph.json")
        self.memory.save_snapshot(self.run.path / "memory_snapshot.json")
        self.mapper.save_state(self.run.artifacts / "map")
        render_run_artifacts(self.run.path)

    def _save_frame(self, frame_id: str, rgb: Any) -> None:
        frames_dir = self.run.artifacts / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(frames_dir / f"{frame_id}.png", rgb)

    def close(self) -> None:
        self.memory.close()
        if hasattr(self.executor, "close"):
            self.executor.close()
        self.run.close()
