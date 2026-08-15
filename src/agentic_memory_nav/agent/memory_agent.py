"""Agent responsible for turning one RGB(+pose) frame into updated graph/knowledge memory.

This is the "memory extraction and construction" half of the realtime loop: it owns
mapping, perception, scene-graph association, knowledge materialization, and episodic
memory writes. It does not decide any navigation action; see `NavigationAgent` for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_memory_nav.common.types import (
    FrameObservation,
    MappingUpdate,
    MemoryItem,
    MemoryType,
    ObjectObservation,
    new_id,
)
from agentic_memory_nav.mapping.mock_mapper import MockMapper
from agentic_memory_nav.memory.knowledge_memory import KnowledgeMemory
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.perception.mock_perception import MockPerception
from agentic_memory_nav.perception.vlm_backend import VLMBackend
from agentic_memory_nav.scene_graph.graph import SceneGraph
from agentic_memory_nav.scene_graph.updater import SceneGraphUpdater


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    frame_id: str
    mapping: MappingUpdate
    observations: list[ObjectObservation]
    graph_nodes: int
    graph_edges: int
    knowledge_facts_created: int


class MemoryAgent:
    """Owns mapping + perception + scene graph + knowledge memory for one robot run."""

    def __init__(
        self,
        scratch_dir: Path | str,
        *,
        mapper: Any | None = None,
        perception: Any | None = None,
        geometry_enricher: Any | None = None,
    ) -> None:
        root = Path(scratch_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.mapper = mapper or MockMapper()
        self.perception = perception or MockPerception()
        self.geometry_enricher = geometry_enricher
        self.graph = SceneGraph()
        self.updater = SceneGraphUpdater(self.graph)
        self.memory = SQLiteMemory(root / "memory.sqlite3")
        self.knowledge = KnowledgeMemory(self.memory)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.mapper.start()
            self._started = True

    def ingest_frame(self, frame: FrameObservation) -> MemorySnapshot:
        """Fold one RGB(+pose) frame into the graph and knowledge memory."""
        self.start()
        mapping = self.mapper.update(frame)
        triples = []
        if isinstance(self.perception, VLMBackend):
            observations, triples = self.perception.analyze(frame, mapping)
        else:
            observations = self.perception.detect(frame, mapping)
        if self.geometry_enricher is not None:
            observations = self.geometry_enricher.enrich(frame, mapping, observations)
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
        return MemorySnapshot(
            frame_id=frame.frame_id,
            mapping=mapping,
            observations=observations,
            graph_nodes=len(self.graph.nodes()),
            graph_edges=len(self.graph.edges()),
            knowledge_facts_created=knowledge_facts_created,
        )

    def close(self) -> None:
        self.memory.close()
