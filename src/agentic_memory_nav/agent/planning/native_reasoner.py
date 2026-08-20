"""Small, deterministic graph reasoner for navigation-goal resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_memory_nav.memory.knowledge_memory import KnowledgeMemory
from agentic_memory_nav.scene_graph.graph import SceneGraph


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    target_id: str | None
    confidence: float
    evidence_ids: list[str]
    requires_verification: bool
    reason: str


class NativeReasoner:
    """Resolve task entities with graph relations and persisted knowledge evidence."""

    def __init__(self, knowledge: KnowledgeMemory) -> None:
        self.knowledge = knowledge

    def resolve(self, graph: SceneGraph, goal: dict[str, Any]) -> ReasoningResult:
        object_label = str(goal.get("object", ""))
        color_value = goal.get("color")
        room_value = goal.get("room")
        color = str(color_value) if color_value is not None else ""
        room_label = str(room_value) if room_value is not None else ""
        candidates = graph.find_nodes(object_label, {"color": color} if color else {})
        if not candidates:
            return ReasoningResult(None, 0.0, [], True, "target has not been observed")

        target = max(candidates, key=lambda node: node.confidence)
        evidence_ids = [target.node_id]
        room_edges = [
            edge
            for edge in graph.relations(target.node_id)
            if edge.relation == "inside"
            and edge.source_id == target.node_id
            and graph.get_node(edge.target_id).label == room_label
        ]
        if room_label and not room_edges:
            return ReasoningResult(
                target.node_id,
                target.confidence * 0.7,
                evidence_ids,
                True,
                "target matches semantics but room containment needs verification",
            )

        if room_edges:
            edge = max(room_edges, key=lambda item: item.confidence)
            evidence_ids.append(edge.edge_id)
            confidence = min(target.confidence, edge.confidence)
        else:
            confidence = target.confidence

        knowledge_facts = self.knowledge.retrieve_subgraph(
            f"{object_label} {color} {room_label}".strip(), limit=10
        )
        evidence_ids.extend(item.memory_id for item in knowledge_facts)
        return ReasoningResult(
            target.node_id,
            confidence,
            evidence_ids,
            False,
            "target and requested room relation are supported by the scene graph",
        )
