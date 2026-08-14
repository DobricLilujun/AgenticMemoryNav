"""Materialize scene-graph nodes and predicates as persistent knowledge facts."""

from __future__ import annotations

from agentic_memory_nav.common.types import MemoryItem, MemoryType, SceneEdge, SceneNode
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.scene_graph.graph import SceneGraph


class KnowledgeMemory:
    def __init__(self, store: SQLiteMemory) -> None:
        self.store = store

    def materialize(self, graph: SceneGraph) -> int:
        """Write the current graph version once; return the number of facts created."""
        created = 0
        for node in graph.nodes():
            created += self._write_node(node, graph.version)
        for edge in graph.edges():
            created += self._write_edge(edge, graph.version)
        return created

    def retrieve_subgraph(self, query: str, limit: int = 10) -> list[MemoryItem]:
        return [
            item
            for item in self.store.retrieve_by_text(query, limit=limit)
            if item.structured_payload.get("knowledge_type") in {"node", "triple"}
        ]

    def _write_node(self, node: SceneNode, graph_version: int) -> int:
        payload = {
            "knowledge_type": "node",
            "entity_id": node.node_id,
            "label": node.label,
            "attributes": node.attributes,
            "graph_version": graph_version,
            "geometry": node.geometry,
        }
        return self._write_once(
            memory_id=f"knowledge_node_{node.node_id}",
            content=f"Entity {node.label} {node.attributes}",
            payload=payload,
            timestamp=node.last_seen,
            location=node.position_3d,
            confidence=node.confidence,
            provenance=node.provenance,
        )

    def _write_edge(self, edge: SceneEdge, graph_version: int) -> int:
        payload = {
            "knowledge_type": "triple",
            "entity_id": edge.edge_id,
            "subject_id": edge.source_id,
            "predicate": edge.relation,
            "object_id": edge.target_id,
            "graph_version": graph_version,
            "evidence": edge.evidence,
        }
        return self._write_once(
            memory_id=f"knowledge_edge_{edge.edge_id}",
            content=f"{edge.source_id} {edge.relation} {edge.target_id}",
            payload=payload,
            timestamp=edge.last_seen,
            location=edge.position_3d,
            confidence=edge.confidence,
            provenance=edge.provenance,
        )

    def _write_once(
        self,
        *,
        memory_id: str,
        content: str,
        payload: dict[str, object],
        timestamp: float,
        location: tuple[float, float, float] | None,
        confidence: float,
        provenance: list[str],
    ) -> int:
        existing = self.store.retrieve_by_entity(memory_id)
        if (
            existing
            and existing[0].structured_payload.get("graph_version") == payload["graph_version"]
        ):
            return 0
        version = existing[0].version + 1 if existing else 1
        self.store.add_observation(
            MemoryItem(
                memory_id=memory_id,
                memory_type=MemoryType.SEMANTIC,
                content=content,
                structured_payload=payload,
                timestamp=timestamp,
                location=location,
                confidence=confidence,
                provenance=provenance,
                version=version,
            )
        )
        return 1
