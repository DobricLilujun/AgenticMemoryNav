"""Open3DSG-inspired directed scene graph with temporal provenance."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from agentic_memory_nav.common.types import SceneEdge, SceneNode, jsonable


class SceneGraph:
    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph[str] = nx.MultiDiGraph()
        self.version = 0

    def upsert_node(self, node: SceneNode) -> None:
        self._graph.add_node(node.node_id, data=node)
        self.version += 1

    def upsert_edge(self, edge: SceneEdge) -> None:
        self._graph.add_edge(edge.source_id, edge.target_id, key=edge.edge_id, data=edge)
        self.version += 1

    def nodes(self) -> list[SceneNode]:
        return [attrs["data"] for _, attrs in self._graph.nodes(data=True)]

    def edges(self) -> list[SceneEdge]:
        return [attrs["data"] for *_, attrs in self._graph.edges(keys=True, data=True)]

    def get_node(self, node_id: str) -> SceneNode:
        return self._graph.nodes[node_id]["data"]

    def find_nodes(self, label: str, attributes: dict[str, str] | None = None) -> list[SceneNode]:
        attributes = attributes or {}
        return [
            node
            for node in self.nodes()
            if node.label == label
            and all(node.attributes.get(key) == value for key, value in attributes.items())
        ]

    def relations(self, node_id: str) -> list[SceneEdge]:
        return [
            edge for edge in self.edges() if edge.source_id == node_id or edge.target_id == node_id
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "nodes": [jsonable(node) for node in self.nodes()],
            "edges": [jsonable(edge) for edge in self.edges()],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
