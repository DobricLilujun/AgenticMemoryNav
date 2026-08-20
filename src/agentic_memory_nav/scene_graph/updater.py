"""Incrementally merge observations and infer directed spatial relations."""

from __future__ import annotations

from math import sqrt

import numpy as np

from agentic_memory_nav.common.types import (
    NodeType,
    ObjectObservation,
    RelationEvidence,
    SceneEdge,
    SceneNode,
    SceneTriple,
    new_id,
)
from agentic_memory_nav.scene_graph.object_association import AssociationDecision, ObjectAssociator
from agentic_memory_nav.scene_graph.graph import SceneGraph


class SceneGraphUpdater:
    def __init__(self, graph: SceneGraph, associator: ObjectAssociator | None = None) -> None:
        self.graph = graph
        self.associator = associator or ObjectAssociator()

    def update(self, observations: list[ObjectObservation]) -> list[AssociationDecision]:
        decisions: list[AssociationDecision] = []
        touched: list[SceneNode] = []
        for observation in observations:
            decision = self.associator.associate(observation, self.graph.nodes())
            decisions.append(decision)
            if decision.accepted and decision.node_id:
                node = self._merge(self.graph.get_node(decision.node_id), observation, decision)
            else:
                node = self._new_node(observation)
            observation.track_id = node.node_id
            self.graph.upsert_node(node)
            touched.append(node)
        self._infer_relations(touched)
        return decisions

    def add_vlm_triples(
        self,
        triples: list[SceneTriple],
        observations: list[ObjectObservation],
    ) -> None:
        """Add valid VLM relation claims without replacing geometric evidence."""
        observation_nodes = {
            observation.observation_id: observation.track_id
            for observation in observations
            if observation.track_id is not None
        }
        for triple in triples:
            source_id = observation_nodes.get(triple.subject_observation_id)
            target_id = observation_nodes.get(triple.object_observation_id)
            if source_id is None or target_id is None or source_id == target_id:
                continue
            evidence = RelationEvidence(
                source="vlm",
                predicate=triple.predicate,
                confidence=triple.confidence,
                timestamp=triple.timestamp,
                source_frame=triple.frame_id,
                provenance=triple.provenance,
            )
            existing = next(
                (
                    edge
                    for edge in self.graph.edges()
                    if edge.source_id == source_id
                    and edge.target_id == target_id
                    and edge.relation == triple.predicate
                ),
                None,
            )
            if existing is not None:
                existing.evidence.append(evidence)
                existing.last_seen = triple.timestamp
                existing.observation_count += 1
                existing.confidence = max(existing.confidence, triple.confidence)
                self.graph.upsert_edge(existing)
                continue
            self.graph.upsert_edge(
                SceneEdge(
                    edge_id=new_id("edge"),
                    source_id=source_id,
                    target_id=target_id,
                    relation=triple.predicate,
                    confidence=triple.confidence,
                    first_seen=triple.timestamp,
                    last_seen=triple.timestamp,
                    source_frame=triple.frame_id,
                    position_3d=self.graph.get_node(source_id).position_3d,
                    evidence=[evidence],
                    provenance=triple.provenance,
                )
            )

    @staticmethod
    def _new_node(observation: ObjectObservation) -> SceneNode:
        node_type = (
            NodeType.ROOM if observation.attributes.get("kind") == "room" else NodeType.OBJECT
        )
        return SceneNode(
            node_id=new_id("node"),
            node_type=node_type,
            label=observation.category,
            attributes=dict(observation.attributes),
            position_3d=observation.center_3d,
            bbox_3d=observation.bbox_3d,
            uncertainty=1.0 - observation.confidence,
            first_seen=observation.timestamp,
            last_seen=observation.timestamp,
            confidence=observation.confidence,
            source_frame=observation.frame_id,
            observation_ids=[observation.observation_id],
            embedding=observation.embedding.tolist() if observation.embedding is not None else None,
            geometry=observation.geometry,
            provenance=[*observation.provenance, observation.observation_id],
        )

    @staticmethod
    def _merge(
        node: SceneNode, observation: ObjectObservation, decision: AssociationDecision
    ) -> SceneNode:
        count = node.observation_count + 1
        old = np.asarray(node.position_3d)
        position = tuple(((old * node.observation_count + observation.center_3d) / count).tolist())
        node.position_3d = position  # type: ignore[assignment]
        node.bbox_3d = observation.bbox_3d
        node.attributes.update(observation.attributes)
        node.last_seen = observation.timestamp
        node.source_frame = observation.frame_id
        node.observation_count = count
        node.observation_ids.append(observation.observation_id)
        if observation.geometry is not None:
            node.geometry = observation.geometry
        node.confidence = 1.0 - (1.0 - node.confidence) * (1.0 - observation.confidence)
        node.uncertainty = 1.0 - node.confidence
        node.provenance.extend(
            [observation.observation_id, f"association_score={decision.score:.3f}"]
        )
        return node

    def _infer_relations(self, touched: list[SceneNode]) -> None:
        all_nodes = self.graph.nodes()
        for source in touched:
            for target in all_nodes:
                if source.node_id == target.node_id:
                    continue
                for relation, confidence in self._relations(source, target):
                    existing = next(
                        (
                            edge
                            for edge in self.graph.edges()
                            if edge.source_id == source.node_id
                            and edge.target_id == target.node_id
                            and edge.relation == relation
                        ),
                        None,
                    )
                    if existing:
                        existing.last_seen = source.last_seen
                        existing.observation_count += 1
                        existing.confidence = max(existing.confidence, confidence)
                        self.graph.upsert_edge(existing)
                    else:
                        self.graph.upsert_edge(
                            SceneEdge(
                                edge_id=new_id("edge"),
                                source_id=source.node_id,
                                target_id=target.node_id,
                                relation=relation,
                                confidence=confidence,
                                first_seen=source.last_seen,
                                last_seen=source.last_seen,
                                source_frame=source.source_frame,
                                position_3d=source.position_3d,
                                evidence=[
                                    RelationEvidence(
                                        source="geometry",
                                        predicate=relation,
                                        confidence=confidence,
                                        timestamp=source.last_seen,
                                        source_frame=source.source_frame,
                                        provenance=[source.source_frame, "geometric_relation_rule"],
                                    )
                                ],
                                provenance=[source.source_frame, "geometric_relation_rule"],
                            )
                        )

    @staticmethod
    def _relations(source: SceneNode, target: SceneNode) -> list[tuple[str, float]]:
        sx, sy, sz = source.position_3d
        tx, ty, tz = target.position_3d
        distance = sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + (sz - tz) ** 2)
        relations: list[tuple[str, float]] = []
        if source.node_type == NodeType.OBJECT and target.node_type == NodeType.ROOM:
            relations.append(("inside", 0.95))
        if source.node_type == NodeType.ROOM and target.node_type == NodeType.OBJECT:
            relations.append(("contains", 0.95))
        if distance < 1.5:
            relations.append(("near", max(0.5, 1.0 - distance / 3.0)))
        if abs(sx - tx) > 0.25:
            relations.append(("left_of" if sx < tx else "right_of", 0.8))
        if abs(sy - ty) > 0.25:
            relations.append(("below" if sy < ty else "above", 0.8))
        if abs(sz - tz) > 0.25:
            relations.append(("in_front_of" if sz < tz else "behind", 0.75))
        return relations
