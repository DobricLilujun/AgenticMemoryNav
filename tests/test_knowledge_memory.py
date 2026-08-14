from agentic_memory_nav.common.types import NodeType, SceneEdge, SceneNode
from agentic_memory_nav.memory.knowledge_memory import KnowledgeMemory
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.scene_graph.graph import SceneGraph


def test_knowledge_memory_materializes_graph_once_per_version(tmp_path):
    graph = SceneGraph()
    kitchen = SceneNode(
        node_id="node_kitchen",
        node_type=NodeType.ROOM,
        label="kitchen",
        attributes={"kind": "room"},
        position_3d=(0.0, 0.0, 0.0),
        bbox_3d=None,
        uncertainty=0.01,
        first_seen=0.0,
        last_seen=1.0,
        confidence=0.99,
        source_frame="frame_0",
    )
    cup = SceneNode(
        node_id="node_cup",
        node_type=NodeType.OBJECT,
        label="cup",
        attributes={"color": "red"},
        position_3d=(1.0, 0.0, 1.0),
        bbox_3d=None,
        uncertainty=0.05,
        first_seen=1.0,
        last_seen=1.0,
        confidence=0.95,
        source_frame="frame_1",
    )
    graph.upsert_node(kitchen)
    graph.upsert_node(cup)
    graph.upsert_edge(
        SceneEdge(
            edge_id="edge_inside",
            source_id=cup.node_id,
            target_id=kitchen.node_id,
            relation="inside",
            confidence=0.95,
            first_seen=1.0,
            last_seen=1.0,
            source_frame="frame_1",
        )
    )
    memory = KnowledgeMemory(SQLiteMemory(tmp_path / "memory.sqlite3"))

    assert memory.materialize(graph) == 3
    assert memory.materialize(graph) == 0
    facts = memory.retrieve_subgraph("cup inside kitchen")

    assert any(fact.structured_payload["knowledge_type"] == "triple" for fact in facts)
