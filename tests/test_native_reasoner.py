from agentic_memory_nav.common.types import NodeType, SceneEdge, SceneNode
from agentic_memory_nav.memory.knowledge_memory import KnowledgeMemory
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.agent.planning.native_reasoner import NativeReasoner
from agentic_memory_nav.scene_graph.graph import SceneGraph


def test_native_reasoner_resolves_goal_with_graph_and_knowledge(tmp_path):
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
    knowledge = KnowledgeMemory(SQLiteMemory(tmp_path / "memory.sqlite3"))
    knowledge.materialize(graph)

    result = NativeReasoner(knowledge).resolve(
        graph, {"object": "cup", "color": "red", "room": "kitchen"}
    )

    assert result.target_id == cup.node_id
    assert result.requires_verification is False
    assert "edge_inside" in result.evidence_ids


def test_native_reasoner_resolves_open_category_without_color(tmp_path):
    graph = SceneGraph()
    sign = SceneNode(
        node_id="node_sign",
        node_type=NodeType.OBJECT,
        label="sign",
        attributes={},
        position_3d=(1.0, 0.0, 1.0),
        bbox_3d=None,
        uncertainty=0.1,
        first_seen=0.0,
        last_seen=1.0,
        confidence=0.8,
        source_frame="frame_0",
    )
    graph.upsert_node(sign)
    knowledge = KnowledgeMemory(SQLiteMemory(tmp_path / "memory.sqlite3"))
    knowledge.materialize(graph)

    result = NativeReasoner(knowledge).resolve(graph, {"object": "sign"})

    assert result.target_id == sign.node_id
    assert result.requires_verification is False


def test_native_reasoner_treats_none_color_as_no_color_constraint(tmp_path):
    graph = SceneGraph()
    cube = SceneNode(
        node_id="node_cube",
        node_type=NodeType.OBJECT,
        label="cube",
        attributes={"color": "red"},
        position_3d=(1.0, 0.0, 1.0),
        bbox_3d=None,
        uncertainty=0.1,
        first_seen=0.0,
        last_seen=1.0,
        confidence=0.8,
        source_frame="frame_0",
    )
    graph.upsert_node(cube)
    knowledge = KnowledgeMemory(SQLiteMemory(tmp_path / "memory.sqlite3"))
    knowledge.materialize(graph)

    result = NativeReasoner(knowledge).resolve(graph, {"object": "cube", "color": None})

    assert result.target_id == cube.node_id
    assert result.requires_verification is False
