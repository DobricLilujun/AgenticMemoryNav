import numpy as np

from agentic_memory_nav.common.types import FrameObservation
from agentic_memory_nav.mapping.mock_mapper import MockMapper
from agentic_memory_nav.perception.mock_perception import MockPerception
from agentic_memory_nav.scene_graph.graph import SceneGraph
from agentic_memory_nav.scene_graph.updater import SceneGraphUpdater


def test_incremental_graph_associates_red_cup():
    graph, mapper, detector = SceneGraph(), MockMapper(), MockPerception()
    updater = SceneGraphUpdater(graph)
    mapper.start()
    for index in range(3):
        frame = FrameObservation(str(index), float(index), np.zeros((32, 32, 3), np.uint8))
        updater.update(detector.detect(frame, mapper.update(frame)))
    cup = graph.find_nodes("cup", {"color": "red"})[0]
    assert cup.observation_count == 2
    assert any(
        edge.source_id == cup.node_id and edge.relation == "inside" for edge in graph.edges()
    )
