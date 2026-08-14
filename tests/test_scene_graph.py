import numpy as np

from agentic_memory_nav.common.types import FrameObservation, InstanceGeometry, SceneTriple
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


def test_graph_preserves_instance_geometry_and_relation_evidence():
    graph, mapper, detector = SceneGraph(), MockMapper(), MockPerception()
    updater = SceneGraphUpdater(graph)
    mapper.start()

    first_frame = FrameObservation("0", 0.0, np.zeros((32, 32, 3), np.uint8))
    updater.update(detector.detect(first_frame, mapper.update(first_frame)))
    second_frame = FrameObservation("1", 1.0, np.zeros((32, 32, 3), np.uint8))
    observations = detector.detect(second_frame, mapper.update(second_frame))
    cup = next(observation for observation in observations if observation.category == "cup")
    cup.geometry = InstanceGeometry(
        instance_id="cup_1",
        artifact_path="artifacts/instance_clouds/cup_1.npz",
        point_count=12,
        centroid_3d=cup.center_3d,
        dimensions_3d=cup.dimensions_3d,
        coordinate_frame="world",
        confidence=0.9,
        mask_provenance=["frame_1", "mask_cup_1"],
    )
    updater.update(observations)

    node = graph.find_nodes("cup", {"color": "red"})[0]
    inside = next(
        edge
        for edge in graph.edges()
        if edge.source_id == node.node_id and edge.relation == "inside"
    )
    assert node.geometry is not None
    assert node.geometry.instance_id == "cup_1"
    assert inside.evidence[0].source == "geometry"


def test_graph_adds_vlm_triple_as_independent_evidence():
    graph, mapper, detector = SceneGraph(), MockMapper(), MockPerception()
    updater = SceneGraphUpdater(graph)
    mapper.start()
    first_frame = FrameObservation("0", 0.0, np.zeros((32, 32, 3), np.uint8))
    updater.update(detector.detect(first_frame, mapper.update(first_frame)))
    frame = FrameObservation("1", 1.0, np.zeros((32, 32, 3), np.uint8))
    observations = detector.detect(frame, mapper.update(frame))
    updater.update(observations)
    cup = next(observation for observation in observations if observation.category == "cup")
    kitchen = next(observation for observation in observations if observation.category == "kitchen")

    updater.add_vlm_triples(
        [
            SceneTriple(
                subject_observation_id=cup.observation_id,
                predicate="on_top_of",
                object_observation_id=kitchen.observation_id,
                confidence=0.8,
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                provenance=["vlm"],
            )
        ],
        observations,
    )

    edge = next(edge for edge in graph.edges() if edge.relation == "on_top_of")
    assert edge.source_id == cup.track_id
    assert edge.target_id == kitchen.track_id
    assert edge.evidence[0].source == "vlm"
