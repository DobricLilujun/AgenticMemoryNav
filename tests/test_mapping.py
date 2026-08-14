import numpy as np

from agentic_memory_nav.common.types import FrameObservation, Pose3D
from agentic_memory_nav.mapping.lingbot_map_adapter import LingBotMapAdapter
from agentic_memory_nav.mapping.mock_mapper import MockMapper


def test_mock_mapper_stream_and_state(tmp_path):
    mapper = MockMapper(keyframe_interval=2)
    mapper.start()
    for index in range(3):
        update = mapper.update(
            FrameObservation(str(index), float(index), np.zeros((16, 16, 3), np.uint8))
        )
    assert update.map_version == 3
    assert len(mapper.buffer.keyframes) == 2
    mapper.save_state(tmp_path)
    restored = MockMapper()
    restored.load_state(tmp_path)
    assert restored.get_global_pointcloud().shape[1] == 3


def test_lingbot_adapter_normalizes_streaming_result_and_persists_state(tmp_path):
    def predictor(frame):
        return {
            "camera_pose": {"position": [1.0, 2.0, 3.0], "yaw": 0.25},
            "depth": [[1.0, 2.0], [3.0, 4.0]],
            "confidence": [[0.9, 0.8], [0.7, 0.6]],
            "local_pointcloud": [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
            "global_pointcloud": [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
            "is_keyframe": True,
        }

    mapper = LingBotMapAdapter(
        checkpoint=tmp_path / "lingbot.pt",
        predictor=predictor,
    )
    mapper.start()
    update = mapper.update(FrameObservation("frame_0", 1.0, np.zeros((2, 2, 3), np.uint8)))

    assert update.map_version == 1
    assert update.camera_pose == Pose3D(position=(1.0, 2.0, 3.0), yaw=0.25)
    assert update.local_pointcloud.shape == (2, 3)
    assert mapper.get_global_pointcloud().shape == (2, 3)
    mapper.save_state(tmp_path / "state")
    restored = LingBotMapAdapter(checkpoint=tmp_path / "lingbot.pt", predictor=predictor)
    restored.load_state(tmp_path / "state")
    assert restored.get_global_pointcloud().shape == (2, 3)


def test_lingbot_adapter_prefers_depth_pose_submaps_over_world_point_head(tmp_path):
    def predictor(frame):
        return {
            "camera_pose": {
                "position": [0.0, 0.0, 0.0],
                "camera_to_world": np.eye(4, dtype=np.float32).tolist(),
            },
            "intrinsics": np.eye(3, dtype=np.float32).tolist(),
            "depth": [[2.0]],
            "confidence": [[1.0]],
            "world_points": [[99.0, 99.0, 99.0]],
            "is_keyframe": True,
        }

    mapper = LingBotMapAdapter(
        checkpoint=tmp_path / "lingbot.pt",
        predictor=predictor,
        local_submap_frames=2,
        local_submap_stride=1,
        local_submap_stability_threshold_m=0.1,
    )
    mapper.start()
    first = mapper.update(FrameObservation("frame_0", 0.0, np.zeros((1, 1, 3), np.uint8)))
    mapper.update(FrameObservation("frame_1", 1.0, np.zeros((1, 1, 3), np.uint8)))

    np.testing.assert_allclose(first.local_pointcloud, np.array([[0.0, 0.0, 2.0]]))
    assert mapper.get_last_committed_submap() is not None
    assert mapper.get_last_committed_submap().stable
