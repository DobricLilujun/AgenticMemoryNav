import numpy as np

from agentic_memory_nav.common.types import MappingUpdate, Pose3D
from agentic_memory_nav.mapping.local_submap import LocalSubmapBuilder


def _update(frame_id: str, center_x: float) -> MappingUpdate:
    cloud = np.array([[center_x, 0.0, 1.0], [center_x + 0.01, 0.0, 1.0]], dtype=np.float32)
    return MappingUpdate(
        frame_id=frame_id,
        timestamp=float(frame_id[-1]),
        camera_pose=Pose3D(position=(center_x, 0.0, 0.0)),
        depth=np.ones((1, 1), dtype=np.float32),
        confidence=np.ones((1, 1), dtype=np.float32),
        local_pointcloud=cloud,
        global_pointcloud=cloud,
        is_keyframe=True,
        map_version=int(frame_id[-1]) + 1,
    )


def test_local_submap_commits_stable_fixed_window():
    builder = LocalSubmapBuilder(window_frames=3, frame_stride=1, stability_threshold_m=0.1)

    assert builder.add(_update("frame_0", 0.0)) is None
    assert builder.add(_update("frame_1", 0.04)) is None
    submap = builder.add(_update("frame_2", 0.08))

    assert submap is not None
    assert submap.stable
    assert submap.frame_ids == ["frame_0", "frame_1", "frame_2"]
    assert submap.points.shape == (6, 3)
    assert submap.geometric_residual_m < 0.1


def test_local_submap_rejects_unstable_window():
    builder = LocalSubmapBuilder(window_frames=3, frame_stride=1, stability_threshold_m=0.1)
    builder.add(_update("frame_0", 0.0))
    builder.add(_update("frame_1", 0.05))
    submap = builder.add(_update("frame_2", 1.0))

    assert submap is not None
    assert not submap.stable
    assert submap.geometric_residual_m > 0.1


def test_local_submap_stride_controls_contributing_frames():
    builder = LocalSubmapBuilder(window_frames=2, frame_stride=2, stability_threshold_m=0.1)

    assert builder.add(_update("frame_0", 0.0)) is None
    assert builder.add(_update("frame_1", 0.02)) is None
    submap = builder.add(_update("frame_2", 0.04))

    assert submap is not None
    assert submap.frame_ids == ["frame_0", "frame_2"]
