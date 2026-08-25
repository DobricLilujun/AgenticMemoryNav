import numpy as np

from agentic_memory_nav.common.types import (
    CameraIntrinsics,
    FrameObservation,
    MappingUpdate,
    ObjectObservation,
    Pose3D,
)
from agentic_memory_nav.agent.geometry.pointcloud_store import PointCloudStore
from agentic_memory_nav.agent.perception.instance_segmentation import (
    BoundingBoxSegmenter,
    InstanceGeometryEnricher,
)


def test_geometry_enricher_backprojects_bbox_mask_to_instance_cloud(tmp_path):
    frame = FrameObservation(
        frame_id="frame_0",
        timestamp=0.0,
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        camera_intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0, 4, 4),
    )
    mapping = MappingUpdate(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        camera_pose=Pose3D(),
        depth=np.ones((4, 4), dtype=np.float32),
        confidence=np.ones((4, 4), dtype=np.float32),
        local_pointcloud=np.empty((0, 3), dtype=np.float32),
        global_pointcloud=np.empty((0, 3), dtype=np.float32),
        is_keyframe=True,
        map_version=1,
    )
    observation = ObjectObservation(
        observation_id="obs_cup",
        category="cup",
        attributes={"color": "red"},
        bbox_2d=(1, 1, 3, 3),
        center_3d=(0.0, 0.0, 0.0),
        dimensions_3d=(0.0, 0.0, 0.0),
        confidence=0.9,
        timestamp=0.0,
        frame_id=frame.frame_id,
    )

    enriched = InstanceGeometryEnricher(
        BoundingBoxSegmenter(), PointCloudStore(tmp_path / "instance_clouds")
    ).enrich(frame, mapping, [observation])

    assert enriched[0].geometry is not None
    assert enriched[0].geometry.point_count == 4
    assert enriched[0].geometry.coordinate_frame == "world"
