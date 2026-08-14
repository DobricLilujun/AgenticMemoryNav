import numpy as np
import pytest

from agentic_memory_nav.common.types import CameraIntrinsics, MappingUpdate, Pose3D
from agentic_memory_nav.geometry.backprojection import backproject_mask
from agentic_memory_nav.geometry.pointcloud_store import PointCloudStore


def test_backproject_mask_filters_invalid_depth_and_transforms_to_world():
    mapping = MappingUpdate(
        frame_id="frame_0",
        timestamp=0.0,
        camera_pose=Pose3D(position=(1.0, 2.0, 3.0)),
        depth=np.array([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32),
        confidence=np.array([[1.0, 1.0], [1.0, 0.25]], dtype=np.float32),
        local_pointcloud=np.empty((0, 3), dtype=np.float32),
        global_pointcloud=np.empty((0, 3), dtype=np.float32),
        is_keyframe=True,
        map_version=1,
    )
    intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0, width=2, height=2)

    points = backproject_mask(np.ones((2, 2), dtype=bool), mapping, intrinsics)

    np.testing.assert_allclose(points, np.array([[1.0, 2.0, 4.0]], dtype=np.float32))


def test_pointcloud_store_persists_geometry_artifact(tmp_path):
    store = PointCloudStore(tmp_path / "instance_clouds")
    points = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32)

    geometry = store.put(
        "cup_1",
        points,
        coordinate_frame="world",
        confidence=0.9,
        mask_provenance=["frame_0", "mask_1"],
    )

    assert geometry.point_count == 2
    assert geometry.artifact_path is not None
    np.testing.assert_allclose(store.get(geometry), points)


def test_pointcloud_store_rejects_nonfinite_points(tmp_path):
    store = PointCloudStore(tmp_path)

    with pytest.raises(ValueError, match="non-finite"):
        store.put(
            "bad",
            np.array([[0.0, 0.0, np.nan]], dtype=np.float32),
            coordinate_frame="world",
            confidence=1.0,
        )
