import numpy as np
import pytest

from agentic_memory_nav.evaluation.pointcloud import evaluate_pointclouds, load_npz_pointcloud
from agentic_memory_nav.evaluation.trajectory_alignment import align_similarity, apply_similarity
from agentic_memory_nav.geometry.ground_truth import backproject_depth_to_world
from agentic_memory_nav.geometry.lidar import flat_scan_to_points


def test_pointcloud_evaluation_reports_perfect_reconstruction():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    metrics = evaluate_pointclouds(points, points, threshold_m=0.01)

    assert metrics["chamfer_l1_m"] == 0.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["alignment"] == "none"


def test_pointcloud_evaluation_requires_explicit_alignment_for_offset_clouds():
    ground_truth = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    prediction = ground_truth + np.array([2.0, -1.0, 0.5], dtype=np.float32)

    unaligned = evaluate_pointclouds(prediction, ground_truth, threshold_m=0.01)
    aligned = evaluate_pointclouds(
        prediction,
        ground_truth,
        threshold_m=0.01,
        alignment="centroid",
    )

    assert unaligned["f1"] == 0.0
    assert aligned["f1"] == 1.0
    assert aligned["chamfer_l1_m"] < 1e-5


def test_load_npz_pointcloud_requires_standard_points_key(tmp_path):
    artifact = tmp_path / "cloud.npz"
    expected = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.savez_compressed(artifact, points=expected)

    np.testing.assert_allclose(load_npz_pointcloud(str(artifact)), expected)


def test_ground_truth_backprojection_uses_camera_to_world_transform():
    depth = np.array([[2.0]], dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, 3] = (1.0, 2.0, 3.0)

    points = backproject_depth_to_world(depth, intrinsics, camera_to_world)

    np.testing.assert_allclose(points, np.array([[1.0, 2.0, 5.0]], dtype=np.float32))


def test_flat_lidar_scan_converts_valid_ranges_to_local_points():
    points = flat_scan_to_points(
        np.array([1.0, 2.0, np.nan], dtype=np.float32),
        np.array([0.0, np.pi / 2], dtype=np.float32),
    )

    np.testing.assert_allclose(
        points,
        np.array([[1.0, 0.0, 0.0], [np.sqrt(2.0), np.sqrt(2.0), 0.0]], dtype=np.float32),
        rtol=1e-5,
    )


def test_flat_lidar_scan_accepts_isaac_degree_azimuth_ranges():
    points = flat_scan_to_points(
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([0.0, 90.0], dtype=np.float32),
    )

    np.testing.assert_allclose(
        points,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_similarity_alignment_recovers_scale_rotation_and_translation():
    estimate = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    reference = np.array([[2.0, -1.0, 3.0], [2.0, 1.0, 3.0], [0.0, -1.0, 3.0]], dtype=np.float32)

    transform = align_similarity(estimate, reference)
    aligned = apply_similarity(estimate, transform)

    np.testing.assert_allclose(aligned, reference, atol=1e-5)
    assert transform.scale == pytest.approx(2.0)
