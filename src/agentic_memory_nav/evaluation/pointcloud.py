"""Ground-truth point-cloud evaluation with explicit coordinate alignment."""

from __future__ import annotations

from typing import Literal

import numpy as np

AlignmentMode = Literal["none", "centroid"]


def load_npz_pointcloud(path: str, key: str = "points") -> np.ndarray:
    """Load a point cloud stored as a compressed NPZ array under `key`."""
    with np.load(path) as artifact:
        if key not in artifact:
            raise ValueError(f"Point-cloud artifact {path!r} does not contain key {key!r}")
        return _validate(artifact[key], path)


def evaluate_pointclouds(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    threshold_m: float = 0.05,
    alignment: AlignmentMode = "none",
    max_points: int = 20_000,
) -> dict[str, float | int | str]:
    """Compare world-coordinate point clouds without silently hiding pose error.

    `alignment="none"` is the benchmark default and requires both clouds to use the
    same coordinate frame. `alignment="centroid"` is diagnostic only: it removes
    global translation but retains scale, rotation, and local reconstruction error.
    """
    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    predicted = _sample(_validate(prediction, "prediction"), max_points)
    reference = _sample(_validate(ground_truth, "ground_truth"), max_points)
    if alignment == "centroid":
        predicted = predicted - predicted.mean(axis=0) + reference.mean(axis=0)
    elif alignment != "none":
        raise ValueError(f"Unsupported alignment mode: {alignment}")

    prediction_to_gt = _nearest_distances(predicted, reference)
    gt_to_prediction = _nearest_distances(reference, predicted)
    precision = float(np.mean(prediction_to_gt <= threshold_m))
    recall = float(np.mean(gt_to_prediction <= threshold_m))
    return {
        "alignment": alignment,
        "threshold_m": threshold_m,
        "predicted_points": len(predicted),
        "ground_truth_points": len(reference),
        "accuracy_m": float(np.mean(prediction_to_gt)),
        "completeness_m": float(np.mean(gt_to_prediction)),
        "chamfer_l1_m": float(np.mean(prediction_to_gt) + np.mean(gt_to_prediction)),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def _validate(points: np.ndarray, name: str) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
        raise ValueError(f"{name} point cloud must have non-empty shape (N, 3)")
    if not np.isfinite(cloud).all():
        raise ValueError(f"{name} point cloud contains non-finite coordinates")
    return cloud


def _sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices]


def _nearest_distances(
    source: np.ndarray,
    target: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    distances = np.empty(len(source), dtype=np.float32)
    for start in range(0, len(source), batch_size):
        batch = source[start : start + batch_size]
        squared = np.sum((batch[:, None, :] - target[None, :, :]) ** 2, axis=2)
        distances[start : start + len(batch)] = np.sqrt(squared.min(axis=1))
    return distances
