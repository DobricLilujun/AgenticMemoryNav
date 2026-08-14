"""Similarity alignment for trajectory and point-cloud calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray


def align_similarity(estimate: np.ndarray, reference: np.ndarray) -> SimilarityTransform:
    """Fit $y = s R x + t$ using corresponding 3D trajectory positions."""
    source = _validate(estimate, "estimate")
    target = _validate(reference, "reference")
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Similarity alignment requires at least three corresponding 3D positions")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right = np.linalg.svd(covariance)
    determinant = np.linalg.det(left @ right)
    correction = np.diag([1.0, 1.0, 1.0 if determinant > 0 else -1.0])
    rotation = left @ correction @ right
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= 1e-12:
        raise ValueError("Estimate trajectory has zero variance")
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def apply_similarity(points: np.ndarray, transform: SimilarityTransform) -> np.ndarray:
    """Apply a fitted similarity transform to points with shape $(N, 3)$."""
    cloud = _validate(points, "points")
    return (transform.scale * (cloud @ transform.rotation.T) + transform.translation).astype(
        np.float32
    )


def _validate(points: np.ndarray, name: str) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError(f"{name} must have finite shape (N, 3)")
    return cloud
