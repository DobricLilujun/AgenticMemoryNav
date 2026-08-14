"""Procedural point-goal navigation episodes for the Isaac Sim PointNav harness.

Episodes are generated over the same fixed obstacle layout used by
`IsaacSimExecutor`'s default procedural scene (no external scene/episode dataset
download required), since Isaac Sim 6.0.1's standalone install does not bundle
Nucleus-hosted sample environments locally.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agentic_memory_nav.common.types import Vector3
from agentic_memory_nav.mapping.occupancy_grid import (
    Obstacle,
    build_occupancy_grid,
    shortest_path_distance,
)

# Mirrors IsaacSimExecutor._add_procedural_obstacles (center_x, center_y, half_x, half_y).
DEFAULT_OBSTACLES: list[Obstacle] = [
    (1.5, 0.0, 0.2, 0.2),
    (-1.5, 1.0, 0.2, 0.2),
    (0.5, -1.5, 0.2, 0.2),
]
DEFAULT_BOUNDS = (-3.0, -3.0, 3.0, 3.0)
DEFAULT_HEIGHT = 0.15


@dataclass(slots=True)
class PointNavEpisode:
    episode_id: str
    start: Vector3
    goal: Vector3
    shortest_path_m: float
    success_threshold_m: float = 0.36


@dataclass(slots=True)
class PointNavDataset:
    num_episodes: int = 10
    seed: int = 0
    resolution: float = 0.1
    success_threshold_m: float = 0.36
    min_geodesic_m: float = 1.0
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS
    obstacles: list[Obstacle] = field(default_factory=lambda: list(DEFAULT_OBSTACLES))

    def generate(self, max_attempts: int = 2000) -> list[PointNavEpisode]:
        grid = build_occupancy_grid(self.bounds, self.obstacles, self.resolution)
        rng = random.Random(self.seed)
        min_x, min_y, max_x, max_y = self.bounds
        episodes: list[PointNavEpisode] = []
        attempts = 0
        while len(episodes) < self.num_episodes and attempts < max_attempts:
            attempts += 1
            start_xy = (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            goal_xy = (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))
            distance = shortest_path_distance(grid, start_xy, goal_xy)
            if distance is None or distance < self.min_geodesic_m:
                continue
            episode_id = f"pointnav_{len(episodes):04d}"
            episodes.append(
                PointNavEpisode(
                    episode_id=episode_id,
                    start=(start_xy[0], DEFAULT_HEIGHT, start_xy[1]),
                    goal=(goal_xy[0], DEFAULT_HEIGHT, goal_xy[1]),
                    shortest_path_m=distance,
                    success_threshold_m=self.success_threshold_m,
                )
            )
        if len(episodes) < self.num_episodes:
            raise RuntimeError(
                f"Could only generate {len(episodes)}/{self.num_episodes} reachable "
                "PointNav episodes; loosen min_geodesic_m or widen bounds."
            )
        return episodes
