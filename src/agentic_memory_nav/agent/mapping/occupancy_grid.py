"""Pure-Python 2D occupancy grid and geodesic shortest-path distance.

Used by the Isaac Sim PointNav and ObjectNav harnesses to compute SPL. Grids are
rasterized from known procedural obstacle geometry rather than queried from a
live Isaac Sim stage, so this module has no Isaac Sim dependency and is fully
unit-testable.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

Obstacle = tuple[float, float, float, float]  # center_x, center_y, half_extent_x, half_extent_y


@dataclass(slots=True)
class OccupancyGrid:
    resolution: float
    min_x: float
    min_y: float
    occupied: np.ndarray  # bool array [rows, cols]; True means blocked

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape  # type: ignore[return-value]

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        row = int((y - self.min_y) / self.resolution)
        col = int((x - self.min_x) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.min_x + (col + 0.5) * self.resolution
        y = self.min_y + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        rows, cols = self.shape
        return 0 <= row < rows and 0 <= col < cols

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and not bool(self.occupied[row, col])


def build_occupancy_grid(
    bounds: tuple[float, float, float, float],
    obstacles: list[Obstacle],
    resolution: float = 0.1,
) -> OccupancyGrid:
    """Rasterize axis-aligned box obstacles into a boolean grid.

    `bounds` is (min_x, min_y, max_x, max_y) in world meters.
    """
    min_x, min_y, max_x, max_y = bounds
    cols = max(1, math.ceil((max_x - min_x) / resolution))
    rows = max(1, math.ceil((max_y - min_y) / resolution))
    occupied = np.zeros((rows, cols), dtype=bool)
    grid = OccupancyGrid(resolution, min_x, min_y, occupied)
    for center_x, center_y, half_x, half_y in obstacles:
        row_start, col_start = grid.world_to_cell(center_x - half_x, center_y - half_y)
        row_end, col_end = grid.world_to_cell(center_x + half_x, center_y + half_y)
        row_start, row_end = sorted((max(row_start, 0), min(row_end, rows - 1)))
        col_start, col_end = sorted((max(col_start, 0), min(col_end, cols - 1)))
        occupied[row_start : row_end + 1, col_start : col_end + 1] = True
    return grid


_NEIGHBORS = [
    (dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
]


def shortest_path_distance(
    grid: OccupancyGrid, start_xy: tuple[float, float], goal_xy: tuple[float, float]
) -> float | None:
    """8-connected Dijkstra shortest path distance in meters, or None if unreachable."""
    start = grid.world_to_cell(*start_xy)
    goal = grid.world_to_cell(*goal_xy)
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return None
    if start == goal:
        return 0.0

    distances = {start: 0.0}
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    visited: set[tuple[int, int]] = set()
    while queue:
        distance, node = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return distance * grid.resolution
        row, col = node
        for delta_row, delta_col, step_cost in _NEIGHBORS:
            neighbor = (row + delta_row, col + delta_col)
            if neighbor in visited or not grid.is_free(*neighbor):
                continue
            candidate = distance + step_cost
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return None
