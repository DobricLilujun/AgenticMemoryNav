from agentic_memory_nav.agent.mapping.occupancy_grid import build_occupancy_grid, shortest_path_distance


def test_shortest_path_direct_when_unobstructed():
    grid = build_occupancy_grid((-2.0, -2.0, 2.0, 2.0), [], resolution=0.1)
    distance = shortest_path_distance(grid, (-1.0, 0.0), (1.0, 0.0))
    assert distance is not None
    assert abs(distance - 2.0) < 0.2


def test_shortest_path_detours_around_obstacle():
    obstacles = [(0.0, 0.0, 1.0, 0.3)]
    grid = build_occupancy_grid((-2.0, -2.0, 2.0, 2.0), obstacles, resolution=0.1)
    direct_grid = build_occupancy_grid((-2.0, -2.0, 2.0, 2.0), [], resolution=0.1)
    blocked_distance = shortest_path_distance(grid, (-1.5, 0.0), (1.5, 0.0))
    direct_distance = shortest_path_distance(direct_grid, (-1.5, 0.0), (1.5, 0.0))
    assert blocked_distance is not None
    assert direct_distance is not None
    assert blocked_distance > direct_distance


def test_shortest_path_unreachable_returns_none():
    # Fully enclose the goal cell with obstacles.
    obstacles = [
        (0.5, 0.0, 0.05, 0.6),
        (-0.5, 0.0, 0.05, 0.6),
        (0.0, 0.5, 0.6, 0.05),
        (0.0, -0.5, 0.6, 0.05),
    ]
    grid = build_occupancy_grid((-2.0, -2.0, 2.0, 2.0), obstacles, resolution=0.1)
    distance = shortest_path_distance(grid, (-1.5, -1.5), (0.0, 0.0))
    assert distance is None
