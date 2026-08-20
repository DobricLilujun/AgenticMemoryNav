from agentic_memory_nav.agent.datasets.pointnav import PointNavDataset


def test_generates_requested_episode_count_deterministically():
    dataset = PointNavDataset(num_episodes=5, seed=1)
    episodes = dataset.generate()
    assert len(episodes) == 5
    ids = {episode.episode_id for episode in episodes}
    assert len(ids) == 5

    repeat = PointNavDataset(num_episodes=5, seed=1).generate()
    assert [e.start for e in episodes] == [e.start for e in repeat]
    assert [e.goal for e in episodes] == [e.goal for e in repeat]


def test_episodes_meet_minimum_geodesic_distance():
    dataset = PointNavDataset(num_episodes=8, seed=2, min_geodesic_m=1.5)
    episodes = dataset.generate()
    assert all(episode.shortest_path_m >= 1.5 for episode in episodes)


def test_episodes_stay_within_bounds():
    bounds = (-3.0, -3.0, 3.0, 3.0)
    dataset = PointNavDataset(num_episodes=6, seed=3, bounds=bounds)
    for episode in dataset.generate():
        for point in (episode.start, episode.goal):
            assert bounds[0] <= point[0] <= bounds[2]
            assert bounds[1] <= point[2] <= bounds[3]
