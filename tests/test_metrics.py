from agentic_memory_nav.evaluation.metrics import goal_reached, success_weighted_path_length


def test_goal_reached_threshold():
    assert goal_reached(0.3, 0.36)
    assert goal_reached(0.36, 0.36)
    assert not goal_reached(0.4, 0.36)


def test_spl_penalizes_longer_paths():
    direct = success_weighted_path_length(True, shortest=2.0, actual=2.0)
    longer = success_weighted_path_length(True, shortest=2.0, actual=4.0)
    failed = success_weighted_path_length(False, shortest=2.0, actual=2.0)
    assert direct == 1.0
    assert 0.0 < longer < 1.0
    assert failed == 0.0
