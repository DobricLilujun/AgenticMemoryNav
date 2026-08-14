import json

from agentic_memory_nav.datasets.objectnav import load_experiments

SAMPLE = {
    "experiments": [
        {
            "name": "kujiale_0020_explore_moved",
            "scene": "kujiale_0020/kujiale_0020.usda",
            "goal": {"task": "explore"},
            "max_runtime": 900.0,
            "remove_assets": ["bottle"],
            "exclude_remove_assets": ["bottle_0012"],
            "robot_start": {"position": [-0.6, 0.0, 0.0]},
        },
        {
            "name": "kujiale_0020_bottle_moved",
            "scene": "kujiale_0020/kujiale_0020.usda",
            "initialmap_experiment": "kujiale_0020_explore_moved",
            "goal": {
                "task": "search",
                "label": "bottle",
                "asset": "bottle_0010",
                "prior_map_object": "bottle_0012",
            },
            "max_runtime": 300.0,
            "remove_assets": ["ornament", "bottle"],
            "exclude_remove_assets": ["bottle_0010"],
            "robot_start": {"position": [-0.6, 0.0, 0.0]},
        },
    ]
}


def test_parses_explore_and_search_experiments(tmp_path):
    path = tmp_path / "experiments.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")

    experiments = load_experiments(path)

    assert set(experiments) == {"kujiale_0020_explore_moved", "kujiale_0020_bottle_moved"}

    explore = experiments["kujiale_0020_explore_moved"]
    assert explore.goal.task == "explore"
    assert explore.goal.label is None
    assert explore.remove_assets == ["bottle"]
    assert explore.exclude_remove_assets == ["bottle_0012"]
    assert explore.robot_start == (-0.6, 0.0, 0.0)

    search = experiments["kujiale_0020_bottle_moved"]
    assert search.goal.task == "search"
    assert search.goal.label == "bottle"
    assert search.goal.asset == "bottle_0010"
    assert search.goal.prior_map_object == "bottle_0012"
    assert search.initialmap_experiment == "kujiale_0020_explore_moved"


def test_missing_file_raises(tmp_path):
    try:
        load_experiments(tmp_path / "missing.json")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
