import asyncio
import json

from agentic_memory_nav.common.logging import configure_logging
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun
from agentic_memory_nav.orchestration.pipeline import NavigationPipeline


def test_red_cup_end_to_end(tmp_path):
    config = {
        "runtime": {"output_root": str(tmp_path), "queue_size": 64, "max_frames": 4},
        "mapping": {"backend": "mock", "keyframe_interval": 2},
        "planning": {"backend": "rule_based", "approach_distance": 0.6},
        "execution": {"backend": "unitree_sim", "max_speed": 0.5, "max_action_timeout": 15.0},
    }
    run = ExperimentRun(tmp_path, config, run_id="test-run")
    pipeline = NavigationPipeline(config, run, configure_logging(run.path / "logs.jsonl"))
    try:
        metrics = asyncio.run(pipeline.run_task("找到厨房里的红色杯子并移动到附近"))
    finally:
        pipeline.close()
    assert metrics["success_rate"] == 1.0
    assert metrics["replanning_count"] >= 1
    assert json.loads((run.path / "scene_graph.json").read_text())["nodes"]
    for name in (
        "metrics.json",
        "trajectory.jsonl",
        "memory_snapshot.json",
        "config.yaml",
        "logs.jsonl",
    ):
        assert (run.path / name).is_file()
