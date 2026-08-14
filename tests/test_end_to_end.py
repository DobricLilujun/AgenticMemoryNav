import asyncio
import json

from agentic_memory_nav.common.logging import configure_logging
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun
from agentic_memory_nav.orchestration.pipeline import NavigationPipeline


def test_red_cup_end_to_end(tmp_path):
    config = {
        "runtime": {"output_root": str(tmp_path), "queue_size": 64, "max_frames": 4},
        "mapping": {"backend": "mock", "keyframe_interval": 2},
        "perception": {"instance_geometry": {"enabled": True}},
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
    graph = json.loads((run.path / "scene_graph.json").read_text())
    assert graph["nodes"]
    cup = next(node for node in graph["nodes"] if node["label"] == "cup")
    assert cup["geometry"]["point_count"] > 0
    assert (run.artifacts / "instance_clouds" / "obs_cup").parent.is_dir()
    memory = json.loads((run.path / "memory_snapshot.json").read_text())
    assert any(item["structured_payload"].get("knowledge_type") == "triple" for item in memory)
    for name in (
        "metrics.json",
        "trajectory.jsonl",
        "memory_snapshot.json",
        "config.yaml",
        "logs.jsonl",
    ):
        assert (run.path / name).is_file()
