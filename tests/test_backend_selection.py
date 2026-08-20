import numpy as np
import pytest

from agentic_memory_nav.common.logging import configure_logging
from agentic_memory_nav.common.types import FrameObservation, MemoryType
from agentic_memory_nav.evaluation.experiment_logger import ExperimentRun
from agentic_memory_nav.agent.lingbot.adapter import LingBotMapAdapter
from agentic_memory_nav.agent.orchestration.pipeline import NavigationPipeline
from agentic_memory_nav.agent.vlm.backend import VLMBackend


def _base_config(tmp_path):
    return {
        "runtime": {"output_root": str(tmp_path), "queue_size": 8, "max_frames": 1},
        "mapping": {"backend": "mock"},
        "perception": {"backend": "mock"},
        "execution": {"backend": "unitree_sim"},
    }


def test_pipeline_selects_vlm_perception_backend(tmp_path):
    config = _base_config(tmp_path)
    config["perception"] = {
        "backend": "vlm",
        "model_id": "demo-vlm",
        "base_url": "http://example.test/v1",
        "api_key": "not-a-secret",
    }
    run = ExperimentRun(tmp_path, config, run_id="backend-selection")
    pipeline = NavigationPipeline(config, run, configure_logging(run.path / "logs.jsonl"))
    try:
        assert isinstance(pipeline.perception, VLMBackend)
        assert pipeline.perception.model_id == "demo-vlm"
    finally:
        pipeline.close()


def test_pipeline_rejects_incomplete_real_backend_configuration(tmp_path):
    config = _base_config(tmp_path)
    config["mapping"] = {"backend": "lingbot_map"}
    run = ExperimentRun(tmp_path, config, run_id="backend-invalid")

    with pytest.raises(ValueError, match="mapping.checkpoint"):
        NavigationPipeline(config, run, configure_logging(run.path / "logs.jsonl"))
    run.close()


def test_pipeline_passes_local_submap_configuration_to_lingbot(tmp_path):
    config = _base_config(tmp_path)
    checkpoint = tmp_path / "lingbot.pt"
    checkpoint.touch()
    config["mapping"] = {
        "backend": "lingbot_map",
        "checkpoint": str(checkpoint),
        "local_submap_frames": 200,
        "local_submap_stride": 3,
        "local_submap_stability_threshold_m": 0.2,
    }
    run = ExperimentRun(tmp_path, config, run_id="submap-config")
    pipeline = NavigationPipeline(config, run, configure_logging(run.path / "logs.jsonl"))
    try:
        assert pipeline.mapper.local_submaps.window_frames == 200
        assert pipeline.mapper.local_submaps.frame_stride == 3
        assert pipeline.mapper.local_submaps.stability_threshold_m == 0.2
    finally:
        pipeline.close()


def test_pipeline_persists_only_stable_depth_pose_submaps(tmp_path):
    config = _base_config(tmp_path)
    config["runtime"]["max_frames"] = 2
    config["mapping"].update(
        {
            "commit_stable_submaps_only": True,
        }
    )
    run = ExperimentRun(tmp_path, config, run_id="stable-submap")
    pipeline = NavigationPipeline(config, run, configure_logging(run.path / "logs.jsonl"))

    def predictor(frame):
        return {
            "camera_pose": {
                "position": [0.0, 0.0, 0.0],
                "camera_to_world": np.eye(4, dtype=np.float32).tolist(),
            },
            "intrinsics": np.eye(3, dtype=np.float32).tolist(),
            "depth": [[2.0]],
            "confidence": [[1.0]],
            "is_keyframe": True,
        }

    pipeline.mapper = LingBotMapAdapter(
        checkpoint=tmp_path / "lingbot.pt",
        predictor=predictor,
        local_submap_frames=2,
        local_submap_stride=1,
        local_submap_stability_threshold_m=0.1,
    )
    try:
        pipeline.mapper.start()
        pipeline.mapper.update(
            FrameObservation("frame_0", 0.0, np.zeros((1, 1, 3), dtype=np.uint8))
        )
        pipeline._persist_stable_submap()
        pipeline.mapper.update(
            FrameObservation("frame_1", 1.0, np.zeros((1, 1, 3), dtype=np.uint8))
        )
        pipeline._persist_stable_submap()
        spatial = [
            item for item in pipeline.memory.all_items() if item.memory_type == MemoryType.SPATIAL
        ]
        assert len(spatial) == 1
        geometry = spatial[0].structured_payload["geometry"]
        assert geometry["point_count"] == 2
        assert spatial[0].structured_payload["geometry_memory_policy"] == "stable_rgb_only"
        assert spatial[0].structured_payload["validation"] == "adjacent_submap_overlap"
        assert any((run.artifacts / "local_submaps").glob("*.npz"))
    finally:
        pipeline.close()
