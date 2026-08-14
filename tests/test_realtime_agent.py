import numpy as np

from agentic_memory_nav.agent.realtime_agent import RealtimeAgent
from agentic_memory_nav.common.types import CameraIntrinsics, FrameObservation, Pose3D


def _frame(index: int) -> FrameObservation:
    return FrameObservation(
        frame_id=f"frame_{index:04d}",
        timestamp=float(index),
        rgb=np.zeros((64, 96, 3), dtype=np.uint8),
        depth=np.full((64, 96), 2.0, dtype=np.float32),
        camera_intrinsics=CameraIntrinsics(80.0, 80.0, 48.0, 32.0, 96, 64),
        camera_pose=Pose3D(),
        robot_pose=Pose3D(),
    )


def test_realtime_agent_replans_from_explore_to_navigate(tmp_path):
    agent = RealtimeAgent(tmp_path, "Find the red cup in the kitchen")
    try:
        first = agent.ingest_frame(_frame(0))
        second = agent.ingest_frame(_frame(1))
    finally:
        agent.close()

    assert first.plan.action.action_type.value == "explore"
    assert second.plan.action.action_type.value == "navigate"
    assert second.plan.action.target is not None
    assert second.graph_nodes >= 2
    assert second.knowledge_facts_created >= 1
