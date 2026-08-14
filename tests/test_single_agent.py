from pathlib import Path

from agentic_memory_nav.agent.single_agent import SingleAgent


def test_single_agent_runs_minimal_task(tmp_path):
    agent = SingleAgent(
        scratch_dir=tmp_path,
        instruction="Find the red cup in the kitchen",
        max_frames=4,
    )

    result = agent.run_task()

    assert result["task_status"] in {"succeeded", "failed"}
    assert result["graph_nodes"] >= 1
    assert result["memory_items"] >= 1
    assert result["plan_action"] in {"explore", "navigate"}
