from agentic_memory_nav.common.types import Pose3D
from agentic_memory_nav.memory.sqlite_store import SQLiteMemory
from agentic_memory_nav.agent.planning.rule_based_fallback import RuleBasedPlanner
from agentic_memory_nav.agent.planning.task_parser import RuleBasedTaskParser
from agentic_memory_nav.scene_graph.graph import SceneGraph


def test_unknown_target_triggers_exploration(tmp_path):
    memory = SQLiteMemory(tmp_path / "memory.db")
    task = RuleBasedTaskParser().parse("Find the red cup in the kitchen")
    plan = RuleBasedPlanner().plan(task, Pose3D(), SceneGraph(), memory)
    assert plan.action.action_type.value == "explore"
    assert plan.replan_required
    memory.close()
