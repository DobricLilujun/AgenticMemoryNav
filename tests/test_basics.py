"""Smoke tests for core agent primitives."""

from pathlib import Path

from agentic_memory_nav.agent.execution.discrete_actions import (
    ACTION_BY_ID,
    DiscreteAction,
    parse_discrete_action,
)
from agentic_memory_nav.agent.planning.task_parser import RuleBasedTaskParser
from agentic_memory_nav.common.config import AppConfig
from agentic_memory_nav.scene_graph.graph import SceneGraph


def test_parse_discrete_action_by_name() -> None:
    assert parse_discrete_action("move_forward") == DiscreteAction.MOVE_FORWARD


def test_parse_discrete_action_by_id() -> None:
    assert parse_discrete_action("2") == DiscreteAction.MOVE_FORWARD


def test_action_id_round_trip() -> None:
    for action_id, action in ACTION_BY_ID.items():
        assert parse_discrete_action(str(action_id)) == action


def test_task_parser_finds_red_cup() -> None:
    parser = RuleBasedTaskParser()
    task = parser.parse("find the red cup in the kitchen")
    assert task.parsed_goal["object"] == "cup"
    assert task.parsed_goal["color"] == "red"
    assert task.parsed_goal["room"] == "kitchen"


def test_scene_graph_starts_empty() -> None:
    graph = SceneGraph()
    assert graph.nodes() == []
    assert graph.edges() == []


def test_config_resolves_repo_relative_paths(tmp_path: Path) -> None:
    config = AppConfig({}, tmp_path / "configs" / "test.yaml")
    assert config.resolve_path("assets/scene.usd") == str((tmp_path / "assets/scene.usd").resolve())
    assert config.resolve_path("omniverse://server/scene.usd") == "omniverse://server/scene.usd"
