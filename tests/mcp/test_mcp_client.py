"""Tests for the fake-robot MCP client and the five evaluation scenarios.

The fake robot is an independent MCP client: it reaches the server *only*
through MCP tools. These tests exercise that path and then run the five
spec scenarios end-to-end.
"""

from __future__ import annotations

import asyncio

from agentic_memory_nav.mcp.client import FakeRobot
from agentic_memory_nav.mcp.eval.scenarios import ALL_SCENARIOS
from agentic_memory_nav.mcp.server.app import build_app


def _robot() -> FakeRobot:
    _app, _engine, _hub, mcp = build_app()
    return FakeRobot.in_process("robot_0", mcp)


def _run(coro):
    return asyncio.run(coro)


def test_create_and_get_canvas() -> None:
    robot = _robot()

    async def main() -> None:
        await robot.connect()
        created = await robot.create()
        assert created["robot_id"] == "robot_0"
        canvas = await robot.get_canvas()
        assert canvas["coordinate_unit"] == "unit_length"
        assert any(r["robot_id"] == "robot_0" for r in canvas["robots"])
        await robot.close()

    _run(main())


def test_execute_by_name_and_by_id() -> None:
    robot = _robot()

    async def main() -> None:
        await robot.connect()
        await robot.reset()
        by_name = await robot.execute(action="move_forward")
        assert by_name["success"]
        by_id = await robot.execute(action_id=2)  # move_forward
        assert by_id["success"]
        assert (await robot.get_canvas())["trajectory_count"] == 2
        await robot.close()

    _run(main())


def test_list_actions_returns_canonical_mapping() -> None:
    robot = _robot()

    async def main() -> None:
        await robot.connect()
        actions = await robot.list_actions()
        mapping = actions.get("actions", actions)
        assert mapping["0"] == "turn_left"
        assert mapping["2"] == "move_forward"
        assert mapping["6"] == "stop"
        await robot.close()

    _run(main())


def test_all_five_scenarios_pass() -> None:
    robot = _robot()

    async def main() -> None:
        await robot.connect()
        failures: list[str] = []
        for title, scenario in ALL_SCENARIOS:
            for check in await scenario(robot):
                if not check.ok:
                    failures.append(f"{title}: {check}")
        await robot.close()
        assert not failures, "\n".join(failures)

    _run(main())