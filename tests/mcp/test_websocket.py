"""Tests for real-time WebSocket / event push.

The :class:`EventHub` is the single mechanism the ``/ws/canvas/main`` route uses
to push events to connected clients. We verify the propagation chain an event
travels: ``engine event -> hub._on_event -> per-connection queue`` and that the
engine version bumps so a polling fallback sees the change too.
"""

from __future__ import annotations

import asyncio

from agentic_memory_nav.mcp.server.app import build_app
from agentic_memory_nav.mcp.server.hub import EventHub


def test_engine_events_reach_subscriber() -> None:
    async def main() -> None:
        _app, engine, _hub, _mcp = build_app()
        received: asyncio.Queue = asyncio.Queue()
        engine.subscribe(received.put_nowait)

        engine.create_robot("r")
        engine.execute_action("r", action="move_forward")
        engine.execute_action("r", action="turn_left")
        engine.add_obstacle("r", {"obstacle_type": "point", "x": 0.5, "radius": 0.1})

        types = [received.get_nowait()["event"] for _ in range(received.qsize())]
        assert "robot_updated" in types
        assert "trajectory_added" in types
        assert "obstacle_added" in types

    asyncio.run(main())


def test_hub_pushes_events_to_connection_queue() -> None:
    async def main() -> None:
        _app, engine, hub, _mcp = build_app()
        # emulate one connected client by attaching its queue to the hub
        queue: asyncio.Queue = asyncio.Queue()
        hub._queues.add(queue)

        engine.create_robot("r")
        engine.execute_action("r", action="move_forward")
        engine.add_obstacle("r", {"obstacle_type": "line", "start": [0.0, 0.0], "end": [1.0, 1.0]})

        assert hub.connection_count == 1
        payloads = [queue.get_nowait() for _ in range(queue.qsize())]
        kinds = {p["event"]["event"] for p in payloads if p.get("type") == "event"}
        assert "trajectory_added" in kinds
        assert "obstacle_added" in kinds

    asyncio.run(main())


def test_version_increments_on_event() -> None:
    async def main() -> None:
        _app, engine, _hub, _mcp = build_app()
        engine.create_robot("r")
        before = engine.version
        engine.execute_action("r", action="move_forward")
        assert engine.version > before

    asyncio.run(main())