"""ASGI application: FastMCP server + WebSocket hub + canvas viewer.

A single ASGI app is served by uvicorn and exposes:

* the MCP tools (via the FastMCP HTTP transport),
* the WebSocket real-time channel ``/ws/canvas/main`` (:class:`EventHub`),
* the self-contained canvas viewer at ``/`` and a polling snapshot at
  ``/api/snapshot`` (compatible with the project's polling-based viewers).

The MCP tools, the WebSocket hub and the viewer all act on one shared
:class:`SimulationEngine`, so a fake-robot MCP client and the browser observe the
same live state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import WebSocketRoute

from agentic_memory_nav.mcp.core.actions import ACTION_BY_ID
from agentic_memory_nav.mcp.core.engine import SimulationEngine
from agentic_memory_nav.mcp.server.hub import EventHub

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def build_app(
    engine: SimulationEngine | None = None,
    *,
    transport: Literal["http", "streamable-http"] = "streamable-http",
) -> tuple[Starlette, SimulationEngine, EventHub, FastMCP]:
    """Build the ASGI app, the shared engine, the hub and the FastMCP server."""

    if engine is None:
        engine = SimulationEngine()

    mcp = FastMCP("infinite-canvas")
    _register_tools(mcp, engine)

    app = mcp.http_app(transport=transport)
    hub = EventHub(engine)

    _register_routes(app, engine, hub)
    return app, engine, hub, mcp


def _register_tools(mcp: FastMCP, engine: SimulationEngine) -> None:
    """Register all MCP tools on the server, each acting on the shared engine."""

    @mcp.tool
    async def create_robot(robot_id: str, start_pose: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a robot (or reset it to a start pose). Returns its pose."""

        return engine.create_robot(robot_id, start_pose or {})

    @mcp.tool
    async def execute_action(
        robot_id: str,
        action: str | None = None,
        action_id: int | None = None,
    ) -> dict[str, Any]:
        """Execute a robot action by name or id and return the transition result."""

        return engine.execute_action(robot_id, action=action, action_id=action_id).to_dict()

    @mcp.tool
    async def add_obstacle(robot_id: str, obstacle: dict[str, Any]) -> dict[str, Any]:
        """Add a point or line obstacle (position/geometry in unit lengths)."""

        return engine.add_obstacle(robot_id, obstacle)

    @mcp.tool
    async def update_obstacle(obstacle_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial update to an existing obstacle."""

        return engine.update_obstacle(obstacle_id, patch)

    @mcp.tool
    async def remove_obstacle(obstacle_id: str) -> dict[str, Any]:
        """Remove an obstacle by id."""

        return engine.remove_obstacle(obstacle_id)

    @mcp.tool
    async def get_canvas() -> dict[str, Any]:
        """Return the full 2D canvas snapshot (robots, trajectory, obstacles, events)."""

        return engine.get_canvas()

    @mcp.tool
    async def reset_simulation(
        robot_id: str | None = None,
        clear_events: bool = False,
    ) -> dict[str, Any]:
        """Reset robot poses, trajectory and obstacle state; optionally clear events."""

        return engine.reset_simulation(robot_id, clear_events)

    @mcp.tool
    async def list_actions() -> dict[str, Any]:
        """Return the canonical action id -> name mapping."""

        return {"actions": {str(k): v for k, v in ACTION_BY_ID.items()}}


def _register_routes(app: Starlette, engine: SimulationEngine, hub: EventHub) -> None:
    """Append the WebSocket channel, the canvas viewer, control and API endpoints.

    State is pushed to the browser over the WebSocket (with a polling fallback); a
    human may drive the robot from the UI through small HTTP POST control
    endpoints. These act on the same engine as the MCP tools, so the HMI and the
    fake-robot MCP client observe identical state.
    """

    async def serve_index(request: Request) -> Response:
        index_path = WEB_DIR / "index.html"
        if index_path.is_file():
            return FileResponse(str(index_path))
        return JSONResponse({"message": "MCP infinite-canvas server running"})

    async def serve_manual(request: Request) -> Response:
        manual_path = WEB_DIR / "manual.html"
        if manual_path.is_file():
            return FileResponse(str(manual_path))
        return JSONResponse({"message": "manual validation console not installed"})

    async def snapshot(request: Request) -> Response:
        return JSONResponse(engine.get_canvas())

    async def health(request: Request) -> Response:
        return JSONResponse(
            {"status": "ok", "version": engine.version, "connections": hub.connection_count}
        )

    async def actions(request: Request) -> Response:
        return JSONResponse({"actions": {str(k): v for k, v in ACTION_BY_ID.items()}})

    def _post(body: dict[str, Any]) -> JSONResponse:
        return JSONResponse(body)

    def _ensure(robot_id: str) -> None:
        if robot_id not in engine.robots:
            engine.create_robot(robot_id)

    async def control_create(request: Request) -> Response:
        body = await request.json()
        return _post(engine.create_robot(body.get("robot_id", "robot_0"), body.get("start_pose")))

    async def control_execute(request: Request) -> Response:
        body = await request.json()
        _ensure(body.get("robot_id", "robot_0"))
        result = engine.execute_action(
            body.get("robot_id", "robot_0"),
            action=body.get("action"),
            action_id=body.get("action_id"),
        )
        return _post(result.to_dict())

    async def control_add_obstacle(request: Request) -> Response:
        body = await request.json()
        _ensure(body.get("robot_id", "robot_0"))
        return _post(engine.add_obstacle(body.get("robot_id", "robot_0"), body.get("obstacle", {})))

    async def control_update_obstacle(request: Request) -> Response:
        body = await request.json()
        return _post(engine.update_obstacle(body.get("obstacle_id", ""), body.get("patch", {})))

    async def control_remove_obstacle(request: Request) -> Response:
        body = await request.json()
        return _post(engine.remove_obstacle(body.get("obstacle_id", "")))

    async def control_reset(request: Request) -> Response:
        body = await request.json()
        return _post(engine.reset_simulation(body.get("robot_id"), body.get("clear_events", False)))

    app.add_route("/", serve_index, methods=["GET"])
    app.add_route("/manual", serve_manual, methods=["GET"])
    app.add_route("/health", health, methods=["GET"])
    app.add_route("/api/snapshot", snapshot, methods=["GET"])
    app.add_route("/api/actions", actions, methods=["GET"])
    app.add_route("/api/create_robot", control_create, methods=["POST"])
    app.add_route("/api/execute_action", control_execute, methods=["POST"])
    app.add_route("/api/add_obstacle", control_add_obstacle, methods=["POST"])
    app.add_route("/api/update_obstacle", control_update_obstacle, methods=["POST"])
    app.add_route("/api/remove_obstacle", control_remove_obstacle, methods=["POST"])
    app.add_route("/api/reset", control_reset, methods=["POST"])
    app.router.routes.append(WebSocketRoute("/ws/canvas/main", hub.handle))