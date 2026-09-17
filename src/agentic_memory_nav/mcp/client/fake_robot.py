"""Fake robot: an independent MCP *client*.

The fake robot is a stand-in for a physical robot. It is a real MCP client: it
talks to the server only through MCP tools (``create_robot``, ``execute_action``,
``add_obstacle``, ``get_canvas`` ...). It never touches the server's internal
state (``robot_state``, ``canvas_state``), so the same code path is exercised that
a real robot over the network would use.

Two connection modes are supported:

* in-process, via the FastMCP server object (deterministic, no network);
* over the network, via an HTTP (streamable-http / SSE) URL.

Both go through the MCP protocol; the in-process mode simply avoids a socket.
"""

from __future__ import annotations

from typing import Any

from fastmcp.client import (
    Client,
    FastMCPTransport,
    SSETransport,
    StreamableHttpTransport,
)

_Client = Client[Any]


def _extract(result: Any) -> dict[str, Any]:
    """Return the dict a tool produced from a FastMCP ``CallToolResult``."""

    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        # FastMCP wraps non-dict returns in {"result": value}; prefer the dict.
        return structured
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            import json

            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return {"text": text}
            return parsed if isinstance(parsed, dict) else {"result": parsed}
    return {}


class FakeRobot:
    """A robot simulated through the MCP tools."""

    def __init__(self, robot_id: str = "robot_0", *, client: _Client | None = None) -> None:
        self.robot_id = robot_id
        self._client = client
        self._owns_client = client is None

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def in_process(cls, robot_id: str, mcp_server: Any) -> "FakeRobot":
        """Create a robot bound to an in-process FastMCP server object."""

        client = Client(transport=FastMCPTransport(mcp_server))
        return cls(robot_id, client=client)

    @classmethod
    def from_url(cls, robot_id: str, url: str, *, transport: str = "streamable-http") -> "FakeRobot":
        """Create a robot bound to a network endpoint."""

        if transport in ("streamable-http", "streamable_http"):
            client = Client(transport=StreamableHttpTransport(url=url))
        elif transport == "sse":
            # fastmcp types only StreamableHttpTransport on Client; SSE is valid at runtime
            client = Client(transport=SSETransport(url=url))  # type: ignore[arg-type]
        else:
            raise ValueError(f"unknown transport: {transport}")
        return cls(robot_id, client=client)

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    async def connect(self) -> "FakeRobot":
        if self._client is None:
            raise RuntimeError("no client configured")
        await self._client.__aenter__()  # type: ignore[no-untyped-call]
        return self

    async def close(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)  # type: ignore[no-untyped-call]
            self._client = None

    async def __aenter__(self) -> "FakeRobot":
        return await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _require(self) -> _Client:
        if self._client is None:
            raise RuntimeError("not connected; call connect() or use as a context manager")
        return self._client

    async def _call(self, tool: str, **arguments: Any) -> dict[str, Any]:
        result = await self._require().call_tool(tool, arguments)
        return _extract(result)

    # ------------------------------------------------------------------
    # MCP tool calls (the ONLY way this robot reaches the server)
    # ------------------------------------------------------------------

    async def create(self, start_pose: dict[str, Any] | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"robot_id": self.robot_id}
        if start_pose is not None:
            args["start_pose"] = start_pose
        return await self._call("create_robot", **args)

    async def execute(self, action: str | None = None, action_id: int | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {"robot_id": self.robot_id}
        if action is not None:
            args["action"] = action
        if action_id is not None:
            args["action_id"] = action_id
        return await self._call("execute_action", **args)

    async def add_obstacle(self, obstacle: dict[str, Any]) -> dict[str, Any]:
        return await self._call("add_obstacle", robot_id=self.robot_id, obstacle=obstacle)

    async def update_obstacle(self, obstacle_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        return await self._call("update_obstacle", obstacle_id=obstacle_id, patch=patch)

    async def remove_obstacle(self, obstacle_id: str) -> dict[str, Any]:
        return await self._call("remove_obstacle", obstacle_id=obstacle_id)

    async def get_canvas(self) -> dict[str, Any]:
        return await self._call("get_canvas")

    async def reset(self, clear_events: bool = False) -> dict[str, Any]:
        return await self._call("reset_simulation", robot_id=self.robot_id, clear_events=clear_events)

    async def list_actions(self) -> dict[str, Any]:
        return await self._call("list_actions")