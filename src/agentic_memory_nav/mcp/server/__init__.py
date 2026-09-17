"""MCP server package: FastMCP tools + WebSocket hub + canvas viewer."""

from __future__ import annotations

from agentic_memory_nav.mcp.server.app import build_app
from agentic_memory_nav.mcp.server.hub import EventHub

__all__ = ["build_app", "EventHub"]