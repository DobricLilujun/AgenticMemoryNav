"""MCP infinite-canvas system.

A 2D robot canvas exposed through an MCP server, driven by a fake robot that is a
pure MCP client. The canvas is updated in real time over a WebSocket and can be
queried on demand with ``get_canvas``.

The dependency-free simulation core lives in :mod:`agentic_memory_nav.mcp.core`;
the MCP server, WebSocket layer, fake-robot client and web viewer wrap it.
"""

from __future__ import annotations

from .core.actions import ACTION_BY_ID
from .core.engine import SimulationEngine

__all__ = ["ACTION_BY_ID", "SimulationEngine"]