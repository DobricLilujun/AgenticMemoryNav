"""Entry point: run the MCP infinite-canvas server with uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from agentic_memory_nav.mcp.presets import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TRANSPORT
from agentic_memory_nav.mcp.server.app import build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MCP infinite-canvas server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port.")
    parser.add_argument(
        "--transport",
        default=DEFAULT_TRANSPORT,
        choices=["streamable-http", "sse"],
        help="MCP transport for the fake-robot client.",
    )
    args = parser.parse_args()

    app, _engine, _hub, _mcp = build_app(transport=args.transport)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()