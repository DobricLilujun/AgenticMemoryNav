"""WebSocket real-time hub for the infinite canvas.

The hub subscribes to the simulation engine's event stream and fans out each
event to every connected ``/ws/canvas/main`` client. A fresh connection also
receives an initial full-canvas snapshot so the viewer can render the current
state immediately, before any further events arrive.

Design notes
------------
* The engine's listener (:meth:`_on_event`) is synchronous and only enqueues a
  payload per connection (``asyncio.Queue``), so it never blocks the event loop.
* Each connection runs a *receiver* task (detects client closes) and a *sender*
  task (drains its queue); when either ends, the connection is cleaned up.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from ..core.engine import SimulationEngine


class EventHub:
    """Bridge the simulation engine to WebSocket clients."""

    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine
        self._queues: set[asyncio.Queue[dict[str, Any] | None]] = set()
        engine.subscribe(self._on_event)

    # ------------------------------------------------------------------
    # engine listener (synchronous)
    # ------------------------------------------------------------------

    def _on_event(self, event: dict[str, Any]) -> None:
        """Push an engine event to every connection queue (non-blocking)."""

        payload = {"type": "event", "event": event}
        for queue in list(self._queues):
            queue.put_nowait(payload)

    def _snapshot_message(self) -> dict[str, Any]:
        return {"type": "snapshot", "canvas": self.engine.get_canvas()}

    # ------------------------------------------------------------------
    # connection lifecycle
    # ------------------------------------------------------------------

    @property
    def connection_count(self) -> int:
        return len(self._queues)

    async def handle(self, websocket: WebSocket) -> None:
        """Full lifecycle for a ``/ws/canvas/main`` connection."""

        try:
            await websocket.accept()
            await websocket.send_json(self._snapshot_message())
        except Exception:  # noqa: BLE001 - client gone before handshake
            await websocket.close()
            return

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._queues.add(queue)
        try:
            async def receiver() -> None:
                try:
                    while True:
                        text = await websocket.receive_text()
                        await self._handle_client_message(websocket, text)
                except Exception:  # noqa: BLE001 - connection closed
                    pass
                queue.put_nowait(None)

            async def sender() -> None:
                while True:
                    payload = await queue.get()
                    if payload is None:
                        return
                    await websocket.send_json(payload)

            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())
            await asyncio.gather(sender_task, receiver_task)
        finally:
            self._queues.discard(queue)

    async def _handle_client_message(self, websocket: WebSocket, message: str) -> None:
        """Handle a client -> server message (e.g. an explicit refresh request)."""

        try:
            data = json.loads(message)
        except (ValueError, TypeError):
            return
        if isinstance(data, dict) and data.get("type") in ("ping", "refresh", "snapshot"):
            await websocket.send_json(self._snapshot_message())