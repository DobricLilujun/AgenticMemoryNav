"""Bounded asynchronous event bus with provenance."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import Any

from agentic_memory_nav.common.types import new_id


class EventType(StrEnum):
    FRAME_RECEIVED = "FrameReceived"
    MAP_UPDATED = "MapUpdated"
    OBJECTS_DETECTED = "ObjectsDetected"
    SCENE_GRAPH_UPDATED = "SceneGraphUpdated"
    MEMORY_UPDATED = "MemoryUpdated"
    PLAN_GENERATED = "PlanGenerated"
    ACTION_ISSUED = "ActionIssued"
    ACTION_FEEDBACK = "ActionFeedback"
    COLLISION_DETECTED = "CollisionDetected"
    GOAL_REACHED = "GoalReached"
    REPLAN_REQUESTED = "ReplanRequested"
    EMERGENCY_STOP = "EmergencyStop"


@dataclass(slots=True)
class Event:
    event_type: EventType
    run_id: str
    producer: str
    sequence_number: int
    payload: dict[str, Any]
    provenance: list[str]
    timestamp: float = field(default_factory=time)
    event_id: str = field(default_factory=lambda: new_id("event"))


class EventBus:
    def __init__(self, max_queue_size: int = 32) -> None:
        self.max_queue_size = max_queue_size
        self._subscribers: list[asyncio.Queue[Event]] = []
        self.published = 0
        self.dropped = 0

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self.max_queue_size)
        self._subscribers.append(queue)
        return queue

    async def publish(self, event: Event, lossless: bool = True) -> None:
        self.published += 1
        for queue in self._subscribers:
            if lossless:
                await queue.put(event)
            elif queue.full():
                try:
                    queue.get_nowait()
                    queue.task_done()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(event)
            else:
                queue.put_nowait(event)
