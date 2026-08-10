"""SQLite structured and vector memory backend."""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

from agentic_memory_nav.common.types import MemoryItem, MemoryType, Vector3, jsonable, new_id
from agentic_memory_nav.memory.vector_store import LocalVectorStore


class SQLiteMemory:
    def __init__(self, path: Path, vector_store: LocalVectorStore | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store or LocalVectorStore()
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                payload TEXT NOT NULL,
                embedding TEXT,
                timestamp REAL NOT NULL,
                x REAL,
                y REAL,
                z REAL,
                confidence REAL NOT NULL,
                decay_score REAL NOT NULL,
                provenance TEXT NOT NULL,
                PRIMARY KEY (memory_id, version)
            );
            CREATE INDEX IF NOT EXISTS idx_memories_time ON memories(timestamp);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            """
        )
        self.connection.commit()

    def add_observation(self, item: MemoryItem) -> MemoryItem:
        if item.embedding is None:
            item.embedding = self.vector_store.embed_text(item.content)
        location = item.location or (None, None, None)
        self.connection.execute(
            """INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                item.memory_id,
                item.version,
                item.memory_type.value,
                item.content,
                json.dumps(jsonable(item.structured_payload), sort_keys=True),
                json.dumps(item.embedding),
                item.timestamp,
                *location,
                item.confidence,
                item.decay_score,
                json.dumps(item.provenance),
            ),
        )
        self.connection.commit()
        return item

    def update_entity(self, entity_id: str, update: dict[str, object]) -> MemoryItem:
        records = self.retrieve_by_entity(entity_id)
        if not records:
            raise KeyError(f"No memory found for entity {entity_id}")
        latest = records[0]
        payload = {**latest.structured_payload, **update}
        item = replace(
            latest,
            structured_payload=payload,
            content=str(update.get("content", latest.content)),
            timestamp=time.time(),
            version=latest.version + 1,
            provenance=[*latest.provenance, f"update:{entity_id}"],
        )
        return self.add_observation(item)

    def retrieve_by_text(self, query: str, limit: int = 5) -> list[MemoryItem]:
        query_tokens = set(query.lower().split())
        query_embedding = self.vector_store.embed_text(query)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self._latest_items():
            text_tokens = set(item.content.lower().split())
            lexical = len(query_tokens & text_tokens) / max(1, len(query_tokens))
            vector = self.vector_store.similarity(query_embedding, item.embedding)
            score = 0.55 * lexical + 0.45 * max(0.0, vector)
            if score > 0:
                scored.append((score, self._with_decay(item)))
        return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]

    def retrieve_by_entity(self, entity_id: str) -> list[MemoryItem]:
        rows = self.connection.execute(
            """SELECT * FROM memories
               WHERE memory_id = ? OR json_extract(payload, '$.entity_id') = ?
               ORDER BY version DESC, timestamp DESC""",
            (entity_id, entity_id),
        ).fetchall()
        return [self._with_decay(self._from_row(row)) for row in rows]

    def retrieve_by_region(self, position: Vector3, radius: float) -> list[MemoryItem]:
        output = []
        for item in self._latest_items():
            if item.location is None:
                continue
            distance = math.dist(position, item.location)
            if distance <= radius:
                output.append(self._with_decay(item))
        return sorted(output, key=lambda item: math.dist(position, item.location or position))

    def retrieve_by_time(self, start: float, end: float) -> list[MemoryItem]:
        rows = self.connection.execute(
            "SELECT * FROM memories WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (start, end),
        ).fetchall()
        return [self._with_decay(self._from_row(row)) for row in rows]

    def retrieve_for_task(self, task: str, limit: int = 10) -> list[MemoryItem]:
        return self.retrieve_by_text(task, limit=limit)

    def consolidate(self) -> int:
        latest = self._latest_items()
        groups: dict[str, list[MemoryItem]] = {}
        for item in latest:
            entity_id = str(item.structured_payload.get("entity_id", ""))
            if entity_id:
                groups.setdefault(entity_id, []).append(item)
        created = 0
        for entity_id, records in groups.items():
            if len(records) < 2:
                continue
            content = max(records, key=lambda item: item.timestamp).content
            provenance = [record.memory_id for record in records]
            self.add_observation(
                MemoryItem(
                    memory_id=new_id("mem"),
                    memory_type=MemoryType.SEMANTIC,
                    content=f"Consolidated fact: {content}",
                    structured_payload={"entity_id": entity_id, "derived": True},
                    timestamp=max(record.timestamp for record in records),
                    location=records[-1].location,
                    confidence=sum(record.confidence for record in records) / len(records),
                    provenance=provenance,
                )
            )
            created += 1
        return created

    def detect_stale(self, max_age_seconds: float, now: float | None = None) -> list[MemoryItem]:
        current = time.time() if now is None else now
        return [item for item in self._latest_items() if current - item.timestamp > max_age_seconds]

    def record_contradiction(self, left: MemoryItem, right: MemoryItem, reason: str) -> MemoryItem:
        return self.add_observation(
            MemoryItem(
                memory_id=new_id("mem"),
                memory_type=MemoryType.UNCERTAINTY,
                content=f"Contradiction: {reason}",
                structured_payload={"left": left.memory_id, "right": right.memory_id},
                timestamp=max(left.timestamp, right.timestamp),
                location=right.location or left.location,
                confidence=min(left.confidence, right.confidence),
                provenance=[left.memory_id, right.memory_id],
            )
        )

    def all_items(self) -> list[MemoryItem]:
        return self._latest_items()

    def save_snapshot(self, path: Path) -> None:
        self.connection.commit()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(
                json.dumps(
                    [jsonable(item) for item in self._latest_items()], indent=2, sort_keys=True
                ),
                encoding="utf-8",
            )
        else:
            shutil.copy2(self.path, path)

    def close(self) -> None:
        self.connection.close()

    def _latest_items(self) -> list[MemoryItem]:
        rows = self.connection.execute(
            """SELECT m.* FROM memories m JOIN (
                   SELECT memory_id, MAX(version) version FROM memories GROUP BY memory_id
               ) latest ON m.memory_id = latest.memory_id AND m.version = latest.version
               ORDER BY m.timestamp DESC"""
        ).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _from_row(row: sqlite3.Row) -> MemoryItem:
        location = None if row["x"] is None else (row["x"], row["y"], row["z"])
        return MemoryItem(
            memory_id=row["memory_id"],
            memory_type=MemoryType(row["memory_type"]),
            content=row["content"],
            structured_payload=json.loads(row["payload"]),
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
            timestamp=row["timestamp"],
            location=location,
            confidence=row["confidence"],
            provenance=json.loads(row["provenance"]),
            decay_score=row["decay_score"],
            version=row["version"],
        )

    @staticmethod
    def _with_decay(item: MemoryItem) -> MemoryItem:
        age = max(0.0, time.time() - item.timestamp)
        return replace(item, decay_score=item.confidence * math.exp(-age / 86_400.0))
