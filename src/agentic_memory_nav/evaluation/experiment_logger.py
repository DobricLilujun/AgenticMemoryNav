"""Run directory and reproducible artifact management."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agentic_memory_nav.common.types import jsonable, new_id


class ExperimentRun:
    def __init__(
        self, output_root: Path, config: dict[str, Any], run_id: str | None = None
    ) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{stamp}_{new_id('run').split('_', 1)[1][:8]}"
        self.path = output_root / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.artifacts = self.path / "artifacts"
        self.artifacts.mkdir()
        self.write_yaml("config.yaml", self._redact(config))
        self._trajectory = (self.path / "trajectory.jsonl").open("w", encoding="utf-8")

    def append_trajectory(self, payload: dict[str, Any]) -> None:
        self._trajectory.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")
        self._trajectory.flush()

    def write_json(self, name: str, payload: Any) -> None:
        (self.path / name).write_text(
            json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
        )

    def write_yaml(self, name: str, payload: Any) -> None:
        (self.path / name).write_text(
            yaml.safe_dump(jsonable(payload), sort_keys=True), encoding="utf-8"
        )

    def close(self) -> None:
        self._trajectory.close()

    @classmethod
    def _redact(cls, value: Any, key: str = "") -> Any:
        lowered = key.lower()
        if any(token in lowered for token in ("api_key", "token", "secret", "password")):
            return "<redacted>"
        if isinstance(value, dict):
            return {item_key: cls._redact(item, str(item_key)) for item_key, item in value.items()}
        return value
