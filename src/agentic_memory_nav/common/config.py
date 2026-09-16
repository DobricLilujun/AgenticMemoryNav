"""YAML configuration loading with deterministic override behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class AppConfig:
    raw: dict[str, Any]
    source: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section {name!r} must be a mapping")
        return value

    def resolve_path(self, value: str | Path | None) -> str | None:
        """Resolve local paths relative to the directory containing this config."""
        if value is None:
            return None
        text = str(value)
        if "://" in text:
            return text
        path = Path(text).expanduser()
        if not path.is_absolute():
            config_relative = self.source.parent / path
            project_relative = self.source.parent.parent / path
            path = config_relative if config_relative.exists() else project_relative
        return str(path.resolve())


def load_config(path: str | Path) -> AppConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Top-level configuration must be a mapping")
    return AppConfig(raw=payload, source=source)
