#!/usr/bin/env python3
"""Validate a completed run and summarize reproducible MVP metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {
    "metrics.json",
    "trajectory.jsonl",
    "scene_graph.json",
    "memory_snapshot.json",
    "config.yaml",
    "logs.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    missing = sorted(name for name in REQUIRED if not (run_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Run is missing required artifacts: {missing}")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    graph = json.loads((run_dir / "scene_graph.json").read_text(encoding="utf-8"))
    memory = json.loads((run_dir / "memory_snapshot.json").read_text(encoding="utf-8"))
    node_ids = {node["node_id"] for node in graph["nodes"]}
    summary = {
        "valid": True,
        "success_rate": metrics["success_rate"],
        "spl": metrics["spl"],
        "graph_consistency": all(
            edge["source_id"] in node_ids and edge["target_id"] in node_ids
            for edge in graph["edges"]
        ),
        "provenance_completeness": sum(bool(item["provenance"]) for item in memory)
        / max(1, len(memory)),
    }
    (run_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
