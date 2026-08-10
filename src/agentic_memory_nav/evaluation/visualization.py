"""Run artifact visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def render_run_artifacts(run_dir: Path) -> None:
    visual_dir = run_dir / "artifacts" / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    _render_trajectory(run_dir / "trajectory.jsonl", visual_dir / "trajectory.png")
    _render_scene_graph(run_dir / "scene_graph.json", visual_dir / "scene_graph.png")
    _render_frame_gallery(run_dir / "artifacts" / "frames", visual_dir / "frames.png")


def _render_trajectory(trajectory_file: Path, output_file: Path) -> None:
    entries = [
        json.loads(line)
        for line in trajectory_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not entries:
        return

    positions = np.asarray([entry["pose"]["position"] for entry in entries], dtype=float)
    frames = [entry["frame_id"] for entry in entries]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.plot(positions[:, 0], positions[:, 2], marker="o", color="#1f77b4", linewidth=2)
    for index, frame_id in enumerate(frames):
        ax.annotate(
            frame_id,
            (positions[index, 0], positions[index, 2]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title("Trajectory (x/z plane)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def _render_scene_graph(graph_file: Path, output_file: Path) -> None:
    payload = json.loads(graph_file.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    if not nodes:
        return

    graph = nx.MultiDiGraph()
    for node in nodes:
        graph.add_node(node["node_id"], **node)
    for edge in edges:
        graph.add_edge(edge["source_id"], edge["target_id"], relation=edge["relation"])

    positions = {
        node["node_id"]: (float(node["position_3d"][0]), float(node["position_3d"][2]))
        for node in nodes
    }
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    nx.draw_networkx_nodes(graph, positions, node_size=900, node_color="#ffd166", ax=ax)
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node["node_id"]: node["label"] for node in nodes},
        font_size=8,
        ax=ax,
    )
    nx.draw_networkx_edges(graph, positions, arrows=True, arrowstyle="-|>", width=1.3, ax=ax)
    edge_labels = {
        (left, right): data["relation"]
        for left, right, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_title("Scene Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)


def _render_frame_gallery(frames_dir: Path, output_file: Path) -> None:
    if not frames_dir.is_dir():
        return
    image_files = sorted(
        [
            path
            for path in frames_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    if not image_files:
        return

    count = min(4, len(image_files))
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4), dpi=160)
    if count == 1:
        axes = [axes]
    for axis, image_file in zip(axes, image_files[:count], strict=True):
        axis.imshow(plt.imread(image_file))
        axis.set_title(image_file.stem)
        axis.axis("off")
    fig.suptitle("Sampled Frames")
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)