"""Unified sub-agent dispatch for Isaac Sim perception/mapping.

The Isaac Sim main process runs in Python 3.12 (Isaac's bundled Python) but the
LingBot-Map model must run in its own isolated Python 3.10 / CUDA venv. Rather than
letting each preview script hand-roll its own subprocess plumbing (the old
``LingBotWorkerPredictor`` lived inline in ``preview_isaacsim_navigation_wasd.py``),
the sub-agents are now dispatched through a single, uniform interface:

* :class:`LingBotMapAgentClient` -- the client for the LingBot map sub-agent. It spawns
  ``lingbot_map_agent.py`` in the isolated venv, waits for the ``ready`` signal, and
  decodes one mapping result per frame request. It *is* the ``LingBotPredictor``
  callable consumed by ``LingBotMapAdapter``.
* :class:`SubAgentDispatcher` -- the unified entry-point. ``start()`` is called once
  when the simulation begins and guarantees every requested sub-agent is up (it fails
  fast if the LingBot map agent cannot start or does not become ready). ``predict()``
  forwards a frame to the map agent; ``close()`` tears the agent down.

This is the "load it when the simulation starts and make sure the agent is open"
requirement: ``SubAgentDispatcher.start()`` is the single load/guarantee point.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import select
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
_AGENT_PATH = Path(__file__).resolve().parent / "agent.py"


class LingBotMapAgentClient:
    """Synchronous client for the LingBot map sub-agent (a ``LingBotPredictor``).

    Spawns ``lingbot_map_agent.py`` in the isolated LingBot venv and communicates over
    stdio pipes: one JSON request line in, one JSON response line out. It is directly
    usable as the ``predictor`` of ``LingBotMapAdapter`` (``client(frame) -> dict``).
    """

    def __init__(
        self,
        python_executable: str,
        checkpoint: str,
        image_size: int,
        keyframe_interval: int,
        *,
        agent_path: str | os.PathLike[str] = _AGENT_PATH,
        start_timeout_s: float = 180.0,
    ) -> None:
        print("Starting LingBot-Map agent; loading checkpoint...", flush=True)
        # The agent must run in a clean environment: Isaac Sim's CARB/ISAAC/LD_* vars
        # clash with the LingBot (torch/CUDA) runtime, so strip them for the child.
        agent_environment = os.environ.copy()
        for variable in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "CARB_APP_PATH",
            "ISAAC_PATH",
            "EXP_PATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
        ):
            agent_environment.pop(variable, None)
        self._process = subprocess.Popen(
            [
                python_executable,
                str(agent_path),
                "--checkpoint",
                checkpoint,
                "--image-size",
                str(image_size),
                "--keyframe-interval",
                str(keyframe_interval),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=agent_environment,
        )
        if self._process.stdout is None:
            raise RuntimeError("LingBot-Map agent stdout is unavailable")
        # Isaac bundled Python (3.12) TextIOWrapper.readline() has no timeout kwarg;
        # wait for the ready line with select() (no keyword arg) then read it.
        deadline = time.monotonic() + start_timeout_s
        ready_line = ""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([self._process.stdout], [], [], remaining)
            if ready:
                ready_line = self._process.stdout.readline()
                break
        if not ready_line:
            raise RuntimeError(
                f"LingBot-Map agent exited during startup: {self._process.poll()}"
            )
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"LingBot-Map agent startup failed: {ready}")
        print("LingBot-Map agent ready.", flush=True)

    def is_ready(self) -> bool:
        """Return True while the agent process is alive (used by the dispatcher)."""
        return self._process is not None and self._process.poll() is None

    def __call__(self, frame: "FrameObservation") -> dict[str, object]:  # noqa: F821
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("LingBot-Map agent pipes are unavailable")
        image_buffer = io.BytesIO()
        Image.fromarray(frame.rgb).save(image_buffer, format="JPEG", quality=85)
        robot_pose = frame.robot_pose
        if robot_pose is None:
            raise ValueError("LingBot-Map agent requires FrameObservation.robot_pose")
        request = {
            "frame_id": frame.frame_id,
            "image": base64_b64encode(zlib_compress(image_buffer.getvalue(), level=1)),
            "isaac_c2w": build_camera_to_world(frame),
            "robot_position": list(robot_pose.position),
            "robot_yaw": robot_pose.yaw,
        }
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"LingBot-Map agent exited with code {self._process.poll()}")
        response = json.loads(response_line)
        if not response.get("ok"):
            raise RuntimeError(
                f"LingBot-Map agent failed: {response.get('error', 'unknown error')}"
            )
        result = response["result"]
        return {
            "camera_pose": json.loads(result["camera_pose"]),
            "depth": self._decode_array(result["depth"]),
            "confidence": self._decode_array(result["confidence"]),
            "intrinsics": self._decode_array(result["intrinsics"]),
            "global_pointcloud": self._decode_array(result["global_pointcloud"]),
        }

    @staticmethod
    def _decode_array(value: str) -> np.ndarray:
        payload = zlib_decompress(base64_b64decode(value))
        return np.load(io.BytesIO(payload), allow_pickle=False)

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.terminate()
        self._process.wait(timeout=10)
        self._process = None


# ---------------------------------------------------------------------------
# Small stdio/serialization helpers kept local so the client has no hard
# dependency on the Isaac-only preview modules.
# ---------------------------------------------------------------------------
def base64_b64encode(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def base64_b64decode(value: str) -> bytes:
    import base64

    return base64.b64decode(value)


def zlib_compress(data: bytes, level: int = 1) -> bytes:
    import zlib

    return zlib.compress(data, level=level)


def zlib_decompress(data: bytes) -> bytes:
    import zlib

    return zlib.decompress(data)


def build_camera_to_world(frame: "FrameObservation") -> list[float]:  # noqa: F821
    """Build the head optical-camera camera-to-world from robot pose + head extrinsics.

    Kept here (rather than in the preview script) so the sub-agent client is a single,
    uniform dispatch path usable by any Isaac main loop.
    """
    from agentic_memory_nav.agent.execution.isaacsim_adapter import _GO2_CAMERA_OFFSET_M

    robot_pose = frame.robot_pose
    if robot_pose is None:
        raise ValueError("Head-camera mapping requires FrameObservation.robot_pose")
    x, y, z = robot_pose.position
    cos_yaw, sin_yaw = np.cos(robot_pose.yaw), np.sin(robot_pose.yaw)
    forward, left, up = _GO2_CAMERA_OFFSET_M
    camera_position = np.array(
        [x + cos_yaw * forward - sin_yaw * left, y + sin_yaw * forward + cos_yaw * left, z + up],
        dtype=np.float32,
    )
    # Isaac optical axes map to base_link (-Y, +Z, -X).
    optical_to_base = np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    base_to_world = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = base_to_world @ optical_to_base
    transform[:3, 3] = camera_position
    return transform.tolist()


class SubAgentDispatcher:
    """Unified loader for Isaac Sim perception/mapping sub-agents.

    ``start()`` is the single "load the sub-agents when the simulation starts" hook. It
    launches each requested sub-agent and *guarantees* it is open: if the LingBot map
    agent cannot start or never becomes ready, ``start()`` raises so the simulation
    aborts cleanly instead of silently running without a mapper.
    """

    def __init__(
        self,
        *,
        lingbot_python: str,
        lingbot_checkpoint: str,
        lingbot_image_size: int,
        lingbot_keyframe_interval: int,
        enable_lingbot_map: bool = True,
    ) -> None:
        self._enable_lingbot_map = enable_lingbot_map
        self.lingbot_client: LingBotMapAgentClient | None = None
        self._lingbot_kwargs = dict(
            python_executable=lingbot_python,
            checkpoint=lingbot_checkpoint,
            image_size=lingbot_image_size,
            keyframe_interval=lingbot_keyframe_interval,
        )

    def start(self) -> None:
        """Launch every enabled sub-agent and guarantee each one is ready."""
        if self._enable_lingbot_map:
            self.lingbot_client = LingBotMapAgentClient(**self._lingbot_kwargs)
            if not self.lingbot_client.is_ready():
                raise RuntimeError("LingBot-Map agent is not ready after start")

    def map_predictor(self) -> "LingBotPredictor":  # type: ignore[name-defined]
        """Return the map-agent client as the ``LingBotPredictor`` for the adapter."""
        if self.lingbot_client is None:
            raise RuntimeError("LingBot-Map agent was not started")
        return self.lingbot_client

    def predict(self, frame: "FrameObservation") -> dict[str, object]:  # noqa: F821
        if self.lingbot_client is None:
            raise RuntimeError("LingBot-Map agent was not started")
        return self.lingbot_client(frame)

    def close(self) -> None:
        if self.lingbot_client is not None:
            self.lingbot_client.close()
            self.lingbot_client = None