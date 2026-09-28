"""Demo agent: drive the MCP infinite-canvas robot with a local VLM to draw a letter.

The agent tests how well a local OpenAI-compatible VLM understands the MCP
canvas: at every step it renders the current canvas, sends the image plus the
recent action history to the VLM, and asks for the next action as one JSON
object. The target letter is configurable (default ``L``); the VLM may answer
``stop`` at any time to end the episode.

Optionally, a reference image of the target letter can be supplied with
``--reference-image``. The VLM then receives both the reference image and the
current canvas, and is asked to imitate the reference shape.

Usage:

    # start the MCP server first
    python -m agentic_memory_nav.mcp.server.main --port 8093

    # draw the letter E from a reference image
    python -m agentic_memory_nav.agent.mcp_love_agent \
        --letter E --reference-image ./letter_e.png --max-steps 40
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib import request

from agentic_memory_nav.mcp.client import FakeRobot
from agentic_memory_nav.mcp.rendering import render_canvas
from agentic_memory_nav.mcp.server.app import build_app

# Valid discrete actions the VLM may choose from.
VALID_ACTIONS = (
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "turn_left_big",
    "turn_right_big",
    "stop",
)

SYSTEM_PROMPT = (
    "You control a 2D robot on an infinite canvas. The robot draws a trajectory "
    "as it moves; your goal is to draw the letter {letter} with the trajectory.\n\n"
    "In the canvas image, the bright arrow marker (with a star) is YOUR current "
    "position and heading — the arrow points in the direction you are facing. "
    "All new motion starts from this marker.\n\n"
    "When a reference image of the letter is provided, imitate its shape as "
    "closely as possible.\n\n"
    "You receive: (1) the current rendered canvas, (2) the most recent actions "
    "(oldest to newest).\n"
    "Reply with ONLY a JSON object, no markdown, no explanation:\n"
    '{{"action": "<one of: move_forward, move_backward, turn_left, turn_right, '
    'turn_left_big, turn_right_big, stop>"}}\n\n'
    "Hints: move_forward moves 1.0 unit along the current heading; turn_left / "
    "turn_right rotate 15 degrees; turn_left_big / turn_right_big rotate 90 "
    "degrees. Choose stop when the letter looks complete. The trajectory shows "
    "what has been drawn so far."
)

# How many recent actions are shown to the VLM each step.
HISTORY = 8


def _image_data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _ask_vlm(
    image_path: str,
    history: list[str],
    *,
    letter: str,
    model_id: str,
    base_url: str,
    api_key: str,
    timeout: float,
    reference_image: str | None,
) -> str:
    """Send the canvas image + recent action history to the VLM and parse the reply."""

    history_text = ", ".join(history) if history else "start"
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Recent actions (oldest to newest): {history_text}"},
    ]
    if reference_image:
        content.append({"type": "text", "text": f"Reference image of the letter {letter}:"})
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(reference_image)}})
    content.append({"type": "text", "text": "Current canvas:"})
    content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_path)}})

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(letter=letter)},
            {"role": "user", "content": content},
        ],
        "temperature": 0.3,
        "max_tokens": 10240,
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"VLM request failed: {exc}")
        if hasattr(exc, "read"):
            try:
                print(exc.read().decode("utf-8"))
            except Exception:
                pass
        raise

    message = result["choices"][0].get("message") or {}
    # With enable_thinking the answer may land in `content` or `reasoning`;
    # collect every candidate string before extracting the JSON action.
    candidates: list[str] = []
    for key in ("content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)
        elif isinstance(value, list):
            candidates.append(
                "".join(
                    str(part.get("text", ""))
                    for part in value
                    if isinstance(part, dict)
                )
            )

    action = ""
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                action = json.loads(candidate[start : end + 1]).get("action", "")
            except json.JSONDecodeError:
                action = ""
        if action:
            break
    return action if action in VALID_ACTIONS else "move_forward"


async def run_agent(
    *,
    letter: str,
    max_steps: int,
    model_id: str,
    base_url: str,
    api_key: str,
    timeout: float,
    url: str | None,
    workdir: Path,
    reference_image: str | None,
) -> None:
    """Drive the robot until the VLM says stop or the step budget is spent."""

    if url:
        robot = FakeRobot.from_url("robot_0", url)
    else:
        _app, _engine, _hub, mcp = build_app()
        robot = FakeRobot.in_process("robot_0", mcp)

    await robot.connect()
    # Same as the viewer's "Reset episode" button: clear trajectory, obstacles
    # and events, then place the robot at the origin so each run starts clean.
    await robot.reset(clear_events=True)
    await robot.create({"x": 0.0, "y": 0.0, "theta": 0.0})

    prev_action = "start"
    history: list[str] = []
    frame_path = workdir / "current.png"
    for step in range(max_steps):
        canvas = await robot.get_canvas()
        render_canvas(canvas, str(frame_path), title=f"step {step}")
        action = _ask_vlm(
            str(frame_path),
            history[-HISTORY:],
            letter=letter,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            reference_image=reference_image,
        )
        print(f"step {step:3d}: history={','.join(history[-HISTORY:]) or 'start'} -> action={action}")
        result = await robot.execute(action)
        if action == "stop":
            break
        history.append(action)
        prev_action = action
        if not result.get("success", True):
            print("  (blocked)")

    canvas = await robot.get_canvas()
    final_path = workdir / f"letter_{letter.lower()}_final.png"
    render_canvas(canvas, str(final_path), title=f"Letter {letter} result")
    print(f"Final canvas: {final_path}")
    await robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-driven MCP letter-drawing agent")
    parser.add_argument("--letter", default="E", help="Target letter to draw")
    parser.add_argument("--reference-image", default="outputs/E.png",
                        help="Optional reference image of the target letter")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--model-id", default="Inferact/Qwen3.8-27B-NVFP4")
    parser.add_argument("--base-url", default="http://10.6.32.16:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--timeout", type=float, default=500.0)
    parser.add_argument("--url", default=None, help="MCP server URL (omit for in-process)")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "mcp_love_agent",
        help="Directory for intermediate canvas renders",
    )
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    asyncio.run(
        run_agent(
            letter=args.letter,
            max_steps=args.max_steps,
            model_id=args.model_id,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            url=args.url,
            workdir=args.workdir,
            reference_image=args.reference_image,
        )
    )


if __name__ == "__main__":
    main()
