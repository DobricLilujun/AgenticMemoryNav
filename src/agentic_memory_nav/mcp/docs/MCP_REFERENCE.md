# MCP Infinite Canvas — API Reference

A real-time 2D robot canvas driven through the Model Context Protocol (MCP). The
server owns a single `SimulationEngine`; robots interact with it exclusively
through MCP tools, and browsers observe the same state over WebSocket / HTTP.

## Conventions

| Item | Value |
| --- | --- |
| Coordinate unit | abstract **unit length** (no m/cm/mm) |
| Heading `theta` | degrees, counter-clockwise positive, `0` faces `+x` |
| Initial pose | `x=0, y=0, theta=0` |
| Move step | `1.0` unit per `move_forward` / `move_backward` |
| Small turn | `±15°` (`turn_left` / `turn_right`) |
| Big turn | `±90°` (`turn_left_big` / `turn_right_big`) |
| Blocked move | red trajectory segment, robot pose unchanged |

## Running the server

```bash
python -m agentic_memory_nav.mcp.server.main --host 127.0.0.1 --port 8093
```

Endpoints:

| Route | Purpose |
| --- | --- |
| `GET /` | Live canvas viewer (WASD/QE drive) |
| `GET /manual` | Manual validation console (Shift blocked-mode toggle) |
| `GET /api/snapshot` | Full canvas JSON snapshot |
| `POST /api/execute_action` | Execute one action |
| `POST /api/add_obstacle` | Add an obstacle |
| `WS /ws/canvas/main` | Real-time event stream |

## MCP tools

### `create_robot`

Create a robot or reset it to a start pose.

```json
{
  "robot_id": "robot_0",
  "start_pose": {"x": 0.0, "y": 0.0, "theta": 0.0}
}
```

Returns `{"robot_id", "pose", "canvas_version"}`.

### `execute_action`

Execute one action by name or id. Exactly one of `action` / `action_id` is
required.

```json
{"robot_id": "robot_0", "action": "move_forward"}
```

Returns an `ActionResult` dict:

| Field | Meaning |
| --- | --- |
| `success` | `false` when a move is blocked by an obstacle |
| `trajectory_added` | `true` for move actions (success or blocked) |
| `pose_before` / `pose_after` | `{x, y, theta}` |
| `blocked_by` | obstacle ids that blocked the move |
| `canvas_version` | monotonically increasing version |

### `list_actions`

Returns the canonical id → name mapping:

| id | name | effect |
| --- | --- | --- |
| 0 | `turn_left` | `+15°` heading only |
| 1 | `turn_right` | `−15°` heading only |
| 2 | `move_forward` | `+1.0` unit, trajectory segment |
| 3 | `turn_left_big` | `+90°` heading only |
| 4 | `turn_right_big` | `−90°` heading only |
| 5 | `move_backward` | `−1.0` unit, trajectory segment |
| 6–8 | `stop`, `look_up`, `look_down` | event only |

### `get_canvas`

Full snapshot:

```json
{
  "canvas_id": "main",
  "frame": "world",
  "coordinate_unit": "unit_length",
  "version": 42,
  "robots": [{"robot_id": "robot_0", "position": [1.0, 0.0], "theta": 0.0, "active": true}],
  "trajectory": [{"start": [0, 0], "end": [1, 0], "action": "move_forward", "success": true, ...}],
  "trajectory_count": 1,
  "obstacles": [...],
  "events": [...]
}
```

Trajectory entries: the latest segment per robot is rendered solid, older ones
dashed; `success=false` renders red.

### `add_obstacle`

Add a point or line obstacle. Line segments sharing a
`properties.group_id` are rendered as one polygon (used for square
meta-obstacles).

```json
{
  "robot_id": "robot_0",
  "obstacle": {
    "type": "line",
    "geometry": {"start": [1, 1], "end": [2, 1]},
    "width": 0.02,
    "label": "",
    "properties": {"group_id": "obs_1", "confirmed": true}
  }
}
```

`properties` rendering flags:

| Flag | Effect |
| --- | --- |
| `confirmed: false` | dashed translucent outline (perceived, not verified) |
| `visible: false` | blocks movement but is never rendered (hidden world) |

### `update_obstacle` / `remove_obstacle`

Patch or delete by `obstacle_id`.

### `reset_simulation`

```json
{"robot_id": "robot_0", "clear_events": true}
```

Resets robot poses, clears trajectory and obstacles; `clear_events` also clears
the event history. This is what the viewer's **Reset episode** button calls, and
what an agent should call immediately after connecting so each run starts from a
clean canvas:

```python
await robot.connect()
await robot.reset(clear_events=True)          # Reset episode
await robot.create({"x": 0, "y": 0, "theta": 0})  # place robot at origin
```

## WebSocket events

`/ws/canvas/main` pushes one JSON message per engine event:

```json
{"type": "event", "event": {"event": "trajectory_added", "canvas_version": 12, "data": {...}}}
```

Event types: `robot_updated`, `trajectory_added`, `obstacle_added`,
`obstacle_updated`, `obstacle_removed`, `simulation_reset`. A new connection
receives a full `{"type": "snapshot", "canvas": ...}` first; viewers also poll
`/api/snapshot` as a fallback.

## Python client

```python
from agentic_memory_nav.mcp.server.app import build_app
from agentic_memory_nav.mcp.client import FakeRobot

app, engine, hub, mcp = build_app()
robot = FakeRobot.in_process("robot_0", mcp)
await robot.connect()
await robot.create({"x": 0, "y": 0, "theta": 0})
await robot.execute("move_forward")
canvas = await robot.get_canvas()
```

Or over the network: `FakeRobot.from_url("robot_0", "http://127.0.0.1:8093/mcp")`.

## Static rendering

`agentic_memory_nav.mcp.rendering` converts a `get_canvas()` snapshot into a PNG
(headless matplotlib):

```python
from agentic_memory_nav.mcp.rendering import render_canvas, render_frames
render_canvas(canvas, "out.png", title="Demo")
render_frames(frames, "frames/", prefix="step")
```

Rendering semantics: green square = start, blue dots = successful steps, red
line + cross = blocked step, cyan star/arrow = robot, red squares = obstacles
(dashed when `confirmed=false`).
