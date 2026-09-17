# MCP Infinite Canvas System

A real-time 2D robot **canvas** driven through the Model Context Protocol. A
*fake robot* is an independent MCP **client**; it reaches the server **only**
through MCP tools (never the server's internal state). Each tool call updates the
shared simulation engine, records a trajectory or event, and is pushed to every
connected viewer over a WebSocket. A human may also drive the same canvas through
small HTTP control endpoints (HMI).

All coordinates, distances and sizes are expressed in an abstract **unit length**
(no meter / cm / mm). `theta` is in degrees, counter-clockwise positive,
`theta = 0` faces `+x`. Initial state is `x = 0, y = 0, theta = 0`.

## Architecture

```
                 ┌───────────────────────────────────────────────┐
                 │              SimulationEngine (core)           │
 fake robot  ──MCP tools──►   pose / heading / trajectory        │
 (client)          ▲               obstacles / events / version   │
                 │   ▲  ─────────────────────────────────────────  │
   viewer  ──WS /ws/canvas/main──►  EventHub (fan-out)            │
   (browser)   │   ▲                                  │
 HMI  ──HTTP POST /api/*──────────────────────────────┘
```

* **One process, one event loop.** The FastMCP HTTP app and the Starlette
  `WebSocketRoute` share a single `SimulationEngine`, so the MCP client, the
  WebSocket and the HMI all observe identical live state.
* **Dependency-free core.** `core/` uses only stdlib `dataclasses` + `math`; it
  is fully unit-testable without any MCP/web stack.

### Packages

| Path | Role |
| --- | --- |
| `core/` | `actions`, `geometry`, `models`, `collision`, `engine` — the state machine |
| `server/` | `app.py` (ASGI build + tools + routes), `hub.py` (WebSocket fan-out), `main.py` (uvicorn entry) |
| `client/` | `fake_robot.py` — an independent MCP client (`in_process` / `from_url`) |
| `eval/` | the five specification scenarios + a runner |
| `web/` | a self-contained canvas viewer (`index.html`) |

## Action mapping

| id | name | effect |
| --- | --- | --- |
| 0 | `turn_left` | rotate `+15°` (heading only) |
| 1 | `turn_right` | rotate `−15°` (heading only) |
| 2 | `move_forward` | advance `move_step`, record trajectory |
| 3 | `turn_left_big` | rotate `+90°` (heading only) |
| 4 | `turn_right_big` | rotate `−90°` (heading only) |
| 5 | `move_backward` | advance `−move_step`, record trajectory |
| 6 | `stop` | no 2-D change (event only) |
| 7 | `look_up` | no 2-D change (event only) |
| 8 | `look_down` | no 2-D change (event only) |

* **Move actions** generate a trajectory segment (black = success, red = blocked by
  an obstacle), update the pose with a green marker, and trigger a WebSocket push.
  A blocked move records a red segment but leaves the pose unchanged. Older
  segments render dashed; the most recent renders solid.
* **Rotation actions** change the heading only — no pose, no trajectory.
* `stop` / `look_up` / `look_down` produce an event only (no 2-D change).

## Manual validation console (`/manual`)

A second viewer page for human-in-the-loop validation. The operator drives the
robot with the keyboard and provides ground-truth feedback for every move:

| key | action |
| --- | --- |
| `W` / `S` | `move_forward` / `move_backward` |
| `A` / `D` | `turn_left` / `turn_right` (15°) |
| `Q` / `E` | `turn_left_big` / `turn_right_big` (90°) |

After every move the console asks *"Was this action blocked by an obstacle?"*
(answer with `Y`/`N` or the buttons):

* **No** → the move executes normally and the path is rendered as a success.
* **Yes** → a hidden virtual blocker forces the engine to record the failed
  (red) segment, the blocker is removed again, and a visible **meta-obstacle**
  square (dashed outline, `confirmed=False`) is reported one step in front of
  the robot.

Two view modes:

* **Global** — auto-fits the whole canvas.
* **Follow** — zooms in on the robot with a configurable radius (default 15
  units) for a focused view while driving.

## WebSocket events

`/ws/canvas/main` pushes: `robot_updated`, `trajectory_added`, `obstacle_added`,
`obstacle_updated`, `obstacle_removed`, `simulation_reset`. A new connection
receives a full snapshot first; the viewer also polls `/api/snapshot` as a
fallback. `get_canvas` returns the complete canvas at any time.

## Run

```bash
# server (MCP + WebSocket + viewer + HMI) on http://127.0.0.1:8710
.venv/bin/python -m agentic_memory_nav.mcp.server.main --port 8710

# open the viewer
open http://127.0.0.1:8710/

# open the manual validation console (WASD drive + ground-truth feedback)
open http://127.0.0.1:8710/manual

# network MCP client (fake robot)
.venv/bin/python -c "import asyncio; \
  from agentic_memory_nav.mcp.client import FakeRobot; \
  asyncio.run(FakeRobot.from_url('robot_0','http://127.0.0.1:8710/mcp').connect())"
```

## Verify

```bash
# five specification scenarios through the in-process MCP client
.venv/bin/python -m agentic_memory_nav.mcp.eval.runner

# unit + integration tests
.venv/bin/python -m pytest tests/mcp/
```

## Scenarios

1. **Three forward moves** → position `(0.75, 0)`, `trajectory_count == 3`.
2. **Big left turn, then forward** → `theta == 90`, position `(0, 0.25)`, one trajectory.
3. **Forward then backward** → net position `(0, 0)`, `trajectory_count == 2`.
4. **Every action callable** — the 9 actions have the right side effects.
5. **Point + line obstacles** — added, perceived, and a move into one is blocked.