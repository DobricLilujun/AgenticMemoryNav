"""Run all evaluation scenarios through the independent MCP client."""

from __future__ import annotations

import asyncio
import sys



from agentic_memory_nav.mcp.client import FakeRobot
from agentic_memory_nav.mcp.eval.scenarios import ALL_SCENARIOS
from agentic_memory_nav.mcp.server.app import build_app


async def run() -> int:
    app, _engine, _hub, mcp = build_app()
    total = 0
    failures = 0
    for title, scenario in ALL_SCENARIOS:
        robot = FakeRobot.in_process("robot_0", mcp)
        await robot.connect()
        try:
            checks = await scenario(robot)
        finally:
            await robot.close()
        total += len(checks)
        failures += sum(1 for c in checks if not c.ok)
        status = "PASS" if all(c.ok for c in checks) else "FAIL"
        print(f"\n[{status}] {title}  ({sum(1 for c in checks if c.ok)}/{len(checks)})")
        for c in checks:
            if not c.ok:
                print(f"    X {c.name} {c.detail}")
    print(f"\n=== {total - failures}/{total} checks passed "
          f"({'ALL PASS' if failures == 0 else str(failures) + ' FAILED'}) ===")
    return 0 if failures == 0 else 1


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
