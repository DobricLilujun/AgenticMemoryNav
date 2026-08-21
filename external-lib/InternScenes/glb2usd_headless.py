#!/usr/bin/env python3
"""Convert a composed InternScenes GLB into USD for Isaac Sim.

The upstream glb2usd.py starts SimulationApp() windowed, which fails to resolve
extensions on a headless-only Isaac Sim install.

Run with Isaac Sim's bundled Python:
    ~/isaacsim/python.sh glb2usd_headless.py --file scene.glb --out scene.usd
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--no-materials", action="store_true")
args = parser.parse_args()

kit = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")

import omni.kit.asset_converter  # noqa: E402


async def convert(in_file: str, out_file: str, load_materials: bool) -> bool:
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = not load_materials
    context.use_meter_as_world_unit = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        in_file, out_file, lambda progress, total: None, context
    )
    while not await task.wait_until_finished():
        await asyncio.sleep(0.1)
    return True


out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Converting {args.file} -> {out_path}")
asyncio.get_event_loop().run_until_complete(
    convert(args.file, str(out_path), not args.no_materials)
)
print(f"CONVERTED {out_path} exists={out_path.exists()}")
kit.close()
