import argparse
import asyncio
from isaacsim import SimulationApp
kit = SimulationApp()

import os
from pathlib import Path
from pxr import Usd, UsdGeom, Gf
import numpy as np

from real2sim_utils.usd_tools import set_usd_prim_orientation


# convert GLB to USD using omni.kit.asset_converter
async def convert(in_file, out_file, load_materials=False):
    import omni.kit.asset_converter
    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials
    converter_context.use_meter_as_world_unit = True
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        in_file, out_file, progress_callback, converter_context
    )
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success

if __name__ == "__main__":
    import omni
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension("omni.kit.asset_converter")
    parser = argparse.ArgumentParser("Convert GLB assets to USD")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single model file to convert.",
    )
    parser.add_argument(
        "--load-materials",
        action="store_true",
        help="If specified, materials will be loaded from meshes",
    )
    parser.add_argument(
        "--dist-folder",
        type=str,
        default="usd",
        help="If specified, converted assets will be placed in this folder.",
    )
    args, unknown_args = parser.parse_known_args()
    if args.file:
        input_path = args.file
        name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = str(Path(args.dist_folder) / f"{name}.usd")
        Path(args.dist_folder).mkdir(parents=True, exist_ok=True)
        
        print(f"Converting single file: {input_path} → {output_path}")
        status = asyncio.get_event_loop().run_until_complete(
            convert(input_path, output_path, args.load_materials)
        )
        if status:
            print(f"Successfully converted: {output_path}")
            # set z up axis
            set_usd_prim_orientation(output_path)
        else:
            print(f"Failed to convert: {input_path}")
    else:
        print("No input specified. Use --file.")
    kit.close()