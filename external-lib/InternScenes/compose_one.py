#!/usr/bin/env python3
"""Compose one InternScenes scene (layout.json + assets) into a single GLB.

Run with the InternScenes-capable venv (needs numpy/trimesh/open3d), e.g.:
    .internScenes-venv/bin/python3 compose_one.py scannet/scene0003_00
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

INTERN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTERN_DIR.parent.parent  # external-lib/InternScenes -> external-lib -> project root
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_name", help="e.g. scannet/scene0003_00")
    parser.add_argument("--no-texture", action="store_true")
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--no-wall", action="store_true")
    parser.add_argument("--no-ceiling", action="store_true")
    args = parser.parse_args()

    # compose_scenes.py derives its data paths from BASE_DIR = Path(os.getcwd()).parent,
    # which must resolve to PROJECT_ROOT (via the data/ and tutorial/ symlinks).
    os.chdir(NOTEBOOK_DIR)
    for p in (str(INTERN_DIR), str(INTERN_DIR / "InternScenes")):
        if p not in sys.path:
            sys.path.insert(0, p)

    from InternScenes.InternScenes_Real2Sim.compose_scenes import SCENE_INFO_DIR, SCENE_SAVE_DIR, SceneComposer

    # compose_one_scene() reads StructureMesh from SCENE_SAVE_DIR, not SCENE_INFO_DIR.
    src_structure = Path(SCENE_INFO_DIR) / args.scene_name / "StructureMesh"
    dst_structure = Path(SCENE_SAVE_DIR) / args.scene_name / "StructureMesh"
    if src_structure.exists() and not dst_structure.exists():
        dst_structure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_structure, dst_structure)

    scene_composer = SceneComposer()
    scene_composer.compose_one_scene(
        args.scene_name,
        use_texture=not args.no_texture,
        add_floor=not args.no_floor,
        add_wall=not args.no_wall,
        add_ceiling=not args.no_ceiling,
    )

    glb_path = Path(SCENE_SAVE_DIR) / args.scene_name / "glb_scene.glb"
    print(f"GLB_PATH={glb_path}")
    print(f"GLB_EXISTS={glb_path.exists()}")
    return 0 if glb_path.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
