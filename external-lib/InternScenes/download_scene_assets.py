#!/usr/bin/env python3
"""Resolve the asset files a single InternScenes layout.json needs.

The full dataset is ~2.46 TB, so scenes are provisioned per-object instead of
mirroring whole asset libraries.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = "InternRobotics/InternScenes"
# These libraries are published only as multi-gigabyte archives, so their objects
# cannot be fetched individually.
ARCHIVE_LIBRARIES = {
    "partnet_mobility": ["asset_library/partnet_mobility/partnet_mobility.tar.gz"],
    "objaverse": [f"asset_library/objaverse/objaverse.tar.gz.{index:02d}" for index in range(10)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", type=Path, help="path to a scene's layout.json")
    parser.add_argument("--dest", type=Path, default=Path("data"))
    parser.add_argument(
        "--with-objaverse",
        action="store_true",
        help="also fetch the 101 GB objaverse archive set",
    )
    parser.add_argument(
        "--no-archives",
        action="store_true",
        help="skip the 0.72 GB partnet_mobility archive and fetch only per-object GLBs",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    objects = json.loads(args.layout.read_text())
    uids = sorted({entry["model_uid"] for entry in objects if "model_uid" in entry})

    per_object: list[str] = []
    archives: list[str] = []
    skipped: collections.Counter[str] = collections.Counter()
    for uid in uids:
        library = uid.split("/")[0]
        if library in ARCHIVE_LIBRARIES:
            wanted = args.with_objaverse if library == "objaverse" else not args.no_archives
            if not wanted:
                skipped[library] += 1
                continue
            for path in ARCHIVE_LIBRARIES[library]:
                if path not in archives:
                    archives.append(path)
            continue
        per_object.append(f"asset_library/{uid}.glb")

    patterns = per_object + archives
    print(f"scene: {args.layout}")
    print(f"objects: {len(objects)}  unique assets: {len(uids)}")
    print(f"per-object GLBs: {len(per_object)}  archives: {len(archives)}")
    for library, count in skipped.items():
        flag = "--with-objaverse" if library == "objaverse" else "drop --no-archives"
        print(f"skipped {count} '{library}' assets (archive-only; {flag})")

    if args.dry_run:
        for pattern in patterns[:10]:
            print("  ", pattern)
        return 0

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(args.dest),
    )
    print(f"downloaded into {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
