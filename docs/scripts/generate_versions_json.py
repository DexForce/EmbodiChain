#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Generate versions.json for multi-version documentation."""

import json
import os
from pathlib import Path


def generate_versions_json(build_dir: str) -> dict:
    """Generate versions.json from build directory structure.

    Args:
        build_dir: Path to the build/html directory

    Returns:
        Dictionary with versions metadata
    """
    build_path = Path(build_dir)

    # Find all version directories (main, v0.1.0, etc.)
    versions = []
    for item in build_path.iterdir():
        if not item.is_dir() or item.name.startswith("_") or item.name.startswith("."):
            continue

        # Check if it's a version directory (has index.html)
        index_file = item / "index.html"
        if index_file.exists():
            versions.append(
                {
                    "name": item.name,
                    "version": item.name,
                    "url": item.name + "/",
                }
            )

    # Sort versions (main last, releases in descending order - newest first)
    def version_key(v):
        name = v["name"]
        if name == "main":
            return (2, "")  # Put main last
        else:
            # Extract version number for sorting (v0.1.3 -> (0, 1, 3))
            parts = name.lstrip("v").split(".")
            try:
                major, minor, patch = map(int, parts[:3])
                # Use negative for descending sort (newest first)
                return (1, (-major, -minor, -patch))
            except (ValueError, IndexError):
                return (1, (0, 0, 0))

    versions.sort(key=version_key)

    return {
        "versions": versions,
        "latest": versions[0]["name"] if versions else "main",
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate versions.json for multi-version docs"
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="build/html",
        help="Path to build/html directory (default: build/html)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="build/html/versions.json",
        help="Output path for versions.json (default: build/html/versions.json)",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.exists():
        print(f"Error: Build directory '{build_dir}' does not exist.")
        return 1

    versions_data = generate_versions_json(args.build_dir)

    # Write versions.json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(versions_data, f, indent=2)

    print(f"Generated versions.json at: {output_path}")
    print(f"Found {len(versions_data['versions'])} versions")
    print(f"Latest: {versions_data['latest']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
