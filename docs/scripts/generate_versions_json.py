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
"""Generate versions.json and root index.html for the docs version selector."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_version(tag: str) -> tuple[int, int, int]:
    """Parse a version tag like 'v1.2.3' into a tuple (1, 2, 3)."""
    match = re.match(r"^v(\d+)\.(\d+)\.(\d+)$", tag)
    if not match:
        return (0, 0, 0)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate versions.json and root index.html for multi-version docs"
    )
    parser.add_argument(
        "--build-dir",
        default="build/html",
        help="Path to build/html directory (default: build/html)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for versions.json (default: <build-dir>/versions.json)",
    )
    parser.add_argument(
        "--latest",
        default=None,
        help="Name of the latest stable version (default: auto-detected from tags, falls back to main)",
    )
    args = parser.parse_args()

    html_dir = Path(args.build_dir)
    output = Path(args.output) if args.output else html_dir / "versions.json"

    if not html_dir.exists():
        print(f"Error: Build directory '{html_dir}' does not exist.")
        raise SystemExit(1)

    versions: list[dict[str, str]] = []

    # Collect tag versions (vX.Y.Z directories), sorted newest-first
    tag_dirs = sorted(
        [d for d in html_dir.glob("v*") if d.is_dir()],
        key=lambda d: parse_version(d.name),
        reverse=True,
    )
    for d in tag_dirs:
        name = d.name
        versions.append({"name": name, "url": f"./{name}/index.html", "type": "tag"})

    # Collect main (dev branch)
    if (html_dir / "main").is_dir():
        versions.append({"name": "main", "url": "./main/index.html", "type": "branch"})

    # Determine latest: explicit arg > newest tag > main
    if args.latest:
        latest = args.latest
    elif versions:
        tag_names = [v["name"] for v in versions if v["type"] == "tag"]
        latest = tag_names[0] if tag_names else "main"
    else:
        latest = "main"

    manifest = {
        "latest": latest,
        "versions": versions,
    }

    # Write versions.json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(f"Generated {output} with {len(versions)} versions (latest: {latest})")

    # Write root index.html redirect
    index_path = html_dir / "index.html"
    index_content = (
        "<!DOCTYPE html>\n"
        "<html><head>\n"
        f"  <title>EmbodiChain Docs</title>\n"
        f'  <meta http-equiv="refresh" content="0; url=./{latest}/index.html">\n'
        "</head></html>\n"
    )
    index_path.write_text(index_content)
    print(f"Generated {index_path} (redirects to ./{latest}/index.html)")


if __name__ == "__main__":
    main()
