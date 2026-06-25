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
"""Validate a generated multi-version documentation site before deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

__all__ = [
    "load_local_manifest",
    "load_live_manifest",
    "validate_docs_site",
]


def load_local_manifest(build_dir: Path) -> dict[str, Any]:
    """Load the generated ``versions.json`` manifest."""
    manifest_path = build_dir / "versions.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing docs manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_live_manifest(site_base_url: str) -> dict[str, Any] | None:
    """Load the currently published ``versions.json`` manifest, if available."""
    manifest_url = f"{site_base_url.rstrip('/')}/versions.json"
    try:
        with urlopen(manifest_url, timeout=30) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"No live docs manifest at {manifest_url}: {exc}", file=sys.stderr)
        return None


def _manifest_names(manifest: dict[str, Any]) -> set[str]:
    return {
        entry["name"]
        for entry in manifest.get("versions", [])
        if isinstance(entry, dict) and entry.get("name")
    }


def _live_tag_names(manifest: dict[str, Any]) -> set[str]:
    return {
        entry["name"]
        for entry in manifest.get("versions", [])
        if (
            isinstance(entry, dict) and entry.get("name") and entry.get("type") == "tag"
        )
    }


def _missing_version_dirs(build_dir: Path, versions: set[str]) -> list[str]:
    return sorted(
        version
        for version in versions
        if not (build_dir / version / "index.html").is_file()
    )


def validate_docs_site(
    build_dir: Path,
    *,
    required_versions: set[str] | None = None,
    preserve_live_tags: bool = False,
    site_base_url: str | None = None,
) -> None:
    """Validate generated docs before publishing.

    Args:
        build_dir: Root of the generated site.
        required_versions: Versions that must be present in the site.
        preserve_live_tags: Whether all currently published tag versions must
            still be present. Use this for ``main`` builds to prevent release
            docs from being overwritten by a branch-only artifact.
        site_base_url: Published site base URL used with ``preserve_live_tags``.

    Raises:
        FileNotFoundError: If required generated files are missing.
        RuntimeError: If the manifest omits required versions.
    """
    build_dir = build_dir.resolve()
    manifest = load_local_manifest(build_dir)
    manifest_names = _manifest_names(manifest)
    required = set(required_versions or set())

    if preserve_live_tags:
        if not site_base_url:
            raise ValueError("site_base_url is required with preserve_live_tags")
        live_manifest = load_live_manifest(site_base_url)
        if live_manifest:
            required.update(_live_tag_names(live_manifest))

    missing_from_manifest = sorted(required - manifest_names)
    if missing_from_manifest:
        raise RuntimeError(
            "Docs manifest is missing required versions: "
            + ", ".join(missing_from_manifest)
        )

    missing_dirs = _missing_version_dirs(build_dir, required)
    if missing_dirs:
        raise FileNotFoundError(
            "Docs site is missing required version directories: "
            + ", ".join(missing_dirs)
        )

    manifest_dir_gaps = _missing_version_dirs(build_dir, manifest_names)
    if manifest_dir_gaps:
        raise FileNotFoundError(
            "Docs manifest references missing version directories: "
            + ", ".join(manifest_dir_gaps)
        )

    print(
        "Validated docs site versions: "
        + ", ".join(sorted(manifest_names, reverse=True))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate generated multi-version docs before deployment"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/html"),
        help="Generated docs site root (default: build/html)",
    )
    parser.add_argument(
        "--require-version",
        action="append",
        default=[],
        help="Version that must be present in versions.json and as a directory",
    )
    parser.add_argument(
        "--preserve-live-tags",
        action="store_true",
        help="Require all tag versions from the live site to be preserved",
    )
    parser.add_argument(
        "--site-base-url",
        default=None,
        help="Published site base URL, e.g. https://org.github.io/EmbodiChain",
    )
    args = parser.parse_args()

    validate_docs_site(
        args.build_dir,
        required_versions=set(args.require_version),
        preserve_live_tags=args.preserve_live_tags,
        site_base_url=args.site_base_url,
    )


if __name__ == "__main__":
    main()
