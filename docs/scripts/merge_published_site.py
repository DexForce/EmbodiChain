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
"""Merge version directories from the live docs site into a local build tree.

CI restores an Actions cache and rebuilds only one version (``main`` or a tag).
Tag-scoped cache entries are not visible on ``main`` pushes, so the cache alone
cannot hold all versions. This script fills *missing* version directories from
the currently published GitHub Pages site (or a local directory in tests).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

_INVALID_ARTIFACT_CHARS = frozenset('"<>:|*?\r\n')

__all__ = [
    "flatten_nested_version_dirs",
    "load_versions_manifest",
    "merge_published_site",
    "normalize_artifact_paths",
]


def load_versions_manifest(
    *,
    site_base_url: str | None = None,
    published_root: Path | None = None,
) -> dict[str, Any] | None:
    """Load ``versions.json`` from a local tree or the live site URL."""
    if published_root is not None:
        manifest_path = published_root / "versions.json"
        if not manifest_path.is_file():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    if not site_base_url:
        return None

    manifest_url = f"{site_base_url.rstrip('/')}/versions.json"
    try:
        with urlopen(manifest_url, timeout=30) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"No published manifest at {manifest_url}: {exc}", file=sys.stderr)
        return None


def _copy_local_version(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    normalize_artifact_paths(dest)


def _safe_artifact_name(name: str) -> str:
    """Return a filesystem-agnostic artifact name for one path component."""
    name = name.split("?", 1)[0].split("#", 1)[0] or "download"
    return "".join("_" if char in _INVALID_ARTIFACT_CHARS else char for char in name)


def normalize_artifact_paths(root: Path) -> list[tuple[Path, Path | None]]:
    """Normalize paths that GitHub artifact upload rejects.

    Recursive ``wget`` mirrors URLs with query strings as literal filenames
    such as ``clipboard.min.js?v=a7894cd8``. Browsers resolve that URL against
    the real file ``clipboard.min.js``, so the mirrored query-string copy is
    redundant and invalid for Actions artifacts.

    Args:
        root: Directory tree to normalize.

    Returns:
        ``(old_path, new_path)`` pairs. ``new_path`` is ``None`` when the
        invalid duplicate was removed because the safe target already existed.
    """
    changes: list[tuple[Path, Path | None]] = []
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        safe_name = _safe_artifact_name(path.name)
        if safe_name == path.name:
            continue

        target = path.with_name(safe_name)
        if target.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            changes.append((path, None))
        else:
            path.rename(target)
            changes.append((path, target))

    return changes


def _cleanup_empty_dirs(root: Path) -> None:
    for directory in sorted(
        root.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def flatten_nested_version_dirs(build_dir: Path) -> list[tuple[Path, Path | None]]:
    """Move nested downloaded version dirs to the build root.

    ``wget -nH`` removes the host component but keeps URL path components. For
    project Pages URLs this can produce ``build/html/EmbodiChain/v0.2.2``. The
    version manifest generator only scans top-level ``v*`` directories, so keep
    release versions directly under ``build/html``.

    Args:
        build_dir: Sphinx output root (``docs/build/html``).

    Returns:
        ``(old_path, new_path)`` pairs. ``new_path`` is ``None`` when a nested
        duplicate was removed because the top-level version already exists.
    """
    build_dir = build_dir.resolve()
    changes: list[tuple[Path, Path | None]] = []
    candidates = [
        candidate
        for candidate in build_dir.rglob("v*")
        if candidate.is_dir()
        and candidate.parent != build_dir
        and (candidate / "index.html").is_file()
    ]
    candidates.sort(key=lambda candidate: len(candidate.parts))

    for candidate in candidates:
        target = build_dir / candidate.name
        if target.exists():
            shutil.rmtree(candidate)
            changes.append((candidate, None))
        else:
            candidate.rename(target)
            changes.append((candidate, target))

    _cleanup_empty_dirs(build_dir)
    return changes


def _download_version_wget(site_base_url: str, version: str, dest: Path) -> None:
    """Download one version subtree with wget (available in CI containers)."""
    url = f"{site_base_url.rstrip('/')}/{version}/"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)

    # -nH: no host-based dirs; -np: stay under version URL; -P: output prefix
    result = subprocess.run(
        [
            "wget",
            "-q",
            "-r",
            "-l",
            "50",
            "-np",
            "-nH",
            "-P",
            str(dest.parent),
            url,
        ],
        check=False,
    )
    if result.returncode != 0:
        print(f"wget failed for {url} (exit {result.returncode})", file=sys.stderr)

    # wget may create dest.parent/<version>/ or preserve extra URL path
    # segments such as dest.parent/EmbodiChain/<version>/; normalize that.
    flatten_nested_version_dirs(dest.parent)
    if not dest.is_dir():
        candidates = [
            candidate
            for candidate in dest.parent.rglob(version)
            if candidate.is_dir() and candidate != dest
        ]
        candidates.sort(key=lambda candidate: len(candidate.parts))
        for candidate in candidates:
            if (candidate / "index.html").is_file():
                candidate.rename(dest)
                break

    if dest.is_dir():
        changes = normalize_artifact_paths(dest)
        if changes:
            print(f"Normalized {len(changes)} artifact path(s) in {version}.")
    elif result.returncode != 0:
        return

    _cleanup_empty_dirs(dest.parent)


def merge_published_site(
    build_dir: Path,
    *,
    site_base_url: str | None = None,
    published_root: Path | None = None,
    skip_versions: frozenset[str] | None = None,
) -> list[str]:
    """Copy missing version dirs from published site into ``build_dir``.

    Args:
        build_dir: Sphinx output root (``docs/build/html``).
        site_base_url: Live Pages base, e.g. ``https://org.github.io/Repo``.
        published_root: Local published tree for tests (``versions.json`` + dirs).
        skip_versions: Version names to leave for a fresh build (e.g. ``main``).

    Returns:
        Names of versions merged from the published site.
    """
    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    skip = skip_versions or frozenset()
    initial_changes = normalize_artifact_paths(build_dir)
    if initial_changes:
        print(
            f"Normalized {len(initial_changes)} existing artifact path(s) in build tree."
        )

    manifest = load_versions_manifest(
        site_base_url=site_base_url,
        published_root=published_root,
    )
    if not manifest:
        print("No published versions manifest; skipping merge.")
        return []

    merged: list[str] = []
    for entry in manifest.get("versions", []):
        name = entry.get("name")
        if not name or name in skip:
            continue
        if (build_dir / name).is_dir():
            continue

        if published_root is not None:
            src = published_root / name
            if not src.is_dir():
                print(
                    f"Published root missing directory {name}; skip.", file=sys.stderr
                )
                continue
            print(f"Merging local published version: {name}")
            _copy_local_version(src, build_dir / name)
            merged.append(name)
        elif site_base_url:
            print(f"Downloading published version: {name}")
            _download_version_wget(site_base_url, name, build_dir / name)
            if (build_dir / name).is_dir():
                merged.append(name)
        else:
            print(
                "Neither published_root nor site_base_url set; cannot merge.",
                file=sys.stderr,
            )

    final_changes = normalize_artifact_paths(build_dir)
    if final_changes:
        print(f"Normalized {len(final_changes)} artifact path(s) in build tree.")

    final_flattened = flatten_nested_version_dirs(build_dir)
    if final_flattened:
        print(f"Flattened {len(final_flattened)} nested version dir(s) in build tree.")

    post_flatten_changes = normalize_artifact_paths(build_dir)
    if post_flatten_changes:
        print(
            f"Normalized {len(post_flatten_changes)} flattened artifact path(s) in build tree."
        )

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge missing doc version dirs from live GitHub Pages into build/html"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/html"),
        help="Local docs build directory (default: build/html)",
    )
    parser.add_argument(
        "--site-base-url",
        default=None,
        help="Published site base URL, e.g. https://org.github.io/EmbodiChain",
    )
    parser.add_argument(
        "--published-root",
        type=Path,
        default=None,
        help="Local directory mirroring published site (for tests)",
    )
    parser.add_argument(
        "--skip-version",
        action="append",
        default=[],
        help="Version to skip (repeatable); rebuilt in the same CI run",
    )
    args = parser.parse_args()

    merged = merge_published_site(
        args.build_dir,
        site_base_url=args.site_base_url,
        published_root=args.published_root,
        skip_versions=frozenset(args.skip_version),
    )
    if merged:
        print(f"Merged versions: {', '.join(merged)}")
    else:
        print("No versions merged from published site.")


if __name__ == "__main__":
    main()
