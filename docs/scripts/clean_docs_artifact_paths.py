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
"""Remove filesystem-hostile wget query-string artifacts from docs builds."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

__all__ = ["clean_docs_artifact_paths"]

_INVALID_ARTIFACT_CHARS = frozenset('"<>|*?\r\n')


def _has_invalid_component(path: Path) -> bool:
    return any(
        any(char in part for char in _INVALID_ARTIFACT_CHARS) for part in path.parts
    )


def clean_docs_artifact_paths(build_dir: Path) -> list[Path]:
    """Remove paths that GitHub artifact upload refuses.

    ``wget`` preserves cache-busting query strings from links such as
    ``clipboard.min.js?v=...`` as literal filenames. Those files are redundant
    on a static Pages site because the real asset is served without the query
    suffix, but ``actions/upload-artifact`` rejects them on portability grounds.

    Args:
        build_dir: Root of the generated docs site.

    Returns:
        Removed paths, relative to ``build_dir``.
    """
    build_dir = build_dir.resolve()
    removed: list[Path] = []
    if not build_dir.exists():
        return removed

    for path in sorted(build_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        relative = path.relative_to(build_dir)
        if not _has_invalid_component(relative):
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        removed.append(relative)

    return sorted(removed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove docs paths rejected by GitHub artifact upload"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/html"),
        help="Generated docs site root (default: build/html)",
    )
    args = parser.parse_args()

    removed = clean_docs_artifact_paths(args.build_dir)
    if removed:
        print("Removed artifact-incompatible docs paths:")
        for path in removed:
            print(f"  {path}")
    else:
        print("No artifact-incompatible docs paths found.")


if __name__ == "__main__":
    main()
