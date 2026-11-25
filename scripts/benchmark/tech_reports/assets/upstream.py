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

"""Resolve licensed NVIDIA assets directly from their pinned upstream files."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from filelock import FileLock

from .download import _file_matches, _relative_path

__all__ = ["download_nvidia_assets"]

_NVIDIA_ROOT = (
    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/5.1"
)


def download_nvidia_assets(
    sources: Path,
    cache_dir: Path,
    *,
    robots: tuple[str, ...],
    local_files_only: bool = False,
) -> dict[str, Path]:
    """Fetch the source USD dependency set for each selected robot.

    The report stores paths and hashes. Original USD contents are downloaded from
    the NVIDIA source with their relative dependency layout preserved.
    """
    manifest = json.loads(sources.read_text())["robots"]
    cache = cache_dir / "nvidia-isaacsim-5.1"
    resolved = {}
    for robot in robots:
        entry = manifest[robot]
        primary = _relative_path(entry["entrypoint"])
        if not any(
            record["source_relative_path"] == primary.as_posix()
            for record in entry["files"]
        ):
            raise ValueError(
                f"NVIDIA entry point is missing from the file inventory: {primary}"
            )
        for record in entry["files"]:
            source = _relative_path(record["source_relative_path"])
            size, digest = record["size_bytes"], record["sha256"]
            target = cache / source
            if _file_matches(target, size, digest):
                continue
            if local_files_only:
                raise FileNotFoundError(
                    f"Missing or invalid cached NVIDIA asset: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(target) + ".lock"):
                if _file_matches(target, size, digest):
                    continue
                url = f"{_NVIDIA_ROOT}/{quote(source.as_posix(), safe='/')}"
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False
                ) as output:
                    pending = Path(output.name)
                try:
                    with (
                        urlopen(url, timeout=60) as response,
                        pending.open("wb") as output,
                    ):
                        remaining = size
                        while remaining:
                            chunk = response.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            output.write(chunk)
                            remaining -= len(chunk)
                        if response.read(1):
                            raise ValueError(f"Unexpected response size for {source}")
                    if not _file_matches(pending, size, digest):
                        raise ValueError(f"NVIDIA asset checksum mismatch: {source}")
                    os.replace(pending, target)
                finally:
                    pending.unlink(missing_ok=True)
        resolved[robot] = cache / primary
    return resolved
