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

"""Read-only provenance collected by workers, never by offline report rebuilds."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
from collections.abc import Iterable
from typing import Any

__all__ = ["git_revision", "package_version", "software_snapshot", "gpu_snapshot"]


def git_revision(path: Path) -> str | None:
    """Read a checkout revision without changing git state."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(name: str) -> str | None:
    """Read distribution metadata without importing the package."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_snapshot(root: Path, source_files: Iterable[Path]) -> dict[str, Any]:
    """Collect Python, source revision and relative-path source hashes."""
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "benchmark_commit": git_revision(root),
        "benchmark_source_sha256": {
            path.relative_to(root)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in sorted(set(source_files))
        },
    }


def gpu_snapshot() -> dict[str, Any]:
    """Read device-level NVIDIA memory snapshots, including other processes."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        devices = []
        for row in csv.reader(result.stdout.splitlines()):
            name, uid, driver, total, used = (v.strip() for v in row)
            devices.append(
                {
                    "name": name,
                    "uuid": uid,
                    "driver": driver,
                    "total_bytes": int(total) * 1024**2,
                    "used_bytes": int(used) * 1024**2,
                }
            )
        return {
            "method": "nvidia-smi device snapshot (includes other processes)",
            "csv": result.stdout.strip(),
            "devices": devices,
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"value": None, "reason": str(exc)}
