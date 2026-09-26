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
import os
import platform
from pathlib import Path
import subprocess
import sys
from collections.abc import Iterable
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None

__all__ = [
    "git_revision",
    "gpu_snapshot",
    "host_snapshot",
    "process_memory_snapshot",
    "package_version",
    "software_snapshot",
]


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


def host_snapshot() -> dict[str, Any]:
    """Collect standard-library host identity and CPU information."""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def process_memory_snapshot() -> dict[str, Any]:
    """Read the process lifetime RSS peak with its platform-dependent method."""
    if resource is None:
        return {
            "value_bytes": None,
            "unit": "byte",
            "scope": "process_lifetime_peak",
            "reason": "resource module unavailable",
        }
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; keep a conservative method label for other hosts.
        return {
            "value_bytes": peak * 1024,
            "unit": "byte",
            "scope": "process_lifetime_peak",
            "method": "resource.getrusage_ru_maxrss_linux_kib",
        }
    except (AttributeError, OSError, ValueError) as exc:
        return {
            "value_bytes": None,
            "unit": "byte",
            "scope": "process_lifetime_peak",
            "reason": str(exc),
        }


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
