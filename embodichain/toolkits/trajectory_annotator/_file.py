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

"""Bounded file fingerprinting shared by file-backed frame adapters."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

__all__ = ["file_signature", "hash_file", "source_identity"]


def file_signature(path: Path) -> tuple[int, int, int, int]:
    """Return immutable file identity fields without reading a source handle."""
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def hash_file(
    path: Path, cancelled: threading.Event, progress: Callable[..., None]
) -> str:
    """Hash in 1 MiB chunks with progress and cooperative cancellation."""
    if not path.is_file():
        raise ValueError("输入路径必须为普通文件。")
    total = path.stat().st_size
    digest = hashlib.sha256()
    completed = 0
    progress(phase="hashing", completed=completed, total=total)
    with path.open("rb") as handle:
        while chunk := handle.read(1024**2):
            if cancelled.is_set():
                raise InterruptedError("初始化已取消。")
            digest.update(chunk)
            completed += len(chunk)
            progress(phase="hashing", completed=completed, total=total)
    return digest.hexdigest()


def source_identity(details: dict[str, Any]) -> str:
    """Bind content and adapter selections to a reproducible identity."""
    return hashlib.sha256(
        json.dumps(details, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
