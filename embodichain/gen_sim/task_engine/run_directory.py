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

"""Collision-safe allocation of human-readable Task Engine run directories."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

__all__ = ["RunDirectory", "reserve_run_directory"]


@dataclass(frozen=True)
class RunDirectory:
    """One reserved run identifier and its not-yet-published destination."""

    run_id: str
    output_root: Path
    path: Path
    created_at: datetime


@contextmanager
def reserve_run_directory(
    output_root: str | Path,
    *,
    now: datetime | None = None,
) -> Iterator[RunDirectory]:
    """Reserve a timestamped child name without creating its destination.

    Args:
        output_root: Persistent task-history directory.
        now: Optional timezone-aware timestamp used by deterministic tests.

    Yields:
        A run directory allocation safe to publish through ArtifactTransaction.
    """
    created_at = now or datetime.now().astimezone()
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Task Engine run timestamps must include a timezone.")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    base = created_at.strftime("%Y%m%d_%H%M%S")
    for collision_index in range(10_000):
        run_id = base if collision_index == 0 else f"{base}_{collision_index:02d}"
        destination = root / run_id
        reservation = root / f".{run_id}.reserve"
        if destination.exists():
            continue
        try:
            reservation.mkdir()
        except FileExistsError:
            continue
        if destination.exists():
            reservation.rmdir()
            continue
        try:
            yield RunDirectory(
                run_id=run_id,
                output_root=root,
                path=destination,
                created_at=created_at,
            )
        finally:
            reservation.rmdir()
        return
    raise RuntimeError("Unable to reserve a Task Engine run directory.")
