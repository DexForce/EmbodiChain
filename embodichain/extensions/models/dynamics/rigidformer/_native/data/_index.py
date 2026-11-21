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

"""Compact author window indexing without materializing repeated paths."""

from __future__ import annotations
from bisect import bisect_right
from pathlib import Path
from typing import Any


class _SceneWindows:
    """Store scene runs while preserving the legacy window-index mapping."""

    def __init__(self, *, random_sampling: bool) -> None:
        self.random_sampling = random_sampling
        self.runs: list[tuple[str, tuple[Path, ...] | None, int, int]] = []
        self.ends: list[int] = []

    def add_scene(
        self, name: str, frames: list[Path], sequence_length: int, stride: int
    ) -> None:
        count = len(range(0, len(frames) - sequence_length + 1, stride))
        if not count:
            return
        self.runs.append(
            (
                name,
                None if self.random_sampling else tuple(frames),
                sequence_length,
                stride,
            )
        )
        self.ends.append((self.ends[-1] if self.ends else 0) + count)

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        run = bisect_right(self.ends, index)
        name, frames, length, stride = self.runs[run]
        if self.random_sampling:
            return {"dataset": name}
        start = (index - (self.ends[run - 1] if run else 0)) * stride
        selected = frames[start : start + length]
        return {
            "dataset": name,
            "frames": [p.as_posix() for p in selected],
            "frame_ids": [int(p.name.split("_")[-1]) for p in selected],
        }
