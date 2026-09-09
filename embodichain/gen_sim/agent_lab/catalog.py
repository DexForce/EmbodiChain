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

"""Read task100 descriptions without imposing their historical recipe labels."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path

__all__ = ["Task", "read_catalog", "write_json"]


@dataclass(frozen=True)
class Task:
    """One source task and its observed asset-preparation state."""

    task_id: str
    instruction: str
    acceptance: str
    source_dir: str
    image: str = ""
    scene_config: str | None = None
    asset_status: str = "unspecified"
    level: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-ready task description."""
        return asdict(self)


def read_catalog(root: str | Path) -> dict[str, Task]:
    """Read the numbered main table, ignoring the separate L4 boundary table."""
    root = Path(root).resolve()
    rows = csv.reader(
        (root / "task100_revised.md").read_text(encoding="utf-8").splitlines(),
        delimiter="|",
    )
    tasks = {}
    level = ""
    for row in rows:
        if row and (heading := re.match(r"## (L[1-4])\b", row[0])):
            level = heading.group(1)
        if len(row) != 7 or not row[1].strip().isdigit():
            continue
        number, instruction, _route, _notes, acceptance = (
            value.strip() for value in row[1:6]
        )
        task_id = f"task{1100 + int(number)}"
        source = root / task_id
        scene = source / "scene_export" / "scene_config.json"
        has_geometry = any(source.glob("scene_generation/*/*.glb"))
        status = (
            "exported"
            if scene.is_file()
            else "intermediate_geometry" if has_geometry else "unprepared"
        )
        tasks[task_id] = Task(
            task_id,
            instruction,
            acceptance,
            str(source),
            str(root / "image" / f"{task_id}.png"),
            str(scene) if scene.is_file() else None,
            status,
            level,
        )
    return tasks


def write_json(path: str | Path, value: object) -> None:
    """Persist one human-readable experiment artifact."""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
