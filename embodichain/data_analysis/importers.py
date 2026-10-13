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

"""Read-only adapters for historical LeRobot metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .schema import DIMENSION_KEYS, unknown_measurement, validate_record

__all__ = ["import_lerobot_metadata"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            rows.append(row)
    return rows


def _info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def import_lerobot_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Adapt LeRobot episode metadata without importing LeRobot or changing source data.

    EmbodiChain sidecars carry richer analysis fields. Standard and legacy
    LeRobot metadata are still imported, with absent analysis facts represented
    as explicit unknown measurements.
    """
    root = Path(path).resolve()
    metadata = root / "meta"
    sidecar = metadata / "embodichain_episodes.jsonl"
    standard = metadata / "episodes.jsonl"
    info = _info(root)
    if sidecar.is_file():
        rows = _read_jsonl(sidecar)
        source_has_sidecar = True
    elif standard.is_file():
        rows = _read_jsonl(standard)
        source_has_sidecar = False
    else:
        total = info.get("total_episodes", 0)
        if type(total) is not int or total < 0:
            raise ValueError(
                "LeRobot info total_episodes must be a non-negative integer."
            )
        rows = [{"episode_index": index} for index in range(total)]
        source_has_sidecar = False
    result = []
    for fallback_index, row in enumerate(rows):
        index = row.get(
            "lerobot_episode_index", row.get("episode_index", fallback_index)
        )
        if type(index) is not int or index < 0:
            raise ValueError(f"Invalid LeRobot episode index {index!r}.")
        supplied_dimensions = row.get("dimensions", {})
        if not isinstance(supplied_dimensions, Mapping):
            raise ValueError(f"Episode {index} dimensions must be a mapping.")
        dimensions = {
            key: supplied_dimensions.get(
                key, unknown_measurement("historical_metadata_unavailable")
            )
            for key in DIMENSION_KEYS
        }
        task = row.get("task", row.get("instruction"))
        if task is None:
            tasks = row.get("tasks", [])
            task = tasks[0] if isinstance(tasks, list) and tasks else "unknown"
        root_key = str(root)
        episode_id = f"lerobot:{root_key}:{index}"
        metrics: dict[str, Any] = {}
        if "length" in row:
            metrics["frames"] = row["length"]
        if "fps" in info:
            metrics["fps"] = info["fps"]
        record = {
            "schema_version": 1,
            "episode_id": episode_id,
            "run_id": str(row.get("run_id", f"lerobot:{root_key}")),
            "candidate_id": str(row.get("candidate_id", episode_id)),
            "attempt_id": int(row.get("attempt_id", 0)),
            "task_id": str(task),
            "robot_id": str(row.get("robot_id", info.get("robot_type", "unknown"))),
            "status": "committed",
            "reason": row.get("reason"),
            "seed": row.get("seed"),
            "parents": list(row.get("parents", [])),
            "dimensions": dimensions,
            "artifacts": {},
            "segments": list(row.get("segments", [])),
            "metrics": metrics,
            "provenance": {
                **dict(row.get("provenance", {})),
                "source": "lerobot",
                "dataset_root": root_key,
                "episode_index": index,
                "historical_metadata": (
                    "available"
                    if source_has_sidecar and bool(supplied_dimensions)
                    else "unavailable"
                ),
            },
        }
        result.append(validate_record(record))
    return result
