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

"""Shared JSONL record helpers for usage and timing summaries.

Both :mod:`llm_usage` and :mod:`timing` persist one JSON object per line and
later aggregate those records into a summary. The read, write, grouping, and
JSON-safety logic is identical between them; only the bucket shape (token
fields vs duration fields) and the per-record accumulation differ. Those
identical pieces live here so the two summaries cannot drift apart.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "read_jsonl_records",
    "write_summary_json",
    "add_grouped_record",
    "json_safe",
]


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank/invalid lines.

    A missing file is treated as empty so a fresh run with no records yet
    still produces a zeroed summary instead of raising.
    """
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Malformed lines are skipped rather than fatal: a partially
            # flushed record must not discard the rest of the run.
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def write_summary_json(
    summary: Mapping[str, Any],
    summary_path: str | Path,
) -> None:
    """Write a summary dict as sorted, indented JSON, creating parent dirs."""
    path = Path(summary_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_grouped_record(
    groups: dict[str, dict[str, Any]],
    key: Any,
    record: Mapping[str, Any],
    empty_bucket: Callable[[], dict[str, Any]],
    add_record: Callable[[dict[str, Any], Mapping[str, Any]], None],
) -> None:
    """Accumulate ``record`` under ``key``, lazily creating a fresh bucket.

    A missing/empty key collapses to ``"unknown"`` so every record is counted
    exactly once even if the producer omitted the grouping field.
    """
    group_key = str(key or "unknown")
    bucket = groups.setdefault(group_key, empty_bucket())
    add_record(bucket, record)


def json_safe(value: Any) -> Any:
    """Recursively coerce ``value`` into something ``json.dumps`` accepts."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return str(value)
