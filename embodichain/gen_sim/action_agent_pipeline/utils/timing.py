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

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any

__all__ = [
    "TIMING_PATH_ENV",
    "TIMING_PROCESS_ENV",
    "TIMING_RUN_ID_ENV",
    "build_timing_summary",
    "configure_timing_tracking",
    "disable_timing_tracking",
    "normalize_timing_stage",
    "record_timing",
    "timing_scope",
    "write_timing_summary",
]

TIMING_PATH_ENV = "EMBODICHAIN_TIMING_PATH"
TIMING_RUN_ID_ENV = "EMBODICHAIN_TIMING_RUN_ID"
TIMING_PROCESS_ENV = "EMBODICHAIN_TIMING_PROCESS"

_TIMING_ENV_KEYS = {
    TIMING_PATH_ENV,
    TIMING_RUN_ID_ENV,
    TIMING_PROCESS_ENV,
}


def configure_timing_tracking(
    *,
    timing_path: str | Path,
    run_id: str,
    process_name: str,
    reset: bool = False,
) -> Path:
    """Configure process-local JSONL timing records."""
    path = Path(timing_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        path.write_text("", encoding="utf-8")
    os.environ[TIMING_PATH_ENV] = path.as_posix()
    os.environ[TIMING_RUN_ID_ENV] = str(run_id)
    os.environ[TIMING_PROCESS_ENV] = str(process_name)
    return path


def disable_timing_tracking() -> None:
    """Disable process-local EmbodiChain timing records."""
    for key in _TIMING_ENV_KEYS:
        os.environ.pop(key, None)


def normalize_timing_stage(stage: str) -> str:
    """Normalize a timing stage into a compact identifier."""
    value = str(stage or "unknown").strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.-")
    return value or "unknown"


@contextmanager
def timing_scope(
    stage: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Record wall-clock duration for a code block when timing is enabled."""
    start = time.perf_counter()
    record_metadata = dict(metadata or {})
    status = "ok"
    error_type = None
    try:
        yield record_metadata
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        duration_s = time.perf_counter() - start
        if error_type is not None:
            record_metadata["error_type"] = error_type
        record_timing(
            stage=stage,
            duration_s=duration_s,
            status=status,
            metadata=record_metadata or None,
        )


def record_timing(
    *,
    stage: str,
    duration_s: float,
    status: str = "ok",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Append one timing record to the configured JSONL file."""
    timing_path = os.getenv(TIMING_PATH_ENV)
    if not timing_path:
        return

    record: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": os.getenv(TIMING_RUN_ID_ENV),
        "process": os.getenv(TIMING_PROCESS_ENV),
        "pid": os.getpid(),
        "stage": normalize_timing_stage(stage),
        "duration_s": float(duration_s),
        "status": str(status or "ok"),
    }
    if metadata:
        record["metadata"] = _json_safe(metadata)

    path = Path(timing_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_timing_summary(timing_path: str | Path) -> dict[str, Any]:
    """Build aggregate duration totals from a JSONL timing file."""
    path = Path(timing_path).expanduser().resolve()
    records = _read_timing_records(path)
    summary: dict[str, Any] = {
        "timing_path": path.as_posix(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": os.getenv(TIMING_RUN_ID_ENV),
        "total": _empty_bucket(),
        "by_stage": {},
        "by_process": {},
    }

    for record in records:
        _add_record(summary["total"], record)
        _add_grouped_record(summary["by_stage"], record.get("stage"), record)
        _add_grouped_record(summary["by_process"], record.get("process"), record)

    _finalize_bucket(summary["total"])
    for group in summary["by_stage"].values():
        _finalize_bucket(group)
    for group in summary["by_process"].values():
        _finalize_bucket(group)

    return summary


def write_timing_summary(
    *,
    timing_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Any]:
    """Write a JSON timing summary and return it."""
    summary = build_timing_summary(timing_path)
    path = Path(summary_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_timing_records(path: Path) -> list[dict[str, Any]]:
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
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _empty_bucket() -> dict[str, float | int]:
    return {
        "calls": 0,
        "ok": 0,
        "errors": 0,
        "total_s": 0.0,
        "min_s": 0.0,
        "max_s": 0.0,
        "avg_s": 0.0,
    }


def _add_grouped_record(
    groups: dict[str, dict[str, float | int]],
    key: Any,
    record: Mapping[str, Any],
) -> None:
    group_key = str(key or "unknown")
    bucket = groups.setdefault(group_key, _empty_bucket())
    _add_record(bucket, record)


def _add_record(bucket: dict[str, float | int], record: Mapping[str, Any]) -> None:
    duration_s = record.get("duration_s")
    if not isinstance(duration_s, (int, float)):
        return

    bucket["calls"] = int(bucket["calls"]) + 1
    if record.get("status") == "error":
        bucket["errors"] = int(bucket["errors"]) + 1
    else:
        bucket["ok"] = int(bucket["ok"]) + 1
    bucket["total_s"] = float(bucket["total_s"]) + float(duration_s)
    bucket["max_s"] = max(float(bucket["max_s"]), float(duration_s))
    bucket["min_s"] = (
        float(duration_s)
        if int(bucket["calls"]) == 1
        else min(float(bucket["min_s"]), float(duration_s))
    )


def _finalize_bucket(bucket: dict[str, float | int]) -> None:
    calls = int(bucket["calls"])
    bucket["avg_s"] = float(bucket["total_s"]) / calls if calls else 0.0


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)
