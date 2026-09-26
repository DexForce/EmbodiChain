# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Convert legacy benchmark rows into the common offline result protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scripts.benchmark.core.artifacts import stable_hash

__all__ = ["convert_legacy_rows", "normalize_legacy_row"]


_SCALAR_METRICS = (
    "cost_time_ms",
    "latency_ms",
    "training_fps",
    "environment_fps",
    "success_rate",
    "peak_gpu_memory_mb",
)


def normalize_legacy_row(
    row: Mapping[str, Any], *, experiment_id: str = "legacy"
) -> dict[str, Any]:
    """Add common identity/status fields while preserving the original row."""
    backend = row.get("backend") or row.get("algorithm") or row.get("planner")
    case_id = row.get("case_id") or row.get("task") or row.get("scenario") or "default"
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("legacy row requires backend, algorithm or planner")
    status = row.get("status")
    if status is None:
        status = (
            "completed"
            if row.get("metrics") or any(key in row for key in _SCALAR_METRICS)
            else "not_run"
        )
    if status not in {"not_run", "completed", "failed", "timeout", "interrupted"}:
        raise ValueError(f"unknown legacy status: {status!r}")
    metrics = dict(row.get("metrics") or {})
    for key in _SCALAR_METRICS:
        if key in row and key not in metrics:
            metrics[key] = row[key]
    normalized = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": str(row.get("run_id") or f"legacy-{stable_hash(dict(row))[:16]}"),
        "case_id": str(case_id),
        "backend": backend,
        "repeat": int(row.get("repeat", 0)),
        "attempt": int(row.get("attempt", 0)),
        "status": status,
        "quality_status": row.get("quality_status", "missing"),
        "task_status": row.get("task_status", "not_evaluated"),
        "data_status": row.get("data_status", "not_evaluated"),
        "metrics": metrics,
        "legacy_source": dict(row),
    }
    for key in ("config_sha256", "hardware_id", "asset_sha256"):
        if key in row:
            normalized[key] = row[key]
    return normalized


def convert_legacy_rows(
    rows: Sequence[Mapping[str, Any]], *, experiment_id: str = "legacy"
) -> list[dict[str, Any]]:
    """Convert all rows without dropping failed or unsupported entries."""
    return [normalize_legacy_row(row, experiment_id=experiment_id) for row in rows]
