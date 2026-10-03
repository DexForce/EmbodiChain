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

"""Versioned contracts for data-analysis episode records."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DIMENSION_KEYS = (
    "asset",
    "pose_x",
    "pose_y",
    "pose_z",
    "yaw",
    "affordance",
    "approach",
    "material",
    "light",
    "trajectory_family",
    "duration_s",
)
MEASUREMENT_SOURCES = frozenset(
    {"measured", "configured", "annotated", "derived", "estimated", "unknown"}
)
STATUSES = frozenset(
    {
        "proposed",
        "planning_failed",
        "rollout_failed",
        "rejected",
        "pending_write",
        "partial_commit",
        "committed",
    }
)
IDENTITY_FIELDS = (
    "episode_id",
    "run_id",
    "candidate_id",
    "attempt_id",
    "task_id",
    "robot_id",
)
REQUIRED_FIELDS = (
    "schema_version",
    *IDENTITY_FIELDS,
    "status",
    "reason",
    "seed",
    "parents",
    "dimensions",
    "artifacts",
    "segments",
    "metrics",
    "provenance",
)

__all__ = [
    "DIMENSION_KEYS",
    "IDENTITY_FIELDS",
    "MEASUREMENT_SOURCES",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "STATUSES",
    "RecordValidationError",
    "unknown_measurement",
    "validate_record",
]


class RecordValidationError(ValueError):
    """Raised when an episode record violates the versioned contract."""


def unknown_measurement(reason: str) -> dict[str, Any]:
    """Create an explicit unavailable measurement."""
    if not reason:
        raise ValueError("Unknown measurements require a non-empty reason.")
    return {
        "value": None,
        "source": "unknown",
        "unit": "",
        "frame": "unknown",
        "scope": "episode",
        "missing_reason": reason,
    }


def _require_type(
    value: object, expected: type | tuple[type, ...], message: str
) -> None:
    if not isinstance(value, expected):
        raise RecordValidationError(message)


def _validate_measurement(name: str, measurement: object) -> None:
    _require_type(
        measurement, Mapping, f"Dimension '{name}' must be a measurement mapping."
    )
    assert isinstance(measurement, Mapping)
    required = {"value", "source", "unit", "frame", "scope"}
    missing = required - set(measurement)
    if missing:
        raise RecordValidationError(
            f"Dimension '{name}' is missing fields: {sorted(missing)}"
        )
    if measurement["source"] not in MEASUREMENT_SOURCES:
        raise RecordValidationError(
            f"Dimension '{name}' has an unsupported source: {measurement['source']!r}"
        )
    for field in ("unit", "frame", "scope"):
        _require_type(
            measurement[field],
            str,
            f"Dimension '{name}' field '{field}' must be a string.",
        )
    if measurement["source"] == "unknown":
        if measurement["value"] is not None:
            raise RecordValidationError(
                f"Unknown dimension '{name}' must have a null value."
            )
        if (
            not isinstance(measurement.get("missing_reason"), str)
            or not measurement["missing_reason"]
        ):
            raise RecordValidationError(
                f"Unknown dimension '{name}' requires missing_reason."
            )


def validate_record(
    record: Mapping[str, Any], *, artifact_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate and return a detached JSON-compatible episode record.

    Args:
        record: Candidate record mapping.
        artifact_root: Directory used to resolve relative declared artifact paths.

    Returns:
        A deep copy safe to persist or return to callers.

    Raises:
        RecordValidationError: If the record or a declared committed artifact is invalid.
    """
    _require_type(record, Mapping, "Episode record must be a mapping.")
    missing = set(REQUIRED_FIELDS) - set(record)
    if missing:
        raise RecordValidationError(
            f"Episode record is missing fields: {sorted(missing)}"
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise RecordValidationError(
            f"Unsupported schema_version {record['schema_version']!r}; expected {SCHEMA_VERSION}."
        )
    for field in (
        "episode_id",
        "run_id",
        "candidate_id",
        "task_id",
        "robot_id",
        "status",
    ):
        _require_type(record[field], str, f"Field '{field}' must be a string.")
        if not record[field]:
            raise RecordValidationError(f"Field '{field}' cannot be empty.")
    if type(record["attempt_id"]) is not int or record["attempt_id"] < 0:
        raise RecordValidationError(
            "Field 'attempt_id' must be a non-negative integer."
        )
    if record["status"] not in STATUSES:
        raise RecordValidationError(f"Unsupported status {record['status']!r}.")
    if record["reason"] is not None and not isinstance(record["reason"], str):
        raise RecordValidationError("Field 'reason' must be a string or null.")
    if record["seed"] is not None and type(record["seed"]) is not int:
        raise RecordValidationError("Field 'seed' must be an integer or null.")
    _require_type(record["parents"], list, "Field 'parents' must be a list.")
    if any(not isinstance(parent, Mapping) for parent in record["parents"]):
        raise RecordValidationError("Every parent must be a mapping.")
    _require_type(
        record["dimensions"], Mapping, "Field 'dimensions' must be a mapping."
    )
    for name, measurement in record["dimensions"].items():
        if name not in DIMENSION_KEYS:
            raise RecordValidationError(f"Unsupported dimension {name!r}.")
        _validate_measurement(name, measurement)
    _require_type(record["artifacts"], Mapping, "Field 'artifacts' must be a mapping.")
    for name, path in record["artifacts"].items():
        if not isinstance(name, str) or not isinstance(path, str) or not path:
            raise RecordValidationError(
                "Artifact names and paths must be non-empty strings."
            )
        if record["status"] == "committed" and artifact_root is not None:
            resolved = Path(path)
            if not resolved.is_absolute():
                resolved = Path(artifact_root) / resolved
            if not resolved.is_file():
                raise RecordValidationError(
                    f"Declared artifact '{name}' does not exist: {resolved}"
                )
    _require_type(record["segments"], list, "Field 'segments' must be a list.")
    if any(not isinstance(segment, Mapping) for segment in record["segments"]):
        raise RecordValidationError("Every segment must be a mapping.")
    for field in ("metrics", "provenance"):
        _require_type(record[field], Mapping, f"Field '{field}' must be a mapping.")
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(
            "Record must contain finite JSON values only."
        ) from exc
    return copy.deepcopy(dict(record))
