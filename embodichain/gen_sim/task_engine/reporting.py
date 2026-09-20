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

"""Tensor-free reports preserving canonical Task Program runtime evidence."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Final, TypeAlias

__all__ = [
    "EXECUTION_REPORT_FILENAME",
    "TASK_PROGRAM_EXECUTION_REPORT_SCHEMA",
    "TaskProgramExecutionReport",
    "validate_execution_report",
    "write_execution_report",
]

EXECUTION_REPORT_FILENAME: Final = "execution_report.json"
TASK_PROGRAM_EXECUTION_REPORT_SCHEMA: Final = "task_program_execution_report/v1"
TaskProgramExecutionReport: TypeAlias = dict[str, Any]


def validate_execution_report(value: Mapping[str, Any]) -> TaskProgramExecutionReport:
    """Validate the stable Task Engine view without reclassifying runtime truth.

    Args:
        value: JSON-compatible execution-report mapping.

    Returns:
        A detached normalized report.

    Raises:
        TypeError: If the report or one of its typed fields has the wrong type.
        ValueError: If required fields, schema identity, or values are invalid.
    """
    try:
        report = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Execution report must contain only finite JSON values."
        ) from exc
    if type(report) is not dict:
        raise TypeError("Execution report must be an exact mapping.")
    required = {
        "schema_version",
        "status",
        "task_id",
        "semantic_call_count",
        "integration_fingerprint",
        "record_dir",
        "environments",
        "runtime_result",
        "failure",
    }
    if set(report) != required:
        raise ValueError(
            "Execution report fields are invalid; "
            f"missing={sorted(required - set(report))}, "
            f"unexpected={sorted(set(report) - required)}."
        )
    if report["schema_version"] != TASK_PROGRAM_EXECUTION_REPORT_SCHEMA:
        raise ValueError("Execution report schema_version is unsupported.")
    if report["status"] not in {"succeeded", "failed", "rejected", "aborted"}:
        raise ValueError("Execution report status is invalid.")
    if not isinstance(report["task_id"], str) or not report["task_id"].strip():
        raise ValueError("Execution report task_id must be non-empty.")
    if (
        type(report["semantic_call_count"]) is not int
        or report["semantic_call_count"] < 0
    ):
        raise ValueError("Execution report semantic_call_count must be non-negative.")
    fingerprint = report["integration_fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Execution report integration_fingerprint is invalid.")
    if not isinstance(report["record_dir"], str) or not report["record_dir"]:
        raise ValueError("Execution report record_dir must be a non-empty path.")
    environments = report["environments"]
    if type(environments) is not list or not environments:
        raise ValueError("Execution report environments must be a non-empty list.")
    for index, environment in enumerate(environments):
        if type(environment) is not dict:
            raise TypeError(
                f"Execution report environments[{index}] must be a mapping."
            )
        expected = {"env_id", "success", "terminal_reason", "semantic_success"}
        if set(environment) != expected:
            raise ValueError(
                f"Execution report environments[{index}] fields are invalid."
            )
        if type(environment["env_id"]) is not int or environment["env_id"] < 0:
            raise ValueError(
                f"Execution report environments[{index}].env_id is invalid."
            )
        if type(environment["success"]) is not bool:
            raise TypeError(
                f"Execution report environments[{index}].success must be bool."
            )
        if not isinstance(environment["terminal_reason"], str):
            raise TypeError(
                f"Execution report environments[{index}].terminal_reason must be string."
            )
        semantics = environment["semantic_success"]
        if type(semantics) is not dict or any(
            type(item) is not bool for item in semantics.values()
        ):
            raise TypeError(
                f"Execution report environments[{index}].semantic_success is invalid."
            )
    if report["failure"] is not None and type(report["failure"]) is not dict:
        raise TypeError("Execution report failure must be a mapping or None.")
    return deepcopy(report)


def write_execution_report(
    output_dir: str | Path,
    value: Mapping[str, Any],
) -> Path:
    """Write one validated execution report.

    Args:
        output_dir: Directory that receives ``execution_report.json``.
        value: JSON-compatible execution-report mapping.

    Returns:
        Absolute path to the written report.

    Raises:
        TypeError: If the report contains an invalid typed field.
        ValueError: If the report violates its schema.
        OSError: If the destination cannot be created or written.
    """
    report = validate_execution_report(value)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / EXECUTION_REPORT_FILENAME
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
