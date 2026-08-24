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

"""Strict validation and atomic publication for execution reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any

from embodichain import __version__ as embodichain_version

from .models import ExecutionReport

__all__ = [
    "EXECUTION_REPORT_FILENAME",
    "EXECUTION_REPORT_SCHEMA",
    "build_execution_provenance",
    "validate_execution_report",
    "write_execution_report",
]

EXECUTION_REPORT_SCHEMA = "action_engine_execution_report_v2"
EXECUTION_REPORT_FILENAME = "execution_report.json"


def build_execution_provenance(
    *,
    episode_seed: int | None = None,
    runtime_arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture the minimum code and runtime context needed to reproduce a run."""
    git_commit, git_dirty = _git_code_state()
    provenance = {
        "episode_seed": episode_seed,
        "embodichain_version": str(embodichain_version),
        "python_version": platform.python_version(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "runtime_arguments": deepcopy(dict(runtime_arguments or {})),
    }
    return _validate_execution_provenance(provenance)


def validate_execution_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the tensor-free, strict-JSON Action Agent result protocol."""
    result = _mapping(value, "ExecutionReport")
    keys = {
        "schema_version",
        "task_id",
        "plan_hash",
        "action_graph_hash",
        "status",
        "run_id",
        "episode_id",
        "provenance",
        "environments",
        "action_count",
        "retry_count",
        "recovery_count",
        "revision_count",
        "failure_events",
        "graph_revisions",
        "record_dir",
        "error",
    }
    _keys(result, keys, "ExecutionReport")
    if result.get("schema_version") != EXECUTION_REPORT_SCHEMA:
        raise ValueError(
            "ExecutionReport.schema_version must be " f"{EXECUTION_REPORT_SCHEMA!r}."
        )
    for key in ("task_id", "run_id", "episode_id"):
        result[key] = _nonempty(result.get(key), f"ExecutionReport.{key}")
    result["provenance"] = _validate_execution_provenance(result.get("provenance"))
    for key in ("plan_hash", "action_graph_hash"):
        result[key] = _digest(result.get(key), f"ExecutionReport.{key}")
    result["status"] = _enum(
        result.get("status"),
        {"succeeded", "failed", "rejected", "aborted"},
        "ExecutionReport.status",
    )

    env_keys = {
        "env_id",
        "success",
        "semantic_success",
        "action_count",
        "retry_count",
        "recovery_count",
        "revision_count",
        "failures",
    }
    environments = []
    for index, raw in enumerate(
        _sequence(result.get("environments"), "ExecutionReport.environments")
    ):
        context = f"ExecutionReport.environments[{index}]"
        environment = _mapping(raw, context)
        _keys(environment, env_keys, context)
        environment["env_id"] = _string(environment.get("env_id"), f"{context}.env_id")
        if not isinstance(environment.get("success"), bool):
            raise ValueError(f"{context}.success must be a boolean.")
        semantic_success = _mapping(
            environment.get("semantic_success"), f"{context}.semantic_success"
        )
        if any(not isinstance(item, bool) for item in semantic_success.values()):
            raise ValueError(f"{context}.semantic_success values must be booleans.")
        environment["semantic_success"] = semantic_success
        for key in ("action_count", "retry_count", "recovery_count", "revision_count"):
            environment[key] = _integer(
                environment.get(key), f"{context}.{key}", minimum=0
            )
        environment["failures"] = _mapping_sequence(
            environment.get("failures"), f"{context}.failures"
        )
        environments.append(environment)
    result["environments"] = environments

    for key in ("action_count", "retry_count", "recovery_count", "revision_count"):
        result[key] = _integer(result.get(key), f"ExecutionReport.{key}", minimum=0)
    result["failure_events"] = _mapping_sequence(
        result.get("failure_events"), "ExecutionReport.failure_events"
    )
    result["graph_revisions"] = _mapping_sequence(
        result.get("graph_revisions"), "ExecutionReport.graph_revisions"
    )
    for key in ("record_dir", "error"):
        if result.get(key) is not None:
            result[key] = _string(result.get(key), f"ExecutionReport.{key}")

    if result["status"] == "rejected" and result["action_count"] != 0:
        raise ValueError("A rejected ExecutionReport must have action_count=0.")
    successes = [environment["success"] for environment in environments]
    if result["status"] == "succeeded" and (
        not successes or not all(successes) or result.get("error") is not None
    ):
        raise ValueError(
            "A succeeded ExecutionReport requires successful environments and no error."
        )
    if result["status"] == "failed" and (
        not successes or all(successes) or result.get("error") is not None
    ):
        raise ValueError(
            "A failed ExecutionReport requires at least one failed environment and no error."
        )
    if result["status"] in {"rejected", "aborted"} and not result.get("error"):
        raise ValueError(
            f"A {result['status']} ExecutionReport requires a non-empty error."
        )
    _json_safe(result, "ExecutionReport")
    return result


def write_execution_report(output_dir: str | Path, value: Any) -> Path:
    """Atomically write a validated execution report into a record directory."""
    payload = value.as_mapping() if isinstance(value, ExecutionReport) else value
    validated = validate_execution_report(payload)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / EXECUTION_REPORT_FILENAME
    encoded = (
        json.dumps(validated, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _validate_execution_provenance(value: Any) -> dict[str, Any]:
    context = "ExecutionReport.provenance"
    result = _mapping(value, context)
    _keys(
        result,
        {
            "episode_seed",
            "embodichain_version",
            "python_version",
            "git_commit",
            "git_dirty",
            "runtime_arguments",
        },
        context,
    )
    seed = result.get("episode_seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError(f"{context}.episode_seed must be an integer or null.")
    result["embodichain_version"] = _nonempty(
        result.get("embodichain_version"), f"{context}.embodichain_version"
    )
    result["python_version"] = _nonempty(
        result.get("python_version"), f"{context}.python_version"
    )
    commit = result.get("git_commit")
    if commit is not None:
        commit = _string(commit, f"{context}.git_commit")
        if len(commit) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise ValueError(
                f"{context}.git_commit must be a lowercase Git object ID or null."
            )
        result["git_commit"] = commit
    dirty = result.get("git_dirty")
    if dirty is not None and not isinstance(dirty, bool):
        raise ValueError(f"{context}.git_dirty must be a boolean or null.")
    arguments = _mapping(
        result.get("runtime_arguments"), f"{context}.runtime_arguments"
    )
    if any(not isinstance(key, str) or not key for key in arguments):
        raise ValueError(f"{context}.runtime_arguments keys must be non-empty strings.")
    _json_safe(arguments, f"{context}.runtime_arguments")
    result["runtime_arguments"] = arguments
    return result


def _git_code_state() -> tuple[str | None, bool | None]:
    repository = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    commit_id = commit.stdout.strip().lower()
    if commit.returncode != 0 or len(commit_id) not in {40, 64}:
        return None, None
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return commit_id, None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return commit_id, dirty


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    return deepcopy(dict(value))


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context} must be a list.")
    return list(value)


def _keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} requires exactly fields {sorted(expected)}; "
            f"received {sorted(value)}."
        )


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string.")
    return value.strip()


def _nonempty(value: Any, context: str) -> str:
    result = _string(value, context)
    if not result:
        raise ValueError(f"{context} must not be empty.")
    return result


def _enum(value: Any, choices: set[str], context: str) -> str:
    result = _string(value, context)
    if result not in choices:
        raise ValueError(f"{context} must be one of {sorted(choices)}.")
    return result


def _integer(value: Any, context: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{context} must be an integer in the allowed range.")
    return value


def _digest(value: Any, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest.")
    return result


def _mapping_sequence(value: Any, context: str) -> list[dict[str, Any]]:
    result = [_mapping(item, context) for item in _sequence(value, context)]
    _json_safe(result, context)
    return result


def _json_safe(value: Any, context: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be finite and JSON serializable.") from error
