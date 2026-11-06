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
from typing import Any, Final, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from .bundle_deployment import LoadedTaskProgramBundle
    from .execution import TaskProgramBundleEpisode

__all__ = [
    "build_bundle_execution_report",
    "build_recovery_summary",
    "verify_bundle_execution_report",
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
        output_dir: Fresh run directory outside every sealed bundle.
        value: JSON-compatible execution-report mapping.

    Returns:
        Absolute path to the written report.

    Raises:
        TypeError: If the report contains an invalid typed field.
        ValueError: If the report violates its schema.
        FileExistsError: The historical execution report already exists.
        OSError: If the destination cannot be created or written.
    """
    from .orchestration.workflow_artifacts import (
        validate_run_output_directory,
        write_run_json,
    )

    report = validate_execution_report(value)
    root = validate_run_output_directory(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / EXECUTION_REPORT_FILENAME
    write_run_json(path, report, overwrite=False)
    return path


def _semantic_success_by_env(
    graph: dict[str, Any],
    runtime_result: dict[str, Any] | None,
    *,
    num_envs: int,
) -> list[dict[str, bool]]:
    """Project verified runtime segment outcomes onto immutable TaskGroups."""
    node_success = {str(node["id"]): [False] * num_envs for node in graph["nodes"]}
    segments = (
        runtime_result.get("segments", []) if type(runtime_result) is dict else []
    )
    if type(segments) is list:
        for segment in segments:
            if type(segment) is not dict:
                continue
            name = segment.get("name")
            if name not in node_success:
                continue
            successes = segment.get("successes")
            active = segment.get("active")
            if type(successes) is not list or len(successes) != num_envs:
                continue
            if type(active) is not list or len(active) != num_envs:
                active = [False] * num_envs
            node_success[str(name)] = [
                is_active is True and success is True
                for is_active, success in zip(active, successes, strict=True)
            ]
    result: list[dict[str, bool]] = []
    for env_id in range(num_envs):
        result.append(
            {
                str(group["id"]): all(
                    node_success[str(node_id)][env_id] for node_id in group["node_ids"]
                )
                for group in graph["task_groups"]
            }
        )
    return result


def _apply_task_spec_verdict(
    runtime_result: dict[str, Any] | None,
    row_success: list[bool],
    terminal_reasons: list[str],
    *,
    acceptance_mode: str = "strict",
) -> tuple[list[bool], list[str]]:
    """Keep legacy success unchanged; new contracts require a verified verdict."""
    if acceptance_mode == "experiment":
        from .experiment_policy import program_completion_mask

        measured = program_completion_mask(runtime_result, len(row_success))
        return [
            a and b for a, b in zip(row_success, measured, strict=True)
        ], terminal_reasons
    if acceptance_mode != "strict":
        raise ValueError("Unknown acceptance mode.")
    if not runtime_result or "task_spec_evaluation" not in runtime_result:
        return row_success, terminal_reasons
    rows = runtime_result["task_spec_evaluation"]["environments"]
    if [r.get("env_id") for r in rows] != list(range(len(row_success))):
        raise ValueError("TaskSpec evaluation environment identities differ.")
    success, reasons = list(row_success), list(terminal_reasons)
    for row in rows:
        index = row["env_id"]
        if success[index] and not (
            row.get("certificate_ready") is True and row.get("status") == "satisfied"
        ):
            success[index] = False
            reasons[index] = (
                "task_spec_failed"
                if row.get("status") == "failed"
                else "task_spec_unverified"
            )
    return success, reasons


def _accepted_rows(
    graph: Mapping[str, Any],
    runtime: dict[str, Any],
    physical: Mapping[str, Any],
    mode: str,
) -> tuple[list[bool], list[str], list[dict[str, bool]]]:
    values, reasons = _apply_task_spec_verdict(
        runtime,
        list(physical["program_success"]),
        list(physical["terminal_reasons"]),
        acceptance_mode=mode,
    )
    semantics = _semantic_success_by_env(graph, runtime, num_envs=len(values))
    cargo = physical.get("cargo_observation")
    if cargo is not None:
        contents = cargo.get("contents", [])
        unavailable = cargo.get("status") == "unavailable"
        for row, accepted in enumerate(values):
            invalid = unavailable or any(
                content["valid_mask"][row] is not True for content in contents
            )
            rejected = any(
                content["observed_mask"][row] is False for content in contents
            )
            if accepted and (invalid or rejected):
                values[row] = False
                reasons[row] = (
                    "cargo_envelope_unavailable" if invalid else "cargo_envelope_failed"
                )
        nodes = {node["id"]: node for node in graph["nodes"]}
        for group in graph["task_groups"]:
            carriers = {
                nodes[name]["call"].get("arguments", {}).get("object")
                for name in group["node_ids"]
                if nodes[name]["task_type"] == "E5"
            }
            for row in range(len(values)):
                if (
                    unavailable
                    and carriers
                    or any(
                        content["carrier"] in carriers
                        and (
                            content["valid_mask"][row] is not True
                            or content["observed_mask"][row] is not True
                        )
                        for content in contents
                    )
                ):
                    semantics[row][group["id"]] = False
    if _initial_plan_rejected(runtime):
        reasons = ["initial_plan_rejected"] * len(values)
    return values, reasons, semantics


def _initial_plan_rejected(runtime: Mapping[str, Any]) -> bool:
    """Read the actual first complete-plan result without another planning call."""
    try:
        masks = runtime["segments"][0]["metadata"]["runtime"]["calls"][0][
            "plan_attempts"
        ][0]["plan_success_mask"]
    except (KeyError, IndexError, TypeError):
        return False
    return (
        type(masks) is list and bool(masks) and all(value is False for value in masks)
    )


def build_bundle_execution_report(
    bundle: LoadedTaskProgramBundle,
    episode: TaskProgramBundleEpisode,
    *,
    record_dir: str | Path,
    acceptance_mode: str = "strict",
) -> TaskProgramExecutionReport:
    """Project one actual episode into the unchanged public execution-report shape.

    Args:
        bundle: Publicly verified immutable source declaration bundle.
        episode: Original executor result and independently collected physical report.
        record_dir: This new run's output directory, outside the sealed bundle.
        acceptance_mode: Original strict or explicitly selected experiment policy.

    Returns:
        Owned report preserving program/effect/strict/cargo facts independently.
        Strict acceptance requires the original certificate status; unavailable
        observations remain explicit and cannot produce success.
    """
    from .bundle_deployment import LoadedTaskProgramBundle
    from .execution import TaskProgramBundleEpisode
    from .bundle_io import BundleSnapshot
    from .contracts import canonical_hash
    import yaml

    if (
        type(bundle) is not LoadedTaskProgramBundle
        or type(episode) is not TaskProgramBundleEpisode
    ):
        raise TypeError(
            "Execution reports require the actual loaded bundle and episode result."
        )
    snapshot = BundleSnapshot(bundle.snapshot.identity, bundle.snapshot.files)
    identity = snapshot.identity.to_dict()
    graph = json.loads(snapshot.files["semantic_task_graph.json"])
    program = yaml.safe_load(snapshot.files["task_program/program.yaml"])
    physical = episode.evidence.to_dict()
    runtime = episode.execution.to_metadata()
    if (
        physical["bundle_digest"] != snapshot.identity.digest
        or physical["program_success"] != runtime["success"]
        or physical["lengths"] != runtime["lengths"]
    ):
        raise ValueError(
            "Physical report differs from its actual bundle/executor result."
        )
    runtime.update(
        bundle_identity=snapshot.identity.digest,
        program_fingerprint=identity["program_fingerprint"],
        acceptance_mode=acceptance_mode,
        physical_run_report=physical,
        task_spec_evaluation=dict(
            schema_version="task_spec_evaluation/v1",
            task_spec_digest=identity["task_spec_digest"],
            graph_digest=canonical_hash(graph),
            program_digest=canonical_hash(program),
            environments=[
                dict(env_id=i, **row)
                for i, row in enumerate(physical["strict_task_spec"])
            ],
        ),
    )
    values, reasons, semantics = _accepted_rows(
        graph, runtime, physical, acceptance_mode
    )
    report = dict(
        schema_version=TASK_PROGRAM_EXECUTION_REPORT_SCHEMA,
        status=(
            "rejected"
            if _initial_plan_rejected(runtime)
            else "succeeded" if all(values) else "failed"
        ),
        task_id=graph["task_id"],
        semantic_call_count=len(graph["nodes"]),
        integration_fingerprint=graph["integration_fingerprint"],
        record_dir=str(Path(record_dir).expanduser().absolute()),
        environments=[
            dict(
                env_id=i,
                success=value,
                terminal_reason=reasons[i],
                semantic_success=semantics[i],
            )
            for i, value in enumerate(values)
        ],
        runtime_result=runtime,
        failure=None,
    )
    return verify_bundle_execution_report(
        report, bundle, acceptance_mode=acceptance_mode
    )


def verify_bundle_execution_report(
    value: Mapping[str, Any],
    bundle: LoadedTaskProgramBundle,
    *,
    acceptance_mode: str = "strict",
    expected_record_dir: str | Path | None = None,
) -> TaskProgramExecutionReport:
    """Validate current-run report identity and summaries without acquiring physics.

    Args:
        value: Original stable execution-report mapping from this new run.
        bundle: Independently verified current bundle, never an old converter input.
        acceptance_mode: Expected selected acceptance policy.
        expected_record_dir: Optional caller-owned exact output path for this attempt.

    Returns:
        Detached validated report. Environment success is the existing selected
        final mask; contradictory declarations raise instead of being rewritten.
        This checks recorded identity/consistency, not authenticity of untrusted
        external measurements or a replacement physical certificate.
    """
    from uuid import UUID
    from .bundle_io import BundleSnapshot
    from .bundle_spec import decode_bundle_task_spec
    from .contracts import canonical_hash
    from ._presentation_contracts import _validate_runtime_evaluation
    import yaml

    if acceptance_mode not in {"strict", "experiment"}:
        raise ValueError("Unknown acceptance mode.")
    report = validate_execution_report(value)
    snapshot = BundleSnapshot(bundle.snapshot.identity, bundle.snapshot.files)
    identity = snapshot.identity.to_dict()
    graph = json.loads(snapshot.files["semantic_task_graph.json"])
    program = yaml.safe_load(snapshot.files["task_program/program.yaml"])
    if (
        report["task_id"] != graph["task_id"]
        or report["semantic_call_count"] != len(graph["nodes"])
        or report["integration_fingerprint"] != graph["integration_fingerprint"]
    ):
        raise ValueError("Execution report differs from its selected semantic bundle.")
    if (
        expected_record_dir is not None
        and Path(report["record_dir"]).absolute()
        != Path(expected_record_dir).expanduser().absolute()
    ):
        raise ValueError("Execution report belongs to another run output directory.")
    runtime = report["runtime_result"]
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("bundle_identity") != snapshot.identity.digest
        or runtime.get("program_fingerprint") != identity["program_fingerprint"]
        or runtime.get("acceptance_mode") != acceptance_mode
    ):
        raise ValueError(
            "Execution report has an old or different run/bundle identity; rebuild and rerun."
        )
    if report["failure"] is not None:
        if report["status"] == "succeeded" or any(
            row["success"] for row in report["environments"]
        ):
            raise ValueError(
                "Failed execution report cannot claim accepted environments."
            )
        return report
    physical = runtime.get("physical_run_report")
    if (
        not isinstance(physical, Mapping)
        or physical.get("schema_version") != "task_engine.physical_run_report/v1"
        or physical.get("bundle_digest") != snapshot.identity.digest
    ):
        raise ValueError(
            "Execution report has no matching independent physical run report."
        )
    try:
        UUID(physical["run_id"])
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        raise ValueError(
            "Physical run report requires its new run identity."
        ) from error
    count = len(report["environments"])
    ids = [row["env_id"] for row in report["environments"]]
    if (
        ids != list(range(count))
        or any(
            type(physical.get(key)) is not list or len(physical[key]) != count
            for key in (
                "episode_ids",
                "program_success",
                "program_completed",
                "terminal_reasons",
                "lengths",
                "strict_task_spec",
            )
        )
        or any(type(item) is not int or item < 0 for item in physical["episode_ids"])
        or any(type(item) is not bool for item in physical["program_success"])
        or any(type(item) is not bool for item in physical["program_completed"])
        or any(type(item) is not int or item < 0 for item in physical["lengths"])
        or physical["program_success"] != runtime.get("success")
        or physical["lengths"] != runtime.get("lengths")
    ):
        raise ValueError("Execution report physical rows/executor facts differ.")
    cargo = physical.get("cargo_observation")
    cargo_cfg = bundle.deployment.integration.registration.cargo_envelope_cfg
    if cargo_cfg is not None and not isinstance(cargo, Mapping):
        raise ValueError(
            "Declared cargo evidence is missing; its result is unavailable."
        )
    if cargo is not None:
        if (
            not isinstance(cargo, Mapping)
            or cargo.get("episode_ids") != physical["episode_ids"]
        ):
            raise ValueError(
                "Cargo evidence belongs to different physical episode rows."
            )
        if cargo.get("status") == "unavailable":
            if not isinstance(cargo.get("reason"), str) or not cargo["reason"]:
                raise ValueError(
                    "Unavailable cargo evidence requires its acquisition reason."
                )
        elif (
            cargo.get("schema_version") != "lab.cargo_envelope_observation/v1"
            or cargo.get("env_ids") != ids
            or type(cargo.get("contents")) is not list
            or any(
                not isinstance(content, Mapping)
                or type(content.get("carrier")) is not str
                or type(content.get("valid_mask")) is not list
                or type(content.get("observed_mask")) is not list
                or len(content["valid_mask"]) != count
                or len(content["observed_mask"]) != count
                or any(type(item) is not bool for item in content["valid_mask"])
                or any(
                    item is not None and type(item) is not bool
                    for item in content["observed_mask"]
                )
                for content in cargo.get("contents", [])
            )
        ):
            raise ValueError("Cargo evidence has invalid physical row channels.")
    evaluation = runtime.get("task_spec_evaluation")
    spec = decode_bundle_task_spec(json.loads(snapshot.files["task_spec.json"]))
    if spec is None:
        if not isinstance(evaluation, Mapping) or any(
            row.get("status") != "unavailable"
            or row.get("certificate_ready") is not False
            for row in physical["strict_task_spec"]
        ):
            raise ValueError(
                "Absent TaskSpec cannot have a normative success certificate."
            )
    else:
        _validate_runtime_evaluation(
            evaluation,
            spec,
            graph,
            ids,
            contract={"program_digest": canonical_hash(program)},
        )
    expected = [
        dict(env_id=i, **row) for i, row in enumerate(physical["strict_task_spec"])
    ]
    if evaluation.get("environments") != expected:
        raise ValueError("Physical and normative report rows differ.")
    values, reasons, semantics = _accepted_rows(
        graph, dict(runtime), dict(physical), acceptance_mode
    )
    for i, row in enumerate(report["environments"]):
        if (
            row["success"] != values[i]
            or row["terminal_reason"] != reasons[i]
            or row["semantic_success"] != semantics[i]
        ):
            raise ValueError(
                "Execution acceptance contradicts the recorded independent facts."
            )
    expected_status = (
        "rejected"
        if _initial_plan_rejected(runtime)
        else "succeeded" if all(values) else "failed"
    )
    if report["status"] != expected_status:
        raise ValueError(
            "Execution status contradicts its environment acceptance masks."
        )
    return report


def build_recovery_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Classify actual workflow recovery independently of normative acceptance.

    Args:
        report: Execution report retaining row identities and the original
            standard runtime workflow recovery ledger.

    Returns:
        Existing summary schema with explicit unknown coverage. Empty workflow
        ledgers never imply that a separate core planning recovery was unused.
        This reads recorded events and never runs recovery or changes success.
    """
    rows = report["environments"]
    count = len(rows)
    result = report.get("runtime_result") or {}
    if not isinstance(result, dict):
        result = {}
    successes = result.get("program_success", result.get("success"))
    if (
        not isinstance(successes, list)
        or len(successes) != count
        or any(type(value) is not bool for value in successes)
    ):
        successes = [None] * count
    segments = result.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    known = [bool(segments)] * count
    covered = [False] * count
    attempts, recovered, reacquired = [0] * count, [0] * count, [0] * count
    traces = []
    roles = {"retry_retained", "reacquire", "retry_reacquired"}
    for segment in segments:
        if not isinstance(segment, dict):
            known = [False] * count
            continue
        active = segment.get("active")
        if (
            not isinstance(active, list)
            or len(active) != count
            or any(type(value) is not bool for value in active)
        ):
            known = [False] * count
            continue
        active_ids = [index for index, value in enumerate(active) if value]
        metadata = segment.get("metadata")
        runtime = metadata.get("runtime") if isinstance(metadata, dict) else None
        if not isinstance(runtime, dict):
            for env_id in active_ids:
                known[env_id] = False
            continue
        env_ids = runtime.get("env_ids", [])
        ledger = runtime.get("workflow_recoveries")
        if (
            runtime.get("kind") != "skill_result"
            or runtime.get("schema_version") != 2
            or not isinstance(env_ids, list)
            or any(
                type(value) is not int or value < 0 or value >= count
                for value in env_ids
            )
            or len(set(env_ids)) != len(env_ids)
            or not isinstance(ledger, list)
        ):
            for env_id in active_ids:
                known[env_id] = False
            continue
        for env_id in active_ids:
            covered[env_id] |= env_id in env_ids
            known[env_id] &= env_id in env_ids
        # Core recovery is not silently reclassified from an empty workflow ledger.
        events, failures = runtime.get("events"), runtime.get("failures")
        if (
            not isinstance(events, list)
            or not isinstance(failures, list)
            or failures
            or any(
                not isinstance(event, dict)
                or type(event.get("kind")) is not str
                or event.get("kind")
                in {"replanned", "recovery_required", "recovery_exhausted"}
                for event in events or []
            )
        ):
            for env_id in active_ids:
                known[env_id] = False
        for trace in ledger:
            traces.append(
                {
                    "segment": segment.get("name"),
                    "env_ids": list(env_ids),
                    "trace": deepcopy(trace),
                }
            )
            if not isinstance(trace, dict):
                for env_id in active_ids:
                    known[env_id] = False
                continue
            masks = trace.get("masks", {})
            valid = (
                type(trace.get("role")) is str
                and trace["role"] in roles
                and isinstance(masks, dict)
                and all(
                    isinstance(masks.get(name), list)
                    and len(masks[name]) == len(env_ids)
                    and all(type(value) is bool for value in masks[name])
                    for name in ("entered", "completed", "failed")
                )
            )
            if not valid:
                for env_id in active_ids:
                    known[env_id] = False
                continue
            for index, env_id in enumerate(env_ids):
                entered, completed, failed = (
                    masks[name][index] for name in ("entered", "completed", "failed")
                )
                if (completed or failed) and not entered or completed and failed:
                    known[env_id] = False
                    continue
                if not entered:
                    continue
                attempts[env_id] += 1
                call = trace.get("call")
                commands = call.get("command_count") if isinstance(call, dict) else None
                if completed and type(commands) is int and commands > 0:
                    recovered[env_id] += 1
                    reacquired[env_id] += int(trace["role"] == "reacquire")
                else:
                    known[env_id] = False
    return {
        "schema_version": "gen_sim.recovery-summary/v1",
        "source": "runtime.workflow_recoveries",
        "scope": "program_completion_not_taskspec_verdict",
        "original_failure_source": "execution_report.json:runtime_result.segments[].metadata.runtime",
        "environments": [
            {
                "env_id": row["env_id"],
                "program_success": successes[index],
                "certified_success": row["success"],
                "completion_kind": (
                    "not_completed"
                    if successes[index] is False
                    else (
                        "recovered_completion"
                        if successes[index] is True and recovered[index]
                        else (
                            "first_pass_completion"
                            if successes[index] is True
                            and known[index]
                            and covered[index]
                            and attempts[index] == 0
                            else "unknown"
                        )
                    )
                ),
                "workflow_recovery_attempt_count": attempts[index],
                "completed_recovery_call_count": recovered[index],
                "completed_reacquisition_call_count": reacquired[index],
            }
            for index, row in enumerate(rows)
        ],
        "workflow_recoveries": traces,
    }
