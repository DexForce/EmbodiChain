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

"""Read-only E-group outcomes derived from the existing execution trace."""

from __future__ import annotations

from copy import deepcopy
from graphlib import TopologicalSorter
from typing import Any

__all__ = ["task_group_results"]


def _row(values: Any, env_id: int | None) -> Any:
    return (
        values[env_id]
        if isinstance(values, list) and env_id is not None and env_id < len(values)
        else None
    )


def _call_result(
    node: dict[str, Any], segment: dict[str, Any], env_id: int
) -> dict[str, Any]:
    metadata = segment.get("metadata")
    runtime = metadata.get("runtime") if isinstance(metadata, dict) else None
    runtime = runtime if isinstance(runtime, dict) else {}
    env_ids = runtime.get("env_ids")
    row_index = (
        (env_ids.index(env_id) if env_id in env_ids else None)
        if isinstance(env_ids, list)
        else env_id
    )
    calls = [
        call
        for call in runtime.get("calls", [])
        if _row(call.get("masks", {}).get("entered"), row_index) is True
    ]
    failures = [
        failure
        for failure in runtime.get("failures", [])
        if _row(failure.get("env_mask"), row_index) is True
    ]
    planning_failure = next(
        (
            event
            for event in runtime.get("events", [])
            if event.get("kind") == "action_planning_failed"
            and _row(event.get("env_mask"), row_index) is True
        ),
        None,
    )
    cause = (
        {
            "phase": "planning",
            "code": planning_failure.get("failure_code"),
            "segment": planning_failure.get("segment_name"),
            "timestamp": planning_failure.get("timestamp"),
            "source": "runtime.events.action_planning_failed",
        }
        if planning_failure
        else deepcopy(failures[0]) if failures else None
    )
    return {
        "id": node["id"],
        "call": deepcopy(node.get("call")),
        "entered": _row(segment.get("active"), env_id) is True,
        "success": _row(segment.get("successes"), env_id),
        "invocation_id": calls[0].get("invocation_id") if calls else None,
        "command_count": sum(call.get("command_count", 0) for call in calls),
        "failure": deepcopy(failures[0]) if failures else None,
        "cause": cause,
        "state_after": (
            {
                "scope": "batch_state_with_explicit_row_address_not_a_projected_state",
                "env_ids": deepcopy(env_ids),
                "row_index": row_index,
                "batch": deepcopy(runtime["task_state"]),
            }
            if runtime.get("task_state") is not None and row_index is not None
            else None
        ),
        "start_step": _row(segment.get("start_steps"), env_id),
        "end_step": _row(segment.get("end_steps"), env_id),
    }


def task_group_results(
    graph: dict[str, Any], runtime_result: dict[str, Any] | None, *, num_envs: int
) -> list[dict[str, dict[str, Any]]]:
    """Summarize entered calls separately from physical/normative qualification.

    A missing segment is not a failure. A verified historical E milestone can
    qualify that E without borrowing a later final pose. Missing certificates
    never turn completed commands into a physical pass.

    Args:
        graph: Validated semantic graph and its E-group dependencies.
        runtime_result: Existing episode trace, or None when unavailable.
        num_envs: Number of globally addressed environment rows.

    Returns:
        Per-environment group mappings with execution status, qualification,
        exact call causes, and explicitly addressed batch-state evidence.
    """
    runtime = runtime_result or {}
    segments = {segment["name"]: segment for segment in runtime.get("segments", [])}
    nodes = {node["id"]: node for node in graph["nodes"]}
    evaluations = {
        row.get("env_id", index): row
        for index, row in enumerate(
            runtime.get("task_spec_evaluation", {}).get("environments", [])
        )
    }
    results = []
    groups_by_id = {str(group["id"]): group for group in graph["task_groups"]}
    ordered_ids = tuple(
        TopologicalSorter(
            {name: group.get("depends_on", []) for name, group in groups_by_id.items()}
        ).static_order()
    )
    for env_id in range(num_envs):
        groups: dict[str, dict[str, Any]] = {}
        evaluation = evaluations.get(env_id, {})
        milestones = {
            value["predicate_id"]: value for value in evaluation.get("milestones", [])
        }
        task_ready = (
            evaluation.get("certificate_ready") is True
            and evaluation.get("status") == "satisfied"
        )
        for group_id in ordered_ids:
            group = groups_by_id[group_id]
            calls = [
                _call_result(nodes[name], segments.get(name, {}), env_id)
                for name in group["node_ids"]
            ]
            failed = next(
                (
                    call["id"]
                    for call in calls
                    if call["entered"] and call["success"] is False
                ),
                None,
            )
            complete = bool(calls) and all(
                call["entered"] and call["success"] is True for call in calls
            )
            blocked = [
                name
                for name in group.get("depends_on", [])
                if groups.get(name, {}).get("execution_status") in {"failed", "blocked"}
            ]
            if failed:
                execution = "failed"
            elif complete:
                execution = "passed"
            elif not any(call["entered"] for call in calls) and blocked:
                execution = "blocked"
            else:
                execution = "unverified"
            milestone = milestones.get("milestone:" + str(group["id"]), {})
            verified = milestone.get("status") == "satisfied" or task_ready
            violated = milestone.get("status") in {"violated", "failed"}
            status = execution
            reason = {
                "failed": "entered_call_failed",
                "blocked": "predecessor_execution_failed",
                "unverified": "incomplete_or_missing_trace",
            }.get(execution, "")
            if execution == "passed":
                status = (
                    "failed" if violated else "passed" if verified else "unverified"
                )
                reason = (
                    "milestone_violated"
                    if violated
                    else (
                        "verified_milestone_or_complete_task"
                        if verified
                        else "commands_completed_without_e_qualification"
                    )
                )
            groups[str(group["id"])] = {
                "id": str(group["id"]),
                "task_type": group.get("task_type"),
                "status": status,
                "execution_status": execution,
                "reason": reason,
                "first_failed_call": failed,
                "blocked_by": blocked if execution == "blocked" else [],
                "task_certificate_ready": task_ready,
                "milestone": deepcopy(milestone) or None,
                "calls": calls,
                "evidence_scope": "existing_runtime_and_exact_historical_milestone_not_new_physical_measurement",
            }
        results.append(groups)
    return results
