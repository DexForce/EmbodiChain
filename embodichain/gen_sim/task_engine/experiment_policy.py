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

"""Explicit GenSim experiment policy, independent of task and scene identities."""

from __future__ import annotations

from collections.abc import Mapping

from typing import Any

__all__: list[str] = []

CONTACT_PROFILE = "contact_experiment"

COMPOSITION_PROFILE = "composition_experiment"

AUTO_PROFILE = "auto"


def validate_native_pick_exploration(
    value: bool, robot_profile: str | None = None
) -> None:
    """Keep the explicit simulation trial separate from other robot defaults."""
    if type(value) is not bool:
        raise TypeError("native_pick_exploration must be a boolean.")
    if (
        value
        and robot_profile is not None
        and robot_profile not in {"dual_ur10", "dual_ur10_robotiq_arg2f_140"}
    ):
        raise ValueError("Native Pick exploration is supported only for dual UR10.")


def validate_profile(value: str) -> None:
    if value not in {"default", AUTO_PROFILE, CONTACT_PROFILE, COMPOSITION_PROFILE}:
        raise ValueError(
            "runtime_profile must be default, auto, contact_experiment or composition_experiment."
        )


def resolve_profile(value: str, graph: Mapping[str, Any]) -> str:
    """Select one bundle-owned policy from recipe semantics, never object identities."""
    validate_profile(value)
    if value != AUTO_PROFILE:
        return value
    pouring = any(
        node.get("task_type") == "E3"
        or node.get("call", {}).get("call_id") == "simulation.pour"
        for node in graph["nodes"]
    )
    if pouring:
        return COMPOSITION_PROFILE
    if all(node.get("task_type") in {"E1", "E2", "E6"} for node in graph["nodes"]):
        return "default"
    return CONTACT_PROFILE


def experimental_release_enabled(profile: str, override: bool | None) -> bool:
    """Keep historical bundles readable while allowing an explicit release gate."""
    validate_profile(profile)
    if override is not None and type(override) is not bool:
        raise TypeError("experimental_open_release must be a boolean or None.")
    return (
        profile in {CONTACT_PROFILE, COMPOSITION_PROFILE}
        if override is None
        else override
    )


def program_completion_mask(
    runtime: Mapping[str, Any] | None, count: int
) -> list[bool]:
    """Require complete measured execution, not a TaskSpec certificate."""
    if not runtime or count != 1:
        return [False] * count
    segments = runtime.get("segments") or []
    accepted = (
        runtime.get("completed") is True
        and runtime.get("completed_by_env") == [True]
        and runtime.get("success") == [True]
        and len(runtime.get("lengths", [])) == 1
        and type(runtime["lengths"][0]) is int
        and runtime["lengths"][0] > 0
        and bool(segments)
        and all(
            segment.get("success") is True
            and segment.get("metadata", {}).get("segment_count") == len(segments)
            and not segment.get("metadata", {}).get("runtime", {}).get("failures")
            and segment.get("metadata", {}).get("validation", {}).get("accepted_mask")
            == [True]
            for segment in segments
        )
    )
    return [bool(accepted)]


def program_completion_summary(
    runtime: Mapping[str, Any] | None,
    strict_report: Mapping[str, Any],
    experiment_report: Mapping[str, Any],
    *,
    acceptance_mode: str,
) -> dict[str, Any]:
    """Report both outcomes from one execution without altering its selected verdict."""
    environments = strict_report["environments"]
    count = len(environments)
    completed = program_completion_mask(runtime, count)
    experiment_rows = experiment_report["environments"]
    certificates = {
        row.get("env_id", index): row.get("certificate_ready") is True
        and row.get("status") == "satisfied"
        for index, row in enumerate(
            ((runtime or {}).get("task_spec_evaluation") or {}).get("environments", [])
        )
    }
    return {
        "schema_version": "gen_sim.program-completion-summary/v1",
        "selected_acceptance_mode": acceptance_mode,
        "experimental_program_completed_by_env": [
            value
            and experiment_report.get("failure") is None
            and row["success"] is True
            and all(flag is True for flag in row.get("semantic_success", {}).values())
            for value, row in zip(completed, experiment_rows, strict=True)
        ],
        "strict_execution_success_by_env": [row["success"] for row in environments],
        "task_spec_certificate_ready_by_env": [
            certificates.get(row.get("env_id", index), False)
            for index, row in enumerate(environments)
        ],
        "affects_selected_success": False,
    }
