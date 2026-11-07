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

"""Offline acquisition requirements selected from exact execution receipts."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from embodichain._json_contract import canonical_hash, json_snapshot
from .handover_evidence import HandOverEvidenceBinding
from .coordinated_evidence import CoordinatedHoldEvidenceBinding
from ._held_evidence import one, frames

__all__ = ["held_observation_request", "resolve_hold_observation_condition"]


def held_observation_request(
    binding: (
        HandOverEvidenceBinding | CoordinatedHoldEvidenceBinding | Mapping[str, Any]
    ),
    runtime: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Return exact acquisition declarations without physical reads.

    Args:
        binding: Runtime declaration with explicit query selection.
        runtime: Original serialized receipts and completed per-row lengths.
        program: Exact program owning the selected execution occurrences.

    Returns:
        Boundary, pose channels and fresh grip queries. Semantic acceptance is
        neither computed nor inferred from the acquisition request.
    """
    if isinstance(binding, Mapping):
        schema = binding.get("schema_version")
        cls = {
            "task_program.handover_evidence_binding/v2": HandOverEvidenceBinding,
            "task_program.coordinated_hold_evidence_binding/v2": CoordinatedHoldEvidenceBinding,
        }.get(schema)
        if cls is None:
            raise ValueError("Unsupported held acquisition declaration version.")
        selected = cls.from_dict(binding)
    elif type(binding) in (HandOverEvidenceBinding, CoordinatedHoldEvidenceBinding):
        selected = type(binding).from_dict(binding)
    else:
        raise TypeError("Unsupported held acquisition declaration.")
    runtime, program = json_snapshot(runtime), json_snapshot(program)
    if canonical_hash(program) != selected.program_digest:
        raise ValueError("Held acquisition belongs to another program.")
    count, operation = len(selected.env_ids), selected.operation
    if type(selected) is HandOverEvidenceBinding:
        from ._handover_evidence import transfer_record, holding_boundary

        record, terminal = transfer_record(
            operation,
            runtime,
            program,
            count,
            selected.observation_kind == "final",
            selected.step_dt,
        )
        if selected.query_kind == "handover_complete":
            return dict(
                object_id=operation["object_id"],
                boundary_frames=record["measurements"]["boundary_frames"],
                capture_pose=False,
                motion_parts=[],
                grips=[],
            )
        segment, _, terminal, _, part = holding_boundary(
            record, terminal, operation, runtime, program, count
        )
        parts, resources = [part], [operation["destination_resource"]]
        fresh = selected.observation_kind == "final" and selected.query_kind == "held"
        pose_kind = selected.observation_kind
    else:
        from ._coordinated_evidence import holding

        _, segment, _, terminal, slots, _, _ = holding(
            operation,
            runtime,
            program,
            count,
            selected.step_dt,
            selected.observation_kind == "final",
        )
        parts = [slots[role][1] for role in ("left", "right")]
        resources = [operation["resources"][role] for role in ("left", "right")]
        fresh, pose_kind = True, "final"
    grips = []
    if fresh:
        for resource in resources:
            expectation = one(
                item
                for item in terminal["effect_spec"]["state_expectations"]
                if item["object_id"] == operation["object_id"]
                and item["resource_id"] == resource
                and item["relation"] == "attached"
            )
            clause = one(
                item
                for item in terminal["effect_spec"]["clauses"]
                if item["expectation_id"] == expectation["expectation_id"]
                and item.get("evidence_kind") == "constraint"
            )
            grips.append(
                dict(
                    resource_id=resource,
                    control_part=clause["source"]["address"]["control_part"],
                    observation_revision=terminal["observation_revision"] + 1,
                )
            )
    return dict(
        object_id=operation["object_id"],
        boundary_frames=frames(segment, count),
        capture_pose=True,
        motion_parts=parts,
        pose_observation_kind=pose_kind,
        grips=grips,
    )


def resolve_hold_observation_condition(
    declaration: Mapping[str, Any], *, motion_parts_count: int
) -> dict[str, Any]:
    """Resolve the existing held-observation defaults from a frozen declaration.

    Args:
        declaration: Explicit owner hold condition, never an execution receipt.
        motion_parts_count: Required number of declared participating motion parts.

    Returns:
        Validated owner parameters with the existing world-up axis made explicit.
    """
    from ._handover_evidence import hold_condition

    selected = hold_condition(
        json_snapshot(declaration), parts_count=motion_parts_count
    )
    result = json_snapshot(vars(selected))
    if result["local_axis"] is not None and result["world_axis"] is None:
        result["world_axis"] = [0.0, 0.0, 1.0]
    return result
