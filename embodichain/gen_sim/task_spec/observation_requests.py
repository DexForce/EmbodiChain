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

"""Pure requests for physical acquisition from sealed held-object declarations."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING
from .canonicalization import canonical_hash, json_snapshot
from .handover_evidence import HandOverEvidenceBinding, evaluate_handover_evidence
from .coordinated_evidence import (
    CoordinatedHoldEvidenceBinding,
    evaluate_coordinated_hold_evidence,
)

if TYPE_CHECKING:
    from .spec import TaskSpec

__all__ = ["held_observation_request"]


def held_observation_request(
    spec: TaskSpec | Mapping[str, Any],
    binding: (
        HandOverEvidenceBinding | CoordinatedHoldEvidenceBinding | Mapping[str, Any]
    ),
    runtime: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Select exact source receipt endpoints and query ordinals without Lab reads.

    Args:
        spec: Frozen normative task specification.
        binding: Exact handover or standalone coordinated-hold declaration.
        runtime: Original serialized execution receipts and per-row lengths.
        program: Complete original program matching the declaration identity.

    Returns:
        Plain request containing the exact boundary, motion parts and fresh grip
        query revisions. An event-only transfer requests no pose or grip capture.
        This is not a physical success report.
    """
    from .spec import TaskSpec
    from ._held_evidence import one, frames

    normalized = TaskSpec.from_dict(spec)
    runtime = json_snapshot(runtime)
    program = json_snapshot(program)
    if isinstance(binding, Mapping):
        schema = binding.get("schema_version")
        cls = (
            HandOverEvidenceBinding
            if schema == "task_spec.handover_evidence_binding/v1"
            else (
                CoordinatedHoldEvidenceBinding
                if schema == "task_spec.coordinated_hold_evidence_binding/v1"
                else None
            )
        )
        if cls is None:
            raise ValueError("Unsupported held-observation binding schema.")
        selected = cls.from_dict(binding)
    elif type(binding) in (HandOverEvidenceBinding, CoordinatedHoldEvidenceBinding):
        selected = type(binding).from_dict(binding)
    else:
        raise TypeError("Held observation requires an exact supported binding.")
    if (
        canonical_hash(normalized.to_dict()) != selected.task_spec_digest
        or canonical_hash(program) != selected.program_digest
    ):
        raise ValueError(
            "Held observation request has another normative/program identity."
        )
    predicate = one(
        item
        for item in (
            *normalized.goal,
            *(
                item
                for milestone in normalized.milestones
                for item in milestone.achieve
            ),
        )
        if item.id == selected.predicate_id
    )
    count = len(selected.env_ids)
    operation = selected.operation
    if type(selected) is HandOverEvidenceBinding:
        evaluate_handover_evidence(normalized, selected, {}, program=program)
        from ._handover_evidence import transfer_record, holding_boundary

        record, terminal = transfer_record(
            operation,
            runtime,
            program,
            count,
            selected.observation_kind == "final",
            selected.step_dt,
        )
        if predicate.name == "handover_complete":
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
        parts = [part]
        resources = [operation["destination_resource"]]
        fresh = selected.observation_kind == "final" and predicate.name == "held"
        pose_kind = selected.observation_kind
    else:
        evaluate_coordinated_hold_evidence(normalized, selected, {}, program=program)
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
        fresh = True
        pose_kind = "final"
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
