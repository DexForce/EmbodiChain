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

"""Common TaskSpec reporting for independently verified physical evidence."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING
from .canonicalization import json_snapshot
from .evidence import (
    SupportedPlacementEvidenceBinding,
    evaluate_supported_placement_evidence,
)
from .condition_evidence import ConditionEvidenceBinding, evaluate_condition_evidence
from .condition_group_evidence import (
    ConditionEvidenceGroupBinding,
    evaluate_condition_group_evidence,
)
from .control_evidence import ControlEvidenceBinding, evaluate_control_evidence
from .pour_evidence import PourEvidenceBinding, evaluate_pour_evidence
from .handover_evidence import HandOverEvidenceBinding, evaluate_handover_evidence
from .coordinated_evidence import (
    CoordinatedHoldEvidenceBinding,
    evaluate_coordinated_hold_evidence,
)

if TYPE_CHECKING:
    from .spec import TaskSpec
    from .validation import EvaluationReport

__all__ = ["evaluate_physical_evidence_report"]


def _binding(
    value: Any,
) -> (
    SupportedPlacementEvidenceBinding
    | ConditionEvidenceBinding
    | ConditionEvidenceGroupBinding
    | ControlEvidenceBinding
    | PourEvidenceBinding
    | HandOverEvidenceBinding
    | CoordinatedHoldEvidenceBinding
):
    if isinstance(value, SupportedPlacementEvidenceBinding):
        return SupportedPlacementEvidenceBinding.from_dict(value)
    if isinstance(value, ConditionEvidenceGroupBinding):
        return ConditionEvidenceGroupBinding.from_dict(value)
    if isinstance(value, ConditionEvidenceBinding):
        return ConditionEvidenceBinding.from_dict(value)
    if isinstance(value, ControlEvidenceBinding):
        return ControlEvidenceBinding.from_dict(value)
    if isinstance(value, PourEvidenceBinding):
        return PourEvidenceBinding.from_dict(value)
    if isinstance(value, HandOverEvidenceBinding):
        return HandOverEvidenceBinding.from_dict(value)
    if isinstance(value, CoordinatedHoldEvidenceBinding):
        return CoordinatedHoldEvidenceBinding.from_dict(value)
    if isinstance(value, Mapping):
        version = value.get("schema_version")
        if version == "task_spec.placement_evidence_binding/v2":
            return SupportedPlacementEvidenceBinding.from_dict(value)
        if version == "task_spec.condition_group_evidence_binding/v1":
            return ConditionEvidenceGroupBinding.from_dict(value)
        if version == "task_spec.condition_evidence_binding/v1":
            return ConditionEvidenceBinding.from_dict(value)
        if version == "task_spec.control_evidence_binding/v1":
            return ControlEvidenceBinding.from_dict(value)
        if version == "task_spec.coordinated_hold_evidence_binding/v1":
            return CoordinatedHoldEvidenceBinding.from_dict(value)
        if version == "task_spec.handover_evidence_binding/v1":
            return HandOverEvidenceBinding.from_dict(value)
        if version == "task_spec.pour_evidence_binding/v1":
            return PourEvidenceBinding.from_dict(value)
    raise ValueError(
        "Unsupported physical evidence binding schema; rebuild the deployment declaration."
    )


def evaluate_physical_evidence_report(
    spec: TaskSpec | Mapping[str, Any],
    bindings: Sequence[
        SupportedPlacementEvidenceBinding
        | ConditionEvidenceBinding
        | ConditionEvidenceGroupBinding
        | ControlEvidenceBinding
        | PourEvidenceBinding
        | HandOverEvidenceBinding
        | CoordinatedHoldEvidenceBinding
        | Mapping[str, Any]
    ],
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    program: Mapping[str, Any],
    num_envs: int,
) -> tuple[EvaluationReport, ...]:
    """Combine verified predicate families without inferring missing observations.

    Args:
        spec: Frozen normative specification.
        bindings: Unique placement/state/control declarations, sharing one
            row, episode and raw-or-historical input contract.
        evidence: Physical/execution snapshots keyed by predicate identity.
        program: Exact serialized program identified by every binding.
        num_envs: Ordered native environment count.

    Returns:
        Existing TaskSpec reports for each environment. Final state cannot fill
        historical milestones, and sparse observations cannot establish temporal
        windows, invariants or predicates outside the declared evidence families.
    """
    from .spec import TaskSpec
    from .validation import evaluate_task_spec

    normalized = TaskSpec.from_dict(spec)
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer.")
    selected = tuple(_binding(item) for item in bindings)
    if len({item.predicate_id for item in selected}) != len(selected):
        raise ValueError("Evidence bindings must identify unique predicates.")
    if any(item.env_ids != tuple(range(num_envs)) for item in selected):
        raise ValueError("Evidence bindings have another environment batch.")
    if len({item.evidence_format for item in selected}) > 1:
        raise ValueError(
            "One report cannot mix raw and historical evidence input contracts."
        )
    if len({item.episode_ids for item in selected}) > 1:
        raise ValueError(
            "Evidence bindings cannot mix declared or historical episodes."
        )
    goals = {item.id: item for item in normalized.goal}
    milestones = {
        item.id: item
        for milestone in normalized.milestones
        for item in milestone.achieve
    }
    goal_rows: list[dict[str, Any]] = [{} for _ in range(num_envs)]
    events: list[dict[int, dict[str, Any]]] = [{} for _ in range(num_envs)]

    def store_observation(
        target: dict[str, Any], key: str, packet: dict[str, Any]
    ) -> None:
        previous = target.get(key)
        if previous is not None:
            if any(
                previous.get(field) != packet.get(field)
                for field in ("status", "value")
            ):
                raise ValueError(
                    "Predicate bindings assign conflicting observations to one key."
                )
            packet["evidence_refs"] = sorted(
                set(previous["evidence_refs"]) | set(packet["evidence_refs"])
            )
        target[key] = packet

    for binding in selected:
        snapshot = json_snapshot(evidence.get(binding.predicate_id, {}))
        verifier = {
            SupportedPlacementEvidenceBinding: evaluate_supported_placement_evidence,
            ConditionEvidenceBinding: evaluate_condition_evidence,
            ConditionEvidenceGroupBinding: evaluate_condition_group_evidence,
            ControlEvidenceBinding: evaluate_control_evidence,
            PourEvidenceBinding: evaluate_pour_evidence,
            HandOverEvidenceBinding: evaluate_handover_evidence,
            CoordinatedHoldEvidenceBinding: evaluate_coordinated_hold_evidence,
        }[type(binding)]
        outcomes = verifier(normalized, binding, snapshot, program=program)
        predicate = goals.get(
            binding.predicate_id, milestones.get(binding.predicate_id)
        )
        assert predicate is not None  # The public verifier checked this identity.
        frame_indices = None
        if binding.predicate_id in milestones:
            frames = snapshot.get("boundary_frames")
            if frames is None:
                frames = (
                    snapshot.get("record", {})
                    .get("measurements", {})
                    .get("boundary_frames")
                )
            if (
                isinstance(frames, list)
                and len(frames) == num_envs
                and all(type(frame) is int and frame > 0 for frame in frames)
            ):
                frame_indices = frames
        for row, outcome in enumerate(outcomes):
            packet = {
                "status": outcome.status.value,
                "value": outcome.observed,
                "reason": outcome.reason,
                "evidence_refs": list(outcome.evidence_refs),
            }
            key = predicate.observation or predicate.id
            if binding.predicate_id in goals:
                store_observation(goal_rows[row], key, packet)
            elif frame_indices is not None:
                store_observation(
                    events[row].setdefault(frame_indices[row], {}), key, packet
                )
    reports = []
    for row in range(num_envs):
        trajectory = [snapshot for _, snapshot in sorted(events[row].items())]
        for condition in normalized.temporal:
            for snapshot in trajectory:
                snapshot[condition.predicate.observation or condition.predicate.id] = {
                    "status": "unavailable",
                    "reason": "No complete control-step trajectory.",
                }
        reports.append(
            evaluate_task_spec(
                normalized,
                {"init": {}, "goal": goal_rows[row], "invariants": {}},
                trajectory=trajectory,
            )
        )
    return tuple(reports)
