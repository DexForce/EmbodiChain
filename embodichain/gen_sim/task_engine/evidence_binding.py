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

"""Compose runtime observation declarations with independently sealed criteria."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from embodichain.gen_sim.task_spec import TaskSpec, canonical_hash, json_snapshot
from embodichain.gen_sim.task_spec.manipulation_evidence import (
    HoldAcceptanceCriteria,
    ManipulationEvidenceBinding,
)
from embodichain.lab.task_program.integrations.evidence import (
    HandOverEvidenceBinding,
    CoordinatedHoldEvidenceBinding,
    CoordinatedTransportEvidenceBinding,
    normalize_handover_evidence,
    normalize_coordinated_hold_evidence,
    normalize_coordinated_transport_evidence,
)

if TYPE_CHECKING:
    from embodichain.gen_sim.task_spec.validation import EvaluationReport

__all__ = ["TaskEngineManipulationBinding", "evaluate_task_engine_evidence"]

_RUNTIME_TYPES = (
    HandOverEvidenceBinding,
    CoordinatedHoldEvidenceBinding,
    CoordinatedTransportEvidenceBinding,
)
_RUNTIME_SCHEMAS = {
    "task_program.handover_evidence_binding/v2": HandOverEvidenceBinding,
    "task_program.coordinated_hold_evidence_binding/v2": CoordinatedHoldEvidenceBinding,
    "task_program.coordinated_transport_evidence_binding/v2": CoordinatedTransportEvidenceBinding,
}


@dataclass(frozen=True, slots=True)
class TaskEngineManipulationBinding:
    """Pair a Lab observation declaration and its explicit TaskSpec acceptance.

    Runtime selection and normative limits have distinct identities. The outer
    declaration seals their association, including both complete snapshots.
    """

    runtime: (
        HandOverEvidenceBinding
        | CoordinatedHoldEvidenceBinding
        | CoordinatedTransportEvidenceBinding
    )
    acceptance: ManipulationEvidenceBinding
    binding_hash: str = ""

    def __post_init__(self) -> None:
        if type(self.runtime) not in _RUNTIME_TYPES:
            raise ValueError(
                "A manipulation declaration requires a supported Lab protocol."
            )
        runtime = type(self.runtime).from_dict(self.runtime)
        acceptance = ManipulationEvidenceBinding.from_dict(self.acceptance)
        for name in (
            "task_spec_digest",
            "program_digest",
            "predicate_id",
            "observation_kind",
            "env_ids",
            "episode_ids",
            "evidence_format",
        ):
            if getattr(runtime, name) != getattr(acceptance, name):
                raise ValueError(
                    "Runtime observation and normative acceptance scopes differ."
                )
        kind = (
            {
                "handover_complete": "handover_complete",
                "held": "receiver_hold",
                "orientation": "receiver_orientation",
            }[runtime.query_kind]
            if isinstance(runtime, HandOverEvidenceBinding)
            else (
                "coordinated_hold"
                if isinstance(runtime, CoordinatedHoldEvidenceBinding)
                else "coordinated_acquisition"
            )
        )
        if (
            acceptance.source_binding_hash != runtime.binding_hash
            or acceptance.kind != kind
        ):
            raise ValueError(
                "Normative acceptance names another runtime observation declaration."
            )
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "acceptance", acceptance)
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Manipulation deployment binding differs from its sealed identity."
            )
        object.__setattr__(self, "binding_hash", expected)

    @property
    def predicate_id(self) -> str:
        """Return the exact predicate shared by both declarations."""
        return self.acceptance.predicate_id

    def _payload(self) -> dict[str, Any]:
        return dict(
            schema_version="task_engine.manipulation_evidence_binding/v1",
            runtime=self.runtime.to_dict(),
            acceptance=self.acceptance.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return independent complete observation and acceptance declarations."""
        return {**self._payload(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls, value: TaskEngineManipulationBinding | Mapping[str, Any]
    ) -> TaskEngineManipulationBinding:
        """Validate a declaration without updating stale nested identities.

        Args:
            value: Existing declaration or its complete serialized snapshot.

        Returns:
            Independently owned, identity-checked declaration.
        """
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if (
            raw.pop("schema_version", None)
            != "task_engine.manipulation_evidence_binding/v1"
        ):
            raise ValueError(
                "Unsupported manipulation deployment binding; regenerate the bundle."
            )
        runtime = raw.pop("runtime")
        protocol = _RUNTIME_SCHEMAS.get(runtime.get("schema_version"))
        if protocol is None:
            raise ValueError(
                "Unsupported runtime evidence protocol; regenerate the bundle."
            )
        return cls(
            runtime=protocol.from_dict(runtime),
            acceptance=ManipulationEvidenceBinding.from_dict(raw.pop("acceptance")),
            **raw,
        )


def _acceptance(
    runtime: Any, kind: str, criteria: HoldAcceptanceCriteria | Mapping[str, Any] | None
) -> TaskEngineManipulationBinding:
    # Called during declaration projection only. Never infer task criteria from
    # runtime receipts, monitor outputs, names or a successful execution result.
    if criteria is None and kind in {
        "receiver_hold",
        "receiver_orientation",
        "coordinated_hold",
    }:
        condition = runtime.operation["hold_condition"]
        required = {
            "translation_drift",
            "rotation_drift",
            "duration",
            "target_position",
            "position_tolerance",
            "local_axis",
            "world_axis",
            "minimum_alignment",
        }
        if not isinstance(condition, dict) or not required.issubset(condition):
            raise ValueError(
                "Held acceptance needs explicit criteria or a complete frozen owner condition."
            )
        from embodichain.lab.task_program.integrations.evidence import (
            resolve_hold_observation_condition,
        )

        condition = resolve_hold_observation_condition(
            condition,
            motion_parts_count=1 if isinstance(runtime, HandOverEvidenceBinding) else 2,
        )
        target = condition["target_position"]
        axis = condition["local_axis"]
        criteria = HoldAcceptanceCriteria(
            condition["translation_drift"],
            condition["rotation_drift"],
            hold_seconds=condition["duration"],
            target_position=target,
            position_tolerance=(
                condition["position_tolerance"] if target is not None else None
            ),
            local_axis=axis,
            world_axis=condition["world_axis"] if axis is not None else None,
            minimum_alignment=(
                condition["minimum_alignment"] if axis is not None else None
            ),
        )
    if criteria is not None:
        criteria = HoldAcceptanceCriteria.from_dict(criteria)
    return TaskEngineManipulationBinding(
        runtime,
        ManipulationEvidenceBinding(
            runtime.task_spec_digest,
            runtime.program_digest,
            runtime.predicate_id,
            kind,
            runtime.observation_kind,
            runtime.env_ids,
            runtime.episode_ids,
            criteria,
            runtime.binding_hash,
            evidence_format=runtime.evidence_format,
        ),
    )


def _normalize(
    spec: TaskSpec, binding: Any, packet: Mapping[str, Any], program: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    from embodichain.gen_sim.task_spec.evidence import SupportedPlacementEvidenceBinding
    from embodichain.gen_sim.task_spec.placement_group_evidence import (
        PlacementEvidenceGroupBinding,
    )
    from embodichain.gen_sim.task_spec.pour_evidence import PourEvidenceBinding
    from embodichain.lab.task_program.integrations.evidence import (
        normalize_placement_evidence,
        normalize_pour_release_evidence,
    )

    if isinstance(binding, TaskEngineManipulationBinding):
        selected = TaskEngineManipulationBinding.from_dict(binding)
        normalizer = {
            HandOverEvidenceBinding: normalize_handover_evidence,
            CoordinatedHoldEvidenceBinding: normalize_coordinated_hold_evidence,
            CoordinatedTransportEvidenceBinding: normalize_coordinated_transport_evidence,
        }[type(selected.runtime)]
        return selected.acceptance, normalizer(
            selected.runtime, packet, program=program
        )
    if isinstance(binding, SupportedPlacementEvidenceBinding):
        predicates = (*spec.goal, *(p for m in spec.milestones for p in m.achieve))
        predicate = next(p for p in predicates if p.id == binding.predicate_id)
        return binding, normalize_placement_evidence(
            binding.to_dict(),
            packet,
            program=program,
            source_family=predicate.arguments.get("task_type"),
        )
    if isinstance(binding, PlacementEvidenceGroupBinding):
        data = json_snapshot(packet)
        if isinstance(data.get("members"), dict):
            for member in binding.members:
                if member.binding_hash in data["members"]:
                    data["members"][member.binding_hash] = _normalize(
                        spec, member, data["members"][member.binding_hash], program
                    )[1]
        return binding, data
    if isinstance(binding, PourEvidenceBinding):
        return binding, normalize_pour_release_evidence(
            binding.to_dict(), packet, program=program
        )
    return binding, json_snapshot(packet)


def evaluate_task_engine_evidence(
    spec: TaskSpec | Mapping[str, Any],
    bindings: Sequence[Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
    num_envs: int,
) -> tuple[EvaluationReport, ...]:
    """Normalize archived runtime facts and delegate task judgments to TaskSpec.

    Args:
        spec: Frozen normative specification.
        bindings: Explicit runtime/acceptance declarations, or other supported bindings.
        evidence: Raw packets keyed by predicate identity.
        program: Complete immutable program declaration.
        num_envs: Exact number of environment rows.

    Returns:
        Per-environment TaskSpec reports, preserving failed and unavailable states.
    """
    from embodichain.gen_sim.task_spec.evidence_report import (
        evaluate_physical_evidence_report,
    )

    normalized = TaskSpec.from_dict(spec)
    pairs = [
        _normalize(normalized, binding, evidence.get(binding.predicate_id, {}), program)
        for binding in bindings
    ]
    return evaluate_physical_evidence_report(
        normalized,
        [p[0] for p in pairs],
        {p[0].predicate_id: p[1] for p in pairs},
        program=program,
        num_envs=num_envs,
    )
