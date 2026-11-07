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

"""Closed finite-count state proofs over the original normative specification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .canonicalization import canonical_hash, json_snapshot
from .condition_evidence import (
    ConditionEvidenceBinding,
    _condition_selection,
    _evaluate_condition_member,
)
from .evidence import _identifier, _digest

if TYPE_CHECKING:
    from .spec import TaskSpec
    from .expressions import PredicateEvaluation

__all__ = ["ConditionEvidenceGroupBinding", "evaluate_condition_group_evidence"]


@dataclass(frozen=True, slots=True)
class ConditionEvidenceGroupBinding:
    """Seal finite E1/E2 orientation or explicit E6 part membership.

    Each member retains an ordinary condition binding against the unchanged
    parent TaskSpec/program. A group has no synthetic child specification and
    cannot replace count selection with a subset. Existing single-role bindings
    retain their original serialization and hashes.

    Attributes:
        task_spec_digest: Identity of the unchanged full normative specification.
        program_digest: Identity of the complete original serialized program.
        predicate_id: One original goal or recorded milestone predicate.
        role_id: Finite-count subject role; an unobserved all-universe is unsupported.
        group_ids: Ordered complete covered execution groups.
        members: Existing per-member bindings retaining native/episode/occurrence identity.
        binding_hash: Identity of all members and group selection, never an old certificate.
    """

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    role_id: str
    group_ids: tuple[str, ...]
    members: tuple[ConditionEvidenceBinding, ...]
    binding_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest"):
            _digest(getattr(self, name), name)
        for name in ("predicate_id", "role_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.group_ids, (tuple, list)):
            raise TypeError("Condition group identities require an ordered sequence.")
        groups = tuple(self.group_ids)
        for value in groups:
            _identifier(value, "group identity")
        members = tuple(
            ConditionEvidenceBinding.from_dict(value) for value in self.members
        )
        if (
            not members
            or len(members) != len(groups)
            or len(set(groups)) != len(groups)
        ):
            raise ValueError(
                "A condition group requires distinct complete member/group identities."
            )
        first = members[0]
        if any(
            member.task_spec_digest != self.task_spec_digest
            or member.program_digest != self.program_digest
            or member.predicate_id != self.predicate_id
            or any(
                getattr(member, name) != getattr(first, name)
                for name in (
                    "observation_kind",
                    "evidence_kind",
                    "evidence_format",
                    "env_ids",
                    "episode_ids",
                )
            )
            for member in members
        ):
            raise ValueError(
                "Condition members have different normative, row, episode or evidence scope."
            )
        if len({value.binding_hash for value in members}) != len(members):
            raise ValueError("A condition group cannot duplicate a member binding.")
        if first.evidence_kind == "orientation":
            logical = {value.entity_id for value in members}
            native = {value.native_uid for value in members}
        else:
            logical = {
                (value.entity_id, value.condition.get("part_id")) for value in members
            }
            native = {(value.native_uid, value.condition["joint"]) for value in members}
        if len(logical) != len(members) or len(native) != len(members):
            raise ValueError(
                "A condition group cannot duplicate a logical/native member channel."
            )
        if first.observation_kind == "recorded" and len(
            {
                (value.boundary_segment_id, value.boundary_source_path)
                for value in members
            }
        ) != len(members):
            raise ValueError(
                "Recorded members require independent execution occurrences."
            )
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "members", members)
        if first.observation_kind == "final" and (
            first.boundary_segment_id is None
            or not first.boundary_source_path
            or len(
                {
                    (value.boundary_segment_id, value.boundary_source_path)
                    for value in members
                }
            )
            != 1
        ):
            raise ValueError(
                "Current count members require one actual final occurrence."
            )
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Condition group binding_hash differs from its complete declaration."
            )
        object.__setattr__(self, "binding_hash", expected)

    @property
    def observation_kind(self) -> str:
        """Return the one shared current-versus-recorded evidence contract."""
        return self.members[0].observation_kind

    @property
    def evidence_format(self) -> str:
        """Return the shared raw or historical input format."""
        return self.members[0].evidence_format

    @property
    def env_ids(self) -> tuple[int, ...]:
        """Return the unchanged ordered environment row identities."""
        return self.members[0].env_ids

    @property
    def episode_ids(self) -> tuple[int, ...] | None:
        """Return the independently bound native episode for every member."""
        return self.members[0].episode_ids

    def _payload(self) -> dict[str, Any]:
        return dict(
            schema_version="task_spec.condition_group_evidence_binding/v1",
            task_spec_digest=self.task_spec_digest,
            program_digest=self.program_digest,
            predicate_id=self.predicate_id,
            role_id=self.role_id,
            group_ids=list(self.group_ids),
            members=[value.to_dict() for value in self.members],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return owned complete membership with its original verified identity."""
        return {**self._payload(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls, value: ConditionEvidenceGroupBinding | Mapping[str, Any]
    ) -> ConditionEvidenceGroupBinding:
        """Decode independently and reject parent or nested declaration drift.

        Args:
            value: Exact composite declaration or previously constructed instance.

        Returns:
            Detached normalized group preserving each valid existing identity.
        """
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if (
            not isinstance(raw, dict)
            or raw.pop("schema_version", None)
            != "task_spec.condition_group_evidence_binding/v1"
        ):
            raise ValueError("Unsupported condition group evidence binding schema.")
        return cls(**raw)


def _selection(spec, selected, program):
    normalized, _, predicate = _condition_selection(
        spec, selected.members[0], program=program
    )
    roles = {value.name: value for value in normalized.roles}
    if (
        predicate.arguments.get("subject") != selected.role_id
        or selected.role_id not in roles
    ):
        raise ValueError("Condition group has another normative subject role.")
    role = roles[selected.role_id]
    if role.quantifier != "count" or role.count != len(selected.members):
        raise ValueError("Condition group must prove the exact declared finite count.")
    family = predicate.arguments.get("task_type")
    kind = selected.members[0].evidence_kind
    if (family, kind) not in {
        ("E1", "orientation"),
        ("E2", "orientation"),
        ("E6", "joint_endpoint"),
    }:
        raise ValueError(
            "Finite state grouping retains only the source E1/E2/E6 scope."
        )
    if (
        family in {"E1", "E2"}
        and selected.observation_kind == "recorded"
        and predicate.name != "object_upright"
    ):
        raise ValueError(
            "The source recorded object count proof requires its upright predicate."
        )
    if (
        family == "E6"
        and len(selected.members) > 1
        and any(
            not isinstance(value.condition.get("part_id"), str)
            or not value.condition["part_id"].startswith("part_")
            for value in selected.members
        )
    ):
        raise ValueError(
            "Multiple source articulation members require explicit stable part identities."
        )
    for value in selected.members[1:]:
        _condition_selection(normalized, value, program=program)
    return normalized, predicate


def evaluate_condition_group_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: ConditionEvidenceGroupBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Require every selected count member without rewriting the parent TaskSpec.

    Args:
        spec: Original immutable normative TaskSpec.
        binding: Complete finite-count subject/group/member binding.
        evidence: Owned member packets keyed by their unchanged binding hashes.
        program: Original serialized program retaining all member occurrences.

    Returns:
        Per-environment aggregate. Any missing or unavailable member remains
        unavailable; a complete measured negative is failed. Recorded members
        retain their individual original policy boundaries and become an event
        only at the latest completed member boundary in that row.
    """
    from .expressions import PredicateEvaluation, EvaluationStatus

    selected = ConditionEvidenceGroupBinding.from_dict(binding)
    normalized, predicate = _selection(spec, selected, program)
    size = len(selected.env_ids)
    reason = None
    outcomes = []
    refs = (selected.binding_hash,)
    try:
        packet = json_snapshot(evidence)
        refs += (canonical_hash(packet),)
        if (
            packet.get("schema_version") != "task_spec.condition_group_observation/v1"
            or packet.get("binding_hash") != selected.binding_hash
            or packet.get("episode_ids")
            != (None if selected.episode_ids is None else list(selected.episode_ids))
        ):
            raise ValueError(
                "Condition group packet has another declaration or episode."
            )
        packets = packet["members"]
        expected = {value.binding_hash for value in selected.members}
        if not isinstance(packets, Mapping) or set(packets) != expected:
            raise ValueError(
                "Complete condition group member evidence was not provided."
            )
        run_ids = [
            packets[value.binding_hash].get("run_id") for value in selected.members
        ]
        if packet.get("run_id") is not None or any(
            value is not None for value in run_ids
        ):
            if (
                not isinstance(packet.get("run_id"), str)
                or not packet["run_id"]
                or any(value != packet["run_id"] for value in run_ids)
            ):
                raise ValueError(
                    "Count members cannot borrow observations from another run."
                )
        if selected.observation_kind == "final":
            times = [
                packets[value.binding_hash]["state_acquisition"]["timestamp"]
                for value in selected.members
            ]
            if any(value != times[0] for value in times):
                raise ValueError(
                    "Current count members require the same final acquisition time."
                )
        if selected.observation_kind == "recorded":
            frames = [
                packets[value.binding_hash].get("boundary_frames")
                for value in selected.members
            ]
            if any(
                not isinstance(row, list)
                or len(row) != size
                or any(type(v) is not int or v <= 0 for v in row)
                for row in frames
            ):
                raise ValueError(
                    "Recorded group lacks an exact boundary for every member."
                )
            if packet.get("boundary_frames") != [
                max(row[i] for row in frames) for i in range(size)
            ]:
                raise ValueError(
                    "Recorded group must use its latest actual member boundary."
                )
        outcomes = [
            _evaluate_condition_member(
                value, predicate, packets[value.binding_hash], program=program
            )
            for value in selected.members
        ]
    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
        IndexError,
    ) as error:
        reason = str(error) or type(error).__name__
    from ._group_evidence import aggregate_member_outcomes

    return aggregate_member_outcomes(
        predicate, outcomes, size=size, refs=refs, reason=reason
    )
