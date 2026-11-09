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

"""Closed finite-count placement proofs over the original normative specification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .canonicalization import canonical_hash, json_snapshot
from .evidence import (
    SupportedPlacementEvidenceBinding,
    _placement_selection,
    _evaluate_placement_member,
    _identifier,
    _digest,
)

if TYPE_CHECKING:
    from .spec import TaskSpec
    from .expressions import PredicateEvaluation

__all__ = ["PlacementEvidenceGroupBinding", "evaluate_placement_group_evidence"]


@dataclass(frozen=True, slots=True)
class PlacementEvidenceGroupBinding:
    """Seal every rigid subject of an original finite E1 placement predicate.

    Each member retains an ordinary placement binding against the unchanged
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
    members: tuple[SupportedPlacementEvidenceBinding, ...]
    binding_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest"):
            _digest(getattr(self, name), name)
        for name in ("predicate_id", "role_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.group_ids, (tuple, list)):
            raise TypeError("Placement group identities require an ordered sequence.")
        groups = tuple(self.group_ids)
        for value in groups:
            _identifier(value, "group identity")
        members = tuple(
            SupportedPlacementEvidenceBinding.from_dict(value) for value in self.members
        )
        if (
            not members
            or len(members) != len(groups)
            or len(set(groups)) != len(groups)
        ):
            raise ValueError(
                "A placement group requires distinct complete member/group identities."
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
                    "evidence_format",
                    "env_ids",
                    "episode_ids",
                )
            )
            for member in members
        ):
            raise ValueError(
                "Placement members have different normative, row, episode or evidence scope."
            )
        if len({value.binding_hash for value in members}) != len(members):
            raise ValueError("A placement group cannot duplicate a member binding.")
        if len({value.object_id for value in members}) != len(members):
            raise ValueError("A placement group cannot duplicate a subject.")
        if len({value.reference_id for value in members}) != 1:
            raise ValueError(
                "A placement group requires one complete reference member."
            )
        native = [
            (
                value.planar_geometry["object_native_uid"]
                if value.planar_geometry is not None
                else (
                    value.directional_geometry["object_native_uid"]
                    if value.directional_geometry is not None
                    else None
                )
            )
            for value in members
        ]
        if all(value is not None for value in native) and len(set(native)) != len(
            native
        ):
            raise ValueError("A placement group cannot duplicate a native subject.")
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
                "Placement group binding_hash differs from its complete declaration."
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
            schema_version="task_spec.placement_group_evidence_binding/v1",
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
        cls, value: PlacementEvidenceGroupBinding | Mapping[str, Any]
    ) -> PlacementEvidenceGroupBinding:
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
            != "task_spec.placement_group_evidence_binding/v1"
        ):
            raise ValueError("Unsupported placement group evidence binding schema.")
        return cls(**raw)


def _selection(spec, selected, program):
    normalized, _, predicate = _placement_selection(
        spec, selected.members[0], program=program
    )
    roles = {value.name: value for value in normalized.roles}
    if (
        predicate.arguments.get("subject") != selected.role_id
        or selected.role_id not in roles
    ):
        raise ValueError("Placement group has another normative subject role.")
    if roles[selected.role_id].quantifier != "count" or roles[
        selected.role_id
    ].count != len(selected.members):
        raise ValueError("Placement group must prove the exact declared finite count.")
    if not roles[predicate.arguments["reference"]].is_single_member:
        raise ValueError("Finite E1 grouping retains its single complete reference.")
    if predicate.arguments.get("task_type") != "E1":
        raise ValueError("Finite placement grouping retains only the source E1 scope.")
    for value in selected.members[1:]:
        _placement_selection(normalized, value, program=program)
    return predicate


def evaluate_placement_group_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: PlacementEvidenceGroupBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Verify complete finite E1 subjects without reducing the original TaskSpec.

    Args:
        spec: Original full normative specification with a finite subject count.
        binding: Every original placement binding and group occurrence.
        evidence: Owned member packets keyed by unchanged member binding hashes.
        program: Full original serialized program.

    Returns:
        Per-environment facts retaining unavailable members and complete measured
        negatives. Normative milestone ordering remains with TaskSpec; a physical
        aggregate cannot erase earlier incomplete member event slots.
    """
    from ._group_evidence import aggregate_member_outcomes

    selected = PlacementEvidenceGroupBinding.from_dict(binding)
    predicate = _selection(spec, selected, program)
    size = len(selected.env_ids)
    reason, outcomes = None, []
    refs = (selected.binding_hash,)
    try:
        packet = json_snapshot(evidence)
        refs += (canonical_hash(packet),)
        if (
            packet.get("schema_version") != "task_spec.placement_group_observation/v1"
            or packet.get("binding_hash") != selected.binding_hash
            or packet.get("episode_ids")
            != (None if selected.episode_ids is None else list(selected.episode_ids))
        ):
            raise ValueError(
                "Placement group packet has another declaration or episode."
            )
        packets = packet["members"]
        if not isinstance(packets, Mapping) or set(packets) != {
            value.binding_hash for value in selected.members
        }:
            raise ValueError(
                "Complete placement group member evidence was not provided."
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
                    "Placement members cannot borrow another run's observations."
                )
        frames = [
            packets[value.binding_hash].get("boundary_frames")
            for value in selected.members
        ]
        if (
            selected.observation_kind == "recorded"
            or selected.evidence_format == "raw_snapshot"
        ):
            if any(
                not isinstance(row, list)
                or len(row) != size
                or any(type(value) is not int or value < 0 for value in row)
                for row in frames
            ):
                raise ValueError("Placement group lacks its exact member boundaries.")
            if selected.observation_kind == "recorded":
                if packet.get("boundary_frames") != [
                    max(row[i] for row in frames) for i in range(size)
                ]:
                    raise ValueError(
                        "Placement group must retain the latest actual member boundary."
                    )
            elif any(row != frames[0] for row in frames):
                raise ValueError(
                    "Current placement members have different final boundaries."
                )
        outcomes = [
            _evaluate_placement_member(
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
    return aggregate_member_outcomes(
        predicate, outcomes, size=size, refs=refs, reason=reason
    )
