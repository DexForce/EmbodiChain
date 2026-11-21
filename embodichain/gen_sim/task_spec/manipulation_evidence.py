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

"""Normative manipulation criteria over normalized, provider-independent facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any

from .canonicalization import canonical_hash, json_snapshot
from .expressions import EvaluationStatus, PredicateEvaluation
from .spec import TaskSpec

__all__ = [
    "HoldAcceptanceCriteria",
    "ManipulationEvidenceBinding",
    "evaluate_manipulation_evidence",
]


def _number(value: Any, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Manipulation measurements and bounds must be finite numbers.")
    if value < 0 or (positive and value == 0):
        raise ValueError("Manipulation bounds have an invalid sign.")
    return float(value)


def _identifier(value: Any) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("Evidence identity requires exact nonempty strings.")


@dataclass(frozen=True, slots=True)
class HoldAcceptanceCriteria:
    """Explicit acceptance limits, independent of an executor's monitors.

    Attributes:
        translation_limit: Maximum attachment translation drift in metres.
        rotation_limit: Maximum attachment rotation drift in radians.
        hold_seconds: Required completely observed stable interval in seconds.
        target_position: Optional world position in metres.
        position_tolerance: Position error bound, required with a target.
        local_axis: Optional object-local orientation axis.
        world_axis: Corresponding world orientation axis.
        minimum_alignment: Minimum dot product for an orientation requirement.
    """

    translation_limit: float
    rotation_limit: float
    hold_seconds: float = 0.0
    target_position: tuple[float, float, float] | None = None
    position_tolerance: float | None = None
    local_axis: tuple[float, float, float] | None = None
    world_axis: tuple[float, float, float] | None = None
    minimum_alignment: float | None = None

    def __post_init__(self) -> None:
        for name in ("translation_limit", "rotation_limit"):
            object.__setattr__(self, name, _number(getattr(self, name), positive=True))
        object.__setattr__(self, "hold_seconds", _number(self.hold_seconds))
        if (self.target_position is None) != (self.position_tolerance is None):
            raise ValueError("A position requirement needs its target and tolerance.")
        if self.position_tolerance is not None:
            object.__setattr__(
                self,
                "position_tolerance",
                _number(self.position_tolerance, positive=True),
            )
        for name in ("target_position", "local_axis", "world_axis"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in value
                )
            ):
                raise ValueError(
                    "Position and axis requirements need three finite coordinates."
                )
            vector = tuple(float(x) for x in value)
            if name != "target_position":
                length = math.sqrt(sum(x * x for x in vector))
                if length == 0:
                    raise ValueError("An orientation axis cannot be zero.")
                if not math.isclose(length, 1.0, rel_tol=1e-15, abs_tol=0.0):
                    vector = tuple(x / length for x in vector)
            object.__setattr__(self, name, vector)
        if any(
            x is not None
            for x in (self.local_axis, self.world_axis, self.minimum_alignment)
        ):
            if (
                self.local_axis is None
                or self.world_axis is None
                or self.minimum_alignment is None
            ):
                raise ValueError(
                    "An orientation requirement needs both axes and its bound."
                )
            alignment = _number(self.minimum_alignment)
            if alignment > 1:
                raise ValueError("Minimum alignment must be in [0, 1].")
            object.__setattr__(self, "minimum_alignment", alignment)

    def to_dict(self) -> dict[str, Any]:
        """Return independent, complete criteria suitable for identity sealing."""
        return json_snapshot(asdict(self))

    @classmethod
    def from_dict(
        cls, value: HoldAcceptanceCriteria | Mapping[str, Any]
    ) -> HoldAcceptanceCriteria:
        """Own a criteria snapshot without reading execution or sensor state."""
        return cls(
            **(value.to_dict() if isinstance(value, cls) else json_snapshot(value))
        )


@dataclass(frozen=True, slots=True)
class ManipulationEvidenceBinding:
    """Bind one predicate to explicit criteria and a normalized evidence scope.

    No native actors, controller parts, SDK schemas or skill phases participate
    in this contract. Program identity is an opaque content identity.
    """

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    kind: str
    observation_kind: str
    env_ids: tuple[int, ...]
    episode_ids: tuple[int, ...] | None
    criteria: HoldAcceptanceCriteria | None
    source_binding_hash: str
    evidence_format: str = "raw_snapshot"
    binding_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest", "source_binding_hash"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(x not in "0123456789abcdef" for x in value)
            ):
                raise ValueError("Evidence requires complete content identities.")
        _identifier(self.predicate_id)
        if self.kind not in {
            "handover_complete",
            "receiver_hold",
            "receiver_orientation",
            "coordinated_hold",
            "coordinated_acquisition",
        }:
            raise ValueError("Unsupported normalized manipulation evidence kind.")
        if self.evidence_format not in {"raw_snapshot", "historical_record"}:
            raise ValueError("Unsupported manipulation evidence input contract.")
        if self.observation_kind not in {"final", "recorded"}:
            raise ValueError("Evidence requires an explicit observation scope.")
        ids = tuple(self.env_ids)
        if (
            not ids
            or ids != tuple(range(len(ids)))
            or any(type(x) is not int for x in ids)
        ):
            raise ValueError("Evidence requires ordered environment identities.")
        object.__setattr__(self, "env_ids", ids)
        if self.episode_ids is not None:
            episodes = tuple(self.episode_ids)
            if len(episodes) != len(ids) or any(
                type(x) is not int or x < 0 for x in episodes
            ):
                raise ValueError("Episode identities must cover every evidence row.")
            object.__setattr__(self, "episode_ids", episodes)
        if self.criteria is not None:
            object.__setattr__(
                self, "criteria", HoldAcceptanceCriteria.from_dict(self.criteria)
            )
        if (
            self.kind in {"receiver_hold", "receiver_orientation", "coordinated_hold"}
            and self.criteria is None
        ):
            raise ValueError(
                "Held predicates require explicit sealed acceptance criteria."
            )
        if (
            self.kind == "coordinated_acquisition"
            and self.observation_kind != "recorded"
        ):
            raise ValueError("Past acquisition cannot certify current holding.")
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Manipulation binding content differs from its sealed identity."
            )
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        return dict(
            schema_version="task_spec.manipulation_evidence_binding/v1",
            task_spec_digest=self.task_spec_digest,
            program_digest=self.program_digest,
            predicate_id=self.predicate_id,
            kind=self.kind,
            observation_kind=self.observation_kind,
            env_ids=list(self.env_ids),
            episode_ids=None if self.episode_ids is None else list(self.episode_ids),
            criteria=None if self.criteria is None else self.criteria.to_dict(),
            source_binding_hash=self.source_binding_hash,
            evidence_format=self.evidence_format,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete normative criteria and their content identity."""
        return {**self._payload(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls, value: ManipulationEvidenceBinding | Mapping[str, Any]
    ) -> ManipulationEvidenceBinding:
        """Rebuild an independent checked snapshot without resigning drift."""
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if (
            raw.pop("schema_version", None)
            != "task_spec.manipulation_evidence_binding/v1"
        ):
            raise ValueError("Unsupported normalized manipulation binding version.")
        return cls(**raw)


def evaluate_manipulation_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: ManipulationEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Evaluate normalized measurements without decoding any runtime protocol.

    Args:
        spec: Independently validated normative task specification.
        binding: Sealed predicate, criteria and final/recorded scope.
        evidence: Normalized facts, validity, interval and provenance references.
        program: Opaque original program snapshot, used only for its identity.

    Returns:
        Per-environment normative outcomes. Missing observations stay unavailable;
        valid measured negatives fail the declared comparison.
    """
    normalized = TaskSpec.from_dict(spec)
    selected = ManipulationEvidenceBinding.from_dict(binding)
    if (
        canonical_hash(normalized.to_dict()) != selected.task_spec_digest
        or canonical_hash(program) != selected.program_digest
    ):
        raise ValueError("Evidence belongs to another normative or program identity.")
    matches = [(p, False) for p in normalized.goal if p.id == selected.predicate_id] + [
        (p, True)
        for m in normalized.milestones
        for p in m.achieve
        if p.id == selected.predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("Evidence must identify one exact normative predicate.")
    predicate, past = matches[0]
    names = {
        "handover_complete": {"handover_complete"},
        "receiver_hold": {"held"},
        "receiver_orientation": {"orientation"},
        "coordinated_hold": {"held_by_both_grippers"},
        "coordinated_acquisition": {"held_by_both_grippers"},
    }
    if (
        predicate.name not in names[selected.kind]
        or predicate.operator != "eq"
        or predicate.expected is not True
        or selected.observation_kind != ("recorded" if past else "final")
    ):
        raise ValueError("Manipulation predicate and evidence scope disagree.")
    count = len(selected.env_ids)
    snapshot = json_snapshot(evidence)
    refs = (
        selected.binding_hash,
        *(
            (snapshot["measurement_hash"],)
            if type(snapshot.get("measurement_hash")) is str
            else ()
        ),
        *tuple(snapshot.get("evidence_refs", ())),
    )
    outcomes = []
    role = next(
        (r for r in normalized.roles if r.name == predicate.arguments.get("subject")),
        None,
    )
    for row in range(count):
        reason = None
        accepted = False
        try:
            if snapshot.get(
                "schema_version"
            ) != "physical_evidence.normalized/v1" or snapshot.get(
                "measurement_hash"
            ) != canonical_hash(
                {k: v for k, v in snapshot.items() if k != "measurement_hash"}
            ):
                raise ValueError(
                    "Normalized measurement identity is missing or changed."
                )
            if role is None or not role.is_single_member:
                raise ValueError(
                    "Complete single-member selector evidence was not provided."
                )
            if "unavailable" in snapshot:
                raise ValueError(str(snapshot["unavailable"]))
            if (
                snapshot.get("kind") != selected.kind
                or snapshot.get("env_ids") != list(selected.env_ids)
                or snapshot.get("episode_ids")
                != (
                    None if selected.episode_ids is None else list(selected.episode_ids)
                )
            ):
                raise ValueError(
                    "Normalized evidence has another semantic or row/episode scope."
                )
            if snapshot.get("source_binding_hash") != selected.source_binding_hash:
                raise ValueError("Measurements have another declared evidence source.")
            frame = snapshot["frame"]
            if (
                frame["env_ids"] != list(selected.env_ids)
                or type(frame["observation_revision"]) is not int
                or frame["observation_revision"] < 0
            ):
                raise ValueError("Normalized frame has another row or revision scope.")
            _number(frame["timestamp"])

            def value(name: str) -> Any:
                batch = frame["batches"][name]
                if (
                    batch["evidence_id"] != name
                    or batch["env_ids"] != frame["env_ids"]
                    or batch["timestamp"] != frame["timestamp"]
                    or batch["observation_revision"] != frame["observation_revision"]
                ):
                    raise ValueError(
                        "Normalized measurement channel has another identity."
                    )
                valid, errors = batch["valid"], batch["acquisition_errors"]
                if (
                    len(valid) != count
                    or len(errors) != count
                    or any(type(x) is not bool for x in valid)
                ):
                    raise ValueError("Normalized evidence validity rows are malformed.")
                if not valid[row] or errors[row] is not None:
                    raise ValueError(errors[row] or "Measurement unavailable.")
                values = batch["values"]
                if not isinstance(values, list) or len(values) != count:
                    raise ValueError(
                        "Measurements do not cover the declared environments."
                    )
                return values[row]

            if selected.kind in {"handover_complete", "coordinated_acquisition"}:
                flag = value(
                    "transfer_complete"
                    if selected.kind == "handover_complete"
                    else "acquired_by_all"
                )
                if type(flag) is not bool:
                    raise ValueError(
                        "Semantic transition measurements require booleans."
                    )
                accepted = flag
            else:
                criteria = selected.criteria
                assert criteria is not None
                elapsed = _number(value("observed_seconds"))
                if elapsed < criteria.hold_seconds:
                    raise ValueError(
                        "The required hold interval was not completely observed."
                    )
                translations, rotations = value("attachment_translation"), value(
                    "attachment_rotation"
                )
                parts = 2 if selected.kind == "coordinated_hold" else 1
                if len(translations) != parts or len(rotations) != parts:
                    raise ValueError(
                        "Attachment measurements have another participant scope."
                    )
                accepted = all(
                    _number(x) <= criteria.translation_limit for x in translations
                ) and all(_number(x) <= criteria.rotation_limit for x in rotations)
                stable = _number(value("stable_seconds"))
                accepted &= stable >= criteria.hold_seconds
                if predicate.name != "orientation":
                    grips = value("grips")
                    if len(grips) != parts or any(type(x) is not bool for x in grips):
                        raise ValueError(
                            "Grip measurements have another participant scope."
                        )
                    accepted &= all(grips)
                if (
                    predicate.name == "orientation"
                    and criteria.minimum_alignment is None
                ):
                    raise ValueError(
                        "An orientation acceptance criterion was not declared."
                    )
                if criteria.minimum_alignment is not None:
                    rotation = value("object_rotation")
                    if len(rotation) != 3 or any(len(line) != 3 for line in rotation):
                        raise ValueError("Orientation measurement has another shape.")
                    if any(
                        type(x) not in (int, float) or not math.isfinite(x)
                        for line in rotation
                        for x in line
                    ):
                        raise ValueError("Orientation measurement is unavailable.")
                    assert (
                        criteria.local_axis is not None
                        and criteria.world_axis is not None
                    )
                    axis = [
                        sum(rotation[i][j] * criteria.local_axis[j] for j in range(3))
                        for i in range(3)
                    ]
                    alignment = sum(axis[i] * criteria.world_axis[i] for i in range(3))
                    accepted &= alignment >= criteria.minimum_alignment
                if criteria.target_position is not None:
                    position = value("object_position")
                    if len(position) != 3 or any(
                        type(x) not in (int, float) or not math.isfinite(x)
                        for x in position
                    ):
                        raise ValueError("Position measurement is unavailable.")
                    distance = math.sqrt(
                        sum(
                            (float(x) - y) ** 2
                            for x, y in zip(
                                position, criteria.target_position, strict=True
                            )
                        )
                    )
                    assert criteria.position_tolerance is not None
                    accepted &= distance <= criteria.position_tolerance
            claims = snapshot.get("claimed_result")
            if claims is not None and (
                len(claims) != count
                or type(claims[row]) is not bool
                or claims[row] != bool(accepted)
            ):
                raise ValueError(
                    "Historical verdict contradicts its normalized measurements."
                )
        except (KeyError, ValueError, TypeError, IndexError) as error:
            reason = str(error) or type(error).__name__
        outcomes.append(
            PredicateEvaluation(
                predicate.id,
                predicate.name,
                (
                    EvaluationStatus.UNAVAILABLE
                    if reason
                    else (
                        EvaluationStatus.SATISFIED
                        if accepted
                        else EvaluationStatus.FAILED
                    )
                ),
                observed=None if reason else bool(accepted),
                expected=True,
                reason=reason
                or "Normalized measurements compared against sealed task acceptance criteria.",
                evidence_refs=refs,
            )
        )
    return tuple(outcomes)
