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

"""Normative qualification of exact rigid tilt/restore motion observations."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from typing import Any, TYPE_CHECKING
from .canonicalization import canonical_hash, json_snapshot
from .evidence import _digest, _identifier, _path, _bool_rows

if TYPE_CHECKING:
    from .spec import TaskSpec
    from .expressions import PredicateEvaluation

__all__ = ["PourEvidenceBinding", "evaluate_pour_evidence"]


@dataclass(frozen=True, slots=True)
class PourEvidenceBinding:
    """Seal a rigid motion criterion and exact stage/tilt/release occurrences.

    Native geometry and runtime clocks are explicit owner declarations. This
    contract retains the source 30-degree tilt/15-degree restore proxy; it never
    qualifies fluid transfer. An operation-event goal uses that operation's own
    boundary even if later operations move or grasp the same object again.
    """

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    operation_id: str
    object_id: str
    reference_id: str
    resource_id: str
    object_native_uid: str
    reference_native_uid: str
    occurrences: tuple[Mapping[str, Any], ...]
    stage_segment_id: str
    pour_segment_id: str
    release_segment_id: str
    boundary_segment_id: str
    observation_kind: str
    evidence_format: str = "raw_snapshot"
    program_fingerprint: str | None = None
    step_dt: float | None = None
    env_ids: tuple[int, ...] = (0,)
    episode_ids: tuple[int, ...] | None = None
    binding_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest"):
            _digest(getattr(self, name), name)
        for name in (
            "predicate_id",
            "operation_id",
            "object_id",
            "reference_id",
            "resource_id",
            "object_native_uid",
            "reference_native_uid",
            "stage_segment_id",
            "pour_segment_id",
            "release_segment_id",
            "boundary_segment_id",
        ):
            _identifier(getattr(self, name), name)
        if (
            self.object_id == self.reference_id
            or self.object_native_uid == self.reference_native_uid
        ):
            raise ValueError(
                "Pour source and receiver must be distinct declared bodies."
            )
        if self.observation_kind not in {
            "final",
            "recorded",
        } or self.evidence_format not in {"raw_snapshot", "historical_record"}:
            raise ValueError(
                "Pour evidence requires explicit boundary and input-format scopes."
            )
        ids = tuple(self.env_ids)
        if (
            not ids
            or ids != tuple(range(len(ids)))
            or any(type(value) is not int for value in ids)
        ):
            raise ValueError("Pour evidence requires ordered native environment rows.")
        object.__setattr__(self, "env_ids", ids)
        if self.episode_ids is not None:
            episodes = tuple(self.episode_ids)
            if len(episodes) != len(ids) or any(
                type(value) is not int or value < 0 for value in episodes
            ):
                raise ValueError(
                    "Pour episode identities differ from the selected native rows."
                )
            object.__setattr__(self, "episode_ids", episodes)
        if self.program_fingerprint is not None:
            _digest(self.program_fingerprint, "program_fingerprint")
        elif self.evidence_format == "raw_snapshot":
            raise ValueError(
                "Raw Pour bindings require the compiled program fingerprint."
            )
        if self.step_dt is not None:
            if (
                type(self.step_dt) not in (int, float)
                or not math.isfinite(self.step_dt)
                or self.step_dt <= 0
            ):
                raise ValueError("Pour recording step_dt must be positive and finite.")
            object.__setattr__(self, "step_dt", float(self.step_dt))
        occurrences = tuple(json_snapshot(value) for value in self.occurrences)
        required = {"segment_id", "segment_name", "source_path", "invocation_id"}
        for item in occurrences:
            if not isinstance(item, Mapping) or set(item) != required:
                raise ValueError(
                    "Pour execution declarations require exact compiled occurrence fields."
                )
            for key in ("segment_id", "segment_name", "invocation_id"):
                _identifier(item[key], key)
            _path(item["source_path"], "source_path")
        identities = [item["segment_id"] for item in occurrences]
        if (
            not identities
            or len(set(identities)) != len(identities)
            or len({item["invocation_id"] for item in occurrences}) != len(occurrences)
        ):
            raise ValueError(
                "Pour occurrences and invocation identities must be independent."
            )
        if (
            any(
                value not in identities
                for value in (
                    self.stage_segment_id,
                    self.pour_segment_id,
                    self.release_segment_id,
                    self.boundary_segment_id,
                )
            )
            or not identities.index(self.stage_segment_id)
            < identities.index(self.pour_segment_id)
            < identities.index(self.release_segment_id)
            or self.boundary_segment_id != identities[-1]
        ):
            raise ValueError(
                "Pour declaration must preserve stage, tilt/restore, release and terminal order."
            )
        object.__setattr__(self, "occurrences", occurrences)
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Pour evidence binding_hash does not match its declaration."
            )
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "task_spec.pour_evidence_binding/v1",
            **{
                item.name: json_snapshot(getattr(self, item.name))
                for item in fields(self)
                if item.name != "binding_hash"
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return an owned declaration while retaining its original content hash."""
        return {**self._payload(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls, value: PourEvidenceBinding | Mapping[str, Any]
    ) -> PourEvidenceBinding:
        """Rebuild and verify an existing or serialized motion-evidence binding.

        Args:
            value: Serialized declaration or existing binding instance.

        Returns:
            Independently owned normalized binding without re-signing changed data.
        """
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if (
            not isinstance(raw, dict)
            or raw.pop("schema_version", None) != "task_spec.pour_evidence_binding/v1"
        ):
            raise ValueError("Unsupported Pour evidence binding version.")
        return cls(**raw)


def evaluate_pour_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: PourEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Qualify the source rigid Pour motion proxy, independently of fluid transfer.

    Args:
        spec: Frozen normative TaskSpec.
        binding: Exact semantic endpoints, operation and compiled occurrences.
        evidence: Raw Lab geometry channels plus runtime receipts/boundary slots,
            or explicitly selected original historical records.
        program: Complete serialized program identified by the declaration.

    Returns:
        Per-environment predicate outcomes. Missing acquisition/release proof
        stays unavailable; valid measured negatives remain failures. Recorded
        milestones additionally require their complete successful group prefix.
    """
    from .spec import TaskSpec
    from .expressions import EvaluationStatus, PredicateEvaluation
    from ._pour_evidence import (
        declared_pour_calls,
        measured_pour_occurrences,
        frames,
        classify_pour_metrics,
        raw_pour_geometry,
        _CRITERION,
        _MINIMUM_TILT,
        _MAXIMUM_RESTORE_ERROR,
    )

    selected = PourEvidenceBinding.from_dict(binding)
    normalized = TaskSpec.from_dict(spec)
    program = json_snapshot(program)
    if (
        canonical_hash(normalized.to_dict()) != selected.task_spec_digest
        or canonical_hash(program) != selected.program_digest
    ):
        raise ValueError("Pour evidence has another normative/program identity.")
    matches = [
        (item, False) for item in normalized.goal if item.id == selected.predicate_id
    ] + [
        (item, True)
        for milestone in normalized.milestones
        for item in milestone.achieve
        if item.id == selected.predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("Pour binding must select one exact declared predicate.")
    predicate, past = matches[0]
    if (
        predicate.name != "pour_motion"
        or predicate.arguments.get("task_type") != "E3"
        or predicate.arguments.get("relation") != "above"
        or predicate.operator != "eq"
        or predicate.expected is not True
        or selected.observation_kind != ("recorded" if past else "final")
    ):
        raise ValueError(
            "This proof qualifies explicit rigid Pour motion, not fluid transfer or another predicate."
        )
    declared = declared_pour_calls(selected, program)
    size = len(selected.env_ids)
    valid = [True] * size
    errors = [None] * size
    flags = [False] * size
    reason = None
    refs = ()
    try:
        roles = {role.name: role for role in normalized.roles}
        if any(
            roles[predicate.arguments[key]].quantifier != "one"
            for key in ("subject", "reference")
        ):
            raise ValueError("Complete selector-universe evidence was not provided.")
        snapshot = json_snapshot(evidence)
        refs = (selected.binding_hash, canonical_hash(snapshot))
        if "unavailable" in snapshot:
            raise ValueError(str(snapshot["unavailable"]))
        segments, calls, eligible = measured_pour_occurrences(
            selected, snapshot, program, declared
        )
        pour_segment = segments[selected.pour_segment_id]
        starts = pour_segment["start_steps"]
        ends = frames(pour_segment, size)
        boundary = frames(segments[selected.boundary_segment_id], size)
        if (
            not isinstance(starts, list)
            or len(starts) != size
            or any(
                type(start) is not int or not 0 <= start < end <= last
                for start, end, last in zip(starts, ends, boundary, strict=True)
            )
        ):
            raise ValueError(
                "Pour runtime interval does not precede its exact terminal boundary."
            )
        descriptor = next(
            item
            for item in selected.occurrences
            if item["segment_id"] == selected.pour_segment_id
        )
        historical = selected.evidence_format == "historical_record"
        if historical:
            if "record" not in snapshot or "geometry_frame" in snapshot:
                raise ValueError(
                    "Historical Pour evidence requires its record without a raw overlay."
                )
            if selected.episode_ids is not None and (
                snapshot.get("episode_ids") != list(selected.episode_ids)
                or any(
                    type(value) is not int for value in snapshot.get("episode_ids", ())
                )
            ):
                raise ValueError("Historical Pour evidence belongs to another episode.")
            record = snapshot["record"]
            metrics = record["measurements"]
            if (
                record.get("uid") != selected.object_id
                or record.get("part_id") is not None
                or record.get("group_id") != selected.operation_id
                or metrics.get("reference_uid") != selected.reference_id
                or metrics.get("pour_node_id") != descriptor["segment_name"]
                or metrics.get("invocation_id") != descriptor["invocation_id"]
                or type(metrics.get("invocation_revision")) is not int
                or metrics["invocation_revision"]
                != calls[selected.pour_segment_id]["invocation_revision"]
            ):
                raise ValueError(
                    "Historical Pour record has another role or invocation identity."
                )
            physical = classify_pour_metrics(metrics, size)
            flags = [
                measured and active
                for measured, active in zip(physical, eligible, strict=True)
            ]
            if (
                not _bool_rows(record.get("accepted"), size)
                or record["accepted"] != flags
            ):
                raise ValueError(
                    "Historical Pour claims differ from their geometry and occurrence eligibility."
                )
            bound = metrics
        else:
            rows, valid, errors = raw_pour_geometry(
                selected, snapshot, program, starts=starts, ends=ends
            )
            from ._release_evidence import measured_release

            release_descriptor = next(
                item
                for item in selected.occurrences
                if item["segment_id"] == selected.release_segment_id
            )
            released, release_valid, _, _ = measured_release(
                segments[selected.release_segment_id]["metadata"]["runtime"],
                segments[selected.boundary_segment_id]["metadata"]["runtime"],
                object_uid=selected.object_id,
                resource=selected.resource_id,
                invocation_id=release_descriptor["invocation_id"],
                num_envs=size,
            )
            for row in range(size):
                if not release_valid[row]:
                    valid[row] = False
                    errors[row] = "Measured Pour release acquisition is unavailable."
                if not valid[row]:
                    continue
                rows[row]["released"] = released[row]
                metrics = dict(
                    scope="measured_pour_motion",
                    criterion=_CRITERION,
                    fluid_transfer_verified=False,
                    minimum_tilt_rad=_MINIMUM_TILT,
                    maximum_restore_error_rad=_MAXIMUM_RESTORE_ERROR,
                    rows=[rows[row]],
                )
                flags[row] = classify_pour_metrics(metrics, 1)[0] and eligible[row]
            bound = snapshot
        if (
            bound.get("start_frames") != starts
            or bound.get("end_frames") != ends
            or bound.get("boundary_frames") != boundary
        ):
            raise ValueError(
                "Pour measurements have another exact operation interval or terminal boundary."
            )
        for key in ("start_frames", "end_frames", "boundary_frames", "peak_frames"):
            values = bound.get(key)
            if (
                not isinstance(values, list)
                or len(values) != size
                or any(type(value) is not int for value in values)
            ):
                raise ValueError("Pour window identities require exact integer rows.")
        if any(
            not start <= peak <= end
            for start, peak, end in zip(starts, bound["peak_frames"], ends, strict=True)
        ):
            raise ValueError("Pour peak lies outside its own observed interval.")
        if past:
            for row, active in enumerate(eligible):
                if not active:
                    valid[row] = False
                    errors[row] = (
                        "The complete Pour operation prefix did not establish this milestone."
                    )
    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
        IndexError,
    ) as error:
        reason = str(error) or type(error).__name__
    return tuple(
        PredicateEvaluation(
            predicate.id,
            predicate.name,
            (
                EvaluationStatus.UNAVAILABLE
                if reason or not valid[row]
                else (
                    EvaluationStatus.SATISFIED
                    if flags[row]
                    else EvaluationStatus.FAILED
                )
            ),
            observed=None if reason or not valid[row] else flags[row],
            expected=True,
            reason=reason
            or errors[row]
            or (
                "Physical Pour geometry acquisition is unavailable."
                if not valid[row]
                else "Rigid tilt/restore motion checked; fluid transfer is not established."
            ),
            evidence_refs=refs,
        )
        for row in range(size)
    )
