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

"""Data-only transfer and receiver-hold qualification for explicit operations."""

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

__all__ = ["CoordinatedHoldEvidenceBinding", "evaluate_coordinated_hold_evidence"]


@dataclass(frozen=True, slots=True)
class CoordinatedHoldEvidenceBinding:
    """Seal exact participants, transfer/boundary occurrences and owner criteria."""

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    operation: Mapping[str, Any]
    observation_kind: str
    step_dt: float
    evidence_format: str = "historical_record"
    env_ids: tuple[int, ...] = (0,)
    episode_ids: tuple[int, ...] | None = None
    program_fingerprint: str | None = None
    object_native_uid: str | None = None
    robot_uid: str | None = None
    binding_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest"):
            _digest(getattr(self, name), name)
        _identifier(self.predicate_id, "predicate_id")
        if self.observation_kind not in {
            "final",
            "recorded",
        } or self.evidence_format not in {"raw_snapshot", "historical_record"}:
            raise ValueError(
                "Coordinated hold evidence requires explicit boundary/input scopes."
            )
        if (
            type(self.step_dt) not in (int, float)
            or not math.isfinite(self.step_dt)
            or self.step_dt <= 0
        ):
            raise ValueError(
                "Coordinated hold evidence requires a positive finite recording cadence."
            )
        object.__setattr__(self, "step_dt", float(self.step_dt))
        ids = tuple(self.env_ids)
        if (
            not ids
            or ids != tuple(range(len(ids)))
            or any(type(value) is not int for value in ids)
        ):
            raise ValueError("Coordinated hold evidence requires ordered native rows.")
        object.__setattr__(self, "env_ids", ids)
        if self.episode_ids is not None:
            episodes = tuple(self.episode_ids)
            if len(episodes) != len(ids) or any(
                type(value) is not int or value < 0 for value in episodes
            ):
                raise ValueError(
                    "Coordinated hold episode identities differ from native rows."
                )
            object.__setattr__(self, "episode_ids", episodes)
        if self.program_fingerprint is not None:
            _digest(self.program_fingerprint, "program_fingerprint")
        for name in ("object_native_uid", "robot_uid"):
            if getattr(self, name) is not None:
                _identifier(getattr(self, name), name)
        if self.evidence_format == "raw_snapshot" and any(
            value is None
            for value in (
                self.program_fingerprint,
                self.episode_ids,
                self.object_native_uid,
                self.robot_uid,
            )
        ):
            raise ValueError(
                "Raw Coordinated hold evidence requires its compiled program, episode and native owners."
            )
        operation = json_snapshot(self.operation)
        if not isinstance(operation, dict) or set(operation) != {
            "operation_id",
            "object_id",
            "resources",
            "boundary",
            "hold_condition",
            "hold_preset",
        }:
            raise ValueError(
                "Coordinated hold declarations require exact operation fields."
            )
        for key in (
            "operation_id",
            "object_id",
        ):
            _identifier(operation[key], key)
        resources = operation["resources"]
        if not isinstance(resources, dict) or set(resources) != {"left", "right"}:
            raise ValueError(
                "Coordinated hold requires explicit left/right resource slots."
            )
        for resource in resources.values():
            _identifier(resource, "resource")
        if resources["left"] == resources["right"]:
            raise ValueError("Coordinated hold participants must be distinct.")
        for key in ("boundary",):
            occurrence = operation[key]
            if not isinstance(occurrence, dict) or set(occurrence) != {
                "segment_id",
                "segment_name",
                "source_path",
                "invocation_id",
            }:
                raise ValueError(
                    "Coordinated hold occurrences require explicit compiled identities."
                )
            for name in ("segment_id", "segment_name", "invocation_id"):
                _identifier(occurrence[name], name)
            _path(occurrence["source_path"], "source_path")
        if operation["hold_condition"] is not None:
            from ._handover_evidence import hold_condition

            cfg = hold_condition(operation["hold_condition"], parts_count=2)
            if cfg.target_position is None:
                raise ValueError("Coordinated hold requires an explicit world target.")
            if cfg.entity != operation["object_id"]:
                raise ValueError("Held condition names another object.")
            _identifier(operation["hold_preset"], "hold_preset")
        else:
            raise ValueError(
                "Coordinated hold requires its named owner hold declaration."
            )
        object.__setattr__(self, "operation", operation)
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Coordinated hold binding_hash does not match its declaration."
            )
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "task_spec.coordinated_hold_evidence_binding/v1",
            **{
                item.name: json_snapshot(getattr(self, item.name))
                for item in fields(self)
                if item.name != "binding_hash"
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return owned declaration data with its original identity."""
        return {**self._payload(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls, value: CoordinatedHoldEvidenceBinding | Mapping[str, Any]
    ) -> CoordinatedHoldEvidenceBinding:
        """Re-decode and verify a serialized declaration or existing instance.

        Args:
            value: Binding or its exact serialized form.

        Returns:
            Independently owned binding; changed data is never automatically signed.
        """
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if (
            not isinstance(raw, dict)
            or raw.pop("schema_version", None)
            != "task_spec.coordinated_hold_evidence_binding/v1"
        ):
            raise ValueError("Unsupported Coordinated hold evidence binding version.")
        return cls(**raw)


def evaluate_coordinated_hold_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: CoordinatedHoldEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Verify both native attachments, disjoint pads and the authored hold target.

    Args:
        spec: Frozen normative specification.
        binding: Explicit two-participant hold declaration.
        evidence: Exact execution receipts with original records or raw channels.
        program: Complete serialized source program matching the binding.

    Returns:
        Per-environment qualification; missing facts remain unavailable.
    """
    from .spec import TaskSpec
    from .expressions import PredicateEvaluation, EvaluationStatus
    from ._coordinated_evidence import holding, record
    from ._held_evidence import frames, raw_held_poses, raw_grip

    selected = CoordinatedHoldEvidenceBinding.from_dict(binding)
    normalized = TaskSpec.from_dict(spec)
    program = json_snapshot(program)
    if (
        canonical_hash(normalized.to_dict()) != selected.task_spec_digest
        or canonical_hash(program) != selected.program_digest
    ):
        raise ValueError("Coordinated hold has another normative/program identity.")
    matches = [
        (item, False) for item in normalized.goal if item.id == selected.predicate_id
    ] + [
        (item, True)
        for milestone in normalized.milestones
        for item in milestone.achieve
        if item.id == selected.predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("Coordinated hold requires one exact predicate.")
    predicate, past = matches[0]
    if (
        predicate.arguments.get("task_type") != "E5"
        or predicate.name != "held_by_both_grippers"
        or predicate.operator != "eq"
        or predicate.expected is not True
        or selected.observation_kind != ("recorded" if past else "final")
    ):
        raise ValueError(
            "Coordinated hold differs from its predicate or boundary scope."
        )
    count = len(selected.env_ids)
    flags = [False] * count
    reason, refs = None, ()
    try:
        role = next(
            role
            for role in normalized.roles
            if role.name == predicate.arguments["subject"]
        )
        if not role.is_single_member:
            raise ValueError("Coordinated hold lacks its complete selector universe.")
        snapshot = json_snapshot(evidence)
        refs = (selected.binding_hash, canonical_hash(snapshot))
        if "unavailable" in snapshot:
            raise ValueError(str(snapshot["unavailable"]))
        raw = selected.evidence_format == "raw_snapshot"
        if (
            raw
            and "record" in snapshot
            or not raw
            and (
                "record" not in snapshot
                or "held_poses" in snapshot
                or "fresh_grips" in snapshot
            )
        ):
            raise ValueError("Coordinated hold cannot mix raw and historical input.")
        if selected.episode_ids is not None and (
            snapshot.get("episode_ids") != list(selected.episode_ids)
            or any(type(x) is not int for x in snapshot["episode_ids"])
        ):
            raise ValueError("Coordinated hold has another episode identity.")
        runtime = {"segments": snapshot["segments"], "lengths": snapshot["lengths"]}
        final = selected.observation_kind == "final"
        if raw:
            if snapshot.get("program_fingerprint") != selected.program_fingerprint:
                raise ValueError("Coordinated hold has another compiled program.")
            _, segment, _, _, slots, _, _ = holding(
                selected.operation, runtime, program, count, selected.step_dt, final
            )
            boundary = frames(segment, count)
            poses = snapshot["held_poses"]
            parts = [slots[key][1] for key in ("left", "right")]
            if poses["motion_parts"] != parts:
                raise ValueError(
                    "Coordinated pose channels have another participant order."
                )
            tcp = {}
            for part in parts:
                object_pose, tcp[part] = raw_held_poses(
                    selected,
                    {**poses, "motion_parts": [part]},
                    boundary,
                    part,
                    observation_kind="final",
                )
            grips = snapshot["fresh_grips"]
            for role in ("left", "right"):
                raw_grip(selected, grips[role], boundary)
        else:
            original = snapshot["record"]
            metrics = original["measurements"]
            if metrics["final"] is not final or metrics["step_dt"] != selected.step_dt:
                raise ValueError("Coordinated record has another boundary or cadence.")
            object_pose, tcp, grips = (
                metrics["object_pose"],
                metrics["tcp_poses"],
                metrics["fresh_grips"],
            )
        expected = record(
            selected.operation,
            runtime,
            program,
            count,
            selected.step_dt,
            final,
            object_pose,
            tcp,
            grips,
        )
        if raw and (
            snapshot.get("boundary_frames")
            != expected["measurements"]["boundary_frames"]
            or any(type(x) is not int for x in snapshot["boundary_frames"])
        ):
            raise ValueError("Coordinated report has another measured boundary.")
        if not raw and canonical_hash(original) != canonical_hash(expected):
            raise ValueError(
                "Coordinated claims differ from exact native measurements."
            )
        flags = expected["accepted"]
    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
        IndexError,
        StopIteration,
    ) as error:
        reason = str(error) or type(error).__name__
    return tuple(
        PredicateEvaluation(
            predicate.id,
            predicate.name,
            (
                EvaluationStatus.UNAVAILABLE
                if reason
                else (
                    EvaluationStatus.SATISFIED
                    if flags[row]
                    else EvaluationStatus.FAILED
                )
            ),
            observed=None if reason else flags[row],
            expected=True,
            reason=reason or "Exact dual attachment and world-target hold checked.",
            evidence_refs=refs,
        )
        for row in range(count)
    )
