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

__all__ = ["HandOverEvidenceBinding", "evaluate_handover_evidence"]


@dataclass(frozen=True, slots=True)
class HandOverEvidenceBinding:
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
                "HandOver evidence requires explicit boundary/input scopes."
            )
        if (
            type(self.step_dt) not in (int, float)
            or not math.isfinite(self.step_dt)
            or self.step_dt <= 0
        ):
            raise ValueError(
                "HandOver evidence requires a positive finite recording cadence."
            )
        object.__setattr__(self, "step_dt", float(self.step_dt))
        ids = tuple(self.env_ids)
        if (
            not ids
            or ids != tuple(range(len(ids)))
            or any(type(value) is not int for value in ids)
        ):
            raise ValueError("HandOver evidence requires ordered native rows.")
        object.__setattr__(self, "env_ids", ids)
        if self.episode_ids is not None:
            episodes = tuple(self.episode_ids)
            if len(episodes) != len(ids) or any(
                type(value) is not int or value < 0 for value in episodes
            ):
                raise ValueError("HandOver episode identities differ from native rows.")
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
                "Raw HandOver evidence requires its compiled program, episode and native owners."
            )
        operation = json_snapshot(self.operation)
        if not isinstance(operation, dict) or set(operation) != {
            "operation_id",
            "object_id",
            "source_resource",
            "destination_resource",
            "transfer",
            "boundary",
            "hold_condition",
            "hold_preset",
        }:
            raise ValueError("HandOver declarations require exact operation fields.")
        for key in (
            "operation_id",
            "object_id",
            "source_resource",
            "destination_resource",
        ):
            _identifier(operation[key], key)
        if operation["source_resource"] == operation["destination_resource"]:
            raise ValueError("HandOver participants must be distinct.")
        for key in ("transfer", "boundary"):
            occurrence = operation[key]
            if not isinstance(occurrence, dict) or set(occurrence) != {
                "segment_id",
                "segment_name",
                "source_path",
                "invocation_id",
            }:
                raise ValueError(
                    "HandOver occurrences require explicit compiled identities."
                )
            for name in ("segment_id", "segment_name", "invocation_id"):
                _identifier(occurrence[name], name)
            _path(occurrence["source_path"], "source_path")
        if operation["hold_condition"] is not None:
            from ._handover_evidence import hold_condition

            cfg = hold_condition(operation["hold_condition"])
            if cfg.entity != operation["object_id"]:
                raise ValueError("Held condition names another object.")
            _identifier(operation["hold_preset"], "hold_preset")
        elif operation["hold_preset"] is not None:
            raise ValueError("A hold preset requires its owner declaration.")
        object.__setattr__(self, "operation", operation)
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError("HandOver binding_hash does not match its declaration.")
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "task_spec.handover_evidence_binding/v1",
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
        cls, value: HandOverEvidenceBinding | Mapping[str, Any]
    ) -> HandOverEvidenceBinding:
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
            != "task_spec.handover_evidence_binding/v1"
        ):
            raise ValueError("Unsupported HandOver evidence binding version.")
        return cls(**raw)


def evaluate_handover_evidence(
    spec: TaskSpec | Mapping[str, Any],
    binding: HandOverEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> tuple[PredicateEvaluation, ...]:
    """Verify transfer phase receipts and optional receiver-held state evidence.

    Args:
        spec: Frozen normative TaskSpec.
        binding: Explicit participants, occurrences and owner hold criterion.
        evidence: Runtime receipts plus an original historical record or raw facts.
        program: Complete serialized program matching the declaration identity.

    Returns:
        Per-environment predicate outcomes. Missing or contradictory physical
        receipts remain unavailable; a measured final loss of grip is negative.
    """
    from .spec import TaskSpec
    from .expressions import PredicateEvaluation, EvaluationStatus
    from ._handover_evidence import transfer_record, hold_records
    from ._held_evidence import matrix, grip_flags

    selected = HandOverEvidenceBinding.from_dict(binding)
    normalized = TaskSpec.from_dict(spec)
    program = json_snapshot(program)
    if (
        canonical_hash(normalized.to_dict()) != selected.task_spec_digest
        or canonical_hash(program) != selected.program_digest
    ):
        raise ValueError("HandOver evidence has another normative/program identity.")
    matches = [
        (item, False) for item in normalized.goal if item.id == selected.predicate_id
    ] + [
        (item, True)
        for milestone in normalized.milestones
        for item in milestone.achieve
        if item.id == selected.predicate_id
    ]
    if len(matches) != 1:
        raise ValueError("HandOver binding must identify one exact predicate.")
    predicate, past = matches[0]
    if (
        predicate.arguments.get("task_type") != "E4"
        or predicate.name not in {"handover_complete", "held", "orientation"}
        or predicate.operator != "eq"
        or predicate.expected is not True
        or selected.observation_kind != ("recorded" if past else "final")
    ):
        raise ValueError("HandOver proof differs from its declared predicate or scope.")
    count = len(selected.env_ids)
    flags = [False] * count
    reason = None
    refs = ()
    try:
        role = next(
            role
            for role in normalized.roles
            if role.name == predicate.arguments["subject"]
        )
        if not role.is_single_member:
            raise ValueError("Complete selector-universe evidence was not provided.")
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
                or "fresh_grip" in snapshot
            )
        ):
            raise ValueError("HandOver input cannot mix raw and historical contracts.")
        if raw and snapshot.get("program_fingerprint") != selected.program_fingerprint:
            raise ValueError("HandOver input has another compiled program identity.")
        if selected.episode_ids is not None and (
            snapshot.get("episode_ids") != list(selected.episode_ids)
            or any(type(value) is not int for value in snapshot.get("episode_ids", ()))
        ):
            raise ValueError("HandOver evidence belongs to another episode.")
        runtime = {"segments": snapshot["segments"], "lengths": snapshot["lengths"]}
        final = selected.observation_kind == "final"
        if not raw:
            record = snapshot["record"]
            metrics = record["measurements"]
            if (
                metrics.get("final") is not final
                or metrics.get("step_dt") != selected.step_dt
            ):
                raise ValueError("HandOver record has another boundary/clock scope.")
        transfer, terminal = transfer_record(
            selected.operation, runtime, program, count, final, selected.step_dt
        )
        if predicate.name == "handover_complete":
            expected = transfer
        else:
            if raw:
                from ._held_evidence import raw_held_poses, frames, one, raw_grip

                pose_snapshot = snapshot["held_poses"]
                boundary_segment = one(
                    value
                    for value in runtime["segments"]
                    if value["metadata"]["program_segment_id"]
                    == selected.operation["boundary"]["segment_id"]
                )
                # The acquisition's explicit part is subsequently checked against
                # the exact terminal receipt by hold_records.
                part = one(pose_snapshot["motion_parts"])
                object_pose, tcp_pose = raw_held_poses(
                    selected, pose_snapshot, frames(boundary_segment, count), part
                )
            else:
                object_pose, tcp_pose = matrix(metrics["object_pose"], count), matrix(
                    metrics["receiver_tcp_pose"], count
                )
            expected = hold_records(
                transfer,
                terminal,
                selected.operation,
                runtime,
                program,
                count,
                object_pose,
                tcp_pose,
            )[predicate.name]
            if raw and expected["measurements"]["motion_part"] != part:
                raise ValueError(
                    "Held pose channel has another resolved receiver endpoint."
                )
            if final and predicate.name == "held":
                holding = next(
                    value
                    for segment in runtime["segments"]
                    for call in segment["metadata"]["runtime"]["calls"]
                    for value in call["effects"]
                    if canonical_hash(value)
                    == expected["measurements"]["holding_receipt_hash"]
                )
                fresh = snapshot["fresh_grip"] if raw else metrics["fresh_grip"]
                if raw:
                    raw_grip(
                        selected, fresh, expected["measurements"]["boundary_frames"]
                    )
                expected["measurements"]["fresh_grip"] = fresh
                expected["accepted"] = grip_flags(
                    uid=selected.operation["object_id"],
                    resource=selected.operation["destination_resource"],
                    sample=fresh,
                    terminal=holding,
                    frames=expected["measurements"]["boundary_frames"],
                    dt=selected.step_dt,
                    count=count,
                )
        if raw and (
            snapshot.get("boundary_frames")
            != expected["measurements"]["boundary_frames"]
            or any(type(value) is not int for value in snapshot["boundary_frames"])
        ):
            raise ValueError("HandOver report has another physical boundary.")
        if not raw and canonical_hash(record) != canonical_hash(expected):
            raise ValueError(
                "HandOver claims differ from their exact receipts and native measurements."
            )
        if not _bool_rows(expected["accepted"], count):
            raise ValueError("HandOver acceptance rows are malformed.")
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
            reason=reason
            or "Exact transfer/held evidence checked without substituting program success.",
            evidence_refs=refs,
        )
        for row in range(count)
    )
