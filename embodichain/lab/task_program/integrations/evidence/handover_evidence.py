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

"""Offline execution-evidence declaration and protocol normalization."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from typing import Any
from embodichain._json_contract import canonical_hash, json_snapshot
from ._values import _digest, _identifier, _path, _bool_rows

__all__ = ["HandOverEvidenceBinding", "normalize_handover_evidence"]


@dataclass(frozen=True, slots=True)
class HandOverEvidenceBinding:
    """Seal exact participants, transfer/boundary occurrences and owner criteria."""

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    operation: Mapping[str, Any]
    observation_kind: str
    step_dt: float
    query_kind: str
    evidence_format: str = "historical_record"
    env_ids: tuple[int, ...] = (0,)
    episode_ids: tuple[int, ...] | None = None
    program_fingerprint: str | None = None
    object_native_uid: str | None = None
    robot_uid: str | None = None
    binding_hash: str = ""

    def __post_init__(self) -> None:
        if self.query_kind not in {"handover_complete", "held", "orientation"}:
            raise ValueError("Unsupported held observation query kind.")
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
            "schema_version": "task_program.handover_evidence_binding/v2",
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
            != "task_program.handover_evidence_binding/v2"
        ):
            raise ValueError("Unsupported HandOver evidence binding version.")
        return cls(**raw)


def normalize_handover_evidence(
    binding: HandOverEvidenceBinding | Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the execution protocol and return facts, without task judgments.

    Args:
        binding: Explicit compiled/native observation declaration.
        evidence: Original raw channels or a frozen historical record.
        program: Exact serialized program owning the declared occurrences.

    Returns:
        Owned measurements, validity, observation boundary and source hashes.
        Missing or contradictory acquisition remains unavailable.
    """
    from ._handover_evidence import transfer_record, hold_records
    from ._held_evidence import matrix, grip_flags
    from types import SimpleNamespace
    from ._values import (
        evidence_packet,
        unavailable_packet,
        validate_historical_claim,
        seal_packet,
    )

    selected = HandOverEvidenceBinding.from_dict(binding)
    program = json_snapshot(program)
    if canonical_hash(program) != selected.program_digest:
        raise ValueError("Execution evidence belongs to another declared program.")
    count = len(selected.env_ids)
    kind = {
        "handover_complete": "handover_complete",
        "held": "receiver_hold",
        "orientation": "receiver_orientation",
    }[selected.query_kind]
    predicate = SimpleNamespace(name=selected.query_kind)
    snapshot = json_snapshot(evidence)
    try:
        snapshot = json_snapshot(evidence)
        if "unavailable" in snapshot:
            raise ValueError(str(snapshot["unavailable"]))
        raw = selected.evidence_format == "raw_snapshot"
        if (
            raw
            and "record" in snapshot
            or (
                not raw
                and (
                    "record" not in snapshot
                    or "held_poses" in snapshot
                    or "fresh_grip" in snapshot
                )
            )
        ):
            raise ValueError("HandOver input cannot mix raw and historical contracts.")
        if raw and snapshot.get("program_fingerprint") != selected.program_fingerprint:
            raise ValueError("HandOver input has another compiled program identity.")
        if selected.episode_ids is not None and (
            snapshot.get("episode_ids") != list(selected.episode_ids)
            or any(
                (type(value) is not int for value in snapshot.get("episode_ids", ()))
            )
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
                    (
                        value
                        for value in runtime["segments"]
                        if value["metadata"]["program_segment_id"]
                        == selected.operation["boundary"]["segment_id"]
                    )
                )
                part = one(pose_snapshot["motion_parts"])
                object_pose, tcp_pose = raw_held_poses(
                    selected, pose_snapshot, frames(boundary_segment, count), part
                )
            else:
                object_pose, tcp_pose = (
                    matrix(metrics["object_pose"], count),
                    matrix(metrics["receiver_tcp_pose"], count),
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
                    (
                        value
                        for segment in runtime["segments"]
                        for call in segment["metadata"]["runtime"]["calls"]
                        for value in call["effects"]
                        if canonical_hash(value)
                        == expected["measurements"]["holding_receipt_hash"]
                    )
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
            or any((type(value) is not int for value in snapshot["boundary_frames"]))
        ):
            raise ValueError("HandOver report has another physical boundary.")
        facts = expected.pop("_normalized_facts", {})
        if selected.query_kind == "handover_complete":
            facts = {"transfer_complete": list(expected["accepted"])}
        elif final and selected.query_kind == "held":
            facts["grips"] = [[value] for value in expected["accepted"]]
        packet = evidence_packet(selected, snapshot, kind)
        packet.update(
            facts=facts, boundary_frames=expected["measurements"]["boundary_frames"]
        )
        if not raw:
            validate_historical_claim(record, expected)
            packet["claimed_result"] = record["accepted"]
        return seal_packet(packet, selected.step_dt)
    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
        IndexError,
        StopIteration,
    ) as error:
        return unavailable_packet(selected, snapshot, kind, error)
