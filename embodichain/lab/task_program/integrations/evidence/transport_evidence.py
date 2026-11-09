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

__all__ = [
    "CoordinatedTransportEvidenceBinding",
    "normalize_coordinated_transport_evidence",
]


@dataclass(frozen=True, slots=True)
class CoordinatedTransportEvidenceBinding:
    """Seal an exact recorded acquisition event and completed execution prefix.

    Args:
        task_spec_digest: Canonical normative specification identity.
        program_digest: Exact declared Task Program identity.
        predicate_id: Goal or milestone predicate receiving the event result.
        operation: Explicit payload, reference, resources and transport occurrence.
        required_execution: Complete ordered execution-prefix declarations.
        boundary_segment_id: Exact recorded boundary segment.
        boundary_source_path: Compiled source path selecting that boundary.
        step_dt: Original control interval in seconds.
        evidence_format: Explicit raw snapshot or historical record contract.
        observation_kind: Recorded event scope; final held states are unsupported.
        env_ids: The original supported native selection, B=1/env0.
        episode_ids: Expected initialized episodes, required for raw snapshots.
        sampled_contact_identity: Complete native/program acquisition lineage.
        binding_hash: Optional existing content identity, checked without re-signing.
    """

    task_spec_digest: str
    program_digest: str
    predicate_id: str
    operation: Mapping[str, Any]
    required_execution: tuple[Mapping[str, Any], ...]
    boundary_segment_id: str
    boundary_source_path: tuple[str | int, ...]
    step_dt: float
    evidence_format: str
    observation_kind: str = "recorded"
    env_ids: tuple[int, ...] = (0,)
    episode_ids: tuple[int, ...] | None = None
    sampled_contact_identity: Mapping[str, Any] | None = None
    binding_hash: str = ""

    @property
    def object_id(self) -> str:
        """Return the exact declared rigid payload identity."""
        return self.operation["object_id"]

    @property
    def reference_id(self) -> str:
        """Return the exact declared rigid placement reference identity."""
        return self.operation["reference_id"]

    def __post_init__(self) -> None:
        for name in ("task_spec_digest", "program_digest"):
            _digest(getattr(self, name), name)
        for name in ("predicate_id", "boundary_segment_id"):
            _identifier(getattr(self, name), name)
        object.__setattr__(
            self,
            "boundary_source_path",
            _path(self.boundary_source_path, "boundary_source_path"),
        )
        if self.observation_kind != "recorded" or self.evidence_format not in {
            "historical_record",
            "raw_snapshot",
        }:
            raise ValueError(
                "Coordinated transport acquisition is a recorded event, not a final held state."
            )
        if tuple(self.env_ids) != (0,) or any(type(x) is not int for x in self.env_ids):
            raise ValueError(
                "Source coordinated transport requires exact native B=1/env0."
            )
        object.__setattr__(self, "env_ids", (0,))
        if (
            type(self.step_dt) not in (int, float)
            or not math.isfinite(self.step_dt)
            or self.step_dt <= 0
        ):
            raise ValueError(
                "Coordinated transport requires its positive source cadence."
            )
        object.__setattr__(self, "step_dt", float(self.step_dt))
        if self.episode_ids is not None:
            epochs = tuple(self.episode_ids)
            if len(epochs) != 1 or type(epochs[0]) is not int or epochs[0] < 0:
                raise ValueError(
                    "Coordinated transport requires one exact episode identity."
                )
            object.__setattr__(self, "episode_ids", epochs)
        operation = json_snapshot(self.operation)
        if not isinstance(operation, dict) or set(operation) != {
            "operation_id",
            "object_id",
            "reference_id",
            "target_id",
            "resources",
            "transport",
        }:
            raise ValueError(
                "Transport acquisition requires its complete explicit operation route."
            )
        for key in ("operation_id", "object_id", "reference_id", "target_id"):
            _identifier(operation[key], key)
        resources = operation["resources"]
        if (
            not isinstance(resources, dict)
            or set(resources) != {"left", "right"}
            or resources["left"] == resources["right"]
        ):
            raise ValueError(
                "Transport acquisition requires two distinct resource slots."
            )
        for value in resources.values():
            _identifier(value, "resource")
        descriptor = operation["transport"]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "segment_id",
            "segment_name",
            "source_path",
            "invocation_id",
        }:
            raise ValueError(
                "Transport acquisition requires exact compiled occurrence identities."
            )
        for key in ("segment_id", "segment_name", "invocation_id"):
            _identifier(descriptor[key], key)
        _path(descriptor["source_path"], "source_path")
        object.__setattr__(self, "operation", operation)
        required = tuple(json_snapshot(item) for item in self.required_execution)
        for item in required:
            if not isinstance(item, dict) or set(item) != {"segment_id", "source_path"}:
                raise ValueError(
                    "Transport prefix requires exact segment/path declarations."
                )
            _identifier(item["segment_id"], "segment_id")
            _path(item["source_path"], "source_path")
        if (
            not required
            or len({item["segment_id"] for item in required}) != len(required)
            or required[-1]
            != {
                "segment_id": self.boundary_segment_id,
                "source_path": list(self.boundary_source_path),
            }
            or {
                "segment_id": descriptor["segment_name"],
                "source_path": descriptor["source_path"],
            }
            not in required
        ):
            raise ValueError(
                "Transport prefix must contain its acquisition operation and terminal group boundary."
            )
        object.__setattr__(self, "required_execution", required)
        if self.evidence_format == "raw_snapshot":
            if self.episode_ids is None or self.sampled_contact_identity is None:
                raise ValueError(
                    "Raw transport requires its explicit episode and native acquisition identity."
                )
            object.__setattr__(
                self,
                "sampled_contact_identity",
                json_snapshot(self.sampled_contact_identity),
            )
            from ._sampled_contacts import validate_sampled_identity

            validate_sampled_identity(self)
            if self.sampled_contact_identity["step_dt"] != self.step_dt:
                raise ValueError(
                    "Raw transport has another declared recording cadence."
                )
        elif self.sampled_contact_identity is not None:
            raise ValueError(
                "Historical transport cannot borrow a raw acquisition declaration."
            )
        expected = canonical_hash(self._payload())
        if self.binding_hash and self.binding_hash != expected:
            raise ValueError(
                "Transport binding_hash differs from its explicit declaration."
            )
        object.__setattr__(self, "binding_hash", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": "task_program.coordinated_transport_evidence_binding/v2",
            **{
                item.name: json_snapshot(getattr(self, item.name))
                for item in fields(self)
                if item.name != "binding_hash"
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return independently owned declaration data with its original identity."""
        return {**self._payload(), "binding_hash": self.binding_hash}

    @classmethod
    def from_dict(
        cls, value: CoordinatedTransportEvidenceBinding | Mapping[str, Any]
    ) -> CoordinatedTransportEvidenceBinding:
        """Decode and verify a declaration without resigning changed data.

        Args:
            value: Existing binding or complete serialized declaration.

        Returns:
            Independently owned verified binding.
        """
        raw = value.to_dict() if isinstance(value, cls) else json_snapshot(value)
        if (
            not isinstance(raw, dict)
            or raw.pop("schema_version", None)
            != "task_program.coordinated_transport_evidence_binding/v2"
        ):
            raise ValueError(
                "Unsupported coordinated transport evidence binding version."
            )
        return cls(**raw)


def normalize_coordinated_transport_evidence(
    binding: CoordinatedTransportEvidenceBinding | Mapping[str, Any],
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
    from ._transport_evidence import transport_proof
    from ._held_evidence import one, frames
    from ._values import (
        evidence_packet,
        unavailable_packet,
        validate_historical_claim,
        seal_packet,
    )

    selected = CoordinatedTransportEvidenceBinding.from_dict(binding)
    program = json_snapshot(program)
    if canonical_hash(program) != selected.program_digest:
        raise ValueError("Execution evidence belongs to another declared program.")
    count = len(selected.env_ids)
    kind = "coordinated_acquisition"
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
                and ("record" not in snapshot or "recorded_contacts" in snapshot)
            )
        ):
            raise ValueError(
                "Transport evidence cannot mix raw and historical contracts."
            )
        if selected.episode_ids is not None and (
            snapshot.get("episode_ids") != list(selected.episode_ids)
            or any((type(x) is not int for x in snapshot["episode_ids"]))
        ):
            raise ValueError("Transport event belongs to another episode.")
        segments = {item["name"]: item for item in snapshot["segments"]}
        if len(segments) != len(snapshot["segments"]):
            raise ValueError("Transport has duplicate execution occurrences.")
        for item in selected.required_execution:
            segment = segments[item["segment_id"]]
            meta = segment["metadata"]
            if (
                meta["program_segment_source_path"] != item["source_path"]
                or meta["runtime"]["env_ids"] != [0]
                or any((type(x) is not int for x in meta["runtime"]["env_ids"]))
                or (segment.get("active") != [True])
                or (segment.get("successes") != [True])
                or (not _bool_rows(segment.get("active"), 1))
                or (not _bool_rows(segment.get("successes"), 1))
            ):
                raise ValueError(
                    "Transport milestone requires the original complete successful execution prefix."
                )
        boundary = frames(segments[selected.boundary_segment_id], 1)
        if snapshot.get("boundary_frames") != boundary or any(
            (type(x) is not int for x in snapshot["boundary_frames"])
        ):
            raise ValueError(
                "Transport event report has another recorded group boundary."
            )
        if raw:
            from ._sampled_contacts import read_sampled_timeline

            payload = snapshot["recorded_contacts"]
            if payload["boundary_frames"] != boundary or any(
                (type(x) is not int for x in payload["boundary_frames"])
            ):
                raise ValueError(
                    "Raw transport acquisition has another report boundary."
                )
            timeline = read_sampled_timeline(selected, payload)
        else:
            timeline = snapshot["contact_timeline"]
        if timeline["step_dt"] != selected.step_dt:
            raise ValueError("Transport event has another source recording cadence.")
        *_, proof = transport_proof(
            selected.operation, {"segments": snapshot["segments"]}, program, timeline
        )
        expected = dict(
            group_id=selected.operation["operation_id"],
            uid=selected.object_id,
            part_id=None,
            accepted=[True],
            measurements=proof,
        )
        packet = evidence_packet(selected, snapshot, kind)
        packet.update(facts={"acquired_by_all": [True]}, boundary_frames=boundary)
        if not raw:
            validate_historical_claim(snapshot["record"], expected)
            packet["claimed_result"] = snapshot["record"]["accepted"]
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
