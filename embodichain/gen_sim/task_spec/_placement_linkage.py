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

"""Pure transfer/transport linkage required before qualifying terminal placement."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any

__all__: list[str] = []


def validate_linkage(binding: Any) -> None:
    from .evidence import _identifier, _path
    from .handover_evidence import HandOverEvidenceBinding

    value = binding.operation_evidence
    if not isinstance(value, Mapping) or set(value) != {
        "family",
        "operation",
        "step_dt",
    }:
        raise ValueError(
            "Placement linkage requires an explicit family, operation and cadence."
        )
    from ._held_evidence import number

    if number(value["step_dt"]) <= 0:
        raise ValueError("Placement linkage requires a positive cadence.")
    operation = value["operation"]
    if value["family"] == "E4":
        HandOverEvidenceBinding(
            binding.task_spec_digest,
            binding.program_digest,
            binding.predicate_id,
            operation,
            binding.observation_kind,
            value["step_dt"],
        )
        if (
            operation["destination_resource"] != binding.resource_id
            or operation["object_id"] != binding.object_id
        ):
            raise ValueError(
                "HandOver placement must release its exact receiver and object."
            )
    elif value["family"] == "E5":
        if not isinstance(operation, Mapping) or set(operation) != {
            "operation_id",
            "object_id",
            "reference_id",
            "target_id",
            "resources",
            "transport",
        }:
            raise ValueError(
                "Coordinated placement requires its exact transport declaration."
            )
        for key in ("operation_id", "object_id", "reference_id", "target_id"):
            _identifier(operation[key], key)
        resources = operation["resources"]
        if (
            not isinstance(resources, Mapping)
            or set(resources) != {"left", "right"}
            or resources["left"] == resources["right"]
        ):
            raise ValueError(
                "Coordinated placement requires two distinct participants."
            )
        for resource in resources.values():
            _identifier(resource, "resource")
        descriptor = operation["transport"]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "segment_id",
            "segment_name",
            "source_path",
            "invocation_id",
        }:
            raise ValueError(
                "Coordinated transport requires exact compiled identities."
            )
        for key in ("segment_id", "segment_name", "invocation_id"):
            _identifier(descriptor[key], key)
        _path(descriptor["source_path"], "source_path")
        if (
            operation["object_id"] != binding.object_id
            or operation["reference_id"] != binding.reference_id
            or binding.resource_id not in resources.values()
            or binding.env_ids != (0,)
            or descriptor["segment_name"] != binding.release_segment_id
            or tuple(descriptor["source_path"]) != binding.release_source_path
            or descriptor["invocation_id"] != binding.release_invocation_id
            or binding.evidence_scope != "native_stable_support"
        ):
            raise ValueError(
                "Coordinated placement requires its source single-row sampled support route."
            )
    else:
        raise ValueError("Unsupported placement operation family.")


def qualify_linkage(
    binding: Any,
    snapshot: Mapping[str, Any],
    program: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    from ._held_evidence import number

    value = binding.operation_evidence
    if snapshot["record"].get("group_id") != value["operation"]["operation_id"]:
        raise ValueError("Placement record belongs to another linked operation.")
    runtime = {"segments": snapshot["segments"], "lengths": snapshot.get("lengths", [])}
    if value["family"] == "E4":
        from ._handover_evidence import transfer_record

        proof, _ = transfer_record(
            value["operation"],
            runtime,
            program,
            len(binding.env_ids),
            binding.observation_kind == "final",
            value["step_dt"],
        )
        if number(proof["measurements"]["transfer_completed_at"]) > number(
            receipt["timestamp"]
        ):
            raise ValueError("HandOver transfer must precede its receiver release.")
    else:
        from ._transport_evidence import transport_proof

        timeline = snapshot["contact_timeline"]
        if timeline["step_dt"] != value["step_dt"]:
            raise ValueError("Coordinated transport has another recording cadence.")
        _, _, _, terminal, proof = transport_proof(
            value["operation"], runtime, program, timeline
        )
        from .canonicalization import canonical_hash

        if canonical_hash(terminal) != canonical_hash(receipt):
            raise ValueError(
                "Coordinated support and transport name different releases."
            )
        metrics = snapshot["record"]["measurements"]
        if binding.evidence_format == "raw_snapshot":
            metrics.update(
                {
                    "transport_scope" if key == "scope" else key: value
                    for key, value in proof.items()
                }
            )
        if any(
            metrics.get("transport_scope" if key == "scope" else key) != expected
            for key, expected in proof.items()
        ):
            raise ValueError(
                "Coordinated placement record differs from its exact transport acquisition."
            )


def transport_release(
    binding: Any,
    snapshot: Mapping[str, Any],
    program: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> tuple[Any, ...]:
    from ._transport_evidence import transport_proof

    operation = binding.operation_evidence["operation"]
    _, _, call, terminal, _ = transport_proof(
        operation,
        {"segments": snapshot["segments"]},
        program,
        snapshot["contact_timeline"],
    )
    state = boundary["metadata"]["runtime"]["task_state"]
    if any(
        value["object_id"] == binding.object_id and any(value["active_mask"])
        for value in [*state["held_objects"], *state["coordinated_held_objects"]]
    ):
        raise ValueError(
            "Coordinated placement is still held at its observation boundary."
        )
    return [True], [True], terminal, call
