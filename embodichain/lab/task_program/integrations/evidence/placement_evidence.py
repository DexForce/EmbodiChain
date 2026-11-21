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

"""Normalize placement observation protocols before normative evaluation."""

from __future__ import annotations
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from embodichain._json_contract import canonical_hash, json_snapshot
from ._held_evidence import one
from ._values import _digest, _identifier, _path
from ._release_evidence import measured_release

__all__ = [
    "normalize_placement_evidence",
    "normalize_pour_release_evidence",
    "validate_placement_declaration",
]

_ALLOWED = frozenset(
    [
        "task_spec_digest",
        "program_digest",
        "predicate_id",
        "object_id",
        "reference_id",
        "resource_id",
        "release_segment_id",
        "release_invocation_id",
        "release_source_path",
        "support_segment_id",
        "stability_preset",
        "support_source_path",
        "boundary_segment_id",
        "boundary_source_path",
        "evidence_scope",
        "observation_kind",
        "evidence_format",
        "condition",
        "required_execution",
        "env_ids",
        "episode_ids",
        "binding_hash",
        "operation_evidence",
        "sampled_contact_identity",
        "planar_geometry",
        "directional_geometry",
        "schema_version",
    ]
)


def _declaration(value: Mapping[str, Any]) -> SimpleNamespace:
    raw = json_snapshot(value)
    if not isinstance(raw, dict) or set(raw) - _ALLOWED:
        raise ValueError("Unknown placement declaration fields.")
    if raw.get("binding_hash") != canonical_hash(
        {k: v for k, v in raw.items() if k != "binding_hash"}
    ):
        raise ValueError("Placement binding_hash differs from its sealed identity.")
    value = {
        **dict(
            operation_evidence=None,
            sampled_contact_identity=None,
            planar_geometry=None,
            directional_geometry=None,
        ),
        **raw,
    }
    for key in (
        "env_ids",
        "episode_ids",
        "release_source_path",
        "support_source_path",
        "boundary_source_path",
    ):
        if value.get(key) is not None:
            value[key] = tuple(value[key])
    if value.get("schema_version") not in {
        f"task_spec.placement_evidence_binding/v{n}" for n in range(2, 8)
    }:
        raise ValueError("Unsupported source placement declaration version.")
    for key in ("task_spec_digest", "program_digest"):
        _digest(value.get(key), key)
    for key in ("predicate_id", "object_id", "reference_id", "resource_id"):
        _identifier(value.get(key), key)
    ids = value.get("env_ids")
    if not ids or ids != tuple(range(len(ids))) or any(type(x) is not int for x in ids):
        raise ValueError("Placement declaration requires ordered native rows.")
    for key in ("release_source_path", "support_source_path", "boundary_source_path"):
        _path(value.get(key), key)
    selected = SimpleNamespace(**value)
    if selected.operation_evidence is not None:
        from ._placement_linkage import validate_linkage

        validate_linkage(selected)
    if selected.sampled_contact_identity is not None:
        from ._sampled_contacts import validate_sampled_identity

        validate_sampled_identity(selected)
    return selected


def validate_placement_declaration(declaration: Mapping[str, Any]) -> None:
    """Reject unsupported source/actor/occurrence declarations before acquisition.

    Args:
        declaration: Complete content-addressed runtime observation declaration.
    """
    _declaration(declaration)


def _segment(snapshot: Mapping[str, Any], identity: str) -> Mapping[str, Any]:
    exact = [
        item
        for item in snapshot["segments"]
        if item["metadata"].get("program_segment_id") == identity
    ]
    return one(
        exact
        if exact
        else [item for item in snapshot["segments"] if item["name"] == identity]
    )


def _release(
    binding: Any,
    snapshot: dict[str, Any],
    program: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> tuple[Any, ...]:
    if (
        binding.operation_evidence is not None
        and binding.operation_evidence["family"] == "E5"
    ):
        from ._placement_linkage import transport_release

        return transport_release(binding, snapshot, program, boundary)
    segment = _segment(snapshot, binding.release_segment_id)
    return measured_release(
        segment["metadata"]["runtime"],
        boundary["metadata"]["runtime"],
        object_uid=binding.object_id,
        resource=binding.resource_id,
        invocation_id=binding.release_invocation_id,
        num_envs=len(binding.env_ids),
    )


def _store_release(
    binding: Any, snapshot: dict[str, Any], result: tuple[Any, ...]
) -> None:
    released, available, receipt, call = result
    if getattr(binding, "evidence_scope", None) == "native_supported_placement":
        from ._release_evidence import release_window

        call = {
            **call,
            "normalized_release_window": release_window(
                call, receipt, len(binding.env_ids)
            ),
        }
    facts = dict(
        object_id=binding.object_id,
        resource_id=binding.resource_id,
        invocation_id=binding.release_invocation_id,
        env_ids=list(binding.env_ids),
        released=released,
        available=available,
        receipt=receipt,
        call=call,
    )
    facts["normalization_hash"] = canonical_hash(facts)
    segment = _segment(snapshot, binding.release_segment_id)
    segment["metadata"]["runtime"]["normalized_release"] = facts


def normalize_placement_evidence(
    declaration: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
    source_family: str | None = None,
) -> dict[str, Any]:
    """Validate raw channels and emit source-bound release/geometry facts.

    Args:
        declaration: Complete serialized source observation declaration.
        evidence: Original raw or historical snapshot.
        program: Exact declared program owning the runtime occurrences.
        source_family: Optional explicit source route needed by its existing protocol.

    Returns:
        An independent snapshot containing normalized facts and source hashes.
        No task predicate or success threshold is evaluated.
    """
    binding = _declaration(declaration)
    if binding.program_digest != canonical_hash(program):
        raise ValueError(
            "Placement observations have another declared program identity."
        )
    snapshot = json_snapshot(evidence)
    snapshot["source_evidence_hash"] = canonical_hash(evidence)
    if "unavailable" in snapshot:
        return snapshot
    prepared = dict(declaration_hash=binding.binding_hash)
    try:
        if binding.operation_evidence is not None:
            from ._placement_linkage import validate_linkage

            validate_linkage(binding)
        if binding.sampled_contact_identity is not None:
            from ._sampled_contacts import validate_sampled_identity

            validate_sampled_identity(binding)
        boundary = _segment(snapshot, binding.boundary_segment_id)
        historical = binding.evidence_format == "historical_record"
        if not historical and binding.evidence_scope == "native_stable_support":
            from ._sampled_contacts import prepare_sampled_support

            metrics, timeline, poses = prepare_sampled_support(
                binding, snapshot, boundary
            )
            prepared.update(metrics=metrics, timeline=timeline, poses=poses)
            snapshot["contact_timeline"], snapshot["boundary_poses"] = timeline, poses
        elif (
            not historical
            and binding.evidence_scope == "measured_planar_landing_region"
        ):
            from ._geometry_protocols import raw_planar_metrics

            prepared["metrics"] = raw_planar_metrics(binding, snapshot, boundary)
        elif not historical and binding.directional_geometry is not None:
            from ._geometry_protocols import raw_directional_metrics

            prepared["metrics"] = raw_directional_metrics(binding, snapshot, boundary)
        if historical:
            prepared["metrics"] = json_snapshot(snapshot["record"]["measurements"])
        elif binding.evidence_scope == "native_supported_landing_region":
            prepared["metrics"] = dict(
                scope=binding.evidence_scope,
                placement_node_id=binding.release_segment_id,
                boundary_node_id=binding.boundary_segment_id,
                observation_kind=binding.observation_kind,
                native_support=snapshot["native_support"],
                landing_region=snapshot["landing_region"],
                support_gap_m=snapshot["support_gap_m"],
                contact_tolerance_m=0.01,
                reference_uid=binding.reference_id,
                relation="on",
                boundary_frames=boundary["end_steps"],
            )
        result = _release(binding, snapshot, program, boundary)
        _store_release(binding, snapshot, result)
        prepared["release"] = list(result)
        metrics = prepared.get("metrics")
        if metrics is not None and binding.directional_geometry is not None:
            from ._geometry_protocols import verify_directional_protocol

            verify_directional_protocol(
                binding, metrics, result[0], boundary, program, family=source_family
            )
            metrics["source_protocol_verified"] = True
        if (
            metrics is not None
            and binding.evidence_scope == "native_supported_landing_region"
        ):
            from ._geometry_protocols import verify_legacy_region_protocol

            temporary = json_snapshot(snapshot)
            temporary["record"] = dict(
                uid=binding.object_id, part_id=None, measurements=metrics
            )
            verify_legacy_region_protocol(
                binding, temporary, boundary, result[3], result[2], result[0], program
            )
            prepared["legacy_region_verified"] = True
        if binding.operation_evidence is not None:
            from ._placement_linkage import qualify_linkage

            prepared["dual_release"] = binding.operation_evidence["family"] == "E5"
            temporary = json_snapshot(snapshot)
            if not historical:
                metrics = prepared.get("metrics")
                if metrics is not None:
                    temporary["record"] = dict(
                        group_id=binding.operation_evidence["operation"][
                            "operation_id"
                        ],
                        uid=binding.object_id,
                        part_id=None,
                        measurements=metrics,
                    )
                else:
                    temporary["record"] = dict(
                        group_id=binding.operation_evidence["operation"][
                            "operation_id"
                        ],
                        uid=binding.object_id,
                        part_id=None,
                        measurements={},
                    )
            qualify_linkage(binding, temporary, program, result[2])
            prepared["prerequisite_verified"] = True
            if "record" in temporary and "metrics" in prepared:
                prepared["metrics"] = temporary["record"]["measurements"]
    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
        IndexError,
        StopIteration,
    ) as error:
        prepared["unavailable"] = str(error) or type(error).__name__
        snapshot["unavailable"] = prepared["unavailable"]
    prepared["normalization_hash"] = canonical_hash(prepared)
    snapshot["normalized_placement"] = prepared
    snapshot["normalization_hash"] = canonical_hash(snapshot)
    return snapshot


def normalize_pour_release_evidence(
    declaration: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a Pour release dependency without deciding the Pour predicate.

    Args:
        declaration: Complete explicit Pour source declaration.
        evidence: Original receipt/measurement snapshot.
        program: Original program, used for its content identity.

    Returns:
        A copied snapshot retaining unavailable acquisition failures.
    """
    raw = json_snapshot(declaration)
    snapshot = json_snapshot(evidence)
    if "unavailable" in snapshot or raw.get("release_segment_id") is None:
        snapshot["normalization_hash"] = canonical_hash(snapshot)
        return snapshot
    if raw.get("binding_hash") != canonical_hash(
        {k: v for k, v in raw.items() if k != "binding_hash"}
    ):
        raise ValueError("Pour binding_hash differs from its sealed declaration.")
    if raw.get("program_digest") != canonical_hash(program):
        raise ValueError("Pour release has another program identity.")
    raw["operation_evidence"] = None
    raw["env_ids"] = tuple(raw["env_ids"])
    release = one(
        x for x in raw["occurrences"] if x["segment_id"] == raw["release_segment_id"]
    )
    raw["release_invocation_id"] = release["invocation_id"]
    binding = SimpleNamespace(**raw)
    try:
        boundary = _segment(snapshot, binding.boundary_segment_id)
        result = _release(binding, snapshot, program, boundary)
        _store_release(binding, snapshot, result)
    except (
        KeyError,
        ValueError,
        TypeError,
        RuntimeError,
        AttributeError,
        IndexError,
        StopIteration,
    ) as error:
        snapshot["unavailable"] = str(error) or type(error).__name__
    snapshot["normalization_hash"] = canonical_hash(snapshot)
    return snapshot
