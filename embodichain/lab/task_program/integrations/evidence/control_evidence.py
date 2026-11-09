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

"""Offline normalization of historical and raw physical control-operation proofs."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from embodichain._json_contract import canonical_hash, json_snapshot
from ._values import _digest, _identifier, evidence_packet, seal_packet
from ._control_records import qualified_control_record
from ._raw_control_evidence import validate_control_acquisition, verify_raw_control

__all__ = ["normalize_control_evidence", "validate_control_declaration"]

_FIELDS = {
    "schema_version",
    "task_spec_digest",
    "program_digest",
    "predicate_id",
    "entity_id",
    "control_kind",
    "observation_kind",
    "operation",
    "evidence_format",
    "env_ids",
    "episode_ids",
    "binding_hash",
}


def _declaration(value: Mapping[str, Any]) -> SimpleNamespace:
    raw = json_snapshot(value)
    if not isinstance(raw, dict):
        raise ValueError("Control observation declaration must be a mapping.")
    version = raw.get("schema_version")
    expected = _FIELDS | (
        {"control_identity", "policy_occurrences"}
        if version == "task_spec.control_evidence_binding/v2"
        else set()
    )
    if (
        version
        not in {
            "task_spec.control_evidence_binding/v1",
            "task_spec.control_evidence_binding/v2",
        }
        or set(raw) != expected
    ):
        raise ValueError("Unsupported or incomplete control observation declaration.")
    for key in ("task_spec_digest", "program_digest", "binding_hash"):
        _digest(raw[key], key)
    if raw["binding_hash"] != canonical_hash(
        {key: value for key, value in raw.items() if key != "binding_hash"}
    ):
        raise ValueError("Control declaration changed under its binding_hash.")
    for key in ("predicate_id", "entity_id"):
        _identifier(raw[key], key)
    if raw["control_kind"] not in {"press", "twist"} or raw["observation_kind"] not in {
        "final",
        "recorded",
    }:
        raise ValueError("Control observation requires an explicit effect and scope.")
    ids = raw["env_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or ids != list(range(len(ids)))
        or any(type(v) is not int for v in ids)
    ):
        raise ValueError("Control observation requires ordered environment identities.")
    episodes = raw["episode_ids"]
    if episodes is not None and (
        not isinstance(episodes, list)
        or len(episodes) != len(ids)
        or any(type(v) is not int or v < 0 for v in episodes)
    ):
        raise ValueError("Control observation has invalid episode identities.")
    historical = version == "task_spec.control_evidence_binding/v1"
    if raw["evidence_format"] != (
        "historical_record" if historical else "raw_snapshot"
    ):
        raise ValueError("Control observation format differs from its sealed version.")
    raw["env_ids"] = tuple(ids)
    raw["episode_ids"] = None if episodes is None else tuple(episodes)
    binding = SimpleNamespace(**raw)
    if not historical:
        validate_control_acquisition(binding)
    return binding


def validate_control_declaration(declaration: Mapping[str, Any]) -> None:
    """Validate an owned control-acquisition declaration without physical access.

    Args:
        declaration: Complete serialized control binding, including its hash.

    Raises:
        ValueError: If format, identity, actor roster or occurrence scope is invalid.
    """
    _declaration(declaration)


def normalize_control_evidence(
    declaration: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck control receipts and emit source-bound physical effect facts.

    Args:
        declaration: Serialized control binding with its original identity.
        evidence: Complete raw acquisition or explicitly selected historical record.
        program: Exact serialized program owning the declared occurrences.

    Returns:
        Owned normalized PhysicalEvidenceFrame/Batch JSON. Legacy incomplete or
        rejected operation proofs remain unavailable. No simulator is constructed
        and no TaskSpec predicate is evaluated.
    """
    binding = _declaration(declaration)
    if binding.program_digest != canonical_hash(program):
        raise ValueError("Control observations have another declared program identity.")
    source = json_snapshot(evidence)
    packet = evidence_packet(binding, source, "control_" + binding.control_kind)
    packet.update(
        evidence_format=binding.evidence_format,
        criterion=binding.operation["route_payload"].get(
            "success_criterion", "contact_backed_press"
        ),
    )
    # Retain a known event boundary even when its proof is unavailable; missing
    # boundaries do not acquire an invented observation or milestone.
    boundaries = source.get("boundary_frames")
    if boundaries is None:
        boundaries = (
            source.get("record", {}).get("measurements", {}).get("boundary_frames")
        )
    if boundaries is not None:
        packet["boundary_frames"] = boundaries
    timestamp = 0.0
    try:
        if "unavailable" in source:
            raise ValueError(str(source["unavailable"]))
        if binding.episode_ids is not None and (
            source.get("episode_ids") != list(binding.episode_ids)
            or any(type(v) is not int for v in source.get("episode_ids", ()))
        ):
            raise ValueError("Control snapshots belong to another declared episode.")
        raw = binding.evidence_format == "raw_snapshot"
        if raw:
            if "record" in source or "control_acquisition" not in source:
                raise ValueError(
                    "Raw control cannot borrow historical claims or omit acquisition."
                )
            verify_raw_control(binding, source, program)
        elif "record" not in source or any(
            key in source
            for key in ("acquisition", "control_acquisition", "physical_evidence")
        ):
            raise ValueError(
                "Historical control replay cannot consume a raw acquisition overlay."
            )
        selected = {"control_kind": binding.control_kind, **binding.operation}
        if raw:
            selected["policy_entity_id"] = binding.control_identity["entity_id"]
        record = qualified_control_record(
            selected, {"segments": source["segments"]}, program, len(binding.env_ids)
        )
        if not raw and source["record"] != record:
            raise ValueError(
                "Historical control claims differ from their measured operation receipts."
            )
        packet["boundary_frames"] = record["measurements"]["boundary_frames"]
        # This is the measured event-window time, independent of its step index.
        timestamp = float(record["measurements"]["control"]["window"][1])
        packet["facts"] = {"control_complete": [True] * len(binding.env_ids)}
    except (
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        StopIteration,
        RuntimeError,
        OverflowError,
    ) as error:
        reason = str(error) or type(error).__name__
        packet.update(
            valid=[False] * len(binding.env_ids),
            acquisition_errors=[reason] * len(binding.env_ids),
            facts={"control_complete": [False] * len(binding.env_ids)},
            unavailable=reason,
        )
    boundary = packet.get("boundary_frames")
    if not (
        isinstance(boundary, list)
        and len(boundary) == len(binding.env_ids)
        and all(type(v) is int and 0 < v <= 2**53 for v in boundary)
    ):
        packet.pop("boundary_frames", None)
    return seal_packet(packet, 0.0, timestamp=timestamp)
