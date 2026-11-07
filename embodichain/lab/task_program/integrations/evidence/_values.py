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

"""Closed JSON identities for offline execution-evidence adapters."""

from __future__ import annotations
from typing import Any
from collections.abc import Mapping
import math

__all__: list[str] = []


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be an exact nonempty identifier.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _path(value: Any, name: str) -> tuple[str | int, ...]:
    if (
        not isinstance(value, (tuple, list))
        or not value
        or any(
            type(item) not in (str, int)
            or (type(item) is str and not item)
            or (type(item) is int and item < 0)
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain exact nonempty source-path components.")
    return tuple(value)


def _bool_rows(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(type(item) is bool for item in value)
    )


def _axis(value: Any, name: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError(f"{name} must contain three finite real values.")
    norm = math.sqrt(sum(v * v for v in value))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"{name} must be finite and nonzero.")
    return tuple(float(v / norm) for v in value)


def evidence_packet(
    binding: Any, source: Mapping[str, Any], kind: str
) -> dict[str, Any]:
    """Start an owned normalized packet with exact original provenance."""
    from embodichain._json_contract import canonical_hash

    return dict(
        kind=kind,
        env_ids=list(binding.env_ids),
        episode_ids=None if binding.episode_ids is None else list(binding.episode_ids),
        source_binding_hash=binding.binding_hash,
        valid=[True] * len(binding.env_ids),
        acquisition_errors=[None] * len(binding.env_ids),
        facts={},
        evidence_refs=[binding.binding_hash, canonical_hash(source)],
    )


def unavailable_packet(
    binding: Any, source: Mapping[str, Any], kind: str, error: Exception
) -> dict[str, Any]:
    packet = evidence_packet(binding, source, kind)
    packet.update(
        valid=[False] * len(binding.env_ids),
        acquisition_errors=[str(error) or type(error).__name__] * len(binding.env_ids),
        unavailable=str(error) or type(error).__name__,
    )
    boundary = source.get("boundary_frames")
    if (
        isinstance(boundary, list)
        and len(boundary) == len(binding.env_ids)
        and all(type(frame) is int and 0 <= frame <= 2**53 for frame in boundary)
        and math.isfinite(max(boundary) * binding.step_dt)
    ):
        packet["boundary_frames"] = boundary
    return seal_packet(packet, binding.step_dt)


def validate_historical_claim(
    claimed: Mapping[str, Any], measured: Mapping[str, Any]
) -> None:
    """Check historical measurements; TaskSpec separately checks its verdict."""
    from embodichain._json_contract import canonical_hash

    if canonical_hash(
        {k: v for k, v in claimed.items() if k != "accepted"}
    ) != canonical_hash({k: v for k, v in measured.items() if k != "accepted"}):
        raise ValueError(
            "Historical claims differ from exact receipts and physical measurements."
        )


def seal_packet(packet: dict[str, Any], step_dt: float) -> dict[str, Any]:
    """Reuse owned Lab frame/batch contracts for derived, normalized facts."""
    import torch
    from embodichain.lab.evidence import PhysicalEvidenceBatch, PhysicalEvidenceFrame
    from embodichain._json_contract import canonical_hash

    ids = torch.tensor(packet["env_ids"], dtype=torch.long)
    boundary = packet.get("boundary_frames", [0] * len(ids))
    timestamp, revision = float(max(boundary) * step_dt), int(max(boundary))
    valid = torch.tensor(packet.pop("valid"), dtype=torch.bool)
    errors = tuple(packet.pop("acquisition_errors"))
    batches = {}
    for name, values in packet.pop("facts").items():
        dtype = (
            torch.bool
            if name in {"grips", "transfer_complete", "acquired_by_all"}
            else torch.float64
        )
        batches[name] = PhysicalEvidenceBatch(
            evidence_id=name,
            values=torch.tensor(values, dtype=dtype),
            valid=valid,
            acquisition_errors=errors,
            timestamp=timestamp,
            env_ids=ids,
            observation_revision=revision,
        )
    frame = PhysicalEvidenceFrame(
        timestamp=timestamp, observation_revision=revision, env_ids=ids, batches=batches
    )
    packet.update(
        schema_version="physical_evidence.normalized/v1",
        frame=frame.to_dict(),
        derivation="validated_source_channels_at_control_boundary",
    )
    packet["measurement_hash"] = canonical_hash(packet)
    return packet
