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

"""Normative prerequisites supplied by an execution evidence adapter."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from .canonicalization import canonical_hash, json_snapshot

__all__: list[str] = []


def validate_linkage(binding: Any) -> None:
    value = json_snapshot(binding.operation_evidence)
    if not isinstance(value, dict) or not value:
        raise ValueError("Placement prerequisite needs an explicit source declaration.")


def _normalized(binding: Any, snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = snapshot["normalized_placement"]
    if normalized.get("declaration_hash") != binding.binding_hash or normalized.get(
        "normalization_hash"
    ) != canonical_hash(
        {k: v for k, v in normalized.items() if k != "normalization_hash"}
    ):
        raise ValueError(
            "Placement facts differ from their declared normalization identity."
        )
    if normalized.get("unavailable"):
        raise ValueError(str(normalized["unavailable"]))
    return normalized


def qualify_linkage(
    binding: Any,
    snapshot: Mapping[str, Any],
    program: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    normalized = _normalized(binding, snapshot)
    if normalized.get("prerequisite_verified") is not True:
        raise ValueError("Required transfer/acquisition prerequisite was not verified.")


def transport_release(
    binding: Any,
    snapshot: Mapping[str, Any],
    program: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> tuple[Any, ...]:
    normalized = _normalized(binding, snapshot)
    return tuple(normalized["release"])
