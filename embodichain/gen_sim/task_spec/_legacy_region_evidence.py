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

"""Explicit legacy table-region load/footprint proof, separate from revision one."""

from __future__ import annotations
from collections.abc import Mapping
import math
from typing import Any
from ._evidence_values import matrix, one, flags
from ._strong_support import _region_flags, _region_poses_match
from ._support_contract import release_window

__all__: list[str] = []


def validate_legacy_region(binding: Any) -> None:
    cfg = binding.condition
    if (
        cfg.get("kind") != "supported_region"
        or cfg.get("support_contract_revision") is not None
        or cfg.get("reference") != "table"
        or binding.reference_id != "table"
        or cfg.get("displacement") is not None
        or cfg.get("target_position") is not None
        or binding.operation_evidence is not None
    ):
        raise ValueError(
            "Legacy region evidence requires its exact bare-table declaration without a strong revision."
        )


def legacy_region_flags(
    binding: Any,
    snapshot: Mapping[str, Any],
    boundary: Mapping[str, Any],
    release_call: Mapping[str, Any],
    receipt: Mapping[str, Any],
    released: list[bool],
    program: Mapping[str, Any],
) -> list[bool]:
    from ._placement_linkage import _normalized

    prepared = _normalized(binding, snapshot)
    if prepared.get("legacy_region_verified") is not True:
        raise ValueError("Region measurements require a verified observation scope.")
    metrics = snapshot["record"]["measurements"]
    count = len(binding.env_ids)
    pose = matrix(snapshot["boundary_poses"][binding.object_id], count)
    reference = matrix(snapshot["boundary_poses"][binding.reference_id], count)
    physical = _region_flags(
        metrics["native_support"],
        metrics["landing_region"],
        pose,
        reference,
        binding.object_id,
        metrics["support_gap_m"],
        count,
        reference_uid=binding.reference_id,
    )
    return [value and free for value, free in zip(physical, released, strict=True)]
