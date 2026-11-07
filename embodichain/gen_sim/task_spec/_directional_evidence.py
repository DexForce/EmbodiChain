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

"""Explicit world-axis side placement and selected-link evidence contracts."""

from __future__ import annotations
from collections.abc import Mapping
import math
from typing import Any
from ._evidence_values import matrix, number, flags, one

from embodichain.compute.geometry.directional import DIRECTIONAL_RELATION_AXES as AXES

__all__: list[str] = []


def validate_directional_geometry(binding: Any) -> None:
    from .evidence import _identifier, _digest

    value = binding.directional_geometry
    if not isinstance(value, Mapping) or set(value) != {
        "relation",
        "program_fingerprint",
        "object_native_uid",
        "reference_native_uid",
        "table_id",
        "table_native_uid",
        "reference_part",
        "reference_link_id",
        "native_link_name",
        "world_displacement",
    }:
        raise ValueError(
            "Directional evidence requires a complete explicit spatial route."
        )
    if value["relation"] not in AXES:
        raise ValueError("Unsupported directional relation.")
    for key in (
        "object_native_uid",
        "reference_native_uid",
        "table_id",
        "table_native_uid",
    ):
        _identifier(value[key], key)
    _digest(value["program_fingerprint"], "program_fingerprint")
    for key in ("reference_part", "reference_link_id", "native_link_name"):
        if value[key] is not None:
            _identifier(value[key], key)
    linked = value["reference_link_id"] is not None
    if (
        linked != (binding.evidence_scope == "bound_link_world_axes")
        or linked != (value["native_link_name"] is not None)
        or not linked
        and value["reference_part"] is not None
    ):
        raise ValueError(
            "Directional geometry differs from its exact rigid/link route."
        )
    if binding.condition.get("kind") != "placement" or binding.condition.get(
        "reference"
    ) != (value["reference_link_id"] or binding.reference_id):
        raise ValueError("Directional condition has another resolved reference.")
    for values in (value["world_displacement"], binding.condition.get("displacement")):
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
        ):
            raise ValueError(
                "Directional placement requires exact finite release and condition offsets."
            )


def directional_flags(
    binding: Any,
    metrics: Mapping[str, Any],
    released: list[bool],
    boundary: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    family: str | None,
) -> list[bool]:
    if metrics.get("source_protocol_verified") is not True:
        raise ValueError(
            "Directional measurements require a verified source declaration."
        )
    table = metrics["tabletop"]
    gaps = metrics["envelope_gaps_m"]
    return [
        free
        and footprint
        and -0.01 <= gap <= 0.05
        and all(axis[row] >= 0 for axis in gaps)
        for row, (free, footprint, gap) in enumerate(
            zip(
                released,
                table["full_footprint_inside"],
                table["support_gap_m"],
                strict=True,
            )
        )
    ]
