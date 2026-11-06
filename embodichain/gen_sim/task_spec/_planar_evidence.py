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

"""Source planar footprint and containment qualification over physical facts."""

from __future__ import annotations
from collections.abc import Mapping
import math
from typing import Any
from ._evidence_values import flags, number, matrix

__all__: list[str] = []
_SUPPORT = 0.05
_PENETRATION = 0.01


def validate_planar_geometry(binding: Any) -> None:
    from .evidence import _identifier, _digest

    value = binding.planar_geometry
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "exact_silhouette",
            "program_fingerprint",
            "object_native_uid",
            "reference_native_uid",
        }
        or type(value["exact_silhouette"]) is not bool
    ):
        raise ValueError("Planar evidence requires an explicit sealed geometry mode.")
    for key in ("object_native_uid", "reference_native_uid"):
        _identifier(value[key], key)
    _digest(value["program_fingerprint"], "program_fingerprint")
    if value["object_native_uid"] == value["reference_native_uid"]:
        raise ValueError("Planar native owners must be distinct.")


def planar_flags(
    binding: Any, metrics: Mapping[str, Any], released: list[bool], relation: str
) -> tuple[list[bool], list[bool]]:
    from .evidence import _bool_rows

    count = len(binding.env_ids)
    exact = binding.planar_geometry["exact_silhouette"]
    if (
        metrics.get("scope") != "measured_planar_landing_region"
        or metrics.get("support_tolerance_m") != _SUPPORT
        or metrics.get("penetration_tolerance_m") != _PENETRATION
        or not _bool_rows(metrics.get("full_footprint_inside"), count)
    ):
        raise ValueError(
            "Planar observations have another footprint or source tolerance."
        )
    for key in ("support_gap_m", "footprint_area_m2"):
        values = metrics[key]
        if (
            not isinstance(values, list)
            or len(values) != count
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
            or key == "footprint_area_m2"
            and any(x <= 0 for x in values)
        ):
            raise ValueError("Planar metrics are incomplete or nonfinite.")
    matrix(metrics["object_pose"], count)
    matrix(metrics["reference_pose"], count)
    if exact != (metrics.get("footprint_kind") == "native_triangle_projection/v1"):
        raise ValueError(
            "Planar evidence cannot substitute a hull for triangle projection."
        )
    available = [True] * count
    if exact:
        for key in ("reference_has_raised_geometry", "reference_aabb_overlap"):
            if not _bool_rows(metrics.get(key), count):
                raise ValueError("Planar containment availability is missing.")
        areas = metrics["outside_floor_area_m2"]
        if (
            not isinstance(areas, list)
            or len(areas) != count
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or x < 0
                for x in areas
            )
        ):
            raise ValueError("Exact footprint outside area is invalid.")
        if relation == "inside":
            available = [
                not (
                    free
                    and raised
                    and overlap
                    and not footprint
                    and -_PENETRATION <= gap <= _SUPPORT
                )
                for free, raised, overlap, footprint, gap in zip(
                    released,
                    metrics["reference_has_raised_geometry"],
                    metrics["reference_aabb_overlap"],
                    metrics["full_footprint_inside"],
                    metrics["support_gap_m"],
                    strict=True,
                )
            ]
    accepted = [
        free and footprint and -_PENETRATION <= gap <= _SUPPORT
        for free, footprint, gap in zip(
            released,
            metrics["full_footprint_inside"],
            metrics["support_gap_m"],
            strict=True,
        )
    ]
    return accepted, available
