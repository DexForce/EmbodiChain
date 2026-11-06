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
from ._held_evidence import flags, number, matrix

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


def raw_planar_metrics(
    binding: Any, snapshot: Mapping[str, Any], boundary: Mapping[str, Any]
) -> dict[str, Any]:
    from .canonicalization import canonical_hash

    count = len(binding.env_ids)
    if (
        snapshot.get("schema_version") != "simulation.planar_observation/v1"
        or snapshot.get("object_id") != binding.object_id
        or snapshot.get("reference_id") != binding.reference_id
        or snapshot.get("observation_kind") != binding.observation_kind
        or snapshot.get("boundary_frames") != boundary["end_steps"]
        or any(type(x) is not int for x in snapshot["boundary_frames"])
    ):
        raise ValueError(
            "Planar geometry has another participant or observation boundary."
        )
    if any(
        snapshot.get(key) != value or type(snapshot.get(key)) is not type(value)
        for key, value in binding.planar_geometry.items()
    ):
        raise ValueError("Planar geometry has another native/program/mode identity.")
    metrics = snapshot["native_planar"]
    frame = snapshot["geometry_frame"]
    number(frame["timestamp"])
    if (
        frame["env_ids"] != list(binding.env_ids)
        or any(type(x) is not int for x in frame["env_ids"])
        or type(frame["observation_revision"]) is not int
        or frame["observation_revision"] < 0
        or frame["clock_domain"] != "simulation_time"
        or frame["revision_kind"] != "planar_geometry_acquisition"
    ):
        raise ValueError("Planar acquisition has another row/clock identity.")
    columns = ["support_gap_m", "footprint_area_m2"] + (
        ["outside_floor_area_m2"] if binding.planar_geometry["exact_silhouette"] else []
    )
    if snapshot["geometry_columns"] != columns:
        raise ValueError("Planar acquisition columns differ from their declared mode.")
    expected = {
        "planar.geometry": [
            [metrics[key][row] for key in columns] for row in range(count)
        ],
        "planar.full_footprint_inside": metrics["full_footprint_inside"],
        "rigid.object_pose": metrics["object_pose"],
        "rigid.reference_pose": metrics["reference_pose"],
    }
    if binding.planar_geometry["exact_silhouette"]:
        expected.update(
            {
                "planar." + key: metrics[key]
                for key in ("reference_has_raised_geometry", "reference_aabb_overlap")
            }
        )
    for key, values in expected.items():
        batch = frame["batches"][key]
        if (
            batch["evidence_id"] != key
            or batch["env_ids"] != list(binding.env_ids)
            or any(type(x) is not int for x in batch["env_ids"])
            or type(batch["timestamp"]) not in (int, float)
            or batch["timestamp"] != frame["timestamp"]
            or type(batch["observation_revision"]) is not int
            or batch["observation_revision"] != frame["observation_revision"]
            or batch["acquisition_errors"] != [None] * count
            or canonical_hash(batch["values"]) != canonical_hash(values)
        ):
            raise ValueError("Planar geometry channel is missing or inconsistent.")
        flags(batch["valid"], count)
        dtype = batch["value_dtype"]
        if dtype not in (
            {"torch.float32", "torch.float64"}
            if key.startswith("rigid.")
            else {"torch.float64"} if key == "planar.geometry" else {"torch.bool"}
        ):
            raise ValueError("Planar acquisition has another source dtype.")
    return {
        **metrics,
        "support_tolerance_m": _SUPPORT,
        "penetration_tolerance_m": _PENETRATION,
    }


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
