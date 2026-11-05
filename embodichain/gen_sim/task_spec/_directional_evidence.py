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
from ._held_evidence import matrix, number, flags, one

__all__: list[str] = []
AXES = {
    "left_of": ((1, -1),),
    "right_of": ((1, 1),),
    "front_of": ((0, -1),),
    "behind": ((0, 1),),
    "front_left_of": ((0, -1), (1, -1)),
    "front_right_of": ((0, -1), (1, 1)),
    "back_left_of": ((0, 1), (1, -1)),
    "back_right_of": ((0, 1), (1, 1)),
}


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


def raw_directional_metrics(
    binding: Any, snapshot: Mapping[str, Any], boundary: Mapping[str, Any]
) -> dict[str, Any]:
    from .canonicalization import canonical_hash

    value = binding.directional_geometry
    count = len(binding.env_ids)
    if (
        snapshot.get("schema_version") != "simulation.directional_observation/v1"
        or snapshot.get("object_id") != binding.object_id
        or snapshot.get("reference_id") != binding.reference_id
        or snapshot.get("observation_kind") != binding.observation_kind
        or snapshot.get("boundary_frames") != boundary["end_steps"]
        or any(type(x) is not int for x in snapshot["boundary_frames"])
    ):
        raise ValueError(
            "Directional acquisition has another native boundary or participants."
        )
    for key in (
        "program_fingerprint",
        "object_native_uid",
        "reference_native_uid",
        "table_id",
        "table_native_uid",
        "reference_part",
        "reference_link_id",
        "native_link_name",
    ):
        if snapshot.get(key) != value[key]:
            raise ValueError(
                "Directional acquisition has another resolved native spatial route."
            )
    if binding.observation_kind == "recorded" and (
        snapshot.get("policy_segment_name") != binding.boundary_segment_id
        or snapshot.get("policy_source_path") != list(binding.boundary_source_path)
    ):
        raise ValueError(
            "Directional policy observation has another compiled boundary."
        )
    axes = [list(pair) for pair in AXES[value["relation"]]]
    if snapshot.get("axes") != axes or any(
        type(x) is not int for pair in snapshot["axes"] for x in pair
    ):
        raise ValueError("Directional channels project another declared world axis.")
    metrics = snapshot["native_directional"]
    table = metrics["tabletop"]
    frame = snapshot["geometry_frame"]
    number(frame["timestamp"])
    if (
        frame["env_ids"] != list(binding.env_ids)
        or any(type(x) is not int for x in frame["env_ids"])
        or type(frame["observation_revision"]) is not int
        or frame["observation_revision"] < 0
        or frame["clock_domain"] != "simulation_time"
        or frame["revision_kind"] != "directional_geometry_acquisition"
    ):
        raise ValueError("Directional acquisition has another row/clock identity.")
    expected = {
        "directional.envelope_gaps": [
            list(row) for row in zip(*metrics["envelope_gaps_m"], strict=True)
        ],
        "rigid.object_pose": metrics["object_pose"],
        "reference.pose": metrics["reference_pose"],
        "table.pose": table["reference_pose"],
        "tabletop.geometry": [
            [table["support_gap_m"][i], table["footprint_area_m2"][i]]
            for i in range(count)
        ],
        "tabletop.full_footprint_inside": table["full_footprint_inside"],
    }
    for key, values in expected.items():
        batch = frame["batches"][key]
        if (
            batch["evidence_id"] != key
            or batch["env_ids"] != list(binding.env_ids)
            or any(type(x) is not int for x in batch["env_ids"])
            or batch["timestamp"] != frame["timestamp"]
            or type(batch["observation_revision"]) is not int
            or batch["observation_revision"] != frame["observation_revision"]
            or batch["acquisition_errors"] != [None] * count
            or canonical_hash(batch["values"]) != canonical_hash(values)
        ):
            raise ValueError("Directional channel is missing or inconsistent.")
        flags(batch["valid"], count)
        allowed = (
            {"torch.bool"}
            if key == "tabletop.full_footprint_inside"
            else (
                {"torch.float64"}
                if key == "tabletop.geometry"
                else {"torch.float32", "torch.float64"}
            )
        )
        if batch.get("value_dtype") not in allowed:
            raise ValueError("Directional channel has another physical dtype.")
    return {
        **metrics,
        "scope": binding.evidence_scope,
        "reference_uid": binding.reference_id,
        "reference_part": value["reference_part"],
        "reference_link_id": value["reference_link_id"],
        "relation": value["relation"],
        "tabletop": {
            **table,
            "support_tolerance_m": 0.05,
            "penetration_tolerance_m": 0.01,
        },
        "boundary_frames": snapshot["boundary_frames"],
    }


def directional_flags(
    binding: Any,
    metrics: Mapping[str, Any],
    released: list[bool],
    boundary: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    family: str | None,
) -> list[bool]:
    import torch
    from .evidence import _bool_rows

    value = binding.directional_geometry
    count = len(binding.env_ids)
    declared = program
    for key in binding.release_source_path:
        declared = declared[key]
    call = declared["steps"]["call"]
    args = call["arguments"]
    if (
        call.get("call_id") != "simulation.place_relative"
        or args.get("object") != binding.object_id
        or args.get("reference") != binding.reference_id
        or args.get("reference_part") != value["reference_part"]
        or args.get("relation") != value["relation"]
        or call["resources"].get("primary") != binding.resource_id
    ):
        raise ValueError("Directional release differs from its declared source route.")
    expected = list(binding.condition["displacement"])
    if (
        family in (None, "E1")
        and value["reference_link_id"] is None
        and value["relation"] in {"left_of", "right_of", "front_of", "behind"}
    ):
        expected[2] += 0.03 - 0.01
    if value["world_displacement"] != expected:
        raise ValueError(
            "Directional release and settled offsets differ from their source contract."
        )
    if (
        metrics.get("scope") != binding.evidence_scope
        or metrics.get("reference_uid") != binding.reference_id
        or metrics.get("reference_part") != value["reference_part"]
        or metrics.get("reference_link_id") != value["reference_link_id"]
        or metrics.get("relation") != value["relation"]
    ):
        raise ValueError("Directional record has another selected spatial reference.")
    object_pose = matrix(metrics["object_pose"], count)
    reference_pose = matrix(metrics["reference_pose"], count)
    table = metrics["tabletop"]
    matrix(table["reference_pose"], count)
    if (
        not torch.allclose(
            matrix(table["object_pose"], count), object_pose, atol=1e-4, rtol=0
        )
        or table.get("scope") != "measured_planar_landing_region"
        or table.get("support_tolerance_m") != 0.05
        or table.get("penetration_tolerance_m") != 0.01
        or not _bool_rows(table.get("full_footprint_inside"), count)
    ):
        raise ValueError(
            "Directional tabletop geometry differs from the observed payload."
        )
    gaps = metrics["envelope_gaps_m"]
    if len(gaps) != len(AXES[value["relation"]]) or any(
        not isinstance(row, list)
        or len(row) != count
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)
        for row in [*gaps, table["support_gap_m"]]
    ):
        raise ValueError("Directional envelope gaps are missing or nonfinite.")
    if binding.observation_kind == "recorded":
        prefix = list(binding.boundary_source_path)
        policy = one(
            p
            for p in boundary["metadata"].get("post_policies", [])
            if p.get("kind") == "wait_stable"
            and type(p.get("policy_index")) is int
            and p.get("source_path") == [*prefix, "post", p["policy_index"]]
            and p.get("result", {}).get("preset") == binding.stability_preset
        )
        result = policy["result"]
        measured = result["measurements"]
        if (
            result.get("kind") != binding.condition["kind"]
            or result.get("required_seconds") != binding.condition["duration"]
            or len(result["stable_seconds"]) != count
            or any(
                number(x) < binding.condition["duration"]
                for x in result["stable_seconds"]
            )
            or measured.get("entity") != binding.object_id
            or measured.get("reference")
            != (value["reference_link_id"] or binding.reference_id)
        ):
            raise ValueError(
                "Recorded directional evidence lacks its authored hold receipt."
            )
        flags(policy["result_mask"], count)
        flags(result["accepted_mask"], count)
        if (
            not torch.allclose(
                matrix(measured["object_pose"], count), object_pose, atol=1e-4, rtol=0
            )
            or not torch.allclose(
                matrix(measured["reference_pose"], count),
                reference_pose,
                atol=1e-4,
                rtol=0,
            )
            or metrics.get("boundary_frames") != boundary["end_steps"]
        ):
            raise ValueError(
                "Directional policy poses differ from the exact recorded boundary."
            )
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
