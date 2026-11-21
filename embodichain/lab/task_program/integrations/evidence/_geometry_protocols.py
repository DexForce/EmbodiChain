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

"""Offline native planar/directional observation decoding."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import math
from ._held_evidence import number, flags, matrix, one
from ._release_evidence import release_window
from ._pose_values import region_poses_match as _region_poses_match
from embodichain.compute.geometry.directional import DIRECTIONAL_RELATION_AXES as AXES

__all__: list[str] = []
_SUPPORT = 0.05
_PENETRATION = 0.01


def raw_planar_metrics(
    binding: Any, snapshot: Mapping[str, Any], boundary: Mapping[str, Any]
) -> dict[str, Any]:
    from embodichain._json_contract import canonical_hash

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


def raw_directional_metrics(
    binding: Any, snapshot: Mapping[str, Any], boundary: Mapping[str, Any]
) -> dict[str, Any]:
    from embodichain._json_contract import canonical_hash

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


def verify_directional_protocol(
    binding: Any,
    metrics: Mapping[str, Any],
    released: list[bool],
    boundary: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    family: str | None,
) -> None:
    import torch
    from ._values import _bool_rows

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
        or (args.get("reference_part") != value["reference_part"])
        or (args.get("relation") != value["relation"])
        or (call["resources"].get("primary") != binding.resource_id)
    ):
        raise ValueError("Directional release differs from its declared source route.")
    expected = list(binding.condition["displacement"])
    if (
        family in (None, "E1")
        and value["reference_link_id"] is None
        and (value["relation"] in {"left_of", "right_of", "front_of", "behind"})
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
        or (metrics.get("reference_link_id") != value["reference_link_id"])
        or (metrics.get("relation") != value["relation"])
    ):
        raise ValueError("Directional record has another selected spatial reference.")
    object_pose = matrix(metrics["object_pose"], count)
    reference_pose = matrix(metrics["reference_pose"], count)
    table = metrics["tabletop"]
    matrix(table["reference_pose"], count)
    if (
        not torch.allclose(
            matrix(table["object_pose"], count), object_pose, atol=0.0001, rtol=0
        )
        or table.get("scope") != "measured_planar_landing_region"
        or table.get("support_tolerance_m") != 0.05
        or (table.get("penetration_tolerance_m") != 0.01)
        or (not _bool_rows(table.get("full_footprint_inside"), count))
    ):
        raise ValueError(
            "Directional tabletop geometry differs from the observed payload."
        )
    gaps = metrics["envelope_gaps_m"]
    if len(gaps) != len(AXES[value["relation"]]) or any(
        (
            not isinstance(row, list)
            or len(row) != count
            or any((type(x) not in (int, float) or not math.isfinite(x) for x in row))
            for row in [*gaps, table["support_gap_m"]]
        )
    ):
        raise ValueError("Directional envelope gaps are missing or nonfinite.")
    if binding.observation_kind == "recorded":
        prefix = list(binding.boundary_source_path)
        policy = one(
            (
                p
                for p in boundary["metadata"].get("post_policies", [])
                if p.get("kind") == "wait_stable"
                and type(p.get("policy_index")) is int
                and (p.get("source_path") == [*prefix, "post", p["policy_index"]])
                and (p.get("result", {}).get("preset") == binding.stability_preset)
            )
        )
        result = policy["result"]
        measured = result["measurements"]
        if (
            result.get("kind") != binding.condition["kind"]
            or result.get("required_seconds") != binding.condition["duration"]
            or len(result["stable_seconds"]) != count
            or any(
                (
                    number(x) < binding.condition["duration"]
                    for x in result["stable_seconds"]
                )
            )
            or (measured.get("entity") != binding.object_id)
            or (
                measured.get("reference")
                != (value["reference_link_id"] or binding.reference_id)
            )
        ):
            raise ValueError(
                "Recorded directional evidence lacks its authored hold receipt."
            )
        flags(policy["result_mask"], count)
        flags(result["accepted_mask"], count)
        if (
            not torch.allclose(
                matrix(measured["object_pose"], count), object_pose, atol=0.0001, rtol=0
            )
            or not torch.allclose(
                matrix(measured["reference_pose"], count),
                reference_pose,
                atol=0.0001,
                rtol=0,
            )
            or metrics.get("boundary_frames") != boundary["end_steps"]
        ):
            raise ValueError(
                "Directional policy poses differ from the exact recorded boundary."
            )
    return None


def verify_legacy_region_protocol(
    binding: Any,
    snapshot: Mapping[str, Any],
    boundary: Mapping[str, Any],
    release_call: Mapping[str, Any],
    receipt: Mapping[str, Any],
    released: list[bool],
    program: Mapping[str, Any],
) -> None:
    count = len(binding.env_ids)
    metrics = snapshot["record"]["measurements"]
    declared = program
    for component in binding.release_source_path:
        declared = declared[component]
    call = declared["steps"]["call"]
    args = call["arguments"]
    if (
        call.get("call_id") != "simulation.place_relative"
        or args.get("object") != binding.object_id
        or args.get("reference") != "table"
        or (args.get("relation") != "on")
        or set(args) - {"object", "reference", "relation", "stage_id"}
        or (call["resources"].get("primary") != binding.resource_id)
    ):
        raise ValueError(
            "Legacy region requires its exact unmodified bare-table release call."
        )
    window = release_window(release_call, receipt, count)
    if binding.evidence_format == "raw_snapshot":
        metrics.update(release_window=window, released=released)
    if (
        metrics.get("scope") != "native_supported_landing_region"
        or metrics.get("placement_node_id") != binding.release_segment_id
        or metrics.get("boundary_node_id") != binding.boundary_segment_id
        or (metrics.get("observation_kind") != binding.observation_kind)
        or (metrics.get("reference_uid") != "table")
        or (metrics.get("relation") != "on")
        or (metrics.get("contact_tolerance_m") != 0.01)
        or (metrics.get("release_window") != window)
        or (metrics.get("released") != released)
    ):
        raise ValueError(
            "Legacy region record differs from its exact release and observation scope."
        )
    native, landing = (metrics["native_support"], metrics["landing_region"])
    if (
        native.get("support_contract_revision") is not None
        or landing.get("footprint_kind") is not None
    ):
        raise ValueError(
            "Legacy region cannot borrow a strong contact revision or triangle-mode proof."
        )
    pose = matrix(snapshot["boundary_poses"][binding.object_id], count)
    reference = matrix(snapshot["boundary_poses"][binding.reference_id], count)
    if binding.observation_kind == "recorded":
        prefix = list(binding.boundary_source_path)
        policy = one(
            (
                value
                for value in boundary["metadata"].get("post_policies", [])
                if value.get("kind") == "wait_stable"
                and type(value.get("policy_index")) is int
                and (
                    value.get("source_path") == [*prefix, "post", value["policy_index"]]
                )
                and (value.get("result", {}).get("preset") == binding.stability_preset)
            )
        )
        result = policy["result"]
        seconds = result["stable_seconds"]
        if (
            result.get("kind") != "supported_region"
            or result.get("support_contract_revision") is not None
            or result.get("required_seconds") != binding.condition["duration"]
            or (len(seconds) != count)
            or any(
                (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < binding.condition["duration"]
                    for value in seconds
                )
            )
        ):
            raise ValueError("Legacy region lacks its actual authored accepted policy.")
        flags(policy["result_mask"], count)
        flags(result["accepted_mask"], count)
        observed = result["measurements"]
        if (
            observed.get("native_support") != native
            or metrics.get("boundary_frames") != boundary["end_steps"]
            or (
                not _region_poses_match(
                    observed.get("landing_region", {}), pose, reference, count
                )
            )
        ):
            raise ValueError(
                "Legacy region policy differs from its exact recorded poses or native support."
            )
