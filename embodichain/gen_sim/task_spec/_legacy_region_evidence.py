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
from ._held_evidence import matrix, one, flags
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
        or args.get("relation") != "on"
        or set(args) - {"object", "reference", "relation", "stage_id"}
        or call["resources"].get("primary") != binding.resource_id
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
        or metrics.get("observation_kind") != binding.observation_kind
        or metrics.get("reference_uid") != "table"
        or metrics.get("relation") != "on"
        or metrics.get("contact_tolerance_m") != 0.01
        or metrics.get("release_window") != window
        or metrics.get("released") != released
    ):
        raise ValueError(
            "Legacy region record differs from its exact release and observation scope."
        )
    native, landing = metrics["native_support"], metrics["landing_region"]
    if (
        native.get("support_contract_revision") is not None
        or landing.get("footprint_kind") is not None
    ):
        raise ValueError(
            "Legacy region cannot borrow a strong contact revision or triangle-mode proof."
        )
    pose = matrix(snapshot["boundary_poses"][binding.object_id], count)
    reference = matrix(snapshot["boundary_poses"][binding.reference_id], count)
    physical = _region_flags(
        native,
        landing,
        pose,
        reference,
        binding.object_id,
        metrics["support_gap_m"],
        count,
        reference_uid="table",
    )
    if binding.observation_kind == "recorded":
        prefix = list(binding.boundary_source_path)
        policy = one(
            value
            for value in boundary["metadata"].get("post_policies", [])
            if value.get("kind") == "wait_stable"
            and type(value.get("policy_index")) is int
            and value.get("source_path") == [*prefix, "post", value["policy_index"]]
            and value.get("result", {}).get("preset") == binding.stability_preset
        )
        result = policy["result"]
        seconds = result["stable_seconds"]
        if (
            result.get("kind") != "supported_region"
            or result.get("support_contract_revision") is not None
            or result.get("required_seconds") != binding.condition["duration"]
            or len(seconds) != count
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < binding.condition["duration"]
                for value in seconds
            )
        ):
            raise ValueError("Legacy region lacks its actual authored accepted policy.")
        flags(policy["result_mask"], count)
        flags(result["accepted_mask"], count)
        observed = result["measurements"]
        if (
            observed.get("native_support") != native
            or metrics.get("boundary_frames") != boundary["end_steps"]
            or not _region_poses_match(
                observed.get("landing_region", {}), pose, reference, count
            )
        ):
            raise ValueError(
                "Legacy region policy differs from its exact recorded poses or native support."
            )
    return [value and free for value, free in zip(physical, released, strict=True)]
