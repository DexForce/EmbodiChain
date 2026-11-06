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
"""Data-only qualification of original Press/Twist operation receipts."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any
from .canonicalization import canonical_hash, json_snapshot

__all__: list[str] = []


def qualified_control_record(
    declaration: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    num_envs: int,
) -> dict[str, Any]:
    """Recheck a source operation using only sealed declaration/receipt data.

    The caller owns TaskSpec role selection and raw acquisition qualification.
    This source receipt checker does not infer an event from final joint state,
    query sensors or relax the original all-active execution requirement.
    """
    declaration, runtime, program = (
        json_snapshot(value) for value in (declaration, runtime, program)
    )
    kind = declaration["control_kind"]
    if kind not in {"press", "twist"}:
        raise ValueError("Control evidence requires an explicit Press or Twist kind.")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("Control evidence requires a positive native row count.")
    route = declaration["route_payload"]
    declared_binding = route["binding"]
    criterion = route.get("success_criterion", "contact_backed_press")
    expected_criteria = (
        {"contact_backed_coarse_turn", "calibrated_setting"}
        if kind == "twist"
        else {"contact_backed_press"}
    )
    if criterion not in expected_criteria:
        raise ValueError(
            "The requested control criterion has no source qualification contract."
        )
    required = declaration["required_execution"]
    if (
        not required
        or len({item["segment_id"] for item in required}) != len(required)
        or required[-1]
        != {
            "segment_id": declaration["terminal_segment_id"],
            "source_path": declaration["terminal_source_path"],
        }
    ):
        raise ValueError(
            "Control evidence requires its full ordered occurrence prefix and terminal boundary."
        )
    segments = {item["name"]: item for item in runtime.get("segments", ())}
    if len(segments) != len(runtime.get("segments", ())):
        raise ValueError("Duplicate control segment receipts.")
    terminal = declaration["terminal_segment_id"]
    for occurrence in declaration["required_execution"]:
        name = occurrence["segment_id"]
        segment = segments.get(name, {})
        identities = segment.get("metadata", {}).get("runtime", {}).get("env_ids", [])
        if (
            segment.get("metadata", {}).get("program_segment_source_path")
            != occurrence["source_path"]
        ):
            raise ValueError("Control prefix has another compiled occurrence.")
        flags = (*segment.get("successes", ()), *segment.get("active", ()))
        if any(type(value) is not int for value in identities) or any(
            type(value) is not bool for value in flags
        ):
            raise ValueError(
                "Control receipt has malformed environment identities or masks."
            )
        if (
            segment.get("metadata", {}).get("runtime", {}).get("env_ids")
            != list(range(num_envs))
            or segment.get("successes") != [True] * num_envs
            or segment.get("active") != [True] * num_envs
        ):
            raise ValueError("Control occurrence did not complete all active segments.")
    segment = segments[terminal]
    frames = segment.get("end_steps")
    if (
        not isinstance(frames, list)
        or len(frames) != num_envs
        or any(type(v) is not int or v <= 0 for v in frames)
    ):
        raise ValueError("Control receipt has no measured terminal boundary.")
    source_path = declaration["terminal_source_path"]
    terminal_item = program
    for component in source_path:
        terminal_item = terminal_item[component]
    expected_preset = declaration["preset"]
    policy_entity = terminal_item["post"][0]["entity"]
    if (
        terminal_item.get("name") != terminal
        or policy_entity
        not in (
            declared_binding["object_id"],
            declaration.get("policy_entity_id", declared_binding["object_id"]),
        )
        or terminal_item.get("post")
        != [
            {
                "kind": "wait_stable",
                "entity": policy_entity,
                "preset": expected_preset,
            }
        ]
    ):
        raise ValueError("Control terminal policy differs from its declared route.")
    receipts = [
        p
        for p in segment.get("metadata", {}).get("post_policies", ())
        if p.get("policy_index") == 0
        and type(p.get("policy_index")) is int
        and p.get("source_path") == [*source_path, "post", 0]
    ]
    if len(receipts) != 1:
        raise ValueError("Control terminal policy receipt is missing or duplicated.")
    receipt = receipts[0]
    result = receipt.get("result", {})
    identity = result.get("control", {})
    expected = {
        "revision": 1,
        "operation_id": route["operation_id"],
        "object_id": declared_binding["object_id"],
        "joint": declared_binding["joint"],
        "link": declared_binding["link"],
        "source_sha256": declared_binding["source_sha256"],
        "route_digest": canonical_hash(route),
        "criterion": criterion,
    }
    if (
        result.get("kind") != "control_operation"
        or type(identity.get("revision")) is not int
        or result.get("preset") != expected_preset
        or any(identity.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("Control receipt identity differs from its occurrence.")
    window = identity.get("window")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in window)
        or not 0 <= window[0] < window[1]
        or type(identity.get("sample_count")) is not int
        or identity["sample_count"] < 2
    ):
        raise ValueError("Control receipt has no valid operation sampling window.")
    if (
        receipt.get("result_mask") != [True] * num_envs
        or any(type(value) is not bool for value in receipt.get("result_mask", ()))
        or result.get("accepted") is not True
        or result.get("terminal") is not True
        or result.get("contact_released") is not True
    ):
        raise ValueError("Control cleanup was not physically accepted.")
    if kind == "twist":

        if (
            result.get("stable") is not True
            or result.get("others_unchanged") is not True
        ):
            raise ValueError("Twist stability or other-joint evidence is missing.")
        path = result.get("contact_path")
        if (
            type(path) not in (int, float)
            or not math.isfinite(path)
            or path < math.radians(15.0)
        ):
            raise ValueError("Twist has insufficient measured contact travel.")
        if criterion == "calibrated_setting" and (
            result.get("target_qpos_reached") is not True
            or result.get("target_error", math.inf) > math.radians(5.0)
        ):
            raise ValueError("Strict setting was not physically reached.")
        clearance = result.get("clearance")
    else:
        if any(
            result.get(key) is not True
            for key in (
                "event_accepted",
                "retreat_verified",
                "housing_contact_released",
            )
        ):
            raise ValueError("Press event or retreat evidence is missing.")
        clearance = result.get("terminal_clearance")
    if (
        type(clearance) not in (int, float)
        or not math.isfinite(clearance)
        or clearance < 0.04
    ):
        raise ValueError("Control terminal clearance is insufficient.")
    return {
        "group_id": route["operation_id"],
        "uid": declared_binding["object_id"],
        "part_id": None,
        "accepted": [True] * num_envs,
        "measurements": {
            "scope": "measured_control_operation",
            "boundary_frames": frames,
            "control": deepcopy(identity),
            "result": deepcopy(result),
        },
    }


def one(values: Any) -> Any:
    values = list(values)
    if len(values) != 1:
        raise ValueError("Held evidence requires one exact occurrence.")
    return values[0]
