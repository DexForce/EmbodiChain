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
"""Operation-bound contact receipts for the canonical TaskSpec observer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from ..contracts import canonical_hash

__all__: list[str] = []


def capture_control_operations(
    deployment: Any,
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    num_envs: int,
) -> dict[str, Any]:
    """Keep every occurrence, including events superseded in terminal goals."""
    result = {}
    for group in graph["task_groups"]:
        if group["task_type"] not in {"E8", "E9"}:
            continue
        try:
            result[group["id"]] = measure_control_operation(
                deployment, graph, group, runtime, program, num_envs
            )
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            result[group["id"]] = {"unavailable": str(error)}
    return result


def stamp_control_result(
    result: dict[str, Any], route: Any, sensor: Any, preset: str
) -> dict[str, Any]:
    """Copy provenance at the post-policy boundary, before any later operation."""
    if not getattr(route, "operation_id", ""):
        return deepcopy(result)
    samples = sensor.samples
    return {
        **deepcopy(result),
        "kind": "control_operation",
        "preset": preset,
        "control": {
            "revision": 1,
            "operation_id": route.operation_id,
            "object_id": route.binding.object_id,
            "joint": route.binding.joint,
            "link": route.binding.link,
            "source_sha256": route.binding.source_sha256,
            "route_digest": canonical_hash(route.payload()),
            "criterion": getattr(route, "success_criterion", "contact_backed_press"),
            "sample_count": len(samples),
            "window": (
                [samples[0].timestamp, samples[-1].timestamp] if samples else None
            ),
        },
    }


def measure_control_operation(
    deployment: Any,
    graph: Mapping[str, Any],
    group: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    num_envs: int,
) -> dict[str, Any]:
    """Qualify the exact terminal receipt, never a program-success shortcut."""
    from .press_runtime import PressFactory
    from .twist_runtime import TwistFactory

    family = group["task_type"]
    factory_type = TwistFactory if family == "E8" else PressFactory
    routes = [
        f.route
        for f in deployment.integration.registration.registered_semantic_lowerer_factories
        if isinstance(f, factory_type) and f.route.operation_id == group["id"]
    ]
    if len(routes) != 1:
        raise ValueError("Control occurrence has no exact registered route.")
    route = routes[0]
    nodes = {node["id"]: node for node in graph["nodes"]}
    segments = {item["name"]: item for item in runtime.get("segments", ())}
    if len(segments) != len(runtime.get("segments", ())):
        raise ValueError("Duplicate control segment receipts.")
    terminal = group["node_ids"][-1]
    for name in group["node_ids"]:
        segment = segments.get(name, {})
        identities = segment.get("metadata", {}).get("runtime", {}).get("env_ids", [])
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
    items = program["program"]["items"]
    index = next(i for i, item in enumerate(items) if item["name"] == terminal)
    expected_preset = route.preset("turned" if family == "E8" else "pressed")
    if items[index].get("post") != [
        {
            "kind": "wait_stable",
            "entity": route.binding.object_id,
            "preset": expected_preset,
        }
    ]:
        raise ValueError("Control terminal policy differs from its route.")
    receipts = [
        p
        for p in segment.get("metadata", {}).get("post_policies", ())
        if p.get("policy_index") == 0
        and type(p.get("policy_index")) is int
        and p.get("source_path") == ["program", "items", index, "post", 0]
    ]
    if len(receipts) != 1:
        raise ValueError("Control terminal policy receipt is missing or duplicated.")
    receipt = receipts[0]
    result = receipt.get("result", {})
    identity = result.get("control", {})
    expected = {
        "revision": 1,
        "operation_id": route.operation_id,
        "object_id": route.binding.object_id,
        "joint": route.binding.joint,
        "link": route.binding.link,
        "source_sha256": route.binding.source_sha256,
        "route_digest": canonical_hash(route.payload()),
        "criterion": getattr(route, "success_criterion", "contact_backed_press"),
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
    if family == "E8":
        from .twist_runtime import COARSE_TURN_THRESHOLD, ANGLE_TOLERANCE

        if (
            result.get("stable") is not True
            or result.get("others_unchanged") is not True
        ):
            raise ValueError("Twist stability or other-joint evidence is missing.")
        path = result.get("contact_path")
        if (
            type(path) not in (int, float)
            or not math.isfinite(path)
            or path < COARSE_TURN_THRESHOLD
        ):
            raise ValueError("Twist has insufficient measured contact travel.")
        if route.success_criterion == "calibrated_setting" and (
            result.get("target_qpos_reached") is not True
            or result.get("target_error", math.inf) > ANGLE_TOLERANCE
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
        "group_id": group["id"],
        "uid": route.binding.object_id,
        "part_id": None,
        "accepted": [True] * num_envs,
        "measurements": {
            "scope": "measured_control_operation",
            "boundary_frames": frames,
            "control": deepcopy(identity),
            "result": deepcopy(result),
        },
    }
