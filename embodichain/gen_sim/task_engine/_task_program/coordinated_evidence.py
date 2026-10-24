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

"""Observe standalone E5 pickup/hold without changing motion or phase contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import torch

from embodichain.utils.math import pose_inv
from ..contracts import canonical_hash
from .held_evidence import (
    delta,
    flags,
    frames,
    fresh_grip,
    grip_flags,
    matrix,
    native_held_poses,
    number,
    one,
    receipt,
    slot,
)
from .stability import StabilityConstraint

__all__: list[str] = []

_CALL_ID = "simulation.coordinated_hold"
_UNAVAILABLE = (
    AttributeError,
    KeyError,
    StopIteration,
    TypeError,
    ValueError,
    RuntimeError,
)


def _occurrence(group, graph, runtime, program, count):
    node = one(value for value in graph["nodes"] if value["id"] in group["node_ids"])
    if (
        group["task_type"] != "E5"
        or group["node_ids"] != [node["id"]]
        or node["task_type"] != "E5"
        or node["task_instance_id"] != group["id"]
        or node["call"].get("kind") != "registered"
        or node["call"].get("call_id") != _CALL_ID
    ):
        raise ValueError(
            "E5 observation supports one exact coordinated_hold operation."
        )
    index, item = one(
        (i, value)
        for i, value in enumerate(program["program"]["items"])
        if value["name"] == node["id"]
    )
    segment = one(value for value in runtime["segments"] if value["name"] == node["id"])
    metadata, calls = segment["metadata"], segment["metadata"]["runtime"]
    segment_id = f"{program['program_id']}:segment:$.program.items[{index}]"
    if (
        item["steps"]["call"] != node["call"]
        or segment.get("success") is not True
        or metadata["task_program_id"] != program["program_id"]
        or metadata["program_segment_id"] != segment_id
        or metadata["program_segment_source_path"] != ["program", "items", index]
        or calls["env_ids"] != list(range(count))
        or any(type(value) is not int for value in calls["env_ids"])
    ):
        raise ValueError("E5 segment differs from its compiled occurrence.")
    flags(segment["successes"], count)
    flags(metadata["validation"]["accepted_mask"], count)
    call = one(calls["calls"])
    if (
        call["semantic_id"] != _CALL_ID
        or call["skill_id"] != "coordinated_pickment"
        or call["status"] != "completed"
        or call["invocation_id"] != f"{program['program_id']}/{segment_id}:0"
        or type(call["invocation_revision"]) is not int
        or call["invocation_revision"] < 0
        or type(call["command_count"]) is not int
        or call["command_count"] <= 0
    ):
        raise ValueError("E5 coordinated invocation did not complete its commands.")
    return node, segment, call, index, item


def _holding(group, graph, runtime, program, constraints, count, dt, final):
    node, segment, call, item_index, item = _occurrence(
        group, graph, runtime, program, count
    )
    uid = node["call"]["arguments"]["object"]
    resources = node["call"]["resources"]
    if (
        type(uid) is not str
        or not uid
        or set(resources) != {"left", "right"}
        or any(
            type(resources[key]) is not str or not resources[key] for key in resources
        )
        or resources["left"] == resources["right"]
    ):
        raise ValueError("E5 requires one object and two distinct bound resources.")
    attempt = one(
        value
        for value in call["plan_attempts"]
        if value["attempt_generation"] == call["active_plan_attempt_generation"]
    )
    planned = number(attempt["planned_at"])
    terminal = one(
        value
        for value in call["effects"]
        if value["boundary"]["kind"] == "terminal"
        and value["effect_spec"]["effect_kind"] == "attach"
        and value["decision"]["success_mask"] == [True] * count
        and number(value["timestamp"]) >= planned
    )
    receipt(terminal, call, count)
    spec = terminal["effect_spec"]
    if (
        spec["semantic_id"] != _CALL_ID
        or spec["skill_id"] != "coordinated_pickment"
        or len(spec["state_expectations"]) != 2
    ):
        raise ValueError("E5 requires an exact terminal dual ATTACH receipt.")
    slots = {}
    for role in ("left", "right"):
        resource = resources[role]
        expectation = one(
            value
            for value in spec["state_expectations"]
            if value["slot_id"] == role and value["resource_id"] == resource
        )
        if expectation["expectation_id"] != role:
            raise ValueError(
                "E5 receipt does not retain its declared participant slots."
            )
        slots[role] = slot(terminal, uid, resource, "attached", count)
    if slots["left"][1] == slots["right"][1]:
        raise ValueError("E5 cannot observe both hands through one motion part.")
    state = segment["metadata"]["runtime"]["task_state"]
    if type(state["batch_size"]) is not int or state["batch_size"] != count:
        raise ValueError("E5 retained state has another batch identity.")
    holders = [value for value in state["held_objects"] if value["object_id"] == uid]
    if (
        len(holders) != 2
        or {value["resource_id"] for value in holders} != set(resources.values())
        or any(value["object_id"] == uid for value in state["coordinated_held_objects"])
    ):
        raise ValueError("E5 boundary must retain exactly its two declared holders.")
    for holder in holders:
        flags(holder["active_mask"], count)
    policy_index, declaration = one(enumerate(item.get("post", [])))
    cfg = StabilityConstraint.decode(constraints["presets"][declaration["preset"]])
    parts = tuple(slots[key][1] for key in ("left", "right"))
    if (
        declaration["kind"] != "wait_stable"
        or declaration["entity"] != uid
        or cfg.kind != "hold"
        or cfg.entity != uid
        or cfg.motion_parts != parts
        or cfg.target_position is None
    ):
        raise ValueError("E5 requires its original two-hand world-target hold policy.")
    policy = one(
        value
        for value in segment["metadata"]["post_policies"]
        if value["source_path"]
        == ["program", "items", item_index, "post", policy_index]
    )
    result = policy["result"]
    if (
        result["kind"] != "hold"
        or result["preset"] != declaration["preset"]
        or number(result["required_seconds"]) != cfg.duration
    ):
        raise ValueError("E5 hold policy differs from its declaration.")
    flags(policy["result_mask"], count)
    flags(result["accepted_mask"], count)
    flags(result["failed_mask"], count, False)
    if len(result["stable_seconds"]) != count or any(
        number(value) < cfg.duration for value in result["stable_seconds"]
    ):
        raise ValueError("E5 did not complete the original hold duration.")
    reported = result["measurements"]
    attachment = torch.as_tensor(
        reported["attachment_translation"], dtype=torch.float64
    )
    position = torch.as_tensor(reported["position_error"], dtype=torch.float64)
    if (
        attachment.shape != (2, count)
        or not torch.isfinite(attachment).all()
        or (attachment < 0).any()
        or (attachment > cfg.translation_drift).any()
        or position.shape != (count,)
        or not torch.isfinite(position).all()
        or (position < 0).any()
        or (position > cfg.position_tolerance).any()
    ):
        raise ValueError(
            "E5 hold has invalid original attachment or world-target measurements."
        )
    completed = number(
        one(
            value
            for value in call["events"]
            if value["kind"] == "action_completed"
            and value.get("invocation_id") == call["invocation_id"]
            and value.get("invocation_revision") == call["invocation_revision"]
        )["timestamp"]
    )
    boundary = frames(segment, count)
    if not planned <= number(terminal["timestamp"]) <= completed or any(
        not math.isclose(
            frame * dt,
            completed + number(result["elapsed_seconds"]),
            rel_tol=0,
            abs_tol=1e-8,
        )
        for frame in boundary
    ):
        raise ValueError("E5 hold observations do not match the group end-step clock.")
    if final and (
        runtime["segments"][-1]["name"] != node["id"] or runtime["lengths"] != boundary
    ):
        raise ValueError("A later execution cannot certify an earlier final E5 hold.")
    return node, segment, call, terminal, slots, cfg, policy


def _record(
    group,
    graph,
    runtime,
    program,
    constraints,
    count,
    dt,
    final,
    object_pose,
    tcp_poses,
    grips,
):
    if dt <= 0:
        raise ValueError("E5 requires a positive canonical control step.")
    node, segment, call, terminal, slots, cfg, policy = _holding(
        group, graph, runtime, program, constraints, count, dt, final
    )
    uid, resources = node["call"]["arguments"]["object"], node["call"]["resources"]
    object_pose = matrix(object_pose, count)
    if set(tcp_poses) != {value[1] for value in slots.values()} or set(grips) != {
        "left",
        "right",
    }:
        raise ValueError("E5 requires one native pose and grip observation per hand.")
    accepted = torch.ones(count, dtype=torch.bool)
    translations, rotations, owned_tcp = {}, {}, {}
    for role in ("left", "right"):
        reference, part = slots[role]
        tcp = matrix(tcp_poses[part], count)
        translation, rotation = delta(pose_inv(object_pose) @ tcp, reference)
        accepted &= (translation <= cfg.translation_drift) & (
            rotation <= cfg.rotation_drift
        )
        accepted &= torch.tensor(
            grip_flags(
                uid=uid,
                resource=resources[role],
                sample=grips[role],
                terminal=terminal,
                frames=frames(segment, count),
                dt=dt,
                count=count,
            ),
            dtype=torch.bool,
        )
        owned_tcp[part] = tcp.tolist()
        translations[role], rotations[role] = translation.tolist(), rotation.tolist()
    # Each grip validator has established the exact outer and inner actor shapes.
    for row in range(count):
        left, right = grips["left"], grips["right"]
        if left["object_actor_ids"][row] != right["object_actor_ids"][row] or not set(
            left["pad_actor_ids"][row]
        ).isdisjoint(right["pad_actor_ids"][row]):
            raise ValueError("E5 requires the same object and disjoint hand actors.")
    position = torch.linalg.vector_norm(
        object_pose[:, :3, 3] - object_pose.new_tensor(cfg.target_position), dim=-1
    )
    accepted &= position <= cfg.position_tolerance
    reported = torch.as_tensor(
        policy["result"]["measurements"]["position_error"], dtype=position.dtype
    )
    if not torch.allclose(reported, position, atol=1e-4, rtol=0):
        raise ValueError("E5 world-target observation differs from its hold boundary.")
    return {
        "group_id": group["id"],
        "uid": uid,
        "part_id": None,
        "accepted": accepted.tolist(),
        "measurements": {
            "scope": "native_coordinated_hold",
            "predicate": "held_by_both_grippers",
            "node_id": node["id"],
            "invocation_id": call["invocation_id"],
            "invocation_revision": call["invocation_revision"],
            "resources": dict(resources),
            "motion_parts": {role: slots[role][1] for role in slots},
            "boundary_frames": frames(segment, count),
            "step_dt": dt,
            "final": final,
            "terminal_attachment_hash": canonical_hash(terminal),
            "hold_policy_hash": canonical_hash(policy),
            "object_pose": object_pose.tolist(),
            "tcp_poses": owned_tcp,
            "attachment_translation": translations,
            "attachment_rotation": rotations,
            "world_target": list(cfg.target_position),
            "position_error": position.tolist(),
            "fresh_grips": deepcopy(grips),
        },
    }


def capture_coordinated_holds(
    env: Any,
    deployment: Any,
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    constraints: Mapping[str, Any],
    num_envs: int,
    *,
    final: bool = False,
) -> dict[str, Any]:
    """Capture the current pickup/hold boundary, not earlier multi-E5 history.

    Fresh contact is sampled only while the group's seeded end-step is the
    current native boundary. Later execution cannot reconstruct that sample.
    """
    result = {}
    target = getattr(env, "unwrapped", env)
    for group in graph["task_groups"]:
        if group["task_type"] != "E5":
            continue
        try:
            dt = number(target.step_dt)
            node, segment, _, terminal, slots, _, _ = _holding(
                group, graph, runtime, program, constraints, num_envs, dt, final
            )
            boundary = frames(segment, num_envs)
            if len(target._traj_steps) != num_envs or any(
                int(target._traj_steps[row]) != frame
                for row, frame in enumerate(boundary)
            ):
                raise ValueError(
                    "Fresh E5 contact requires the group's current native boundary."
                )
            uid = node["call"]["arguments"]["object"]
            object_pose, tcp = native_held_poses(
                env,
                deployment,
                uid,
                boundary,
                tuple(value[1] for value in slots.values()),
                num_envs,
                True,
            )
            grips = {
                role: fresh_grip(
                    env,
                    uid=uid,
                    resource=node["call"]["resources"][role],
                    terminal=terminal,
                    frames=boundary,
                    dt=dt,
                    count=num_envs,
                )
                for role in ("left", "right")
            }
            result[group["id"]] = _record(
                group,
                graph,
                runtime,
                program,
                constraints,
                num_envs,
                dt,
                final,
                object_pose,
                tcp,
                grips,
            )
        except _UNAVAILABLE as error:
            result[group["id"]] = {"unavailable": str(error)}
    return result


def coordinated_record_matches(
    record: Mapping[str, Any],
    group: Mapping[str, Any],
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    constraints: Mapping[str, Any],
    num_envs: int,
    *,
    final: bool = False,
) -> bool:
    """Recompute both participants and original hold, never trust accepted labels."""
    try:
        metrics = record["measurements"]
        if metrics["final"] is not final:
            return False
        expected = _record(
            group,
            graph,
            runtime,
            program,
            constraints,
            num_envs,
            number(metrics["step_dt"]),
            final,
            metrics["object_pose"],
            metrics["tcp_poses"],
            metrics["fresh_grips"],
        )
        return canonical_hash(record) == canonical_hash(expected)
    except _UNAVAILABLE:
        return False
