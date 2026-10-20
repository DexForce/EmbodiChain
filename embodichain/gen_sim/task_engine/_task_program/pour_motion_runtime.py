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
"""Native tilt/restore witnesses for the distinct motion-only E3 predicate."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import torch

from embodichain.utils.math import matrix_from_quat, pose_inv
from .held_evidence import matrix, one, delta
from .placement_evidence import measured_release
from .pour_evidence import (
    PourProxyContract,
    NativePourWindow,
    PourHoldEvidence,
    evaluate_pour_pose_proxy,
)

__all__: list[str] = []


def observer_policy(angle_rad: float, position_tolerance_m: float) -> dict[str, Any]:
    """Declare the revision-one motion policy without changing fluid semantics."""
    return validate_policy(
        {
            "revision": 1,
            "angle_rad": angle_rad,
            "tilt_tolerance_rad": math.radians(5),
            "restore_tolerance_rad": math.radians(5),
            "position_tolerance_m": position_tolerance_m,
        }
    )


def validate_policy(value: Any) -> dict[str, Any]:
    keys = {
        "revision",
        "angle_rad",
        "tilt_tolerance_rad",
        "restore_tolerance_rad",
        "position_tolerance_m",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or type(value["revision"]) is not int
        or value["revision"] != 1
    ):
        raise ValueError(
            "Pour motion requires its explicit revision-one observer policy."
        )
    for name in keys - {"revision"}:
        if type(value[name]) not in (int, float) or not math.isfinite(value[name]):
            raise ValueError("Pour motion policy values must be finite real numbers.")
    if (
        min(value[n] for n in keys - {"revision", "angle_rad"}) <= 0
        or not value["tilt_tolerance_rad"] + value["restore_tolerance_rad"]
        < abs(value["angle_rad"])
        < math.pi
    ):
        raise ValueError(
            "Pour motion requires distinct tilt/restore tolerance regions."
        )
    return dict(value)


def _step(time: Any, dt: float) -> int:
    if type(time) not in (int, float) or not math.isfinite(time) or time < 0:
        raise ValueError("Pour event clock is invalid.")
    result = round(time / dt)
    if not math.isclose(time, result * dt, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Pour event is not on the native control grid.")
    return result


def _inputs(
    group: Mapping,
    graph: Mapping,
    runtime: Mapping,
    program: Mapping,
    constraints: Mapping,
    dt: float,
):
    policy = validate_policy(constraints.get("pour_motion_observer"))
    if (
        not math.isfinite(dt)
        or dt <= 0
        or group["task_type"] != "E3"
        or group["success"].get("type") != "pour_motion"
    ):
        raise ValueError("Only a declared motion-only E3 occurrence is eligible.")
    nodes = {n["id"]: n for n in graph["nodes"]}
    selected = [nodes[key] for key in group["node_ids"]]
    kinds = [n["call"].get("call_id", n["call"]["kind"]) for n in selected]
    if kinds != [
        "pick",
        "simulation.move_held_object",
        "simulation.pour",
        "simulation.move_held_object",
        "place",
        "simulation.park",
    ]:
        raise ValueError(
            "Pour motion requires the complete explicit return/release/Park recipe."
        )
    source, resource = selected[0]["call"]["object"], selected[0]["call"]["resources"]
    receiver = selected[1]["call"]["arguments"]["reference"]
    if (
        set(resource) != {"primary"}
        or any(n["call"]["resources"] != resource for n in selected)
        or any(n["call"]["arguments"].get("object") != source for n in selected[1:4])
        or selected[4]["call"]["object"] != source
    ):
        raise ValueError("Pour motion recipe changed source or owning resource.")
    segments = {s["name"]: s for s in runtime["segments"]}
    if len(segments) != len(runtime["segments"]) or any(
        not segments[n["id"]]["success"] for n in selected
    ):
        raise ValueError("The full Pour motion recipe has not completed.")
    items = {
        item["name"]: (index, item)
        for index, item in enumerate(program["program"]["items"])
    }
    index, item = items[selected[2]["id"]]
    call = one(segments[selected[2]["id"]]["metadata"]["runtime"]["calls"])
    invocation = f"{program['program_id']}/{program['program_id']}:segment:$.program.items[{index}]:0"
    if (
        call["invocation_id"] != invocation
        or call["semantic_id"] != "simulation.pour"
        or call["skill_id"] != "pour"
        or call["call"] != selected[2]["call"]
    ):
        raise ValueError(
            "Pour motion call identity differs from its compiled occurrence."
        )
    attempt = one(
        a
        for a in call["plan_attempts"]
        if a["attempt_generation"] == call["active_plan_attempt_generation"]
    )
    meta = attempt["planner_diagnostics"]["metadata"]["gen_sim_pour_geometry"]
    if (
        meta["revision"] != 2
        or meta["source_entity_id"] != source
        or meta["receiver_entity_id"] != receiver
        or meta["proxy_position_tolerance_m"] != policy["position_tolerance_m"]
        or not math.isclose(
            abs(meta["effective_angle_rad"]), abs(policy["angle_rad"]), abs_tol=1e-6
        )
    ):
        raise ValueError("Pour geometry differs from its declared observer policy.")
    entry, peak = matrix(meta["entry_pose"], 1), matrix(meta["peak_pose"], 1)
    if not math.isclose(
        float(delta(entry, peak)[1][0]), abs(policy["angle_rad"]), abs_tol=1e-5
    ):
        raise ValueError(
            "Pour planned tilt magnitude differs from the registered motion."
        )
    start = one(
        e["timestamp"]
        for e in call["events"]
        if e["kind"] == "trajectory_segment_entered" and e.get("segment_name") == "pour"
    )
    end = one(
        e["timestamp"] for e in call["events"] if e["kind"] == "trajectory_completed"
    )
    lo, hi = _step(start, dt), _step(end, dt)
    receiver_pose = matrix(meta["receiver_pose"], 1)[0]
    pivot = torch.as_tensor(meta["pivot_world"], dtype=torch.float64).reshape(3)
    anchor = (pose_inv(receiver_pose) @ torch.cat((pivot, pivot.new_ones(1))))[:3]
    contract = PourProxyContract(
        source,
        receiver,
        resource["primary"],
        invocation,
        call["invocation_revision"],
        0,
        attempt["planned_at"],
        lo,
        hi,
        dt,
        matrix(meta["observed_source_entry_pose"], 1)[0],
        peak[0],
        tuple(torch.as_tensor(meta["pivot_object"]).reshape(3).tolist()),
        tuple(anchor.tolist()),
        policy["tilt_tolerance_rad"],
        policy["restore_tolerance_rad"],
        policy["position_tolerance_m"],
    )
    terminal = one(e for e in call["effects"] if e["boundary"]["kind"] == "terminal")
    clause = one(
        c
        for c in terminal["effect_spec"]["clauses"]
        if c.get("evidence_kind") == "constraint"
    )
    part = clause["source"]["address"]["control_part"]
    return contract, call, part, selected, segments, items


def _return_matches(
    source: str, selected: list, segments: Mapping, items: Mapping, pose: Any
) -> None:
    terminal = segments[selected[-1]["id"]]
    if measured_release(selected[4], source, segments, terminal, 1) != [True]:
        raise ValueError("Pour return did not physically release its source.")
    index, item = items[selected[-1]["id"]]
    validator = one(
        v
        for v in item.get("validators", [])
        if v["kind"] == "object_near_target" and v["object"] == source
    )
    records = terminal["metadata"]["validation"]
    receipt = one(
        v
        for v in records["validators"]
        if v["kind"] == "object_near_target" and v["result"]["object_id"] == source
    )
    value = receipt["result"]
    expected_path = [
        "program",
        "items",
        index,
        "validators",
        receipt["validator_index"],
    ]
    if (
        value["target_id"] != validator["target"]
        or value["position_tolerance"] != validator["position_tolerance"]
        or value["source_path"] != expected_path
        or receipt["source_path"] != expected_path
        or records["accepted_mask"] != [True]
        or receipt["result_mask"] != [True]
        or value["accepted_mask"] != [True]
    ):
        raise ValueError("Pour return validator differs from its original declaration.")
    native = matrix(pose, 1)[0, :3, 3]
    reported = torch.as_tensor(value["object_position"], dtype=torch.float64).reshape(3)
    target = torch.as_tensor(value["target_position"], dtype=torch.float64).reshape(3)
    if (
        not torch.allclose(native, reported, atol=1e-6, rtol=0)
        or not torch.isfinite(target).all()
        or torch.linalg.vector_norm(native - target) > validator["position_tolerance"]
    ):
        raise ValueError("Native Pour return position fails its original validator.")
    settled = one(
        v
        for v in terminal["metadata"]["post_policies"]
        if v["kind"] == "wait_stable" and v["result"]["entity_id"] == source
    )
    declaration = one(
        v for v in item["post"] if v["kind"] == "wait_stable" and v["entity"] == source
    )
    if (
        settled["result_mask"] != [True]
        or settled["result"]["preset"] != declaration["preset"]
        or settled["result"]["state"]["settled_mask"] != [True]
        or settled["result"]["state"]["timeout_mask"] != [False]
    ):
        raise ValueError("Pour return lacks its original native settling acceptance.")


def record_matches(
    record: Mapping,
    group: Mapping,
    graph: Mapping,
    runtime: Mapping,
    program: Mapping,
    constraints: Mapping,
) -> bool:
    """Recompute the raw native witness instead of trusting its accepted flag."""
    try:
        m = record["measurements"]
        contract, call, part, selected, segments, items = _inputs(
            group, graph, runtime, program, constraints, m["step_dt"]
        )
        if (
            record["group_id"] != group["id"]
            or record["uid"] != contract.source_uid
            or record["part_id"] is not None
            or m["scope"] != "native_pour_motion"
            or m["integration_fingerprint"] != graph["integration_fingerprint"]
            or m["boundary_frames"] != segments[selected[-1]["id"]]["end_steps"]
        ):
            return False
        window = NativePourWindow(
            contract.source_uid,
            contract.receiver_uid,
            0,
            tuple(m["steps"]),
            torch.as_tensor(m["source_poses"], dtype=torch.float64),
            torch.as_tensor(m["receiver_poses"], dtype=torch.float64),
        )
        hold = PourHoldEvidence(
            tuple(call["plan_attempts"]),
            tuple(call["effects"]),
            tuple(m["native_grips"]),
            m["object_actor_id"],
            tuple(m["pad_actor_ids"]),
            part,
        )
        result = evaluate_pour_pose_proxy(contract, window, hold)
        _return_matches(
            contract.source_uid, selected, segments, items, m["return_pose"]
        )
        return (
            type(record["accepted"]) is list
            and record["accepted"] == [True]
            and result["accepted"] is True
        )
    except (KeyError, ValueError, TypeError, StopIteration, IndexError, RuntimeError):
        return False


def capture_motion(
    env: Any, graph: Mapping, runtime: Mapping, program: Mapping, constraints: Mapping
) -> dict[str, Any]:
    """Read existing causal trajectory and grip buffers before the runner resets."""
    groups = [
        g
        for g in graph["task_groups"]
        if g["task_type"] == "E3" and g["success"].get("type") == "pour_motion"
    ]
    if not groups:
        return {}
    target = getattr(env, "unwrapped", env)
    from .grip_evidence import SENSOR_UID

    result = {}
    for group in groups:
        try:
            if (
                target.num_envs != 1
                or target._traj_steps.tolist() != runtime["lengths"]
                or runtime["lengths"] != [runtime["length"]]
            ):
                raise ValueError(
                    "Pour motion requires its complete B1 native recording."
                )
            dt = float(target.step_dt)
            c, call, part, selected, segments, items = _inputs(
                group, graph, runtime, program, constraints, dt
            )
            observer = target.sim.get_sensor(SENSOR_UID)._gen_sim_grip_observer
            trace = observer.snapshot_trace()
            actors = trace["actor_bindings"]
            obj_ids = torch.as_tensor(actors["object_actor_ids"][c.source_uid]).reshape(
                -1
            )
            pad_ids = torch.as_tensor(actors["pad_actor_ids"][part]).reshape(-1)
            if obj_ids.numel() != 1 or pad_ids.numel() != 2:
                raise ValueError("Pour motion lacks exact native actor bindings.")

            def pose_values(uid: str, start: int, end: int) -> torch.Tensor:
                if end >= runtime["length"]:
                    raise ValueError("Pour native motion window exceeds its recording.")
                values = (
                    target._traj_buffer["states"]["rigid_objects", uid, "pose"][
                        0, start : end + 1
                    ]
                    .detach()
                    .cpu()
                    .double()
                )
                poses = torch.eye(4, dtype=torch.float64).repeat(len(values), 1, 1)
                poses[:, :3, :3], poses[:, :3, 3] = (
                    matrix_from_quat(values[:, 3:]),
                    values[:, :3],
                )
                return poses

            boundary = segments[selected[-1]["id"]]["end_steps"][0]
            if boundary < runtime["length"]:
                returned = pose_values(c.source_uid, boundary, boundary)
            elif boundary == runtime["length"]:
                returned = (
                    target.sim.get_rigid_object(c.source_uid)
                    .get_local_pose(to_matrix=True)
                    .detach()
                    .cpu()
                    .double()
                )
            else:
                raise ValueError("Pour return boundary exceeds its native recording.")
            m = {
                "scope": "native_pour_motion",
                "integration_fingerprint": graph["integration_fingerprint"],
                "step_dt": dt,
                "steps": list(range(c.start_step, c.end_step + 1)),
                "source_poses": pose_values(
                    c.source_uid, c.start_step, c.end_step
                ).tolist(),
                "receiver_poses": pose_values(
                    c.receiver_uid, c.start_step, c.end_step
                ).tolist(),
                "native_grips": [
                    deepcopy(q)
                    for q in trace["trace"]
                    if q["object_id"] == c.source_uid
                    and q["control_part"] == part
                    and c.planned_at <= q["timestamp"] <= c.end_step * dt + 1e-8
                ],
                "object_actor_id": int(obj_ids[0]),
                "pad_actor_ids": pad_ids.tolist(),
                "return_pose": returned.tolist(),
                "boundary_frames": [boundary],
            }
            record = {
                "group_id": group["id"],
                "uid": c.source_uid,
                "part_id": None,
                "accepted": [True],
                "measurements": m,
            }
            if not record_matches(record, group, graph, runtime, program, constraints):
                raise ValueError("Native Pour motion/return witness was not qualified.")
            result[group["id"]] = record
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            StopIteration,
            RuntimeError,
            IndexError,
        ) as error:
            result[group["id"]] = {"unavailable": str(error)}
    return result
