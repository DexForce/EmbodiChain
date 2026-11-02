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

"""Measured E3 tilt/restore events, never liquid-transfer evidence."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from shapely.geometry import Point
import torch
import trimesh

from embodichain.utils.math import xyz_quat_to_4x4_matrix
from .held_evidence import frames, one
from .placement_evidence import _native_mesh, _valid_pose, measured_release
from .placement_proposals import _projection
from .pour_geometry import _top

__all__: list[str] = []

# This coarse action proxy is deliberately distinct from fluid transfer.
_CRITERION = "measured_tilt_restore/v1"
_MINIMUM_TILT = math.radians(30.0)
_MAXIMUM_RESTORE_ERROR = math.radians(15.0)
_UNAVAILABLE = (
    AttributeError,
    KeyError,
    StopIteration,
    TypeError,
    ValueError,
    RuntimeError,
)


def _occurrence(group, graph, runtime, program, count):
    nodes = {n["id"]: n for n in graph["nodes"]}
    segments = {s["name"]: s for s in runtime["segments"]}
    if len(segments) != len(runtime["segments"]) or group["task_type"] != "E3":
        raise ValueError("Pour motion requires unambiguous E3 execution boundaries.")
    selected = [nodes[key] for key in group["node_ids"]]
    pour = one(n for n in selected if n["call"].get("call_id") == "simulation.pour")
    uid = pour["call"]["arguments"]["object"]
    stage = one(
        n
        for n in selected
        if n["call"].get("call_id") == "simulation.move_held_object"
        and n["call"]["arguments"].get("object") == uid
        and n["call"]["arguments"].get("reference") is not None
    )
    release = one(
        n
        for n in selected
        if n["call"].get("kind") == "place" and n["call"]["object"] == uid
    )
    if not selected.index(stage) < selected.index(pour) < selected.index(release):
        raise ValueError("Pour motion requires staging, tilt/restore, then release.")
    resource = pour["call"]["resources"]["primary"]
    eligible = [True] * count
    for node in selected:
        index, item = one(
            (i, v)
            for i, v in enumerate(program["program"]["items"])
            if v["name"] == node["id"]
        )
        segment = segments[node["id"]]
        metadata = segment["metadata"]
        call = one(metadata["runtime"]["calls"])
        segment_id = f"{program['program_id']}:segment:$.program.items[{index}]"
        if (
            node["task_type"] != "E3"
            or node["task_instance_id"] != group["id"]
            or (
                node in (stage, pour, release)
                and node["call"]["resources"].get("primary") != resource
            )
            or item["steps"]["call"] != node["call"]
            or metadata["task_program_id"] != program["program_id"]
            or metadata["program_segment_id"] != segment_id
            or metadata["program_segment_source_path"] != ["program", "items", index]
            or metadata["runtime"]["env_ids"] != list(range(count))
            or any(type(v) is not int for v in metadata["runtime"]["env_ids"])
            or call["invocation_id"] != f"{program['program_id']}/{segment_id}:0"
            or type(call["invocation_revision"]) is not int
            or call["invocation_revision"] < 0
            or call["status"] != "completed"
            or type(call["command_count"]) is not int
            or call["command_count"] <= 0
            or call["semantic_id"] != node["call"].get("call_id", node["call"]["kind"])
        ):
            raise ValueError("Pour motion differs from its exact compiled occurrence.")
        for mask in (
            segment["successes"],
            segment["active"],
            metadata["validation"]["accepted_mask"],
        ):
            if (
                not isinstance(mask, list)
                or len(mask) != count
                or any(type(value) is not bool for value in mask)
            ):
                raise ValueError("Pour eligibility requires aligned boolean masks.")
            eligible = [a and b for a, b in zip(eligible, mask, strict=True)]
    pour_call = one(segments[pour["id"]]["metadata"]["runtime"]["calls"])
    if pour_call["skill_id"] != "pour":
        raise ValueError("Pour motion requires the actual Pour skill.")
    return pour, stage, release, segments, pour_call, eligible


def _classify(metrics, count):
    rows = metrics["rows"]
    if (
        metrics["scope"] != "measured_pour_motion"
        or metrics["criterion"] != _CRITERION
        or metrics["fluid_transfer_verified"] is not False
        or metrics["minimum_tilt_rad"] != _MINIMUM_TILT
        or metrics["maximum_restore_error_rad"] != _MAXIMUM_RESTORE_ERROR
        or len(rows) != count
    ):
        raise ValueError("Pour motion evidence has another criterion or batch.")
    accepted = []
    for row in rows:
        angles = (row["maximum_tilt_rad"], row["restore_error_rad"])
        checks = (row["above_receiver"], row["released"])
        distance, clearance = (
            row["receiver_projection_distance_m"],
            row["receiver_top_clearance_m"],
        )
        if (
            any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or not 0 <= v <= math.pi
                for v in angles
            )
            or any(type(v) is not bool for v in checks)
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (distance, clearance)
            )
            or distance < 0
            or checks[0] != (distance == 0 and clearance >= 0)
        ):
            raise ValueError("Pour motion contains invalid native measurements.")
        accepted.append(
            angles[0] >= _MINIMUM_TILT
            and angles[1] <= _MAXIMUM_RESTORE_ERROR
            and all(checks)
        )
    return accepted


def capture_pour_motions(
    env: Any,
    deployment: Any,
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    num_envs: int,
) -> dict[str, Any]:
    """Read recorded source/receiver poses at this occurrence, before reset.

    The gravity-top point is a geometric alignment proxy, not a detected spout
    or liquid-transfer witness. Missing recording or release evidence remains
    unavailable rather than falling back to command completion.
    """
    if not any(group["task_type"] == "E3" for group in graph["task_groups"]):
        return {}
    target = getattr(env, "unwrapped", env)
    bindings = {
        v.entity_id: v
        for v in deployment.integration.registration.scene_binding.rigid_objects
    }
    result = {}
    for group in graph["task_groups"]:
        if group["task_type"] != "E3":
            continue
        try:
            pour, stage, release, segments, call, eligible = _occurrence(
                group, graph, runtime, program, num_envs
            )
            uid = pour["call"]["arguments"]["object"]
            reference = stage["call"]["arguments"]["reference"]
            if uid == reference:
                raise ValueError(
                    "Pour source and receiver must be distinct native bodies."
                )
            boundary = segments[group["node_ids"][-1]]
            terminal_frames = frames(boundary, num_envs)
            starts = segments[pour["id"]]["start_steps"]
            ends = frames(segments[pour["id"]], num_envs)
            vertices, _ = _native_mesh(target.sim, bindings[uid], num_envs)
            receiving, faces = _native_mesh(target.sim, bindings[reference], num_envs)
            states = target._traj_buffer["states"]["rigid_objects"]
            source = states[bindings[uid].simulation_uid]["pose"]
            receiver = states[bindings[reference].simulation_uid]["pose"]
            released = measured_release(release, uid, segments, boundary, num_envs)
            if (
                not isinstance(starts, list)
                or len(starts) != num_envs
                or source.ndim != 3
                or receiver.ndim != 3
                or source.shape[0] != num_envs
                or receiver.shape[0] != num_envs
                or source.shape[-1] != 7
                or receiver.shape[-1] != 7
            ):
                raise ValueError("Pour motion lacks its recorded native trajectory.")
            rows, peaks = [], []
            for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
                if (
                    type(start) is not int
                    or not 0
                    <= start
                    < end
                    <= terminal_frames[row]
                    <= int(target._traj_steps[row])
                    or end >= min(source.shape[1], receiver.shape[1])
                ):
                    raise ValueError(
                        "Pour motion window has no exact post-action slot."
                    )
                poses = xyz_quat_to_4x4_matrix(
                    source[row, start : end + 1].detach().cpu()
                )
                _valid_pose(poses, len(poses))
                up, pivot = _top(vertices[row].numpy(), poses[0, :3, :3].numpy())
                up_world = poses[:, :3, :3] @ torch.as_tensor(up).to(poses)
                tilts = torch.acos(up_world[:, 2].clamp(-1, 1))
                peak = int(tilts.argmax())
                peaks.append(start + peak)
                reference_pose = xyz_quat_to_4x4_matrix(
                    receiver[row, start + peak].detach().cpu()[None]
                )
                _valid_pose(reference_pose, 1)
                world = (
                    receiving[row].to(reference_pose) @ reference_pose[0, :3, :3].T
                    + reference_pose[0, :3, 3]
                )
                anchor = (
                    poses[peak, :3, :3] @ torch.as_tensor(pivot).to(poses)
                    + poses[peak, :3, 3]
                )
                projection = _projection(
                    trimesh.Trimesh(world.numpy(), faces[row].numpy(), process=False)
                )
                if projection.is_empty or not projection.is_valid:
                    raise ValueError(
                        "Pour receiver has no valid native mesh projection."
                    )
                above = bool(
                    projection.covers(Point(anchor[:2].tolist()))
                    and anchor[2] >= world[:, 2].max()
                )
                rows.append(
                    {
                        "maximum_tilt_rad": float(tilts[peak]),
                        "restore_error_rad": float(tilts[-1]),
                        "anchor_world_m": anchor.tolist(),
                        "receiver_projection_distance_m": float(
                            projection.distance(Point(anchor[:2].tolist()))
                        ),
                        "receiver_top_clearance_m": float(
                            anchor[2] - world[:, 2].max()
                        ),
                        "above_receiver": above,
                        "released": released[row],
                    }
                )
            metrics = {
                "scope": "measured_pour_motion",
                "criterion": _CRITERION,
                "fluid_transfer_verified": False,
                "reference_uid": reference,
                "pour_node_id": pour["id"],
                "invocation_id": call["invocation_id"],
                "invocation_revision": call["invocation_revision"],
                "boundary_frames": terminal_frames,
                "start_frames": starts,
                "end_frames": ends,
                "peak_frames": peaks,
                "minimum_tilt_rad": _MINIMUM_TILT,
                "maximum_restore_error_rad": _MAXIMUM_RESTORE_ERROR,
                "rows": rows,
            }
            result[group["id"]] = {
                "group_id": group["id"],
                "uid": uid,
                "part_id": None,
                "accepted": [
                    valid and active
                    for valid, active in zip(
                        _classify(metrics, num_envs), eligible, strict=True
                    )
                ],
                "measurements": metrics,
            }
        except _UNAVAILABLE as error:
            result[group["id"]] = {"unavailable": str(error) or type(error).__name__}
    return result


def pour_record_matches(
    record: Mapping[str, Any],
    predicate: Any,
    roles: Mapping[str, Any],
    group: Mapping[str, Any],
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    count: int,
) -> bool:
    """Bind measured historical events to their source, receiver and exact call."""
    try:
        pour, stage, _, segments, call, eligible = _occurrence(
            group, graph, runtime, program, count
        )
        metrics = record["measurements"]
        reference = stage["call"]["arguments"]["reference"]
        starts, ends = segments[pour["id"]]["start_steps"], frames(
            segments[pour["id"]], count
        )
        return bool(
            predicate.name == "pour_motion"
            and predicate.arguments.get("task_type") == "E3"
            and predicate.arguments.get("relation") == "above"
            and roles.get(predicate.arguments.get("reference"))
            == [{"uid": reference, "part_id": None}]
            and {"uid": record["uid"], "part_id": None}
            in roles.get(predicate.arguments.get("subject"), [])
            and record["uid"] == pour["call"]["arguments"]["object"]
            and record["part_id"] is None
            and record["group_id"] == group["id"]
            and metrics["reference_uid"] == reference
            and metrics["pour_node_id"] == pour["id"]
            and metrics["invocation_id"] == call["invocation_id"]
            and metrics["invocation_revision"] == call["invocation_revision"]
            and metrics["boundary_frames"]
            == frames(segments[group["node_ids"][-1]], count)
            and metrics["start_frames"] == starts
            and metrics["end_frames"] == ends
            and len(metrics["peak_frames"]) == count
            and all(
                type(p) is int and s <= p <= e
                for s, p, e in zip(starts, metrics["peak_frames"], ends, strict=True)
            )
            and record["accepted"]
            == [
                valid and active
                for valid, active in zip(
                    _classify(metrics, count), eligible, strict=True
                )
            ]
            and all(type(v) is bool for v in record["accepted"])
        )
    except _UNAVAILABLE:
        return False
