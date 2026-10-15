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

"""Observe authored E4 transfer and holding contracts without changing execution."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import torch

from embodichain.utils.math import pose_inv, xyz_quat_to_4x4_matrix
from ..contracts import canonical_hash
from .placement_evidence import _valid_pose
from .stability import StabilityConstraint

__all__: list[str] = []

_UNAVAILABLE = (
    AttributeError,
    KeyError,
    StopIteration,
    TypeError,
    ValueError,
    RuntimeError,
)


def _fresh_grip(
    env: Any, record: Mapping[str, Any], terminal: Mapping[str, Any], count: int
) -> dict[str, Any]:
    from embodichain.lab.task_program.semantics.evidence import (
        BinaryEffectEvidenceQuery,
        EffectEvidenceCollectionContext,
    )
    from embodichain.lab.task_program.semantics.effects import (
        BinaryEffectClause,
        BinaryEvidenceKind,
        ControlPartEvidenceAddress,
        EffectEvidenceSourceRef,
        HeldObjectRelation,
        HeldObjectStateExpectation,
    )
    from .grip_evidence import NativeGripObserver, SENSOR_UID

    target = getattr(env, "unwrapped", env)
    observer = target.sim.get_sensor(SENSOR_UID)._gen_sim_grip_observer
    if not isinstance(observer, NativeGripObserver):
        raise ValueError("Final E4 requires its existing native grip observer.")
    destination = record["measurements"]["destination_resource"]
    expectation = _one(
        value
        for value in terminal["effect_spec"]["state_expectations"]
        if value["object_id"] == record["uid"]
        and value["resource_id"] == destination
        and value["relation"] == "attached"
    )
    clause = _one(
        value
        for value in terminal["effect_spec"]["clauses"]
        if value["expectation_id"] == expectation["expectation_id"]
        and value.get("evidence_kind") == "constraint"
        and value.get("expected") is True
    )
    address = clause["source"]["address"]
    part = address["control_part"]
    query = BinaryEffectEvidenceQuery(
        BinaryEffectClause(
            clause["clause_id"],
            clause["expectation_id"],
            EffectEvidenceSourceRef(
                "builtin.control_part",
                "1",
                ControlPartEvidenceAddress(part, "constraint"),
            ),
            BinaryEvidenceKind.CONSTRAINT,
            True,
        ),
        HeldObjectStateExpectation(
            expectation["expectation_id"],
            HeldObjectRelation.ATTACHED,
            record["uid"],
            expectation["slot_id"],
            destination,
            destination,
        ),
    )
    frames = record["measurements"]["boundary_frames"]
    context = EffectEvidenceCollectionContext(
        max(frames) * record["measurements"]["step_dt"],
        terminal["observation_revision"] + 1,
        torch.arange(count, dtype=torch.long),
    )
    observer(query, context)
    return {
        "query": deepcopy(observer.trace[-1]),
        "object_actor_ids": observer.object_actors[record["uid"]].tolist(),
        "pad_actor_ids": observer.pad_actors[part].tolist(),
        "timestamp_scope": "collection_query_clock_not_native_impulse_refresh",
    }


def _grip_flags(
    record: Mapping[str, Any], terminal: Mapping[str, Any], count: int
) -> list[bool]:
    from .grip_evidence import classify_contacts

    metrics, uid = record["measurements"], record["uid"]
    sample = metrics["fresh_grip"]
    query = sample["query"]
    expectation = _one(
        value
        for value in terminal["effect_spec"]["state_expectations"]
        if value["object_id"] == uid
        and value["resource_id"] == metrics["destination_resource"]
        and value["relation"] == "attached"
    )
    clause = _one(
        value
        for value in terminal["effect_spec"]["clauses"]
        if value["expectation_id"] == expectation["expectation_id"]
        and value.get("evidence_kind") == "constraint"
    )
    if (
        query["object_id"] != uid
        or query["control_part"] != clause["source"]["address"]["control_part"]
        or query["expected"] is not True
        or query["dropped_contacts"] != 0
        or query["timestamp"] != max(metrics["boundary_frames"]) * metrics["step_dt"]
        or query["observation_revision"] != terminal["observation_revision"] + 1
        or sample["timestamp_scope"]
        != "collection_query_clock_not_native_impulse_refresh"
    ):
        raise ValueError("Final E4 grip query has another participant or boundary.")
    actors, pads = sample["object_actor_ids"], sample["pad_actor_ids"]
    if len(actors) != count or len(pads) != count or len(query["rows"]) != count:
        raise ValueError("Final E4 grip actor inventory has another batch.")
    flags = []
    for env_id, row in enumerate(query["rows"]):
        inventory = [*actors[env_id], *pads[env_id]]
        if (
            len(actors[env_id]) != 1
            or len(pads[env_id]) != 2
            or len(set(inventory)) != 3
            or any(type(value) is not int for value in inventory)
            or type(row["env_id"]) is not int
            or row["env_id"] != env_id
        ):
            raise ValueError("Final E4 grip lacks exact object/pad actors.")
        raw = row["raw_contacts"]
        if any(type(value) is not int for pair in raw["pairs"] for value in pair):
            raise ValueError("Final E4 grip contact actor IDs are invalid.")
        pairs = torch.tensor(raw["pairs"], dtype=torch.long).reshape(-1, 2)
        normals = torch.tensor(raw["normals"], dtype=torch.float64).reshape(-1, 3)
        impulses = torch.tensor(raw["impulses"], dtype=torch.float64)
        if len(pairs) != len(normals) or impulses.shape != (len(pairs),):
            raise ValueError("Final E4 grip contact arrays disagree.")
        value, valid = classify_contacts(
            pairs, normals, impulses, target=actors[env_id][0], pads=tuple(pads[env_id])
        )
        if (
            not valid
            or row["finite"] is not True
            or row["valid"] is not True
            or row["value"] is not value
        ):
            raise ValueError("Final E4 native grip constraint is unavailable.")
        flags.append(value)
    return flags


def _holding_receipt(
    record: Mapping[str, Any], runtime: Mapping[str, Any]
) -> Mapping[str, Any]:
    metrics = record["measurements"]
    segment = _one(
        value for value in runtime["segments"] if value["name"] == metrics["node_id"]
    )
    return _one(
        value
        for call in segment["metadata"]["runtime"]["calls"]
        for value in call["effects"]
        if canonical_hash(value) == metrics["holding_receipt_hash"]
    )


def _one(values: Any) -> Any:
    values = list(values)
    if len(values) != 1:
        raise ValueError("E4 evidence requires one exact occurrence.")
    return values[0]


def _flags(values: Any, count: int, expected: bool = True) -> None:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(value) is not bool or value is not expected for value in values)
    ):
        raise ValueError("E4 evidence masks are invalid or not accepted.")


def _number(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("E4 evidence requires finite nonnegative clocks/metrics.")
    return float(value)


def _matrix(value: Any, count: int) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float64).cpu()
    _valid_pose(result, count)
    return result


def _delta(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    translation = torch.linalg.vector_norm(first[:, :3, 3] - second[:, :3, 3], dim=-1)
    rotation = first[:, :3, :3].transpose(-1, -2) @ second[:, :3, :3]
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return translation, torch.acos(cosine)


def _call(
    node: Mapping[str, Any],
    segment: Mapping[str, Any],
    program: Mapping[str, Any],
    count: int,
) -> Mapping[str, Any]:
    index, item = _one(
        (i, item)
        for i, item in enumerate(program["program"]["items"])
        if item["name"] == node["id"]
    )
    metadata, runtime = segment["metadata"], segment["metadata"]["runtime"]
    segment_id = f"{program['program_id']}:segment:$.program.items[{index}]"
    if (
        segment["name"] != node["id"]
        or segment.get("success") is not True
        or metadata["task_program_id"] != program["program_id"]
        or metadata["program_segment_id"] != segment_id
        or metadata["program_segment_source_path"] != ["program", "items", index]
        or item["steps"]["call"] != node["call"]
        or runtime["env_ids"] != list(range(count))
        or any(type(value) is not int for value in runtime["env_ids"])
    ):
        raise ValueError("E4 segment differs from its compiled occurrence.")
    _flags(segment["successes"], count)
    _flags(metadata["validation"]["accepted_mask"], count)
    call = _one(runtime["calls"])
    if (
        call["status"] != "completed"
        or call["invocation_id"] != f"{program['program_id']}/{segment_id}:0"
        or type(call["invocation_revision"]) is not int
        or call["invocation_revision"] < 0
        or type(call["command_count"]) is not int
        or call["command_count"] <= 0
    ):
        raise ValueError("E4 invocation did not complete its actual commands.")
    return call


def _receipt(receipt: Mapping[str, Any], call: Mapping[str, Any], count: int) -> None:
    spec, monitor = receipt["effect_spec"], receipt["monitor"]
    if (
        spec["invocation_id"] != call["invocation_id"]
        or spec["invocation_revision"] != call["invocation_revision"]
        or type(spec["invocation_revision"]) is not int
        or spec["env_ids"] != list(range(count))
        or any(type(value) is not int for value in spec["env_ids"])
        or monitor["monitor_id"] != "builtin.composite_effect"
        or monitor["revision"] != "1"
        or type(receipt["observation_revision"]) is not int
    ):
        raise ValueError("E4 physical receipt has another identity or monitor.")
    _number(receipt["timestamp"])
    _flags(receipt["decision"]["success_mask"], count)
    _flags(receipt["decision"]["failure_mask"], count, False)
    for clause in spec["clauses"]:
        expectation = _one(
            value
            for value in spec["state_expectations"]
            if value["expectation_id"] == clause["expectation_id"]
        )
        is_pose = clause.get("expectation") in {"matched", "separated"}
        endpoint = _one(
            value
            for value in call["resolved_core_policy"]["endpoints"]
            if value["slot_id"] == expectation["slot_id"]
            and value["endpoint_id"] == ("motion" if is_pose else "grasp")
            and value["resource_id"] == expectation["resource_id"]
            and value["task_state_key"] == expectation["task_state_key"]
        )
        address = clause["source"]["address"]
        if address["control_part"] != endpoint["target_id"] or address["channel"] != (
            "pose_relation" if is_pose else "constraint"
        ):
            raise ValueError("E4 samples do not belong to the resolved participant.")


def _slot(
    receipt: Mapping[str, Any], uid: str, resource: str, relation: str, count: int
) -> tuple[torch.Tensor, str]:
    spec = receipt["effect_spec"]
    expectation = _one(
        value
        for value in spec["state_expectations"]
        if value["object_id"] == uid
        and value["resource_id"] == resource
        and value["relation"] == relation
        and value["task_state_key"] == resource
    )
    decision = _one(
        value
        for value in receipt["decision"]["expectations"]
        if value["expectation_id"] == expectation["expectation_id"]
    )
    _flags(decision["satisfied_mask"], count)
    _flags(decision["contradicted_mask"], count, False)
    clauses = [
        value
        for value in spec["clauses"]
        if value["expectation_id"] == expectation["expectation_id"]
    ]
    pose = _one(
        value
        for value in clauses
        if value.get("expectation")
        == ("matched" if relation == "attached" else "separated")
    )
    constraint = _one(
        value
        for value in clauses
        if value.get("evidence_kind") == "constraint"
        and value.get("expected") is (relation == "attached")
    )
    samples = [receipt["evidence"][value["clause_id"]] for value in (pose, constraint)]
    for clause, sample in zip((pose, constraint), samples, strict=True):
        if (
            clause["source"]["provider_id"] != "builtin.control_part"
            or clause["source"]["revision"] != "1"
            or sample["env_ids"] != list(range(count))
            or sample["timestamp"] != receipt["timestamp"]
            or sample["observation_revision"] != receipt["observation_revision"]
            or sample.get("acquisition_errors") != [None] * count
        ):
            raise ValueError("E4 physical samples differ from their receipt.")
        _flags(sample["valid_mask"], count)
    _flags(samples[1]["values"], count, relation == "attached")
    return (
        _matrix(samples[0]["object_to_endpoint"], count),
        pose["source"]["address"]["control_part"],
    )


def _frames(segment: Mapping[str, Any], count: int) -> list[int]:
    frames = segment["end_steps"]
    if (
        not isinstance(frames, list)
        or len(frames) != count
        or any(type(frame) is not int or frame <= 0 for frame in frames)
    ):
        raise ValueError("E4 has no exact recorded end-step boundary.")
    return frames


def _holder(segment: Mapping[str, Any], uid: str, destination: str, count: int) -> None:
    state = segment["metadata"]["runtime"]["task_state"]
    if type(state["batch_size"]) is not int or state["batch_size"] != count:
        raise ValueError("E4 holder state has another batch identity.")
    holders = [value for value in state["held_objects"] if value["object_id"] == uid]
    holder = _one(holders)
    if holder["resource_id"] != destination or any(
        value["object_id"] == uid for value in state["coordinated_held_objects"]
    ):
        raise ValueError("E4 boundary does not retain its exclusive receiver.")
    _flags(holder["active_mask"], count)


def _transfer(
    group: Mapping[str, Any],
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    count: int,
    final: bool,
    dt: float,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    node = _one(
        value
        for value in graph["nodes"]
        if value["id"] in group["node_ids"] and value["call"]["kind"] == "hand_over"
    )
    segment = _one(
        value for value in runtime["segments"] if value["name"] == node["id"]
    )
    call = _call(node, segment, program, count)
    if call["semantic_id"] != "hand_over":
        raise ValueError("E4 requires the typed HandOver invocation.")
    uid, resources = node["call"]["object"], node["call"]["resources"]
    source, destination = resources["source"], resources["destination"]
    if source == destination:
        raise ValueError("E4 requires distinct participants.")
    attempt = _one(
        value
        for value in call["plan_attempts"]
        if value["attempt_generation"] == call["active_plan_attempt_generation"]
    )
    planned = _number(attempt["planned_at"])
    effects = [
        value
        for value in call["effects"]
        if value["effect_spec"]["invocation_id"] == call["invocation_id"]
        and value["effect_spec"]["invocation_revision"] == call["invocation_revision"]
        and _number(value["timestamp"]) >= planned
    ]
    terminal = _one(
        value
        for value in effects
        if value["boundary"]["kind"] == "terminal"
        and value["effect_spec"]["effect_kind"] == "transfer"
        and value["decision"]["success_mask"] == [True] * count
    )
    gate = _one(
        value
        for value in reversed(effects)
        if value["boundary"].get("gate_id") == "destination_acquired"
        and value["boundary"].get("segment_name") == "handover_release"
        and value["decision"]["success_mask"] == [True] * count
    )
    for value in (terminal, gate):
        _receipt(value, call, count)
        _slot(value, uid, destination, "attached", count)
    _slot(terminal, uid, source, "detached", count)
    if any(
        value["boundary"]["kind"] == "in_flight_guard"
        and any(value["decision"]["failure_mask"])
        for value in effects
    ):
        raise ValueError("E4 lost its held-object guard during execution.")
    events = [
        value
        for value in call["events"]
        if value.get("invocation_id") == call["invocation_id"]
        and value.get("invocation_revision") == call["invocation_revision"]
        and _number(value["timestamp"]) >= planned
    ]

    def event(kind: str, segment_name: str) -> float:
        return _number(
            _one(
                value
                for value in events
                if value["kind"] == kind and value.get("segment_name") == segment_name
            )["timestamp"]
        )

    acquired = event("phase_effect_gate_satisfied", "handover_release")
    released = event("trajectory_segment_entered", "handover_release")
    retreat = event("trajectory_segment_entered", "source_retreat")
    finished = event("trajectory_completed", "source_retreat")
    completed = event("action_completed", "source_retreat")
    if (
        not planned
        <= acquired
        == gate["timestamp"]
        <= released
        <= retreat
        <= finished
        <= terminal["timestamp"]
        <= completed
    ):
        raise ValueError("E4 transfer receipts do not follow its actual phase order.")
    _holder(segment, uid, destination, count)
    measurements = {
        "scope": "native_handover_transfer",
        "predicate": "handover_complete",
        "transfer_node_id": node["id"],
        "node_id": node["id"],
        "invocation_id": call["invocation_id"],
        "invocation_revision": call["invocation_revision"],
        "source_resource": source,
        "destination_resource": destination,
        "boundary_frames": _frames(segment, count),
        "final": final,
        "step_dt": dt,
        "terminal_transfer_hash": canonical_hash(terminal),
        "acquisition_receipt_hash": canonical_hash(gate),
        "transfer_completed_at": completed,
    }
    return {
        "group_id": group["id"],
        "uid": uid,
        "part_id": None,
        "accepted": [True] * count,
        "measurements": measurements,
    }, terminal


def _hold(
    record: Mapping[str, Any],
    terminal: Mapping[str, Any],
    group: Mapping[str, Any],
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    constraints: Mapping[str, Any],
    count: int,
    object_pose: torch.Tensor,
    tcp_pose: torch.Tensor,
) -> dict[str, Any]:
    metrics = dict(record["measurements"])
    uid, destination = record["uid"], metrics["destination_resource"]
    node = _one(
        value for value in graph["nodes"] if value["id"] == group["node_ids"][-1]
    )
    segment = _one(
        value for value in runtime["segments"] if value["name"] == node["id"]
    )
    call = _call(node, segment, program, count)
    reference, motion_part = _slot(terminal, uid, destination, "attached", count)
    if node["id"] != metrics["transfer_node_id"]:
        if (
            node["call"].get("call_id") != "gen_sim.align_held"
            or node["call"]["arguments"]["object"] != uid
            or node["call"]["resources"]["primary"] != destination
        ):
            raise ValueError("E4 terminal holding continuation is not supported.")
        terminal = _one(
            value
            for value in call["effects"]
            if value["boundary"]["kind"] == "terminal"
            and value["effect_spec"]["effect_kind"] == "attach"
            and value["decision"]["success_mask"] == [True] * count
        )
        _receipt(terminal, call, count)
        reference, motion_part = _slot(terminal, uid, destination, "attached", count)
    _holder(segment, uid, destination, count)
    item_index, item = _one(
        (i, value)
        for i, value in enumerate(program["program"]["items"])
        if value["name"] == node["id"]
    )
    declarations = item.get("post", [])
    policy = None
    cfg = None
    if declarations:
        policy_index, declaration = _one(enumerate(declarations))
        cfg = StabilityConstraint.decode(constraints["presets"][declaration["preset"]])
        if (
            declaration["kind"] != "wait_stable"
            or declaration["entity"] != uid
            or cfg.kind != "hold"
            or cfg.entity != uid
            or cfg.motion_parts != (motion_part,)
        ):
            raise ValueError("E4 hold preset does not belong to its receiver.")
        policy = _one(
            value
            for value in segment["metadata"]["post_policies"]
            if value["source_path"]
            == ["program", "items", item_index, "post", policy_index]
        )
        result = policy["result"]
        if (
            result["kind"] != "hold"
            or result["preset"] != declaration["preset"]
            or _number(result["required_seconds"]) != cfg.duration
        ):
            raise ValueError("E4 hold policy differs from its declaration.")
        for values in (policy["result_mask"], result["accepted_mask"]):
            _flags(values, count)
        _flags(result["failed_mask"], count, False)
        if len(result["stable_seconds"]) != count or any(
            _number(value) < cfg.duration for value in result["stable_seconds"]
        ):
            raise ValueError("E4 did not finish its authored hold duration.")
        elapsed = _number(result["elapsed_seconds"])
    else:
        if (
            node["id"] != metrics["transfer_node_id"]
            or segment["metadata"]["post_policies"]
        ):
            raise ValueError("E4 has no supported terminal holding evidence.")
        elapsed = 0.0
    completed = _number(
        _one(
            value
            for value in call["events"]
            if value["kind"] == "action_completed"
            and value.get("invocation_id") == call["invocation_id"]
            and value.get("invocation_revision") == call["invocation_revision"]
        )["timestamp"]
    )
    frames = _frames(segment, count)
    if any(
        not math.isclose(
            frame * metrics["step_dt"], completed + elapsed, rel_tol=0, abs_tol=1e-8
        )
        for frame in frames
    ):
        raise ValueError(
            "E4 holding evidence differs from the recorded clock boundary."
        )
    if metrics["final"] and (
        runtime["segments"][-1]["name"] != node["id"] or runtime["lengths"] != frames
    ):
        raise ValueError("A later execution cannot certify an earlier final E4 hold.")
    relative = pose_inv(object_pose) @ tcp_pose
    translation, rotation = _delta(relative, reference)
    params = terminal["monitor"]["resolved_params"]
    distance = (
        cfg.translation_drift
        if cfg
        else _number(params["attached_translation_threshold"])
    )
    angle = (
        cfg.rotation_drift if cfg else _number(params["attached_rotation_threshold"])
    )
    if not bool(((translation <= distance) & (rotation <= angle)).all()):
        raise ValueError(
            "Native E4 receiver pose differs from its verified attachment."
        )
    metrics.update(
        scope="native_handover_hold",
        predicate="held",
        node_id=node["id"],
        boundary_frames=frames,
        hold_policy_hash=canonical_hash(policy),
        holding_receipt_hash=canonical_hash(terminal),
        object_pose=object_pose.tolist(),
        receiver_tcp_pose=tcp_pose.tolist(),
        motion_part=motion_part,
    )
    result = {"held": {**record, "measurements": metrics}}
    if cfg is not None:
        reported = policy["result"]["measurements"]
        attachment = torch.as_tensor(
            reported["attachment_translation"], dtype=torch.float64
        )
        if (
            attachment.shape != (1, count)
            or not torch.isfinite(attachment).all()
            or (attachment < 0).any()
            or (attachment > cfg.translation_drift).any()
        ):
            raise ValueError("E4 hold lacks valid native attachment measurements.")
        if cfg.local_axis is not None:
            axis = object_pose.new_tensor(cfg.local_axis)
            world = object_pose.new_tensor(cfg.world_axis or (0.0, 0.0, 1.0))
            alignment = ((object_pose[:, :3, :3] @ axis) * world).sum(-1)
            measured = torch.as_tensor(reported["alignment"], dtype=alignment.dtype)
            if (
                measured.shape != alignment.shape
                or not torch.isfinite(measured).all()
                or not torch.allclose(measured, alignment, atol=1e-4, rtol=0)
                or not (alignment >= cfg.minimum_alignment).all()
            ):
                raise ValueError(
                    "E4 boundary orientation differs from its native hold."
                )
            result["orientation"] = {
                **record,
                "measurements": {
                    **metrics,
                    "predicate": "orientation",
                    "alignment": alignment.tolist(),
                },
            }
    return result


def _native_poses(
    env: Any,
    deployment: Any,
    uid: str,
    frames: list[int],
    motion_part: str,
    count: int,
    final: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = getattr(env, "unwrapped", env)
    binding = _one(
        value
        for value in deployment.integration.registration.scene_binding.rigid_objects
        if value.entity_id == uid
    )
    counts = target._traj_steps
    states = target._traj_buffer["states"]
    objects, joints = (
        states["rigid_objects"][binding.simulation_uid]["pose"],
        states["robot"]["qpos"],
    )
    if (
        len(counts) != count
        or objects.ndim != 3
        or objects.shape[0] != count
        or objects.shape[-1] != 7
        or joints.ndim != 3
        or joints.shape[0] != count
        or any(
            frame > int(counts[row]) or frame >= min(objects.shape[1], joints.shape[1])
            for row, frame in enumerate(frames)
        )
    ):
        raise ValueError("E4 requires its seeded end-step pose and robot state slots.")
    # Recorder rows are pre-action; only the seeded end_step is post-segment.
    pose = xyz_quat_to_4x4_matrix(
        torch.stack([objects[row, frame] for row, frame in enumerate(frames)])
        .detach()
        .cpu()
    )
    qpos = torch.stack([joints[row, frame] for row, frame in enumerate(frames)])
    robot = target.robot
    tcp = (
        robot.compute_fk(
            qpos=qpos[:, robot.get_joint_ids(name=motion_part)],
            name=motion_part,
            to_matrix=True,
        )
        .detach()
        .cpu()
    )
    recorded = (_matrix(pose, count), _matrix(tcp, count))
    if not final:
        return recorded
    native = target.sim.get_rigid_object(binding.simulation_uid)
    fresh_pose = _matrix(native.get_local_pose(to_matrix=True).detach().cpu(), count)
    fresh_tcp = _matrix(
        robot.compute_fk(
            qpos=robot.get_qpos()[:, robot.get_joint_ids(name=motion_part)],
            name=motion_part,
            to_matrix=True,
        )
        .detach()
        .cpu(),
        count,
    )
    if not all(
        torch.allclose(first, last, atol=1e-4, rtol=0)
        for first, last in zip(recorded, (fresh_pose, fresh_tcp), strict=True)
    ):
        raise ValueError("Final native E4 poses do not match their recorded boundary.")
    return fresh_pose, fresh_tcp


def capture_handover_operations(
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
    """Measure E4 events and optional authored holds at their own boundaries."""
    result = {}
    target = getattr(env, "unwrapped", env)
    for group in graph["task_groups"]:
        if group["task_type"] != "E4":
            continue
        records = {
            name: {"unavailable": "No matching E4 native observation."}
            for name in ("handover_complete", "held", "orientation")
        }
        result[group["id"]] = records
        try:
            dt = _number(target.step_dt)
            if dt <= 0:
                raise ValueError("E4 requires the canonical control step duration.")
            transfer, terminal = _transfer(
                group, graph, runtime, program, num_envs, final, dt
            )
            records["handover_complete"] = transfer
        except _UNAVAILABLE as error:
            for name in records:
                records[name] = {"unavailable": str(error)}
            continue
        try:
            boundary = _one(
                value
                for value in runtime["segments"]
                if value["name"] == group["node_ids"][-1]
            )
            _, part = _slot(
                terminal,
                transfer["uid"],
                transfer["measurements"]["destination_resource"],
                "attached",
                num_envs,
            )
            poses = _native_poses(
                env,
                deployment,
                transfer["uid"],
                _frames(boundary, num_envs),
                part,
                num_envs,
                final,
            )
            records.update(
                _hold(
                    transfer,
                    terminal,
                    group,
                    graph,
                    runtime,
                    program,
                    constraints,
                    num_envs,
                    *poses,
                )
            )
        except _UNAVAILABLE as error:
            records["held"] = records["orientation"] = {"unavailable": str(error)}
            continue
        if final:
            try:
                held = records["held"]
                holding_receipt = _holding_receipt(held, runtime)
                held["measurements"]["fresh_grip"] = _fresh_grip(
                    env, held, holding_receipt, num_envs
                )
                held["accepted"] = _grip_flags(held, holding_receipt, num_envs)
            except _UNAVAILABLE as error:
                records["held"] = {"unavailable": str(error)}
    return result


def handover_record_matches(
    record: Mapping[str, Any],
    predicate_name: str,
    group: Mapping[str, Any],
    graph: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    constraints: Mapping[str, Any],
    num_envs: int,
    *,
    final: bool = False,
) -> bool:
    """Recheck occurrence receipts, policy and native metrics instead of flags."""
    try:
        metrics = record["measurements"]
        if metrics["final"] is not final:
            return False
        transfer, terminal = _transfer(
            group, graph, runtime, program, num_envs, final, _number(metrics["step_dt"])
        )
        expected = (
            transfer
            if predicate_name == "handover_complete"
            else _hold(
                transfer,
                terminal,
                group,
                graph,
                runtime,
                program,
                constraints,
                num_envs,
                _matrix(metrics["object_pose"], num_envs),
                _matrix(metrics["receiver_tcp_pose"], num_envs),
            )[predicate_name]
        )
        if final and predicate_name == "held":
            expected["measurements"]["fresh_grip"] = metrics["fresh_grip"]
            expected["accepted"] = _grip_flags(
                expected, _holding_receipt(expected, runtime), num_envs
            )
        return canonical_hash(record) == canonical_hash(expected)
    except _UNAVAILABLE:
        return False
