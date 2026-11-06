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

"""Pure transfer/hold checks over explicit HandOver operation descriptors."""

from __future__ import annotations
from collections.abc import Mapping
from types import SimpleNamespace
import math
from typing import Any
from embodichain._json_contract import canonical_hash
from ._held_evidence import (
    one as _one,
    flags as _flags,
    number as _number,
    matrix as _matrix,
    delta as _delta,
    receipt as _receipt,
    slot as _slot,
    frames as _frames,
)
from ._values import _axis

__all__: list[str] = []


def execution_occurrence(
    declaration: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    count: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    segment = _one(
        value
        for value in runtime["segments"]
        if value["metadata"]["program_segment_id"] == declaration["segment_id"]
    )
    item = program
    for component in declaration["source_path"]:
        item = item[component]
    metadata = segment["metadata"]
    observed = metadata["runtime"]
    if (
        segment["name"] != declaration["segment_name"]
        or item.get("name") != declaration["segment_name"]
        or segment.get("success") is not True
        or metadata["task_program_id"] != program["program_id"]
        or metadata["program_segment_source_path"] != declaration["source_path"]
        or observed["env_ids"] != list(range(count))
        or any(type(value) is not int for value in observed["env_ids"])
    ):
        raise ValueError("HandOver segment differs from its compiled occurrence.")
    _flags(segment["successes"], count)
    _flags(metadata["validation"]["accepted_mask"], count)
    call = _one(observed["calls"])
    if (
        call["status"] != "completed"
        or call["invocation_id"] != declaration["invocation_id"]
        or type(call["invocation_revision"]) is not int
        or call["invocation_revision"] < 0
        or type(call["command_count"]) is not int
        or call["command_count"] <= 0
    ):
        raise ValueError("HandOver invocation did not complete its actual commands.")
    return segment, call, item["steps"]["call"]


def hold_condition(
    value: Mapping[str, Any], *, parts_count: int = 1
) -> SimpleNamespace:
    if not isinstance(value, Mapping) or value.get("kind") != "hold":
        raise ValueError("Held observations require an explicit owner hold condition.")
    defaults = dict(
        duration=3.0,
        translation_drift=0.02,
        rotation_drift=math.pi / 18,
        local_axis=None,
        world_axis=None,
        minimum_alignment=0.8660254037844386,
        motion_parts=(),
        target_position=None,
        position_tolerance=0.04,
    )
    from ._values import _identifier

    _identifier(value.get("entity"), "hold entity")
    # Other stability modes are not silently interpreted as a held-axis proof.
    inactive = dict(
        reference=None,
        reference_axis=None,
        displacement=None,
        reference_half_extents=None,
        object_bottom=0.0,
        reference_top=0.0,
        axis_direction_free=False,
        support_tolerance=0.01,
        support_contract_revision=None,
        com_tolerance=None,
        support_sensor=None,
    )
    if set(value) - {*defaults, *inactive, "kind", "entity", "timeout"}:
        raise ValueError("Unknown held condition fields.")
    if any(value.get(name, default) != default for name, default in inactive.items()):
        raise ValueError(
            "This held proof does not support another geometric condition."
        )
    selected = {name: value.get(name, default) for name, default in defaults.items()}
    timeout = value.get("timeout", 12.0)
    if (
        type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or timeout < selected["duration"]
    ):
        raise ValueError("Held timeout must cover its authored duration.")
    parts = selected["motion_parts"]
    if not isinstance(parts, (tuple, list)) or len(parts) != parts_count:
        raise ValueError("HandOver hold requires one explicit receiver motion part.")
    for part in parts:
        _identifier(part, "motion_part")
    if len(set(parts)) != len(parts):
        raise ValueError("Held motion parts must be distinct.")
    target = selected["target_position"]
    if target is not None and (
        not isinstance(target, (list, tuple))
        or len(target) != 3
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in target)
    ):
        raise ValueError("Held target requires three finite coordinates.")
    if target is not None:
        selected["target_position"] = tuple(float(x) for x in target)
    if selected["local_axis"] is None and target is None:
        raise ValueError(
            "HandOver hold requires an explicit axis or position criterion."
        )
    for name in (
        "duration",
        "translation_drift",
        "rotation_drift",
        "position_tolerance",
    ):
        if (
            type(selected[name]) not in (int, float)
            or not math.isfinite(selected[name])
            or selected[name] <= 0
        ):
            raise ValueError("Hold bounds must be positive finite source values.")
    if (
        type(selected["minimum_alignment"]) not in (int, float)
        or not math.isfinite(selected["minimum_alignment"])
        or not 0 <= selected["minimum_alignment"] <= 1
    ):
        raise ValueError("Hold alignment must be finite in [0,1].")
    for name in ("local_axis", "world_axis"):
        if selected[name] is not None:
            selected[name] = _axis(selected[name], name)
    selected["motion_parts"] = tuple(selected["motion_parts"])
    return SimpleNamespace(**selected, kind="hold", entity=value["entity"])


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


def transfer_record(
    operation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    count: int,
    final: bool,
    dt: float,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    segment, call, declared = execution_occurrence(
        operation["transfer"], runtime, program, count
    )
    if (
        declared.get("kind") != "hand_over"
        or declared.get("object") != operation["object_id"]
        or declared["resources"].get("source") != operation["source_resource"]
        or declared["resources"].get("destination") != operation["destination_resource"]
    ):
        raise ValueError("HandOver declaration differs from its grounded participants.")
    if call["semantic_id"] != "hand_over":
        raise ValueError("E4 requires the typed HandOver invocation.")
    uid, resources = declared["object"], declared["resources"]
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
        "transfer_node_id": operation["transfer"]["segment_name"],
        "node_id": operation["transfer"]["segment_name"],
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
        "group_id": operation["operation_id"],
        "uid": uid,
        "part_id": None,
        "accepted": [True] * count,
        "measurements": measurements,
    }, terminal


def holding_boundary(
    record: Mapping[str, Any],
    terminal: Mapping[str, Any],
    operation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    count: int,
) -> tuple[Any, ...]:
    """Select the original receiver receipt without reading physical state."""
    metrics = record["measurements"]
    uid, destination = record["uid"], metrics["destination_resource"]
    segment, call, declared = execution_occurrence(
        operation["boundary"], runtime, program, count
    )
    name = operation["boundary"]["segment_name"]
    reference, motion_part = _slot(terminal, uid, destination, "attached", count)
    if name != metrics["transfer_node_id"]:
        if (
            declared.get("call_id") != "gen_sim.align_held"
            or declared["arguments"]["object"] != uid
            or declared["resources"]["primary"] != destination
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
    return segment, call, terminal, reference, motion_part


def hold_records(
    record: Mapping[str, Any],
    terminal: Mapping[str, Any],
    operation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    program: Mapping[str, Any],
    count: int,
    object_pose: Any,
    tcp_pose: Any,
) -> dict[str, Any]:
    import torch
    from embodichain.utils.math import pose_inv

    metrics = dict(record["measurements"])
    uid, destination = record["uid"], metrics["destination_resource"]
    segment, call, terminal, reference, motion_part = holding_boundary(
        record, terminal, operation, runtime, program, count
    )
    name = operation["boundary"]["segment_name"]
    item = program
    for component in operation["boundary"]["source_path"]:
        item = item[component]
    declarations = item.get("post", [])
    policy = None
    cfg = None
    if declarations:
        policy_index, declaration = _one(enumerate(declarations))
        cfg = hold_condition(operation["hold_condition"])
        if declaration["preset"] != operation["hold_preset"]:
            raise ValueError("HandOver hold selects another named owner condition.")
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
            == [*operation["boundary"]["source_path"], "post", policy_index]
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
        if name != metrics["transfer_node_id"] or segment["metadata"]["post_policies"]:
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
        runtime["segments"][-1]["name"] != name or runtime["lengths"] != frames
    ):
        raise ValueError("A later execution cannot certify an earlier final E4 hold.")
    relative = pose_inv(object_pose) @ tcp_pose
    translation, rotation = _delta(relative, reference)
    metrics.update(
        scope="native_handover_hold",
        predicate="held",
        node_id=name,
        boundary_frames=frames,
        hold_policy_hash=canonical_hash(policy),
        holding_receipt_hash=canonical_hash(terminal),
        object_pose=object_pose.tolist(),
        receiver_tcp_pose=tcp_pose.tolist(),
        motion_part=motion_part,
    )
    result = {"held": {**record, "measurements": metrics}}
    facts = dict(
        attachment_translation=[[float(x)] for x in translation],
        attachment_rotation=[[float(x)] for x in rotation],
        object_position=object_pose[:, :3, 3].tolist(),
        object_rotation=object_pose[:, :3, :3].tolist(),
        observed_seconds=[elapsed] * count,
        stable_seconds=(
            list(policy["result"]["stable_seconds"]) if policy else [0.0] * count
        ),
        grips=[[True] for _ in range(count)],
    )
    result["held"]["_normalized_facts"] = facts
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
            ):
                raise ValueError(
                    "E4 boundary orientation differs from its native hold."
                )
            result["orientation"] = {
                **record,
                "_normalized_facts": facts,
                "measurements": {
                    **metrics,
                    "predicate": "orientation",
                    "alignment": alignment.tolist(),
                },
            }
    return result
