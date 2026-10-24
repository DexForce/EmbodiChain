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

"""Native E5 acquisition and dual release for explicitly bound on-placement."""

from __future__ import annotations

import math
from typing import Any

import torch

from embodichain.utils.math import xyz_quat_to_4x4_matrix
from ..contracts import canonical_hash
from .held_evidence import flags, frames, one, receipt
from .grip_evidence import classify_contacts
from .placement_evidence import measure_planar_placement
from .stable_support_evidence import capture_stable_support

__all__: list[str] = []
_CALL = "simulation.coordinated_transport"


def _occurrence(group, graph, runtime, program):
    nodes = {node["id"]: node for node in graph["nodes"]}
    node = one(
        nodes[key]
        for key in group["node_ids"]
        if nodes[key]["call"].get("call_id") == _CALL
    )
    index, item = one(
        (index, item)
        for index, item in enumerate(program["program"]["items"])
        if item["name"] == node["id"]
    )
    segment = one(value for value in runtime["segments"] if value["name"] == node["id"])
    call = one(segment["metadata"]["runtime"]["calls"])
    identity = f"{program['program_id']}:segment:$.program.items[{index}]"
    if (
        group["task_type"] != "E5"
        or node["task_type"] != "E5"
        or node["task_instance_id"] != group["id"]
        or item["steps"]["call"] != node["call"]
        or node["call"]["arguments"].get("relation") != "on"
        or set(node["call"]["resources"]) != {"left", "right"}
        or len(set(node["call"]["resources"].values())) != 2
        or segment["metadata"]["task_program_id"] != program["program_id"]
        or segment["metadata"]["program_segment_id"] != identity
        or segment["metadata"]["program_segment_source_path"]
        != ["program", "items", index]
        or segment["metadata"]["runtime"]["env_ids"] != [0]
        or call["invocation_id"] != f"{program['program_id']}/{identity}:0"
        or call["semantic_id"] != _CALL
        or call["skill_id"] != "coordinated_pickment"
        or call["status"] != "completed"
        or type(call["command_count"]) is not int
        or call["command_count"] <= 0
    ):
        raise ValueError("Coordinated placement differs from its exact compiled route.")
    flags(segment["successes"], 1)
    flags(segment["metadata"]["validation"]["accepted_mask"], 1)
    releases = [
        value
        for value in call["effects"]
        if value["boundary"]["kind"] == "terminal"
        and value["effect_spec"]["effect_kind"] == "release"
        and value["decision"]["success_mask"] == [True]
    ]
    terminal = releases[-1] if releases else None
    if terminal is None:
        raise ValueError(
            "Coordinated placement has no successful dual release receipt."
        )
    receipt(terminal, call, 1)
    if (
        terminal["effect_spec"].get("semantic_id") != _CALL
        or terminal["effect_spec"].get("skill_id") != "coordinated_pickment"
    ):
        raise ValueError("Coordinated release has another semantic owner.")
    expectations = terminal["effect_spec"]["state_expectations"]
    if len(expectations) != 2:
        raise ValueError("Coordinated release must cover exactly both hands.")
    parts = {}
    for role, resource in node["call"]["resources"].items():
        expectation = one(
            value
            for value in expectations
            if value["slot_id"] == role
            and value["resource_id"] == resource
            and value["object_id"] == node["call"]["arguments"]["object"]
            and value["relation"] == "detached"
        )
        clause = one(
            value
            for value in terminal["effect_spec"]["clauses"]
            if value["expectation_id"] == expectation["expectation_id"]
            and value.get("evidence_kind") == "constraint"
            and value.get("expected") is False
        )
        sample = terminal["evidence"][clause["clause_id"]]
        if (
            sample["timestamp"] != terminal["timestamp"]
            or sample["observation_revision"] != terminal["observation_revision"]
            or sample["env_ids"] != [0]
        ):
            raise ValueError("Coordinated release sample has another native boundary.")
        flags(sample["valid_mask"], 1)
        flags(sample["values"], 1, False)
        parts[role] = clause["source"]["address"]["control_part"]
    if len(set(parts.values())) != 2:
        raise ValueError("Coordinated release cannot reuse one hand twice.")
    return node, segment, call, terminal, parts


def _native_grips(timeline, sample, uid, parts):
    if sample["dropped_contacts"] != 0 or len(sample["rows"]) != 1:
        raise ValueError("Coordinated native contacts are incomplete.")
    raw = sample["rows"][0]
    if any(type(value) is not int for pair in raw["pairs"] for value in pair):
        raise ValueError("Coordinated native actor IDs must be integers.")
    pairs = torch.tensor(raw["pairs"], dtype=torch.long).reshape(-1, 2)
    normals = torch.tensor(raw["normals"], dtype=torch.float64).reshape(-1, 3)
    impulses = torch.tensor(raw["impulses"], dtype=torch.float64)
    if normals.shape != (len(pairs), 3) or impulses.shape != (len(pairs),):
        raise ValueError("Coordinated raw contact arrays have different lengths.")
    if (
        not torch.isfinite(normals).all()
        or not torch.isfinite(impulses).all()
        or (impulses < 0).any()
        or (torch.linalg.vector_norm(normals[impulses > 0], dim=-1) <= 1e-8).any()
    ):
        raise ValueError("Coordinated raw contact payload is invalid.")
    actors = timeline["object_actors"][uid][0]
    if len(actors) != 1:
        raise ValueError("Coordinated object actor is ambiguous.")
    values = {}
    inventory = [
        actor for part in parts.values() for actor in timeline["pad_actors"][part][0]
    ]
    if (
        len(inventory) != 4
        or len(set(inventory)) != 4
        or any(type(actor) is not int for actor in [actors[0], *inventory])
    ):
        raise ValueError("Coordinated hands require disjoint native pad actors.")
    for role, part in parts.items():
        pads = timeline["pad_actors"][part][0]
        if len(pads) != 2 or len(set([actors[0], *pads])) != 3:
            raise ValueError("Coordinated pad actors are ambiguous.")
        value, valid = classify_contacts(
            pairs, normals, impulses, target=actors[0], pads=tuple(pads)
        )
        values[role] = (value, valid)
    return values


def transport_proof(group, graph, runtime, program, timeline):
    """Recompute acquisition and release from exact call and raw native samples."""
    if timeline.get("errors"):
        raise ValueError("Coordinated contact timeline reported data loss.")
    node, segment, call, terminal, parts = _occurrence(group, graph, runtime, program)
    uid = node["call"]["arguments"]["object"]
    dt = timeline["step_dt"]
    if type(dt) not in (float, int) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Coordinated contact clock is invalid.")
    samples = {value["step"]: value for value in timeline["samples"]}
    if len(samples) != len(timeline["samples"]):
        raise ValueError("Duplicate native contact boundary.")
    if any(
        type(step) is not int
        or step < 0
        or type(value["time"]) not in (int, float)
        or not math.isfinite(value["time"])
        or not math.isclose(value["time"], step * dt, abs_tol=1e-8, rel_tol=0)
        for step, value in samples.items()
    ):
        raise ValueError("Coordinated native sample has an inconsistent step/time.")
    release_step = round(terminal["timestamp"] / dt)
    if not math.isclose(
        release_step * dt, terminal["timestamp"], abs_tol=1e-8, rel_tol=0
    ):
        raise ValueError("Coordinated release is not on the recorded control clock.")
    release = samples[release_step]
    if release["time"] != release_step * dt:
        raise ValueError("Coordinated release sample has a stale timestamp.")
    grips = _native_grips(timeline, release, uid, parts)
    for role, part in parts.items():
        opening = release["hand_opening"][part][0]
        measured = torch.tensor(opening["measured_qpos"])
        target = torch.tensor(opening["open_qpos"])
        if (
            not opening["finite"]
            or measured.numel() == 0
            or not torch.isfinite(measured).all()
            or not torch.isfinite(target).all()
            or measured.shape != target.shape
            or not (0 < opening["tolerance"] <= 0.1)
        ):
            raise ValueError("Coordinated release lacks measured driving joints.")
        if (
            grips[role] != (False, True)
            or not ((measured - target).abs() < opening["tolerance"]).all()
        ):
            raise ValueError("Coordinated hand is not physically open and released.")

    def phase(name):
        return one(
            event["timestamp"]
            for event in call["events"]
            if event["kind"] == "trajectory_segment_entered"
            and event.get("segment_name") == name
            and event.get("invocation_id") == call["invocation_id"]
            and event.get("invocation_revision") == call["invocation_revision"]
        )

    start, end = phase("lift"), phase("move")
    acquired = [
        sample
        for sample in samples.values()
        if start <= sample["time"] <= end
        and all(
            value == (True, True)
            for value in _native_grips(timeline, sample, uid, parts).values()
        )
    ]
    if not acquired:
        raise ValueError("No measured simultaneous two-hand grip during this lift.")
    return (
        node,
        segment,
        call,
        terminal,
        {
            "scope": "measured_coordinated_transport/v1",
            "invocation_id": call["invocation_id"],
            "invocation_revision": call["invocation_revision"],
            "release_step": release_step,
            "acquired_step": acquired[0]["step"],
            "contact_timeline_digest": canonical_hash(timeline),
            "release_receipt_digest": canonical_hash(terminal),
        },
    )


def capture_transport_placement(
    target: Any,
    deployment: Any,
    graph: dict,
    runtime: dict,
    program: dict,
    group: dict,
    member: dict,
    boundary: dict,
    count: int,
    *,
    recorded: bool,
) -> dict:
    """Measure supported placement only for the configured E5 on route."""
    if count != 1:
        raise ValueError(
            "Coordinated placement observation currently requires one environment."
        )
    reader = getattr(target, "_gen_sim_acceptance_contacts", None)
    if reader is None:
        raise ValueError("Coordinated placement has no recorded native contacts.")
    timeline = reader.snapshot()
    node, segment, call, terminal, proof = transport_proof(
        group, graph, runtime, program, timeline
    )
    uid, reference = (
        node["call"]["arguments"]["object"],
        node["call"]["arguments"]["reference"],
    )
    factory = one(
        value
        for value in deployment.integration.registration.registered_semantic_lowerer_factories
        if value.call_id == _CALL
    )
    one(
        route
        for route in factory.routes
        if route.object_id == uid
        and route.target_id == node["call"]["arguments"]["target"]
        and route.reference_entity_id == reference
        and route.relation == "on"
    )
    held = boundary["metadata"]["runtime"]["task_state"]
    if any(
        value["object_id"] == uid and any(value["active_mask"])
        for value in [*held["held_objects"], *held["coordinated_held_objects"]]
    ):
        raise ValueError(
            "Coordinated placement is still held at its observation boundary."
        )
    bindings = {
        value.entity_id: value
        for value in deployment.integration.registration.scene_binding.rigid_objects
    }
    observed_frames = frames(boundary, count)
    if recorded:
        states = target._traj_buffer["states"]["rigid_objects"]

        def pose(name):
            values = states[bindings[name].simulation_uid]["pose"]
            if values.shape[1] <= observed_frames[0]:
                raise ValueError("Coordinated placement lacks its exact boundary slot.")
            return xyz_quat_to_4x4_matrix(values[:, observed_frames[0]].detach().cpu())

    else:

        def pose(name):
            return (
                target.sim.get_rigid_object(bindings[name].simulation_uid)
                .get_local_pose(to_matrix=True)
                .detach()
                .cpu()
            )

    constraint = dict(deployment.integration.adapter_factory.constraints)[
        f"gen_sim.{node['id']}.stable"
    ]
    accepted, metrics = capture_stable_support(
        target,
        bindings,
        uid,
        reference,
        observed_frames[0],
        constraint,
        terminal["timestamp"],
    )
    try:
        _, landing = measure_planar_placement(
            target.sim,
            bindings[uid],
            bindings[reference],
            pose(uid),
            pose(reference),
            exact_silhouette=True,
        )
    except ValueError as error:
        landing = {"unavailable": str(error)}
    metrics.update(
        {key: value for key, value in proof.items() if key != "scope"},
        transport_scope=proof["scope"],
        reference_uid=reference,
        relation="on",
        released=[True],
        boundary_frames=observed_frames,
        observation_kind="recorded" if recorded else "final",
        placement_node_id=node["id"],
        landing_region_diagnostic=landing,
    )
    return {
        "group_id": group["id"],
        "uid": uid,
        "part_id": member["object_part_id"],
        "accepted": accepted,
        "measurements": metrics,
    }


def capture_transport_holds(
    graph: dict, runtime: dict, program: dict, timeline: dict | None
) -> dict:
    """Retain historical E5 acquisition independently of its later release."""
    result = {}
    nodes = {node["id"]: node for node in graph["nodes"]}
    for group in graph["task_groups"]:
        if group["task_type"] != "E5" or not any(
            nodes[key]["call"].get("call_id") == _CALL for key in group["node_ids"]
        ):
            continue
        try:
            if timeline is None:
                raise ValueError("No coordinated native contact timeline.")
            node, _, _, _, proof = transport_proof(
                group, graph, runtime, program, timeline
            )
            result[group["id"]] = {
                "group_id": group["id"],
                "uid": node["call"]["arguments"]["object"],
                "part_id": None,
                "accepted": [True],
                "measurements": proof,
            }
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as error:
            result[group["id"]] = {"unavailable": str(error)}
    return result


def transport_record_matches(
    record,
    predicate,
    roles,
    group,
    graph,
    runtime,
    program,
    timeline,
    *,
    acquisition=False,
):
    """Rebind the native observations, rather than accepting cached proof flags."""
    try:
        node, _, _, terminal, proof = transport_proof(
            group, graph, runtime, program, timeline
        )
        metrics = record["measurements"]
        uid = node["call"]["arguments"]["object"]
        if (
            record["group_id"] != group["id"]
            or record["uid"] != uid
            or record["part_id"] is not None
            or any(
                metrics.get(
                    "transport_scope" if key == "scope" and not acquisition else key
                )
                != value
                for key, value in proof.items()
            )
        ):
            return False
        if {"uid": uid, "part_id": None} not in roles.get(
            predicate.arguments.get("subject"), []
        ):
            return False
        if acquisition:
            return predicate.name == "held_by_both_grippers" and record["accepted"] == [
                True
            ]
        reference = node["call"]["arguments"]["reference"]
        return (
            predicate.arguments.get("task_type") == "E5"
            and predicate.arguments.get("relation") == "on"
            and roles.get(predicate.arguments.get("reference"))
            == [{"uid": reference, "part_id": None}]
            and metrics["reference_uid"] == reference
            and metrics["relation"] == "on"
            and metrics["released"] == [True]
            and metrics["release_timestamp"] == terminal["timestamp"]
            and metrics["placement_node_id"] == node["id"]
        )
    except (KeyError, ValueError, TypeError, AttributeError, RuntimeError):
        return False
