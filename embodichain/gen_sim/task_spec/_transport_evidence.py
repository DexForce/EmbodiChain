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

from .canonicalization import canonical_hash
from ._held_evidence import flags, one, receipt

__all__: list[str] = []
_CALL = "simulation.coordinated_transport"


def _occurrence(operation, runtime, program):
    descriptor = operation["transport"]
    segment = one(
        value
        for value in runtime["segments"]
        if value["metadata"]["program_segment_id"] == descriptor["segment_id"]
    )
    metadata = segment["metadata"]
    item = program
    for component in descriptor["source_path"]:
        item = item[component]
    declared = item["steps"]["call"]
    call = one(metadata["runtime"]["calls"])
    if (
        segment["name"] != descriptor["segment_name"]
        or item["name"] != descriptor["segment_name"]
        or metadata["task_program_id"] != program["program_id"]
        or metadata["program_segment_source_path"] != descriptor["source_path"]
        or metadata["runtime"]["env_ids"] != [0]
        or call["invocation_id"] != descriptor["invocation_id"]
        or call["status"] != "completed"
        or type(call["command_count"]) is not int
        or call["command_count"] <= 0
    ):
        raise ValueError("Coordinated placement differs from its exact compiled route.")
    flags(segment["successes"], 1)
    flags(metadata["validation"]["accepted_mask"], 1)
    if (
        declared.get("call_id") != _CALL
        or declared["arguments"].get("relation") != "on"
        or declared["arguments"].get("object") != operation["object_id"]
        or declared["arguments"].get("reference") != operation["reference_id"]
        or declared["arguments"].get("target") != operation["target_id"]
        or declared.get("resources") != operation["resources"]
        or call.get("semantic_id") != _CALL
        or call.get("skill_id") != "coordinated_pickment"
    ):
        raise ValueError(
            "Coordinated transport differs from its declared participants and route."
        )
    node = {"id": operation["transport"]["segment_name"], "call": declared}
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
    import torch
    from embodichain.compute.geometry.grip_contacts import classify_contacts

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


def transport_proof(operation, runtime, program, timeline):
    """Recompute acquisition and release from exact call and raw native samples."""
    import torch

    if timeline.get("errors"):
        raise ValueError("Coordinated contact timeline reported data loss.")
    node, segment, call, terminal, parts = _occurrence(operation, runtime, program)
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
