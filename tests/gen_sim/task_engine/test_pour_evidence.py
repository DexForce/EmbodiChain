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
"""CPU native-record fixtures, not simulation or material-transfer evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.pour_evidence import (
    NativePourWindow,
    PourHoldEvidence,
    PourProxyContract,
    evaluate_pour_pose_proxy,
)


def rotation(angle: float, axis: str = "x") -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def fixture():
    source = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
    source[:, 2, 3] = 1
    source[1, :3, :3] = rotation(math.pi / 6)
    source[2, :3, :3] = rotation(math.pi / 3)
    source[3, :3, :3] = rotation(math.pi / 6)
    receiver = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
    receiver[:, 2, 3] = 0.5
    contract = PourProxyContract(
        "source",
        "receiver",
        "left",
        "program/pour:0",
        0,
        0,
        0.04,
        2,
        6,
        0.04,
        source[0].clone(),
        source[2].clone(),
        (0, 0, 0),
        (0, 0, 0.5),
        0.20,
        0.20,
        0.02,
    )
    window = NativePourWindow(
        "source", "receiver", 0, (2, 3, 4, 5, 6), source, receiver
    )
    effects, queries = [], []
    for revision, (step, kind) in enumerate(
        [
            (2, "phase_effect_gate"),
            (3, "in_flight_guard"),
            (4, "in_flight_guard"),
            (5, "in_flight_guard"),
            (6, "in_flight_guard"),
            (6, "terminal"),
        ]
    ):
        timestamp = step * contract.step_dt
        spec = {
            "semantic_id": "simulation.pour",
            "skill_id": "pour",
            "effect_kind": "attach",
            "invocation_id": contract.invocation_id,
            "invocation_revision": 0,
            "env_ids": [0],
            "state_expectations": [
                {
                    "expectation_id": "primary",
                    "object_id": "source",
                    "resource_id": "left",
                    "relation": "attached",
                }
            ],
            "clauses": [
                {
                    "clause_id": "grip",
                    "expectation_id": "primary",
                    "evidence_kind": "constraint",
                    "expected": True,
                    "source": {
                        "provider_id": "builtin.control_part",
                        "revision": "1",
                        "address": {
                            "type": "address",
                            "control_part": "left_eef",
                            "channel": "constraint",
                        },
                    },
                }
            ],
        }
        effects.append(
            {
                "monitor": {"monitor_id": "builtin.composite_effect", "revision": "1"},
                "effect_spec": spec,
                "timestamp": timestamp,
                "observation_revision": revision,
                "boundary": {
                    "kind": kind,
                    **({"segment_name": "pour"} if kind != "terminal" else {}),
                },
                "evidence": {
                    "grip": {
                        "timestamp": timestamp,
                        "observation_revision": revision,
                        "env_ids": [0],
                        "valid_mask": [True],
                        "values": [True],
                    }
                },
                "decision": {"success_mask": [True], "failure_mask": [False]},
            }
        )
        queries.append(
            {
                "object_id": "source",
                "control_part": "left_eef",
                "timestamp": timestamp,
                "observation_revision": revision,
                "expected": True,
                "dropped_contacts": 0,
                "rows": [
                    {
                        "env_id": 0,
                        "finite": True,
                        "valid": True,
                        "value": True,
                        "raw_contacts": {
                            "pairs": [[10, 21], [10, 22]],
                            "normals": [[1, 0, 0], [-1, 0, 0]],
                            "impulses": [1, 1],
                        },
                    }
                ],
            }
        )
    hold = PourHoldEvidence(
        (
            {
                "planned_at": 0.04,
                "invocation_id": contract.invocation_id,
                "invocation_revision": 0,
                "attempt_generation": 7,
                "plan_success_mask": [True],
            },
        ),
        tuple(effects),
        tuple(queries),
        10,
        (21, 22),
        "left_eef",
    )
    return contract, window, hold


def test_measured_pose_proxy_is_distinct_from_unmodeled_fluid_and_return():
    result = evaluate_pour_pose_proxy(*fixture())
    assert result["status"] == "satisfied" and result["accepted"] is True
    assert result["fluid_status"] == "unmodeled"
    assert result["measurements"]["peak_step"] == 4
    assert result["measurements"]["restore_step"] == 6
    assert "release" not in result and "park" not in result


@pytest.mark.parametrize("axis,sign", [("z", 1), ("x", -1)])
def test_wrong_axis_or_sign_with_correct_unsigned_tilt_is_rejected(axis, sign):
    c, w, h = fixture()
    w.source_poses[1:4, :3, :3] = rotation(sign * math.pi / 3, axis)
    assert evaluate_pour_pose_proxy(c, w, h)["status"] == "rejected"


def test_restore_must_follow_peak_and_terminal_pose_must_remain_restored():
    c, w, h = fixture()
    w.source_poses[-1, :3, :3] = rotation(math.pi / 3)
    assert evaluate_pour_pose_proxy(c, w, h)["accepted"] is False
    w.source_poses[:] = c.entry_pose
    assert evaluate_pour_pose_proxy(c, w, h)["accepted"] is False


@pytest.mark.parametrize("offset,accepted", [(0.019, True), (0.021, False)])
def test_explicit_receiver_position_tolerance_uses_measured_window(offset, accepted):
    c, w, h = fixture()
    w.receiver_poses[2, 0, 3] += offset
    assert evaluate_pour_pose_proxy(c, w, h)["accepted"] is accepted


def test_world_yaw_and_translation_preserve_receiver_relative_measurement():
    c, w, h = fixture()
    world = torch.eye(4, dtype=torch.float64)
    world[:3, :3] = rotation(0.7, "z")
    world[:3, 3] = torch.tensor([0.3, -0.2, 0.8])
    c = replace(c, entry_pose=world @ c.entry_pose, peak_pose=world @ c.peak_pose)
    w = replace(
        w, source_poses=world @ w.source_poses, receiver_poses=world @ w.receiver_poses
    )
    assert evaluate_pour_pose_proxy(c, w, h)["accepted"] is True


def test_nonzero_body_pivot_and_rotated_receiver_use_full_frame_composition():
    c, w, h = fixture()
    pivot = torch.tensor([0.02, -0.01, 0.10], dtype=torch.float64)
    world_anchor = torch.tensor([0.1, -0.2, 1.0], dtype=torch.float64)
    initial = rotation(0.4, "z") @ rotation(-0.3)
    w.source_poses[:, :3, :3] = initial @ w.source_poses[:, :3, :3]
    w.source_poses[:, :3, 3] = world_anchor - w.source_poses[:, :3, :3] @ pivot
    w.receiver_poses[:, :3, :3] = rotation(0.8, "z") @ rotation(0.2)
    receiver = w.receiver_poses[0]
    anchor = receiver[:3, :3].T @ (world_anchor - receiver[:3, 3])
    c = replace(
        c,
        entry_pose=w.source_poses[0].clone(),
        peak_pose=w.source_poses[2].clone(),
        pivot_object=tuple(pivot.tolist()),
        receiver_anchor_local=tuple(anchor.tolist()),
    )
    result = evaluate_pour_pose_proxy(c, w, h)
    assert result["accepted"] is True
    assert max(result["measurements"]["receiver_position_error_m"]) < 1e-12


@pytest.mark.parametrize("error,accepted", [(0.199, True), (0.201, False)])
def test_explicit_full_R_peak_tolerance_is_not_an_unsigned_angle_test(error, accepted):
    c, w, h = fixture()
    w.source_poses[2, :3, :3] = rotation(math.pi / 3 + error)
    assert evaluate_pour_pose_proxy(c, w, h)["accepted"] is accepted


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "receiver",
        "env",
        "missing_step",
        "duplicate_step",
        "nan",
        "reflection",
    ],
)
def test_missing_or_foreign_native_pose_evidence_is_unavailable(damage):
    c, w, h = fixture()
    if damage == "source":
        w = replace(w, source_uid="foreign")
    elif damage == "receiver":
        w = replace(w, receiver_uid="foreign")
    elif damage == "env":
        w = replace(w, env_id=1)
    elif damage == "missing_step":
        w = replace(w, steps=(2, 3, 5, 6))
    elif damage == "duplicate_step":
        w = replace(w, steps=(2, 3, 3, 5, 6))
    elif damage == "nan":
        w.source_poses[2, 0, 3] = math.nan
    else:
        w.source_poses[2, :3, :3] *= -1
    assert evaluate_pour_pose_proxy(c, w, h)["status"] == "unavailable"


@pytest.mark.parametrize(
    "damage",
    [
        "guard_gap",
        "duplicate_guard",
        "no_gate",
        "no_terminal",
        "native_gap",
        "dropped",
        "wrong_actor",
        "wrong_resource",
        "foreign_invocation",
        "invalid_native",
        "ambiguous_plan",
        "cross_plan",
        "duplicate_native",
    ],
)
def test_incomplete_or_mismatched_hold_evidence_cannot_certify(damage):
    c, w, h = fixture()
    effects, queries, attempts = (
        list(deepcopy(h.effects)),
        list(deepcopy(h.native_grips)),
        list(deepcopy(h.plan_attempts)),
    )
    if damage == "guard_gap":
        effects.pop(2)
    elif damage == "duplicate_guard":
        effects.append(deepcopy(effects[2]))
    elif damage == "no_gate":
        effects.pop(0)
    elif damage == "no_terminal":
        effects.pop()
    elif damage == "native_gap":
        queries.pop(2)
    elif damage == "dropped":
        queries[2]["dropped_contacts"] = 1
    elif damage == "wrong_actor":
        h = replace(h, object_actor_id=11)
    elif damage == "wrong_resource":
        effects[2]["effect_spec"]["state_expectations"][0]["resource_id"] = "right"
    elif damage == "foreign_invocation":
        effects[2]["effect_spec"]["invocation_id"] = "other/pour:0"
    elif damage == "invalid_native":
        queries[2]["rows"][0]["valid"] = False
    elif damage == "ambiguous_plan":
        attempts.append(deepcopy(attempts[0]))
    elif damage == "duplicate_native":
        queries.append(deepcopy(queries[2]))
    else:
        attempts.append({**attempts[0], "planned_at": 0.16})
    h = replace(
        h,
        effects=tuple(effects),
        native_grips=tuple(queries),
        plan_attempts=tuple(attempts),
    )
    assert evaluate_pour_pose_proxy(c, w, h)["status"] == "unavailable"


def test_monitor_warmup_is_not_an_invented_instantaneous_loss_gate():
    c, w, h = fixture()
    h.effects[2]["decision"]["success_mask"] = [False]
    assert evaluate_pour_pose_proxy(c, w, h)["accepted"] is True


@pytest.mark.parametrize("step", [3, 4, 6])
def test_every_post_ack_guard_including_first_and_end_is_required(step):
    c, w, h = fixture()
    h = replace(
        h,
        effects=tuple(
            e
            for e in h.effects
            if not (
                e["boundary"]["kind"] == "in_flight_guard"
                and math.isclose(e["timestamp"], step * c.step_dt)
            )
        ),
    )
    assert evaluate_pour_pose_proxy(c, w, h)["status"] == "unavailable"


@pytest.mark.parametrize("damage", ["earlier", "pending", "foreign_phase"])
def test_first_dispatch_requires_same_frame_successful_native_acquisition(damage):
    c, w, h = fixture()
    gate = h.effects[0]
    if damage == "earlier":
        gate["timestamp"] -= c.step_dt
        gate["evidence"]["grip"]["timestamp"] = gate["timestamp"]
        h.native_grips[0]["timestamp"] = gate["timestamp"]
    elif damage == "pending":
        gate["decision"]["success_mask"] = [False]
    else:
        gate["boundary"]["segment_name"] = "entry_hold"
    assert evaluate_pour_pose_proxy(c, w, h)["status"] == "unavailable"


def test_explicit_guard_failure_rejects_even_with_valid_native_contacts():
    c, w, h = fixture()
    h.effects[2]["decision"]["failure_mask"] = [True]
    assert evaluate_pour_pose_proxy(c, w, h)["status"] == "rejected"


def test_confirmed_loss_remains_rejected_when_it_stops_later_evidence_collection():
    c, w, h = fixture()
    h.effects[2]["decision"]["failure_mask"] = [True]
    h = replace(h, effects=h.effects[:3], native_grips=h.native_grips[:3])
    result = evaluate_pour_pose_proxy(c, w, h)
    assert result["status"] == "rejected"
    assert result["reason"] == "held_guard_contradicted"


def test_real_clear_contact_loss_is_reclassified_not_overwritten_by_cached_success():
    c, w, h = fixture()
    h.effects[2]["decision"]["failure_mask"] = [True]
    h.effects[2]["decision"]["success_mask"] = [False]
    h.effects[2]["evidence"]["grip"]["values"] = [False]
    native = h.native_grips[2]["rows"][0]
    native["value"] = False
    native["raw_contacts"] = {"pairs": [], "normals": [], "impulses": []}
    result = evaluate_pour_pose_proxy(c, w, h)
    assert result["status"] == "rejected"
    assert result["reason"] == "held_guard_contradicted"


@pytest.mark.parametrize(
    "damage",
    [
        "semantic_id",
        "skill_id",
        "effect_kind",
        "monitor",
        "actor_type",
        "env_type",
        "unplanned",
    ],
)
def test_foreign_call_monitor_or_coerced_actor_ids_are_not_native_proof(damage):
    c, w, h = fixture()
    if damage in {"semantic_id", "skill_id", "effect_kind"}:
        h.effects[2]["effect_spec"][damage] = "foreign"
    elif damage == "monitor":
        h.effects[2]["monitor"]["monitor_id"] = "program_boolean"
    elif damage == "actor_type":
        h.native_grips[2]["rows"][0]["raw_contacts"]["pairs"][0][0] = 10.5
    elif damage == "env_type":
        h.effects[2]["effect_spec"]["env_ids"] = [False]
    else:
        h.plan_attempts[0]["plan_success_mask"] = [False]
    assert evaluate_pour_pose_proxy(c, w, h)["status"] == "unavailable"


def test_unknown_generation_is_not_fabricated_and_inputs_are_not_mutated():
    c, w, h = fixture()
    h.plan_attempts[0].pop("attempt_generation")
    before = deepcopy((c, w, h))
    rng = torch.random.get_rng_state().clone()
    result = evaluate_pour_pose_proxy(c, w, h)
    assert result["accepted"] is True
    assert result["measurements"]["coverage"]["attempt_generation"] is None
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(w.source_poses, before[1].source_poses)
    assert h.effects == before[2].effects


@pytest.mark.parametrize(
    "field",
    ["tilt_tolerance_rad", "restore_tolerance_rad", "position_tolerance_m", "step_dt"],
)
@pytest.mark.parametrize("value", [0, -1, True, math.nan])
def test_invalid_explicit_tolerance_or_dt_is_unavailable(field, value):
    c, w, h = fixture()
    assert (
        evaluate_pour_pose_proxy(replace(c, **{field: value}), w, h)["status"]
        == "unavailable"
    )


def test_no_program_boolean_or_sampling_callback_is_accepted_as_evidence():
    c, w, h = fixture()
    with pytest.raises(TypeError):
        evaluate_pour_pose_proxy(c, w, h, program_success=True)


@pytest.mark.parametrize("peak_angle", [0.0, 0.3, 0.4])
def test_stationary_trace_cannot_pass_overlapping_entry_peak_tolerance_balls(
    peak_angle,
):
    c, w, h = fixture()
    peak = c.entry_pose.clone()
    peak[:3, :3] = rotation(peak_angle)
    w.source_poses[:] = c.entry_pose
    result = evaluate_pour_pose_proxy(replace(c, peak_pose=peak), w, h)
    assert result["status"] == "unavailable"
    assert result["accepted"] is None
    assert "tolerance balls overlap" in result["reason"]


def test_retained_native_post_ack_protocol_full_window_golden():
    path = Path(__file__).parent / "fixtures/pour_native_window.json.gz"
    with gzip.open(path, "rt") as stream:
        record = json.load(stream)
    contract_values = record["contract"]
    for key in ("entry_pose", "peak_pose"):
        contract_values[key] = torch.tensor(contract_values[key], dtype=torch.float64)
    for key in ("pivot_object", "receiver_anchor_local"):
        contract_values[key] = tuple(contract_values[key])
    contract = PourProxyContract(**contract_values)
    window_values = record["window"]
    window_values["steps"] = tuple(window_values["steps"])
    for key in ("source_poses", "receiver_poses"):
        window_values[key] = torch.tensor(window_values[key], dtype=torch.float64)
    window = NativePourWindow(**window_values)
    hold_values = record["hold"]
    for key in ("plan_attempts", "effects", "native_grips", "pad_actor_ids"):
        hold_values[key] = tuple(hold_values[key])
    hold = PourHoldEvidence(**hold_values)
    assert record["provenance"]["timestamps_actors_poses_effects_modified"] is False
    assert record["provenance"]["new_World_IK_native_calls"] == 0
    assert len(record["provenance"]["raw_sha256"]["execution_report.json"]) == 64
    assert window.steps == tuple(range(299, 443))
    guards = [e for e in hold.effects if e["boundary"]["kind"] == "in_flight_guard"]
    assert len(guards) == 143
    assert guards[0]["timestamp"] == 12.0
    assert guards[-1]["timestamp"] == 17.68
    result = evaluate_pour_pose_proxy(contract, window, hold)
    assert result["status"] == "satisfied" and result["accepted"] is True
    assert result["fluid_status"] == "unmodeled"
    coverage = result["measurements"]["coverage"]
    assert coverage["acquisition_step"] == 299
    assert coverage["guard_start_step"] == 300
    assert coverage["guard_end_step"] == 442
    assert coverage["guard_samples"] == 143
    assert coverage["native_actor_ids"] == (44, 39, 40)
    without_end = replace(
        hold,
        effects=tuple(
            e
            for e in hold.effects
            if not (
                e["boundary"]["kind"] == "in_flight_guard" and e["timestamp"] == 17.68
            )
        ),
    )
    assert (
        evaluate_pour_pose_proxy(contract, window, without_end)["status"]
        == "unavailable"
    )
