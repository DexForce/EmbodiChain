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

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.coordinated_evidence import (
    capture_coordinated_holds,
    coordinated_record_matches,
)
from embodichain.gen_sim.task_engine._task_program.grip_evidence import (
    NativeGripObserver,
)


def pose(count=1, *, x=0.0, z=0.0):
    result = torch.eye(4, dtype=torch.float64).repeat(count, 1, 1)
    result[:, 0, 3], result[:, 2, 3] = x, z
    return result


@pytest.fixture(params=[1, 2])
def holding(request):
    count = request.param
    call_spec = {
        "kind": "registered",
        "call_id": "simulation.coordinated_hold",
        "arguments": {
            "object": "tray",
            "target": "target",
            "world_displacement": [0.0, 0.0, 0.14],
        },
        "resources": {"left": "left", "right": "right"},
    }
    node = {
        "id": "hold",
        "task_instance_id": "e5",
        "task_type": "E5",
        "call": call_spec,
    }
    group = {"id": "e5", "task_type": "E5", "node_ids": ["hold"]}
    post = {"kind": "wait_stable", "entity": "tray", "preset": "stable"}
    program = {
        "program_id": "program",
        "program": {
            "kind": "sequence",
            "items": [
                {
                    "kind": "segment",
                    "name": "hold",
                    "steps": {"kind": "invoke", "call": call_spec},
                    "post": [post],
                }
            ],
        },
    }
    invocation = "program/program:segment:$.program.items[0]:0"
    expectations, clauses, evidence = [], [], {}
    for role in ("left", "right"):
        expectations.append(
            {
                "expectation_id": role,
                "object_id": "tray",
                "slot_id": role,
                "resource_id": role,
                "task_state_key": role,
                "relation": "attached",
            }
        )
        for channel in ("pose", "constraint"):
            cid = role + "." + channel
            clause = {
                "clause_id": cid,
                "expectation_id": role,
                "source": {
                    "provider_id": "builtin.control_part",
                    "revision": "1",
                    "address": {
                        "control_part": role
                        + ("_arm" if channel == "pose" else "_eef"),
                        "channel": (
                            "pose_relation" if channel == "pose" else "constraint"
                        ),
                    },
                },
            }
            sample = {
                "env_ids": list(range(count)),
                "timestamp": 1.2,
                "observation_revision": 8,
                "valid_mask": [True] * count,
                "acquisition_errors": [None] * count,
            }
            if channel == "pose":
                clause["expectation"] = "matched"
                sample["object_to_endpoint"] = pose(
                    count, x=-0.1 if role == "left" else 0.1
                ).tolist()
            else:
                clause.update(evidence_kind="constraint", expected=True)
                sample["values"] = [True] * count
            clauses.append(clause)
            evidence[cid] = sample
    terminal = {
        "timestamp": 1.2,
        "observation_revision": 8,
        "effect_spec": {
            "semantic_id": "simulation.coordinated_hold",
            "effect_kind": "attach",
            "skill_id": "coordinated_pickment",
            "invocation_id": invocation,
            "invocation_revision": 0,
            "env_ids": list(range(count)),
            "state_expectations": expectations,
            "clauses": clauses,
        },
        "monitor": {
            "monitor_id": "builtin.composite_effect",
            "revision": "1",
            "resolved_params": {
                "attached_translation_threshold": 0.06,
                "attached_rotation_threshold": 3.0,
            },
        },
        "evidence": evidence,
        "decision": {
            "success_mask": [True] * count,
            "failure_mask": [False] * count,
            "expectations": [
                {
                    "expectation_id": role,
                    "satisfied_mask": [True] * count,
                    "contradicted_mask": [False] * count,
                }
                for role in ("left", "right")
            ],
        },
        "boundary": {"kind": "terminal"},
    }
    call = {
        "semantic_id": "simulation.coordinated_hold",
        "skill_id": "coordinated_pickment",
        "status": "completed",
        "command_count": 30,
        "invocation_id": invocation,
        "invocation_revision": 0,
        "active_plan_attempt_generation": 0,
        "plan_attempts": [{"attempt_generation": 0, "planned_at": 0.0}],
        "effects": [terminal],
        "events": [
            {
                "kind": "action_completed",
                "timestamp": 1.2,
                "invocation_id": invocation,
                "invocation_revision": 0,
            }
        ],
        "resolved_core_policy": {
            "endpoints": [
                {
                    "slot_id": role,
                    "endpoint_id": endpoint,
                    "resource_id": role,
                    "task_state_key": role,
                    "target_id": role + ("_arm" if endpoint == "motion" else "_eef"),
                }
                for role in ("left", "right")
                for endpoint in ("motion", "grasp")
            ]
        },
    }
    policy = {
        "source_path": ["program", "items", 0, "post", 0],
        "result_mask": [True] * count,
        "result": {
            "kind": "hold",
            "preset": "stable",
            "required_seconds": 3.0,
            "elapsed_seconds": 3.0,
            "stable_seconds": [3.0] * count,
            "accepted_mask": [True] * count,
            "failed_mask": [False] * count,
            "measurements": {
                "position_error": [0.0] * count,
                "attachment_translation": [[0.0] * count, [0.0] * count],
            },
        },
    }
    segment = {
        "name": "hold",
        "success": True,
        "successes": [True] * count,
        "end_steps": [105] * count,
        "metadata": {
            "task_program_id": "program",
            "program_segment_id": "program:segment:$.program.items[0]",
            "program_segment_source_path": ["program", "items", 0],
            "validation": {"accepted_mask": [True] * count},
            "post_policies": [policy],
            "runtime": {
                "env_ids": list(range(count)),
                "calls": [call],
                "task_state": {
                    "batch_size": count,
                    "held_objects": [
                        {
                            "object_id": "tray",
                            "resource_id": role,
                            "active_mask": [True] * count,
                        }
                        for role in ("left", "right")
                    ],
                    "coordinated_held_objects": [],
                },
            },
        },
    }
    robot = SimpleNamespace(
        get_qpos=Mock(return_value=torch.zeros(count, 2)),
        get_joint_ids=lambda name: [0] if name == "left_arm" else [1],
        compute_fk=Mock(
            side_effect=lambda **kw: pose(
                count, x=-0.1 if kw["name"] == "left_arm" else 0.1, z=0.8
            )
        ),
    )
    sensor = SimpleNamespace(
        dropped_contacts=0,
        update=Mock(),
        data={
            "is_valid": torch.ones(count, 4, dtype=torch.bool),
            "user_ids": torch.tensor([[[1, 2], [1, 3], [1, 4], [1, 5]]]).repeat(
                count, 1, 1
            ),
            "normal": torch.tensor(
                [[[1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]]]
            ).repeat(count, 1, 1),
            "impulse": torch.ones(count, 4),
        },
    )
    sensor.get_data = lambda: sensor.data
    NativeGripObserver(
        sensor,
        {"tray": torch.tensor([[1]]).repeat(count, 1)},
        {
            "left_eef": torch.tensor([[2, 3]]).repeat(count, 1),
            "right_eef": torch.tensor([[4, 5]]).repeat(count, 1),
        },
    )
    object_poses = torch.zeros(count, 106, 7)
    object_poses[:, :, 2], object_poses[:, :, 6] = 0.8, 1.0
    native = SimpleNamespace(get_local_pose=Mock(return_value=pose(count, z=0.8)))
    env = SimpleNamespace(
        step_dt=0.04,
        robot=robot,
        _traj_steps=torch.tensor([105] * count),
        _traj_buffer={
            "states": {
                "rigid_objects": {"native_tray": {"pose": object_poses}},
                "robot": {"qpos": torch.zeros(count, 106, 2)},
            }
        },
        sim=SimpleNamespace(
            get_sensor=lambda uid: sensor, get_rigid_object=lambda uid: native
        ),
    )
    deployment = SimpleNamespace(
        integration=SimpleNamespace(
            registration=SimpleNamespace(
                scene_binding=SimpleNamespace(
                    rigid_objects=[
                        SimpleNamespace(entity_id="tray", simulation_uid="native_tray")
                    ]
                )
            )
        )
    )
    return SimpleNamespace(
        env=env,
        deployment=deployment,
        graph={"nodes": [node], "task_groups": [group]},
        runtime={"segments": [segment], "lengths": [105] * count},
        program=program,
        constraints={
            "presets": {
                "stable": {
                    "kind": "hold",
                    "entity": "tray",
                    "target_position": [0.0, 0.0, 0.8],
                    "motion_parts": ["left_arm", "right_arm"],
                    "position_tolerance": 0.05,
                }
            }
        },
        group=group,
        node=node,
        segment=segment,
        call=call,
        terminal=terminal,
        policy=policy,
        sensor=sensor,
        count=count,
        native=native,
    )


def capture(fixture, final=True):
    return capture_coordinated_holds(
        fixture.env,
        fixture.deployment,
        fixture.graph,
        fixture.runtime,
        fixture.program,
        fixture.constraints,
        fixture.count,
        final=final,
    )["e5"]


def matches(record, fixture, final=True):
    return coordinated_record_matches(
        record,
        fixture.group,
        fixture.graph,
        fixture.runtime,
        fixture.program,
        fixture.constraints,
        fixture.count,
        final=final,
    )


def test_complete_dual_hold_uses_single_body_and_qpos_snapshot(holding):
    record = capture(holding)
    assert record.get("accepted") == [True] * holding.count
    assert matches(record, holding)
    holding.native.get_local_pose.assert_called_once_with(to_matrix=True)
    holding.env.robot.get_qpos.assert_called_once_with()
    assert holding.sensor.update.call_count == 2
    assert record["measurements"]["fresh_grips"]["left"]["query"]["timestamp"] == 4.2
    assert (
        record["measurements"]["fresh_grips"]["right"]["query"]["observation_revision"]
        == 9
    )


@pytest.mark.parametrize("role", ["left", "right"])
def test_either_hand_contact_loss_cannot_be_certified(holding, role):
    indices = (0, 1) if role == "left" else (2, 3)
    holding.sensor.data["impulse"][:, indices] = 0
    record = capture(holding)
    assert record.get("accepted") != [True] * holding.count
    if "accepted" in record:
        assert record["accepted"] == [False] * holding.count
        forged = deepcopy(record)
        forged["accepted"] = [True] * holding.count
        assert not matches(forged, holding)


@pytest.mark.parametrize(
    "case",
    [
        "uid",
        "resource",
        "revision",
        "env",
        "duration",
        "occurrence",
        "shared_resource",
        "holder",
        "nonfinite",
        "clock",
        "missing_seed",
        "pose",
    ],
)
def test_wrong_or_missing_evidence_rejects_original_hold(holding, case):
    if case == "uid":
        holding.terminal["effect_spec"]["state_expectations"][0]["object_id"] = "other"
    elif case == "resource":
        holding.terminal["effect_spec"]["state_expectations"][0][
            "resource_id"
        ] = "right"
    elif case == "revision":
        holding.terminal["effect_spec"]["invocation_revision"] = True
    elif case == "env":
        holding.terminal["effect_spec"]["env_ids"] = [True] * holding.count
    elif case == "duration":
        holding.policy["result"]["stable_seconds"][0] = 2.99
    elif case == "occurrence":
        holding.call["invocation_id"] += "/old"
    elif case == "shared_resource":
        holding.node["call"]["resources"]["left"] = "right"
    elif case == "holder":
        holding.segment["metadata"]["runtime"]["task_state"]["held_objects"].pop()
    elif case == "nonfinite":
        holding.sensor.data["normal"][0, 0, 0] = float("nan")
    elif case == "clock":
        holding.segment["end_steps"][0] -= 1
    elif case == "missing_seed":
        holding.env._traj_buffer["states"]["robot"]["qpos"] = torch.zeros(
            holding.count, 105, 2
        )
    elif case == "pose":
        holding.native.get_local_pose.return_value[:, 0, 3] = 0.1
    assert "unavailable" in capture(holding)


def test_later_execution_cannot_supply_current_final_contact_for_old_hold(holding):
    holding.runtime["segments"].append({"name": "later"})
    holding.runtime["lengths"] = [106] * holding.count
    assert "unavailable" in capture(holding)


@pytest.mark.parametrize(
    "field",
    ["accepted", "resource", "revision", "time", "contact", "target", "final", "pads"],
)
def test_record_match_recomputes_raw_evidence_instead_of_accepting_tags(holding, field):
    record = capture(holding)
    assert matches(record, holding)
    if field == "accepted":
        record["accepted"] = [1] * holding.count
    elif field == "resource":
        record["measurements"]["resources"]["left"] = "right"
    elif field == "revision":
        record["measurements"]["fresh_grips"]["left"]["query"][
            "observation_revision"
        ] -= 1
    elif field == "time":
        record["measurements"]["fresh_grips"]["right"]["query"]["timestamp"] -= 0.04
    elif field == "contact":
        record["measurements"]["fresh_grips"]["left"]["query"]["rows"][0][
            "raw_contacts"
        ]["impulses"] = [0.0] * 4
    elif field == "target":
        record["measurements"]["world_target"][0] += 0.1
    elif field == "final":
        record["measurements"]["final"] = False
    elif field == "pads":
        record["measurements"]["fresh_grips"]["right"]["pad_actor_ids"] = deepcopy(
            record["measurements"]["fresh_grips"]["left"]["pad_actor_ids"]
        )
    assert not matches(record, holding)


def test_milestone_is_separate_from_final_and_requires_own_live_boundary(holding):
    milestone = capture(holding, final=False)
    assert milestone["accepted"] == [True] * holding.count
    assert matches(milestone, holding, final=False)
    assert not matches(milestone, holding, final=True)
    holding.env._traj_steps += 1
    assert "unavailable" in capture(holding, final=False)


@pytest.mark.parametrize("role", ["left", "right"])
@pytest.mark.parametrize("field", ["object_actor_ids", "pad_actor_ids"])
def test_short_actor_batch_is_rejected_without_index_error(holding, role, field):
    record = capture(holding)
    record["measurements"]["fresh_grips"][role][field].pop()
    assert not matches(record, holding)


def test_original_nondefault_hold_duration_is_not_replaced(holding):
    holding.constraints["presets"]["stable"]["duration"] = 1.6
    holding.policy["result"].update(
        required_seconds=1.6,
        elapsed_seconds=1.6,
        stable_seconds=[1.6] * holding.count,
    )
    holding.segment["end_steps"] = [70] * holding.count
    holding.runtime["lengths"] = [70] * holding.count
    holding.env._traj_steps.fill_(70)
    assert capture(holding)["accepted"] == [True] * holding.count


def test_capture_does_not_mutate_receipts_or_random_stream(holding):
    previous = deepcopy(holding.runtime)
    random_state = torch.random.get_rng_state().clone()
    assert capture(holding)["accepted"] == [True] * holding.count
    assert holding.runtime == previous
    assert torch.equal(torch.random.get_rng_state(), random_state)


def test_other_family_and_transport_are_not_claimed(holding):
    holding.group["task_type"] = "E4"
    assert (
        capture_coordinated_holds(
            holding.env,
            holding.deployment,
            holding.graph,
            holding.runtime,
            holding.program,
            holding.constraints,
            holding.count,
        )
        == {}
    )
    holding.group["task_type"] = "E5"
    holding.node["call"]["call_id"] = "simulation.coordinated_transport"
    assert "unavailable" in capture(holding)
