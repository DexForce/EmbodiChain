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
import pytest
import torch
from embodichain.gen_sim.task_engine._task_program.handover_evidence import (
    capture_handover_operations,
    handover_record_matches,
)


def pose(x=0.0):
    value = torch.eye(4)[None]
    value[:, 0, 3] = x
    return value


@pytest.fixture
def handover():
    node = {
        "id": "handover",
        "task_instance_id": "e4",
        "task_type": "E4",
        "call": {
            "kind": "hand_over",
            "object": "object",
            "resources": {"source": "left", "destination": "right"},
        },
    }
    group = {"id": "e4", "task_type": "E4", "node_ids": ["handover"]}
    program = {
        "program_id": "program",
        "program": {
            "kind": "sequence",
            "items": [
                {
                    "name": "handover",
                    "steps": {"kind": "invoke", "call": deepcopy(node["call"])},
                }
            ],
        },
    }
    invocation = "program/program:segment:$.program.items[0]:0"
    expectations, clauses, evidence = [], [], {}
    for slot, resource, relation, expected, offset in (
        ("source", "left", "detached", False, 0.2),
        ("destination", "right", "attached", True, 0.1),
    ):
        expectations.append(
            {
                "expectation_id": slot,
                "object_id": "object",
                "slot_id": slot,
                "resource_id": resource,
                "task_state_key": resource,
                "relation": relation,
            }
        )
        for channel in ("pose", "constraint"):
            cid = slot + "." + channel
            clause = {
                "clause_id": cid,
                "expectation_id": slot,
                "source": {
                    "provider_id": "builtin.control_part",
                    "revision": "1",
                    "address": {
                        "control_part": resource
                        + ("_arm" if channel == "pose" else "_eef"),
                        "channel": (
                            "pose_relation" if channel == "pose" else "constraint"
                        ),
                    },
                },
            }
            sample = {
                "env_ids": [0],
                "timestamp": 1.2,
                "observation_revision": 8,
                "valid_mask": [True],
                "acquisition_errors": [None],
            }
            if channel == "pose":
                clause["expectation"] = "separated" if slot == "source" else "matched"
                sample["object_to_endpoint"] = pose(offset).tolist()
            else:
                clause.update(evidence_kind="constraint", expected=expected)
                sample["values"] = [expected]
            clauses.append(clause)
            evidence[cid] = sample
    receipt = {
        "timestamp": 1.2,
        "observation_revision": 8,
        "effect_spec": {
            "semantic_id": "hand_over",
            "effect_kind": "transfer",
            "invocation_id": invocation,
            "invocation_revision": 0,
            "env_ids": [0],
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
            "success_mask": [True],
            "failure_mask": [False],
            "expectations": [
                {
                    "expectation_id": slot,
                    "satisfied_mask": [True],
                    "contradicted_mask": [False],
                }
                for slot in ("source", "destination")
            ],
        },
        "boundary": {"kind": "terminal"},
    }
    gate = deepcopy(receipt)
    gate["timestamp"] = 0.4
    gate["effect_spec"]["effect_kind"] = "attach"
    gate["effect_spec"]["state_expectations"] = [expectations[1]]
    gate["effect_spec"]["clauses"] = [
        clause
        for clause in gate["effect_spec"]["clauses"]
        if clause["expectation_id"] == "destination"
    ]
    gate["decision"]["expectations"] = [gate["decision"]["expectations"][1]]
    gate["evidence"] = {
        key: value
        for key, value in gate["evidence"].items()
        if key.startswith("destination.")
    }
    gate["boundary"] = {
        "kind": "phase_effect_gate",
        "gate_id": "destination_acquired",
        "segment_name": "handover_release",
    }
    for sample in gate["evidence"].values():
        sample["timestamp"] = 0.4
    events = [
        {
            "kind": kind,
            "timestamp": timestamp,
            "segment_name": segment,
            "invocation_id": invocation,
            "invocation_revision": 0,
        }
        for kind, timestamp, segment in (
            ("phase_effect_gate_satisfied", 0.4, "handover_release"),
            ("trajectory_segment_entered", 0.4, "handover_release"),
            ("trajectory_segment_entered", 0.8, "source_retreat"),
            ("trajectory_completed", 1.08, "source_retreat"),
            ("action_completed", 1.2, "source_retreat"),
        )
    ]
    call = {
        "semantic_id": "hand_over",
        "status": "completed",
        "command_count": 30,
        "invocation_id": invocation,
        "invocation_revision": 0,
        "active_plan_attempt_generation": 0,
        "resolved_core_policy": {
            "endpoints": [
                {
                    "slot_id": slot,
                    "endpoint_id": endpoint,
                    "resource_id": resource,
                    "task_state_key": resource,
                    "target_id": resource
                    + ("_arm" if endpoint == "motion" else "_eef"),
                }
                for slot, resource in (("source", "left"), ("destination", "right"))
                for endpoint in ("motion", "grasp")
            ]
        },
        "plan_attempts": [{"attempt_generation": 0, "planned_at": 0.0}],
        "effects": [gate, receipt],
        "events": events,
    }
    segment = {
        "name": "handover",
        "success": True,
        "successes": [True],
        "start_steps": [0],
        "end_steps": [30],
        "metadata": {
            "task_program_id": "program",
            "program_segment_source_path": ["program", "items", 0],
            "program_segment_id": "program:segment:$.program.items[0]",
            "runtime": {
                "env_ids": [0],
                "calls": [call],
                "task_state": {
                    "batch_size": 1,
                    "held_objects": [
                        {
                            "object_id": "object",
                            "resource_id": "right",
                            "active_mask": [True],
                        }
                    ],
                    "coordinated_held_objects": [],
                },
            },
            "post_policies": [],
            "validation": {"accepted_mask": [True]},
        },
    }
    object_poses = torch.zeros(1, 131, 7)
    object_poses[:, :, 6] = 1
    robot = SimpleNamespace(
        get_qpos=lambda: torch.zeros(1, 2),
        get_joint_ids=lambda name: [0, 1],
        compute_fk=lambda **kwargs: pose(0.1),
    )
    from embodichain.gen_sim.task_engine._task_program.grip_evidence import (
        NativeGripObserver,
    )

    sensor = SimpleNamespace(
        dropped_contacts=0,
        update=lambda: None,
        data={
            "is_valid": torch.tensor([[True, True]]),
            "user_ids": torch.tensor([[[1, 2], [1, 3]]]),
            "normal": torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
            "impulse": torch.tensor([[0.1, 0.1]]),
        },
    )
    sensor.get_data = lambda: sensor.data
    NativeGripObserver(
        sensor, {"object": torch.tensor([[1]])}, {"right_eef": torch.tensor([[2, 3]])}
    )
    env = SimpleNamespace(
        step_dt=0.04,
        robot=robot,
        _traj_steps=torch.tensor([30]),
        _traj_buffer={
            "states": {
                "rigid_objects": {"native_object": {"pose": object_poses}},
                "robot": {"qpos": torch.zeros(1, 131, 2)},
            }
        },
        sim=SimpleNamespace(
            get_sensor=lambda uid: sensor,
            get_rigid_object=lambda uid: SimpleNamespace(
                get_local_pose=lambda to_matrix: pose()
            ),
        ),
    )
    deployment = SimpleNamespace(
        integration=SimpleNamespace(
            registration=SimpleNamespace(
                scene_binding=SimpleNamespace(
                    rigid_objects=[
                        SimpleNamespace(
                            entity_id="object", simulation_uid="native_object"
                        )
                    ]
                )
            )
        )
    )
    return SimpleNamespace(
        env=env,
        deployment=deployment,
        graph={"nodes": [node], "task_groups": [group]},
        runtime={"segments": [segment], "lengths": [30]},
        program=program,
        constraints={"presets": {}},
        segment=segment,
        call=call,
        receipt=receipt,
        group=group,
        sensor=sensor,
    )


def capture(fixture, *, final=False):
    return capture_handover_operations(
        fixture.env,
        fixture.deployment,
        fixture.graph,
        fixture.runtime,
        fixture.program,
        fixture.constraints,
        1,
        final=final,
    )["e4"]


def declared_hold(fixture):
    fixture.program["program"]["items"][0]["post"] = [
        {"kind": "wait_stable", "entity": "object", "preset": "hold"}
    ]
    fixture.constraints["presets"]["hold"] = {
        "kind": "hold",
        "entity": "object",
        "motion_parts": ["right_arm"],
        "local_axis": [0, 0, 1],
        "world_axis": [0, 0, 1],
        "minimum_alignment": 0.98,
    }
    fixture.segment["end_steps"] = [105]
    fixture.runtime["lengths"] = [105]
    fixture.env._traj_steps = torch.tensor([105])
    fixture.segment["metadata"]["post_policies"] = [
        {
            "kind": "wait_stable",
            "source_path": ["program", "items", 0, "post", 0],
            "result_mask": [True],
            "result": {
                "kind": "hold",
                "preset": "hold",
                "required_seconds": 3.0,
                "elapsed_seconds": 3.0,
                "stable_seconds": [3.0],
                "accepted_mask": [True],
                "failed_mask": [False],
                "measurements": {"alignment": [1.0], "attachment_translation": [[0.0]]},
            },
        }
    ]


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("with_hold", [False, True])
def test_existing_transfer_and_optional_authored_hold(handover, final, with_hold):
    if with_hold:
        declared_hold(handover)
    records = capture(handover, final=final)
    for name in ("handover_complete", "held", *(("orientation",) if with_hold else ())):
        assert records[name]["accepted"] == [True]
        assert handover_record_matches(
            records[name],
            name,
            handover.group,
            handover.graph,
            handover.runtime,
            handover.program,
            handover.constraints,
            1,
            final=final,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "source_not_released",
        "receiver_not_attached",
        "wrong_object",
        "wrong_invocation",
        "wrong_env",
        "stale_sample",
        "missing_terminal",
        "failed_retreat",
        "nonfinite_pose",
        "foreign_hand_evidence",
        "wrong_channel",
    ],
)
def test_transfer_requires_its_exact_physical_receipt(handover, invalid):
    receipt = handover.receipt
    if invalid == "source_not_released":
        receipt["evidence"]["source.constraint"]["values"] = [True]
    elif invalid == "receiver_not_attached":
        receipt["evidence"]["destination.constraint"]["values"] = [False]
    elif invalid == "wrong_object":
        receipt["effect_spec"]["state_expectations"][1]["object_id"] = "other"
    elif invalid == "wrong_invocation":
        receipt["effect_spec"]["invocation_id"] = "other"
    elif invalid == "wrong_env":
        receipt["effect_spec"]["env_ids"] = [1]
    elif invalid == "stale_sample":
        receipt["evidence"]["destination.pose"]["timestamp"] = 1.16
    elif invalid == "missing_terminal":
        handover.call["effects"].pop()
    elif invalid == "failed_retreat":
        handover.call["status"] = "failed"
    elif invalid == "foreign_hand_evidence":
        receipt["effect_spec"]["clauses"][3]["source"]["address"][
            "control_part"
        ] = "left_eef"
    elif invalid == "wrong_channel":
        receipt["effect_spec"]["clauses"][2]["source"]["address"][
            "channel"
        ] = "constraint"
    else:
        receipt["evidence"]["destination.pose"]["object_to_endpoint"][0][0][0] = float(
            "nan"
        )
    assert "accepted" not in capture(handover)["handover_complete"]


def test_final_pose_and_suffix_do_not_certify_a_previous_hold(handover):
    handover.env.robot.compute_fk = lambda **kwargs: pose(0.3)
    assert "accepted" not in capture(handover, final=True)["held"]
    handover.env.robot.compute_fk = lambda **kwargs: pose(0.1)
    later = deepcopy(handover.segment)
    later["name"] = "later"
    handover.runtime["segments"].append(later)
    assert "accepted" not in capture(handover, final=True)["held"]
    assert capture(handover)["handover_complete"]["accepted"] == [True]


def test_matcher_rechecks_receipts_instead_of_trusting_flags(handover):
    record = capture(handover)["held"]
    record["measurements"]["boundary_frames"] = [29]
    assert not handover_record_matches(
        record,
        "held",
        handover.group,
        handover.graph,
        handover.runtime,
        handover.program,
        handover.constraints,
        1,
    )
    record = capture(handover)["held"]
    handover.receipt["evidence"]["source.constraint"]["values"] = [True]
    assert not handover_record_matches(
        record,
        "held",
        handover.group,
        handover.graph,
        handover.runtime,
        handover.program,
        handover.constraints,
        1,
    )
