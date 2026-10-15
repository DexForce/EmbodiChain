# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
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

from embodichain.gen_sim.task_engine._task_program import held_evidence as held
from embodichain.gen_sim.task_engine._task_program.grip_evidence import (
    NativeGripObserver,
)


def pose(x=0.0):
    result = torch.eye(4)[None]
    result[:, 0, 3] = x
    return result


@pytest.fixture
def dual_hold():
    endpoints, expectations, clauses, evidence = [], [], [], {}
    for resource, offset in (("left", 0.1), ("right", 0.2)):
        expectations.append(
            {
                "expectation_id": resource,
                "object_id": "body",
                "slot_id": resource,
                "resource_id": resource,
                "task_state_key": resource,
                "relation": "attached",
            }
        )
        for channel in ("pose", "constraint"):
            cid = resource + "." + channel
            is_pose = channel == "pose"
            part = resource + ("_arm" if is_pose else "_eef")
            endpoints.append(
                {
                    "slot_id": resource,
                    "resource_id": resource,
                    "task_state_key": resource,
                    "endpoint_id": "motion" if is_pose else "grasp",
                    "target_id": part,
                }
            )
            clause = {
                "clause_id": cid,
                "expectation_id": resource,
                "source": {
                    "provider_id": "builtin.control_part",
                    "revision": "1",
                    "address": {
                        "control_part": part,
                        "channel": "pose_relation" if is_pose else "constraint",
                    },
                },
            }
            sample = {
                "env_ids": [0],
                "timestamp": 0.2,
                "observation_revision": 8,
                "valid_mask": [True],
                "acquisition_errors": [None],
            }
            if is_pose:
                clause["expectation"] = "matched"
                sample["object_to_endpoint"] = pose(offset).tolist()
            else:
                clause.update(evidence_kind="constraint", expected=True)
                sample["values"] = [True]
            clauses.append(clause)
            evidence[cid] = sample
    terminal = {
        "timestamp": 0.2,
        "observation_revision": 8,
        "effect_spec": {
            "invocation_id": "program/invocation",
            "invocation_revision": 0,
            "env_ids": [0],
            "semantic_id": "simulation.coordinated_hold",
            "effect_kind": "attach",
            "state_expectations": expectations,
            "clauses": clauses,
        },
        "monitor": {"monitor_id": "builtin.composite_effect", "revision": "1"},
        "evidence": evidence,
        "decision": {
            "success_mask": [True],
            "failure_mask": [False],
            "expectations": [
                {
                    "expectation_id": resource,
                    "satisfied_mask": [True],
                    "contradicted_mask": [False],
                }
                for resource in ("left", "right")
            ],
        },
    }
    sensor = SimpleNamespace(
        dropped_contacts=0,
        update=lambda: None,
        data={
            "is_valid": torch.tensor([[True] * 4]),
            "user_ids": torch.tensor([[[1, 2], [1, 3], [1, 4], [1, 5]]]),
            "normal": torch.tensor(
                [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]
            ),
            "impulse": torch.tensor([[0.1] * 4]),
        },
    )
    sensor.get_data = lambda: sensor.data
    NativeGripObserver(
        sensor,
        {"body": torch.tensor([[1]])},
        {"left_eef": torch.tensor([[2, 3]]), "right_eef": torch.tensor([[4, 5]])},
    )
    reads = {"qpos": 0, "body": 0, "fk": []}
    qpos = torch.tensor([[0.1, 0.2]])

    def get_qpos():
        reads["qpos"] += 1
        return qpos

    def get_pose(*, to_matrix):
        reads["body"] += 1
        return pose()

    def fk(*, qpos, name, to_matrix):
        reads["fk"].append((name, qpos.clone()))
        return pose(qpos[0, 0])

    robot = SimpleNamespace(
        get_qpos=get_qpos,
        get_joint_ids=lambda name: [0] if name == "left_arm" else [1],
        compute_fk=fk,
    )
    object_rows = torch.zeros(1, 16, 7)
    object_rows[..., 6] = 1
    env = SimpleNamespace(
        robot=robot,
        _traj_steps=torch.tensor([10]),
        _traj_buffer={
            "states": {
                "rigid_objects": {"native_body": {"pose": object_rows}},
                "robot": {"qpos": qpos[:, None].expand(1, 16, 2)},
            }
        },
        sim=SimpleNamespace(
            get_sensor=lambda uid: sensor,
            get_rigid_object=lambda uid: SimpleNamespace(get_local_pose=get_pose),
        ),
    )
    deployment = SimpleNamespace(
        integration=SimpleNamespace(
            registration=SimpleNamespace(
                scene_binding=SimpleNamespace(
                    rigid_objects=[
                        SimpleNamespace(entity_id="body", simulation_uid="native_body")
                    ]
                )
            )
        )
    )
    call = {
        "invocation_id": "program/invocation",
        "invocation_revision": 0,
        "resolved_core_policy": {"endpoints": endpoints},
    }
    return SimpleNamespace(
        env=env,
        deployment=deployment,
        terminal=terminal,
        call=call,
        sensor=sensor,
        reads=reads,
    )


def test_one_receipt_verifies_two_explicit_held_resources(dual_hold):
    held.receipt(dual_hold.terminal, dual_hold.call, 1)
    for resource, offset in (("left", 0.1), ("right", 0.2)):
        relative, part = held.slot(dual_hold.terminal, "body", resource, "attached", 1)
        assert part == resource + "_arm"
        torch.testing.assert_close(relative, pose(offset).double())


@pytest.mark.parametrize("resource", ["left", "right"])
def test_fresh_grip_has_explicit_participant_without_e4_shape(dual_hold, resource):
    sample = held.fresh_grip(
        dual_hold.env,
        uid="body",
        resource=resource,
        terminal=dual_hold.terminal,
        frames=[10],
        dt=0.04,
        count=1,
    )
    assert sample["query"]["control_part"] == resource + "_eef"
    assert sample["query"]["timestamp"] == 0.4
    assert sample["query"]["observation_revision"] == 9
    assert held.grip_flags(
        uid="body",
        resource=resource,
        sample=sample,
        terminal=dual_hold.terminal,
        frames=[10],
        dt=0.04,
        count=1,
    ) == [True]


@pytest.mark.parametrize(
    "invalid",
    [
        "wrong_resource",
        "stale_clock",
        "wrong_revision",
        "one_pad",
        "nonfinite",
        "drops",
        "unknown_actor",
    ],
)
def test_fresh_grip_rechecks_raw_identity_and_contacts(dual_hold, invalid):
    sample = held.fresh_grip(
        dual_hold.env,
        uid="body",
        resource="left",
        terminal=dual_hold.terminal,
        frames=[10],
        dt=0.04,
        count=1,
    )
    query, row = sample["query"], sample["query"]["rows"][0]
    if invalid == "wrong_resource":
        query["control_part"] = "right_eef"
    elif invalid == "stale_clock":
        query["timestamp"] = 0.2
    elif invalid == "wrong_revision":
        query["observation_revision"] = 8
    elif invalid == "drops":
        query["dropped_contacts"] = 1
    elif invalid == "unknown_actor":
        sample["pad_actor_ids"] = [[98, 99]]
    elif invalid == "nonfinite":
        row["raw_contacts"]["impulses"][0] = float("nan")
    elif invalid == "one_pad":
        for field in ("pairs", "normals", "impulses"):
            row["raw_contacts"][field] = row["raw_contacts"][field][:1]
    with pytest.raises(ValueError):
        held.grip_flags(
            uid="body",
            resource="left",
            sample=sample,
            terminal=dual_hold.terminal,
            frames=[10],
            dt=0.04,
            count=1,
        )


def test_dual_native_endpoints_share_one_fresh_body_and_qpos_read(dual_hold):
    body, tcps = held.native_held_poses(
        dual_hold.env,
        dual_hold.deployment,
        "body",
        [10],
        ("left_arm", "right_arm"),
        1,
        True,
    )
    torch.testing.assert_close(body, pose().double())
    torch.testing.assert_close(tcps["left_arm"], pose(0.1).double())
    torch.testing.assert_close(tcps["right_arm"], pose(0.2).double())
    assert dual_hold.reads["body"] == dual_hold.reads["qpos"] == 1
    assert [part for part, _ in dual_hold.reads["fk"]] == [
        "left_arm",
        "right_arm",
        "left_arm",
        "right_arm",
    ]


def test_native_snapshot_rejects_missing_seeded_boundary(dual_hold):
    with pytest.raises(ValueError, match="seeded end-step"):
        held.native_held_poses(
            dual_hold.env,
            dual_hold.deployment,
            "body",
            [16],
            ("left_arm", "right_arm"),
            1,
            False,
        )


def test_native_snapshot_rejects_fresh_robot_drift(dual_hold):
    dual_hold.env.robot.get_qpos = lambda: torch.tensor([[0.1, 0.4]])
    with pytest.raises(ValueError, match="recorded boundary"):
        held.native_held_poses(
            dual_hold.env,
            dual_hold.deployment,
            "body",
            [10],
            ("left_arm", "right_arm"),
            1,
            True,
        )


def test_receipt_rejects_foreign_endpoint_channel(dual_hold):
    terminal = deepcopy(dual_hold.terminal)
    terminal["effect_spec"]["clauses"][0]["source"]["address"][
        "control_part"
    ] = "right_arm"
    with pytest.raises(ValueError, match="resolved participant"):
        held.receipt(terminal, dual_hold.call, 1)
