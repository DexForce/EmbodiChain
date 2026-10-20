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

from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import place_branch as branch
from embodichain.gen_sim.task_engine._task_program import place_transport as transport
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.plans import TimedTrajectory, TrajectorySegment
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceOptions


def _fixture(monkeypatch):
    xyz = torch.tensor(
        [
            [
                [0.1, 0.15, 1.0],
                [0.2, 0.3, 1.0],
                [0.2, 0.3, 0.95],
                [0.2, 0.3, 0.9],
                [0.2, 0.3, 0.95],
                [0.2, 0.3, 1.0],
            ]
        ]
    )
    poses = torch.eye(4).repeat(1, 6, 1, 1)
    poses[..., :3, 3] = xyz
    start = torch.tensor([[0.0, 0.0, 1.0, 0.3, 0.4]])
    arm = JointPositionTarget(control_part="arm", joint_ids=(0, 1, 2))

    def fk(qpos, **_):
        result = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        result[..., :3, 3] = qpos
        return result

    robot = NS(
        uid="robot",
        compute_fk=fk,
        compute_batch_fk=fk,
        compute_ik=lambda pose, **_: (torch.tensor([True]), pose[:, :3, 3].clone()),
        get_qpos_limits=lambda **_: torch.tensor([[[-2.0, 2.0]] * 3]),
        get_solver=lambda _: NS(
            get_iteration_params=lambda: {"pos_eps": 0.0001, "rot_eps": 0.0001}
        ),
    )
    endpoint = NS(task_state_key="left", require_target=lambda _: arm)
    context = NS(
        robot=NS(qpos=start),
        require_control_dt=lambda: 0.04,
        get_held_object=lambda _: NS(semantics=NS(entity_id="spoon")),
    )
    request = NS(
        goal=NS(xpos=object()),
        binding=NS(endpoint=lambda *_: endpoint),
        skill_options=PlaceOptions(hand_interp_steps=12, cartesian_waypoint_count=2),
        motion_policy=NS(sample_count=140),
        recovery_policy=NS(action_timeout=30.0),
    )
    action = NS(
        robot=robot,
        motion_generator=NS(planner=NS(cfg=NS(sim_instance_id="sim"))),
        recovery_scene_vertices={"spoon": torch.zeros(1, 4, 3)},
    )
    scope = branch._Scope(
        action,
        request,
        context,
        velocity_phase_allocation=True,
        groups=[poses[:, :4], poses[:, 4:]],
    )
    monkeypatch.setattr(
        transport, "resolve_pose_goal", lambda *_args, **_kwargs: poses[:, 3]
    )
    monkeypatch.setattr(
        transport.motion, "_joint_velocity_limits", lambda *_: torch.full((1, 3), 10.0)
    )
    qposes, contact = transport.prepare_contact(scope, poses, xyz, arm)
    scope.qposes, scope.contact = qposes, contact
    scope.receipt["free_transport"] = {}
    full = start[:, None].repeat(1, 200, 1)
    full[:, 77:149, :3] = contact.positions[:, -1:]
    full[:, 149:, :3] = contact.positions[:, :1]
    original = NS(
        segments=(
            TrajectorySegment("approach", 0, 77),
            TrajectorySegment("release", 77, 149),
            TrajectorySegment("retract", 149, 200),
        ),
        joint_trajectory=TimedTrajectory.from_uniform_step(
            full, step_dt=0.04, env_ids=torch.tensor([0])
        ),
        scene_dependency_monitor_until={"support": 77, "fixed": 0},
    )
    prefix = torch.lerp(
        start[:, None, :3],
        contact.positions[:, :1],
        torch.linspace(0, 1, 41)[None, :, None],
    )
    monkeypatch.setattr(transport, "_free_prefix", Mock(return_value=prefix))
    return scope, original, arm, poses, xyz, prefix


def test_internal_midpoint_is_replaced_but_exact_contact_and_release_survive(
    monkeypatch,
):
    scope, original, arm, poses, xyz, prefix = _fixture(monkeypatch)
    contact = scope.contact
    assert contact.positions.shape == (1, 39, 3)
    assert torch.equal(contact.desired[0], poses[:, 1:4])
    assert torch.equal(contact.desired[1], poses[:, 4:])
    assert torch.equal(contact.positions[:, [0, 19, 38]], contact.qposes[:, :3])
    assert torch.allclose(contact.qposes, xyz[:, 1:])
    before = original.joint_trajectory.positions.clone()
    full, lengths, dt = transport.compose(scope, original, arm)
    assert full.shape == (1, 202, 5)
    assert lengths == {"approach": 79, "release": 72, "retract": 51}
    assert torch.equal(full[:, :41, :3], prefix)
    assert torch.equal(full[:, 40:79, :3], contact.positions)
    assert torch.equal(full[:, 79:151], before[:, 77:149])
    assert torch.equal(full[:, :, 3:], before[:, :1, 3:].expand(-1, 202, -1))
    assert torch.equal(original.joint_trajectory.positions, before)
    assert contact.monitor_until == {"support": 79, "fixed": 0}
    assert dt[0, 0] == 0 and torch.all(dt[:, 1:] == 0.04)
    assert scope.receipt["timing_unchanged"] is False
    assert scope.receipt["free_transport"]["total_seconds"] == pytest.approx(8.04)


@pytest.mark.parametrize(
    "damage", ["goal_sequence", "caller_via", "topology", "ik", "fk"]
)
def test_contact_preparation_fails_closed_without_exact_original_contract(
    monkeypatch, damage
):
    scope, _, arm, poses, xyz, _ = _fixture(monkeypatch)
    if damage == "goal_sequence":
        monkeypatch.setattr(
            transport, "resolve_pose_goal", lambda *_args, **_kwargs: poses[:, 2:4]
        )
    elif damage == "caller_via":
        poses = poses.clone()
        poses[:, 0, 0, 3] += 0.1
    elif damage == "topology":
        scope.groups[0] = poses[:, :3]
    elif damage == "ik":
        monkeypatch.setattr(
            transport.motion, "resolve_seeded_ik_path", lambda *_args, **_kwargs: None
        )
    elif damage == "fk":
        scope.action.robot.compute_batch_fk = lambda qpos, **_: torch.zeros(
            *qpos.shape[:-1], 4, 4
        )
    with pytest.raises(ValueError):
        transport.prepare_contact(scope, poses, xyz, arm)


@pytest.mark.parametrize(
    "damage",
    ["passive", "release", "timeout", "monitor", "short_retract", "speed", "grid"],
)
def test_composition_cannot_weaken_existing_phase_or_safety_contract(
    monkeypatch, damage
):
    scope, original, arm, _, _, prefix = _fixture(monkeypatch)
    scope.receipt["free_transport"] = {}
    if damage == "passive":
        original.joint_trajectory.positions[:, 1, 3] += 0.1
    elif damage == "release":
        original.joint_trajectory.positions[:, 77, 0] += 0.1
    elif damage == "timeout":
        scope.request.recovery_policy.action_timeout = 0.1
    elif damage == "monitor":
        original.scene_dependency_monitor_until["support"] = 50
    elif damage == "short_retract":
        original.segments = (
            *original.segments[:2],
            TrajectorySegment("retract", 149, 179),
        )
        original.joint_trajectory = TimedTrajectory.from_uniform_step(
            original.joint_trajectory.positions[:, :179],
            step_dt=0.04,
            env_ids=torch.tensor([0]),
        )
    elif damage == "speed":
        prefix[:, 2, 0] += 1.0
    elif damage == "grid":
        original.joint_trajectory.dt[:, 50] *= 2
    with pytest.raises(ValueError):
        transport.compose(scope, original, arm)


@pytest.mark.parametrize("failed", [False, True])
def test_native_payload_lifetime_is_closed_on_success_and_failure(monkeypatch, failed):
    scope, _, arm, _, _, prefix = _fixture(monkeypatch)
    monkeypatch.undo()
    times = torch.full((1, 41), 0.04)
    times[:, 0] = 0
    generator = NS(
        generate=Mock(
            return_value=NS(
                success=torch.tensor([not failed]),
                positions=prefix,
                dt=times,
            )
        ),
        planner=NS(close=Mock()),
    )
    factory = Mock(return_value=generator)
    monkeypatch.setattr(transport, "_PayloadMotionGenerator", factory)
    monkeypatch.setattr(
        transport,
        "SimulationManager",
        NS(get_instance=lambda _: NS(get_rigid_object=lambda _: object())),
    )
    monkeypatch.setattr(transport, "world_sources", lambda *_: {"table": object()})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_: nullcontext())
    scope.action.robot.uid = "robot"
    if failed:
        with pytest.raises(ValueError, match="free transport failed"):
            transport._free_prefix(scope, arm)
    else:
        result = transport._free_prefix(scope, arm)
        assert torch.equal(result[:, 0], scope.context.robot.qpos[:, :3])
        assert torch.equal(result[:, -1], scope.contact.qposes[:, 0])
    generator.planner.close.assert_called_once()
    assert factory.call_args.kwargs["observation"].vertices.shape == (4, 3)
    assert set(factory.call_args.args[0].world.rigid_objects) == {"table"}
