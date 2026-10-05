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

from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import actions, motion
from embodichain.gen_sim.task_engine._task_program.grasp_filter import (
    TaskGraspPoseGenerator,
)
from embodichain.lab.sim.atomic_actions import ObjectSemantics, PickUpOptions
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics, TimedTrajectory
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy
from embodichain.lab.sim.atomic_actions.primitives.pick_up import GraspGoal, PickUp
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, TaskState


@dataclass(frozen=True)
class _Request:
    goal: GraspGoal
    skill_options: PickUpOptions
    motion_policy: MotionPolicy
    binding: object


@dataclass(frozen=True)
class _Plan:
    plan_success: torch.Tensor
    expected_effects: StateDelta
    diagnostics: PlannerDiagnostics
    joint_trajectory: TimedTrajectory


@pytest.fixture
def setup_pick(monkeypatch):
    affordance = AntipodalAffordance(
        mesh_vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
        ),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
    )
    affordance.set_custom_config(actions._PICK_GRASP_RULE, ("can", "upright"))
    semantics = ObjectSemantics(affordance=affordance, geometry={}, entity_id="can")
    endpoint = SimpleNamespace(
        task_state_key="left",
        require_target=lambda _: SimpleNamespace(
            joint_ids=tuple(range(7)), control_part="left_arm", target_id="hand"
        ),
    )
    request = _Request(
        GraspGoal(semantics),
        PickUpOptions(),
        MotionPolicy(
            strategy="ik_interp", sample_count=260, dynamic_collision_mode="off"
        ),
        SimpleNamespace(endpoint=lambda *args: endpoint),
    )
    context = SimpleNamespace(
        batch_size=1,
        robot=SimpleNamespace(qpos=torch.zeros(1, 8)),
        task=TaskState(batch_size=1, device=torch.device("cpu")),
    )
    generator = Mock(spec=TaskGraspPoseGenerator)
    generator.for_pick.return_value = generator
    generator.accepts_grasp_poses = Mock(return_value=torch.ones(1, dtype=torch.bool))
    robot = SimpleNamespace(
        get_qpos_limits=lambda name: torch.tensor([[[-1.0, 1.0]]]).expand(1, 7, 2)
    )
    action = actions.GenSimPickUp()
    action._planning_services = SimpleNamespace(
        device=torch.device("cpu"),
        robot=robot,
        motion_generator=SimpleNamespace(direct_transport_planning=False),
        grasp_pose_generator=lambda _: generator,
    )
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.ones(1, 8)
    )

    def make_plan(goal, success):
        pose = torch.eye(4)[None] if goal.grasp_xpos is None else goal.grasp_xpos
        held = HeldObjectState(semantics, object_to_eef=pose, grasp_xpos=pose)
        trajectory = TimedTrajectory.from_uniform_step(
            torch.zeros(1, 3, 8), env_ids=torch.tensor([0]), step_dt=0.04
        )
        return _Plan(
            torch.tensor([success]),
            StateDelta(held_object_updates={"left": held}),
            PlannerDiagnostics(
                backend="probe",
                metadata={"affordance_sample": {"grasp": {"candidate_ids": [0]}}},
            ),
            trajectory,
        )

    def install_planner(planner):
        def wrapped(self, request, context):
            actions._PICK_SELECTION.get().constrained = True
            return planner(self, request, context)

        monkeypatch.setattr(PickUp, "_plan", wrapped)

    return SimpleNamespace(
        action=action,
        request=request,
        context=context,
        generator=generator,
        make_plan=make_plan,
        install_planner=install_planner,
    )


def test_failed_constrained_pick_retries_one_roll_and_preserves_rng(
    monkeypatch, setup_pick
):
    probe = setup_pick
    calls = []
    rng_after_first = None

    def plan(self, request, context):
        nonlocal rng_after_first
        calls.append(request)
        torch.rand(3)
        result = probe.make_plan(request.goal, len(calls) == 2)
        if len(calls) == 1:
            rng_after_first = torch.random.get_rng_state().clone()
        return result

    probe.install_planner(plan)
    result = probe.action._plan(probe.request, probe.context)
    assert len(calls) == 2
    assert calls[1].skill_options is probe.request.skill_options
    assert calls[1].motion_policy is probe.request.motion_policy
    expected = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))[None]
    torch.testing.assert_close(calls[1].goal.grasp_xpos, expected)
    held = result.expected_effects.held_object_updates["left"]
    torch.testing.assert_close(held.grasp_xpos, expected)
    torch.testing.assert_close(held.object_to_eef, expected)
    assert result.plan_success.tolist() == [True]
    assert result.diagnostics.metadata["affordance_sample"]["grasp"][
        "candidate_ids"
    ] == [0]
    torch.testing.assert_close(
        torch.tensor(
            result.diagnostics.metadata["affordance_sample"]["grasp"]["selected_poses"]
        ),
        expected,
    )
    assert result.diagnostics.metadata["gen_sim_pick_roll_recovery"]["accepted"] is True
    assert torch.equal(torch.random.get_rng_state(), rng_after_first)
    assert probe.context.task.held_objects == {}
    assert actions._PICK_SELECTION.get() is None


def test_successful_constrained_pick_is_returned_unchanged(monkeypatch, setup_pick):
    probe = setup_pick
    first = probe.make_plan(probe.request.goal, True)
    planner = Mock(return_value=first)
    probe.install_planner(planner)
    assert probe.action._plan(probe.request, probe.context) is first
    planner.assert_called_once()
    probe.generator.for_pick.assert_not_called()


def test_only_call_local_constrained_sampling_enables_recovery(monkeypatch, setup_pick):
    probe = setup_pick
    affordance = probe.request.goal.semantics.affordance
    poses = torch.eye(4)[None, None]
    candidates = SimpleNamespace(
        poses=poses, costs=torch.zeros(1, 1), valid=torch.ones(1, 1, dtype=torch.bool)
    )
    monkeypatch.setattr(
        affordance, "get_grasp_candidates", Mock(return_value=candidates)
    )
    monkeypatch.setattr(affordance, "sample_candidates", Mock(return_value=object()))
    monkeypatch.setattr(
        probe.action,
        "_select_feasible_grasp_variants",
        Mock(return_value=(poses, candidates.valid)),
    )
    selection = actions._PickSelection(request=probe.request)
    token = actions._PICK_SELECTION.set(selection)
    try:
        probe.action._resolve_grasp_pose(
            probe.request.goal.semantics,
            torch.eye(4)[None],
            torch.zeros(1, 7),
            object(),
            "hand",
            probe.request.skill_options,
            torch.tensor([0.0, 0.0, -1.0]),
            SimpleNamespace(affordance_sampling=None, env_ids=torch.tensor([0])),
            sample_key="same_candidate",
        )
        assert selection.constrained is True
    finally:
        actions._PICK_SELECTION.reset(token)


def test_roll_planning_exception_restores_rng_and_call_scope(monkeypatch, setup_pick):
    probe = setup_pick
    calls = 0
    rng_after_first = None

    def plan(self, request, context):
        nonlocal calls, rng_after_first
        calls += 1
        torch.rand(3)
        if calls == 2:
            raise RuntimeError("alternate planner failure")
        rng_after_first = torch.random.get_rng_state().clone()
        return probe.make_plan(request.goal, False)

    probe.install_planner(plan)
    with pytest.raises(RuntimeError, match="alternate planner failure"):
        probe.action._plan(probe.request, probe.context)
    assert torch.equal(torch.random.get_rng_state(), rng_after_first)
    assert actions._PICK_SELECTION.get() is None


@pytest.mark.parametrize(
    "exemption",
    [
        "unmarked",
        "explicit",
        "batch",
        "motion_gen",
        "collision",
        "downstream",
        "region",
    ],
)
def test_roll_recovery_does_not_change_other_pick_contracts(
    monkeypatch, setup_pick, exemption
):
    probe = setup_pick
    request = probe.request
    if exemption == "unmarked":
        request.goal.semantics.affordance.custom_config.clear()
    elif exemption == "explicit":
        request = replace(request, goal=replace(request.goal, grasp_xpos=torch.eye(4)))
    elif exemption == "batch":
        probe.context.batch_size = 2
    elif exemption == "motion_gen":
        request = replace(request, motion_policy=MotionPolicy(strategy="motion_gen"))
    elif exemption == "collision":
        request = replace(
            request,
            motion_policy=replace(
                request.motion_policy, dynamic_collision_mode="required"
            ),
        )
    elif exemption == "downstream":
        request = replace(
            request,
            skill_options=replace(
                request.skill_options, downstream_object_target_poses=(torch.eye(4),)
            ),
        )
    else:
        request = replace(
            request,
            skill_options=replace(request.skill_options, pick_object_part="top"),
        )
    first = probe.make_plan(request.goal, False)
    planner = Mock(return_value=first)
    probe.install_planner(planner)
    assert probe.action._plan(request, probe.context) is first
    planner.assert_called_once()


@pytest.mark.parametrize(
    "rejection", ["geometry", "planning", "velocity", "joint_limits"]
)
def test_alternate_roll_fails_closed_after_final_trajectory_checks(
    monkeypatch, setup_pick, rejection
):
    probe = setup_pick
    first = probe.make_plan(probe.request.goal, False)
    calls = []
    if rejection == "geometry":
        probe.generator.accepts_grasp_poses.return_value = torch.zeros(
            1, dtype=torch.bool
        )

    def plan(self, request, context):
        calls.append(request)
        if len(calls) == 1:
            return first
        second = probe.make_plan(request.goal, rejection != "planning")
        positions = second.joint_trajectory.positions.clone()
        if rejection == "velocity":
            positions[:, 1, -1] = 0.2
        elif rejection == "joint_limits":
            positions[:, :, 0] = 1.1
        return replace(
            second,
            joint_trajectory=replace(second.joint_trajectory, positions=positions),
        )

    probe.install_planner(plan)
    result = probe.action._plan(probe.request, probe.context)
    assert len(calls) == (1 if rejection == "geometry" else 2)
    assert result.plan_success.tolist() == [False]
    torch.testing.assert_close(
        result.joint_trajectory.positions, first.joint_trajectory.positions
    )
    assert (
        result.diagnostics.metadata["gen_sim_pick_roll_recovery"]["accepted"] is False
    )


def test_alternate_roll_is_applied_in_the_calibrated_grasp_frame(
    monkeypatch, setup_pick
):
    probe = setup_pick
    calibration = torch.eye(4)
    calibration[0, 3] = 0.02
    request = replace(
        probe.request,
        skill_options=replace(
            probe.request.skill_options, grasp_frame_to_eef=calibration
        ),
    )
    calls = []

    def plan(self, request, context):
        calls.append(request)
        return probe.make_plan(request.goal, len(calls) == 2)

    probe.install_planner(plan)
    result = probe.action._plan(request, probe.context)
    expected = torch.eye(4)[None]
    expected[:, :3, :2] *= -1
    expected[:, 0, 3] = -0.04
    torch.testing.assert_close(calls[1].goal.grasp_xpos, expected)
    assert result.plan_success.tolist() == [True]
