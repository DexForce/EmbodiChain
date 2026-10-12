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
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import pick_path as path
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.plans import (
    PlannerDiagnostics,
    TimedTrajectory,
    TrajectorySegment,
)


def _ownership():
    action = NS(
        path_simulation=object(),
        pick_purposes={"p:0": "ordinary"},
        path_excluded_invocations=(),
    )
    options = NS(
        fixed_object_to_eef=None,
        rotate_upright=None,
        obj_upright_direction=None,
        pick_object_part="center",
    )
    goal = NS(
        object_pose=None,
        grasp_xpos=None,
        semantics=NS(affordance=AntipodalAffordance()),
    )
    request = NS(
        invocation_id="p:0",
        goal=goal,
        skill_options=options,
        motion_policy=NS(strategy="ik_interp", dynamic_collision_mode="off"),
    )
    context = NS(batch_size=1, env_ids=torch.tensor([0]))
    selection = NS(constrained=False, direction=torch.tensor([[0.0, 0.0, -1.0]]))
    return action, request, context, selection


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "unmarked",
        "pour",
        "explicit",
        "fixed",
        "upright",
        "ends",
        "constrained",
        "batch",
        "dynamic",
        "cartesian",
    ],
)
def test_only_declared_ordinary_gensim_calls_are_routed(damage) -> None:
    action, request, context, selection = _ownership()
    if damage == "unmarked":
        action.path_simulation = None
    if damage == "pour":
        action.pick_purposes["p:0"] = "pour"
    if damage == "explicit":
        request.goal.grasp_xpos = torch.eye(4)
    if damage == "fixed":
        request.skill_options.fixed_object_to_eef = torch.eye(4)
    if damage == "upright":
        request.skill_options.rotate_upright = "up"
    if damage == "ends":
        request.skill_options.pick_object_part = "positive"
    if damage == "constrained":
        selection.constrained = True
    if damage == "batch":
        context.batch_size = 2
    if damage == "dynamic":
        request.motion_policy.dynamic_collision_mode = "replan"
    if damage == "cartesian":
        action.path_excluded_invocations = ("p:0",)
    assert path.eligible(action, request, context, selection) is (damage is None)


@pytest.mark.parametrize("damage", [None, "sideways", "rotation", "nonfinite"])
def test_cartesian_qualification_checks_intermediate_samples(damage) -> None:
    desired = torch.eye(4)[None, None].repeat(1, 5, 1, 1)
    desired[0, :, 2, 3] = torch.linspace(0.2, 0.0, 5)
    actual = desired.clone()
    if damage == "sideways":
        actual[0, 2, 0, 3] += 0.03
    if damage == "rotation":
        actual[0, 2, :3, :3] = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
    if damage == "nonfinite":
        actual[0, 2, 0, 3] = float("nan")
    assert path._pose_path_valid(actual, desired, 0.005, 0.0005) is (damage is None)


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    joint_trajectory: TimedTrajectory
    segments: tuple
    expected_effects: object
    diagnostics: PlannerDiagnostics
    scene_dependency_monitor_until: dict
    scene_dependency_end_segment: object = None
    effect_candidates: object = None
    effect_verification: object = None
    replannable: bool = True


def _fixture(monkeypatch):
    action, request, context, selection = _ownership()
    original = torch.tensor(
        [
            [
                [0.0, 0.3, 0.5, 0.4],
                [0.2, 0.2, 0.5, 0.4],
                [0.3, 0.1, 0.5, 0.4],
                [0.2, 0.0, 0.5, 0.4],
                [0.2, 0.0, 0.5, 0.4],
                [0.2, 0.0, 0.6, 0.4],
                [0.2, 0.1, 0.6, 0.4],
                [0.2, 0.2, 0.6, 0.4],
            ]
        ]
    )
    grasp = torch.eye(4)[None]
    grasp[0, 0, 3] = 0.2
    held = NS(grasp_xpos=grasp, semantics=NS(entity_id="item"))
    first = Plan(
        torch.tensor([True]),
        TimedTrajectory.from_uniform_step(
            original, env_ids=context.env_ids, step_dt=0.04
        ),
        (
            TrajectorySegment("approach", 0, 4),
            TrajectorySegment("close", 4, 6),
            TrajectorySegment("lift", 6, 8),
        ),
        NS(held_object_updates={"arm": held}),
        PlannerDiagnostics(backend="test"),
        {"item": 3},
    )
    target = JointPositionTarget("arm", (0, 1))
    hand = JointPositionTarget("grip", (2,))
    request.binding = NS(
        endpoint=lambda _slot, endpoint: NS(
            task_state_key="arm",
            require_target=lambda _t: target if endpoint == "motion" else hand,
        )
    )
    request.goal.semantics.entity_id = "item"
    request.skill_options.pre_grasp_distance = 0.2
    request.skill_options.grasp_commit_fraction = 0.75
    request.recovery_policy = NS(action_timeout=30.0)
    context.robot = NS(qpos=original[:, 0])
    context.require_control_dt = lambda: 0.04
    robot = NS(
        device=torch.device("cpu"),
        get_solver=lambda _p: NS(
            get_iteration_params=lambda: {"pos_eps": 0.005, "rot_eps": 0.0005}
        ),
    )

    def fk(*, qpos, **_kw):
        result = torch.eye(4)[None, None].repeat(1, qpos.shape[1], 1, 1)
        result[..., 0, 3] = qpos[..., 0]
        result[..., 2, 3] = qpos[..., 1]
        return result

    robot.compute_batch_fk = fk
    action.robot = robot

    def generate(states, *, options):
        torch.rand(1)
        if options.preserve_cartesian_samples:
            q = torch.tensor([[[0.2, 0.0], [0.2, 0.1], [0.2, 0.2]]])
        else:
            q = torch.stack((options.start_qpos, states[0].qpos), 1)
        dt = torch.full(q.shape[:2], 0.04)
        dt[:, 0] = 0
        return NS(success=torch.tensor([True]), positions=q, dt=dt)

    action.motion_generator = NS(_generate_checked=generate)

    def build(_r, _c, **kw):
        segments, start = [], 0
        for name, count in kw["segment_lengths"].items():
            segments.append(TrajectorySegment(name, start, start + count))
            start += count
        return replace(
            first,
            joint_trajectory=kw["trajectory"],
            segments=tuple(segments),
            diagnostics=kw["diagnostics"],
            scene_dependency_monitor_until=kw["scene_dependency_monitor_until"],
        )

    action.build_plan = build
    action.failed_plan = Mock(
        return_value=replace(first, plan_success=torch.tensor([False]))
    )
    monkeypatch.setattr(
        path, "_joint_velocity_limits", lambda *_a: torch.full((1, 2), 10.0)
    )
    monkeypatch.setattr(path, "_inspect_pick_plan", lambda *_a: {"valid": True})
    monkeypatch.setattr(path, "check_free_motion", lambda *_a: {"valid": True})
    return action, request, context, first, selection


def test_rebuilt_path_preserves_grasp_suffix_effects_passive_joints_and_rng(
    monkeypatch,
) -> None:
    data = _fixture(monkeypatch)
    state = torch.random.get_rng_state()
    result = path.rebuild(*data)
    assert result.plan_success.all()
    assert torch.equal(
        result.joint_trajectory.positions[:, 4:],
        data[3].joint_trajectory.positions[:, 4:],
    )
    assert torch.equal(
        result.joint_trajectory.positions[:, :, 3],
        data[3].joint_trajectory.positions[:, :, 3],
    )
    assert result.expected_effects is data[3].expected_effects
    assert (
        result.diagnostics.metadata["gen_sim_pick_path"]["provider"] == "joint_interp"
    )
    assert torch.equal(state, torch.random.get_rng_state())


def test_collision_unavailable_fails_closed_without_dispatching_original(
    monkeypatch,
) -> None:
    data = _fixture(monkeypatch)
    monkeypatch.setattr(path, "check_free_motion", lambda *_a: {"valid": False})
    result = path.rebuild(*data)
    assert not result.plan_success.any()
    assert result.diagnostics.failure.code == "gen_sim_pick_path_unqualified"


def test_failed_cartesian_samples_use_bounded_local_ik_and_fail_closed(
    monkeypatch,
) -> None:
    data = _fixture(monkeypatch)
    failed = NS(success=torch.tensor([False]), positions=None, dt=None)
    generate = Mock(return_value=failed)
    local = Mock(return_value=None)
    data[0].motion_generator._generate_checked = generate
    monkeypatch.setattr(path, "_local_place_ik", local)
    result = path.rebuild(*data)
    assert not result.plan_success.any()
    assert generate.call_count == 2
    assert local.call_count == 2
    assert all(
        call.kwargs["options"].preserve_cartesian_samples
        for call in generate.call_args_list
    )


def test_failed_cartesian_samples_accept_local_ik_recovery(monkeypatch) -> None:
    data = _fixture(monkeypatch)
    failed = NS(success=torch.tensor([False]), positions=None, dt=None)
    original_generate = data[0].motion_generator._generate_checked

    def generate(states, *, options):
        if options.preserve_cartesian_samples:
            return failed
        return original_generate(states, options=options)

    data[0].motion_generator._generate_checked = generate
    recovered = NS(
        success=torch.tensor([True]),
        positions=torch.tensor([[[0.2, 0.0], [0.2, 0.1], [0.2, 0.2]]]),
        dt=torch.tensor([[0.0, 0.04, 0.04]]),
    )
    local = Mock(return_value=recovered)
    monkeypatch.setattr(path, "_local_place_ik", local)

    result = path.rebuild(*data)

    assert result.plan_success.all()
    assert local.call_count == 1
    assert (
        result.diagnostics.metadata["gen_sim_pick_path"]["attempts"][0][
            "contact_local_ik_valid"
        ]
        is True
    )


def test_collision_planner_connects_exact_branch_only_after_direct_path_rejection(
    monkeypatch,
) -> None:
    data = _fixture(monkeypatch)
    checks = Mock(side_effect=[{"valid": True}, {"valid": False}, {"valid": True}])
    monkeypatch.setattr(path, "check_free_motion", checks)
    prefix = torch.tensor([[[0.0, 0.3], [0.1, 0.4], [0.2, 0.2]]])
    monkeypatch.setattr(
        path,
        "_curobo_prefix",
        lambda *_a: NS(
            success=torch.tensor([True]),
            positions=prefix,
            dt=torch.tensor([[0.0, 0.04, 0.04]]),
        ),
    )
    result = path.rebuild(*data)
    assert result.plan_success.all()
    assert result.diagnostics.metadata["gen_sim_pick_path"]["provider"] == "curobo"
    assert torch.equal(
        result.joint_trajectory.positions[:, 5:],
        data[3].joint_trajectory.positions[:, 4:],
    )


def test_different_joint_branches_cannot_be_spliced() -> None:
    original = torch.zeros(1, 8, 3)
    prefix = torch.zeros(1, 3, 2)
    contact = torch.zeros(1, 4, 2)
    contact[:, 0, 0] = 0.1
    with pytest.raises(ValueError, match="different joint branches"):
        path._compose_approach(original, 4, [0, 1], prefix, contact)


@pytest.mark.parametrize("limit", [0.0, float("nan"), float("inf")])
def test_invalid_active_velocity_limits_fail_closed(monkeypatch, limit) -> None:
    data = _fixture(monkeypatch)
    monkeypatch.setattr(
        path, "_joint_velocity_limits", lambda *_a: torch.full((1, 2), limit)
    )
    result = path.rebuild(*data)
    assert not result.plan_success.any()


def test_duration_budget_is_checked_before_building_large_prefix(monkeypatch) -> None:
    data = _fixture(monkeypatch)
    data[1].recovery_policy.action_timeout = 0.1
    monkeypatch.setattr(
        path,
        "_curobo_prefix",
        Mock(side_effect=AssertionError("must not initialize planner")),
    )
    result = path.rebuild(*data)
    assert not result.plan_success.any()


def test_already_qualified_original_retains_exact_plan(monkeypatch) -> None:
    data = list(_fixture(monkeypatch))
    positions = data[3].joint_trajectory.positions.clone()
    positions[:, 2, 0] = 0.2
    data[3] = replace(
        data[3],
        joint_trajectory=TimedTrajectory.from_uniform_step(
            positions, env_ids=data[2].env_ids, step_dt=0.04
        ),
    )
    assert path.rebuild(*data) is data[3]
