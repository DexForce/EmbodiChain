# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import pytest
import torch
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

from embodichain.gen_sim.task_engine._task_program.pick_path import _compose_approach
from embodichain.gen_sim.task_engine._task_program import pick_path, motion
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget


@pytest.mark.parametrize(
    "enabled,scene_clear,robot_safe,expected",
    [
        (True, False, True, True),
        (False, False, True, False),
        (False, True, True, True),
        (True, False, False, False),
    ],
)
def test_continuous_pick_rebuild_keeps_native_mode_and_robot_guards(
    monkeypatch, enabled, scene_clear, robot_safe, expected
):
    @dataclass(frozen=True)
    class Plan:
        plan_success: Any
        joint_trajectory: Any
        segments: tuple
        expected_effects: Any
        diagnostics: Any
        scene_dependency_monitor_until: dict
        scene_dependency_end_segment: dict
        effect_candidates: tuple = ()
        effect_verification: Any = None
        replannable: bool = True

    full = torch.zeros(1, 7, 3)
    full[0, :, 0] = torch.tensor([0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3])
    full[0, :, 1] = torch.tensor([-0.002, -0.001, 0, 0, 0.7, 0.7, 0.7])
    full[:, :, 2] = 0.37
    grasp = torch.eye(4)[None]
    grasp[:, 0, 3] = 0.2
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="object"), grasp_xpos=grasp
    )
    first = Plan(
        torch.tensor([True]),
        SimpleNamespace(positions=full),
        tuple(
            SimpleNamespace(name=n, start=a, stop=b)
            for n, a, b in (("approach", 0, 3), ("close", 3, 5), ("lift", 5, 7))
        ),
        SimpleNamespace(held_object_updates={"left": held}),
        PlannerDiagnostics("test"),
        {},
        {},
    )
    endpoints = {
        name: SimpleNamespace(
            task_state_key="left",
            require_target=lambda _, name=name: JointPositionTarget(
                name, (0,) if name == "motion" else (1,)
            ),
        )
        for name in ("motion", "grasp")
    }
    request = SimpleNamespace(
        binding=SimpleNamespace(endpoint=lambda _, name: endpoints[name]),
        goal=SimpleNamespace(semantics=held.semantics),
        skill_options=SimpleNamespace(
            pre_grasp_distance=0.1, grasp_commit_fraction=0.8
        ),
        recovery_policy=SimpleNamespace(action_timeout=30),
    )
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=full[:, 0].clone()),
        env_ids=torch.tensor([0]),
        require_control_dt=lambda: 0.04,
    )

    def fk(*, qpos, **_):
        poses = torch.eye(4).expand(*qpos.shape[:2], 4, 4).clone()
        poses[..., 0, 3] = qpos[..., 0]
        return poses

    robot = SimpleNamespace(
        device="cpu",
        get_qpos_limits=lambda **_: torch.tensor([[[0, 0.7]]]),
        compute_batch_fk=fk,
        get_solver=lambda _: object(),
    )

    def generate(states, *, options):
        count = options.sample_count
        q = (
            torch.linspace(0.2, 0.1, count)
            if options.is_linear
            else torch.linspace(0, 0.1, count)
        )
        dt = torch.full((1, count), 0.04)
        dt[:, 0] = 0
        return SimpleNamespace(
            success=torch.tensor([True]), positions=q[None, :, None], dt=dt
        )

    def build(*_, **kwargs):
        cursor, segments = 0, []
        for name, count in kwargs["segment_lengths"].items():
            segments.append(
                SimpleNamespace(name=name, start=cursor, stop=cursor + count)
            )
            cursor += count
        return replace(
            first,
            joint_trajectory=kwargs["trajectory"],
            segments=tuple(segments),
            diagnostics=kwargs["diagnostics"],
        )

    action = SimpleNamespace(
        native_pick_exploration=enabled,
        robot=robot,
        path_simulation=object(),
        motion_generator=SimpleNamespace(_generate_checked=generate),
        build_plan=build,
        failed_plan=lambda *_a, **_k: replace(
            first, plan_success=torch.tensor([False])
        ),
    )
    monkeypatch.setattr(
        pick_path,
        "_pose_validation_parameters",
        lambda _: {"pos_eps": 0.001, "rot_eps": 0.001},
    )
    monkeypatch.setattr(
        pick_path, "_joint_velocity_limits", lambda *_: torch.ones(1, 1)
    )
    monkeypatch.setattr(
        pick_path,
        "_inspect_pick_plan",
        lambda _r, _q, _c, plan: {"valid": robot_safe and plan is not first},
    )
    monkeypatch.setattr(
        pick_path,
        "check_free_motion",
        lambda _r, _s, positions, _p: {"valid": scene_clear or len(positions) == 1},
    )
    monkeypatch.setattr(
        pick_path,
        "_curobo_joint_prefix",
        lambda *_: SimpleNamespace(
            success=torch.tensor([False]), positions=None, dt=None
        ),
    )
    result = pick_path.rebuild(
        action,
        request,
        context,
        first,
        SimpleNamespace(direction=torch.tensor([[1.0, 0, 0]])),
    )
    assert bool(result.plan_success[0]) is expected
    if expected:
        positions = result.joint_trajectory.positions
        assert (positions[:, :-4, 1] >= 0).all()
        assert torch.equal(positions[:, -4:], full[:, 3:])
        assert torch.equal(
            positions[:, :, 2], torch.full_like(positions[:, :, 2], 0.37)
        )
        assert ("gen_sim_native_pick_exploration" in result.diagnostics.metadata) is (
            enabled and not scene_clear
        )
    assert full[0, 0, 1] == pytest.approx(-0.002)


@pytest.mark.parametrize("initial_hand", [-0.002, 0.0, 0.7, 0.700128])
def test_rebuilt_approach_preserves_opening_before_close_and_passive_joints(
    initial_hand: float,
) -> None:
    original = torch.zeros(1, 8, 4)
    original[:, :, 3] = 0.37
    original[:, :4, 2] = torch.linspace(initial_hand, 0.0, 4)
    original[:, 4:7, 2] = torch.linspace(0.0, 0.7, 3)
    original[:, 7:, 2] = 0.7
    original[:, :4, :2] = torch.linspace(0.0, 0.4, 4)[None, :, None]
    before = original.clone()
    prefix = torch.linspace(0.0, 0.2, 4)[None, :, None].expand(1, 4, 2).clone()
    contact = torch.linspace(0.2, 0.4, 3)[None, :, None].expand(1, 3, 2).clone()
    bounds = torch.tensor([[[0.0, 0.7]]])
    result = _compose_approach(original, 4, [0, 1], prefix, contact, [2], bounds)
    approach_count = len(prefix[0]) + len(contact[0]) - 1
    assert torch.equal(result[:, :4, :2], prefix)
    assert torch.equal(result[:, 4:approach_count, :2], contact[:, 1:])
    assert torch.equal(result[:, approach_count:], original[:, 4:])
    assert torch.equal(result[:, :, 3], torch.full_like(result[:, :, 3], 0.37))
    assert result[0, 0, 2] == max(0.0, min(initial_hand, 0.7))
    assert (
        (result[:, :approach_count, 2] >= 0) & (result[:, :approach_count, 2] <= 0.7)
    ).all()
    assert result[0, 0, 0] == original[0, 0, 0]
    assert result[0, approach_count - 1, 2] == result[0, approach_count, 2] == 0.0
    assert (result[0, 1:approach_count, 2] <= result[0, : approach_count - 1, 2]).all()
    assert torch.equal(original, before)


@dataclass(frozen=True)
class _TrialPlan:
    diagnostics: PlannerDiagnostics
    joint_trajectory: Any
    segments: tuple
    commands: object
    expected_effects: object


def _trial_inputs(enabled: bool = True) -> tuple:
    plan = _TrialPlan(
        PlannerDiagnostics("test"),
        SimpleNamespace(positions=torch.zeros(1, 3, 2)),
        (SimpleNamespace(name="approach", stop=2),),
        object(),
        object(),
    )
    action = SimpleNamespace(
        native_pick_exploration=enabled,
        robot=SimpleNamespace(device="cpu"),
        path_simulation=object(),
    )
    context = SimpleNamespace(robot=SimpleNamespace(qpos=torch.zeros(1, 2)))
    request = SimpleNamespace(
        binding=SimpleNamespace(
            endpoint=lambda *args: SimpleNamespace(
                require_target=lambda cls: JointPositionTarget("arm", (0,))
            )
        )
    )
    return action, request, context, plan


def test_native_trial_is_opt_in_and_preserves_all_issued_commands(monkeypatch) -> None:
    action, request, context, plan = _trial_inputs()
    monkeypatch.setattr(pick_path, "_inspect_pick_plan", lambda *args: {"valid": True})
    monkeypatch.setattr(
        pick_path, "check_free_motion", lambda *args: {"valid": False, "kind": "scene"}
    )
    result = pick_path._native_scene_attempt(action, request, context, plan)
    assert result.commands is plan.commands
    assert result.expected_effects is plan.expected_effects
    assert result.joint_trajectory is plan.joint_trajectory
    audit = result.diagnostics.metadata["gen_sim_native_pick_exploration"]
    assert audit["planned_scene_clearance_certified"] is False
    assert audit["scene_precheck"]["valid"] is False
    action.native_pick_exploration = False
    monkeypatch.setattr(
        pick_path,
        "_inspect_pick_plan",
        lambda *args: pytest.fail("Default path must not enter exploration."),
    )
    assert pick_path._native_scene_attempt(action, request, context, plan) is None


@pytest.mark.parametrize(
    "value,upper,reason",
    [(0.2, 1.0, "final_velocity_rejected"), (0.01, 0.005, "joint_limits_rejected")],
)
def test_native_trial_does_not_bypass_command_guards(
    monkeypatch, value: float, upper: float, reason: str
) -> None:
    action, request, context, plan = _trial_inputs()
    plan.joint_trajectory.positions[:, 1:, 0] = value
    plan.joint_trajectory.dt = torch.tensor([[0.0, 0.04, 0.04]])
    plan.segments[0].start = 0
    plan.segments[0].stop = 3
    context.env_ids = torch.tensor([0])
    endpoints = {
        name: SimpleNamespace(
            task_state_key="left",
            require_target=lambda cls, name=name: JointPositionTarget(
                name, (0,) if name == "motion" else (1,)
            ),
        )
        for name in ("motion", "grasp")
    }
    request.binding.endpoint = lambda slot, name: endpoints[name]
    action.robot.get_qpos_limits = lambda name, env_ids: torch.tensor([[[-1.0, upper]]])
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.ones(1, 2)
    )
    observed = []
    inspect = pick_path._inspect_pick_plan

    def capture(*args):
        result = inspect(*args)
        observed.append(result)
        return result

    monkeypatch.setattr(pick_path, "_inspect_pick_plan", capture)
    monkeypatch.setattr(
        pick_path,
        "check_free_motion",
        lambda *args: pytest.fail("Unsafe commands cannot reach a Native trial."),
    )
    assert pick_path._native_scene_attempt(action, request, context, plan) is None
    assert observed[0]["reason"] == reason
