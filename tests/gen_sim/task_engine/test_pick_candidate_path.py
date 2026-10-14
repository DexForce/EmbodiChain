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

from embodichain.gen_sim.task_engine._task_program import (
    pick_candidate_path as candidate,
)
from embodichain.gen_sim.task_engine._task_program.actions import (
    GenSimPickUp,
    _PickSelection,
    _PICK_SELECTION,
)
from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
    GraspStrategy,
    select_grasp,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    GraspGoal,
    JointPositionTarget,
    ObjectSemantics,
    PickUpOptions,
)
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
)
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp


@dataclass
class Request:
    invocation_id: str
    goal: object
    skill_options: object
    binding: object
    motion_policy: object
    recovery_policy: object


@dataclass
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics
    joint_trajectory: object = None
    segments: tuple = ()


def _pool():
    poses = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
    poses[0, :, 0, 3] = torch.tensor([0.2, 0.4])
    return AffordancePoseCandidates(
        poses, torch.tensor([[0.0, 1.0]]), torch.tensor([[True, False]])
    )


def test_raw_geometry_observer_survives_empty_endpoint_ik_pool():
    raw, filtered = [], []
    pool = _pool()
    affordance = AntipodalAffordance()
    affordance.get_grasp_candidates = Mock(return_value=pool)
    select_grasp(
        affordance,
        object(),
        torch.eye(4)[None],
        (GraspStrategy(torch.tensor([[0.0, 0.0, -1.0]]), None, torch.tensor([True])),),
        NS(affordance_sampling=None, env_ids=torch.tensor([0])),
        "test",
        feasible=lambda poses, direction: (poses, torch.zeros_like(pool.valid)),
        candidate_observer=filtered.append,
        raw_candidate_observer=raw.append,
    )
    assert torch.equal(raw[0].valid, pool.valid)
    assert not filtered[0].valid.any()


def _fixture(monkeypatch):
    qpos = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.7]])
    arm, hand = JointPositionTarget("arm", (0, 1, 2)), JointPositionTarget("hand", (3,))
    endpoint = NS(require_target=lambda _type: arm, task_state_key="arm")
    hand_endpoint = NS(
        require_target=lambda _type: hand,
        task_state_key="arm",
        joint_positions=lambda name, **kw: torch.tensor(
            [[0.0 if name == "open" else 0.1]]
        ),
    )
    request = Request(
        invocation_id="pick:0",
        goal=GraspGoal(ObjectSemantics(AntipodalAffordance(), {}, "item")),
        skill_options=PickUpOptions(
            hand_interp_steps=3,
            grasp_settle_steps=2,
            pre_grasp_distance=0.1,
            lift_height=0.1,
        ),
        binding=NS(
            endpoint=lambda slot, name: endpoint if name == "motion" else hand_endpoint
        ),
        motion_policy=NS(
            sample_count=18, strategy="ik_interp", dynamic_collision_mode="off"
        ),
        recovery_policy=NS(action_timeout=10.0),
    )
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=NS(qpos=qpos),
        require_control_dt=lambda: 0.04,
    )

    def fk(qpos, **kw):
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., :3, 3] = qpos
        return poses

    robot = NS(
        device=torch.device("cpu"),
        compute_fk=fk,
        compute_batch_fk=fk,
        get_solver=lambda part: NS(
            get_iteration_params=lambda: {"pos_eps": 0.005, "rot_eps": 0.0005}
        ),
        get_qpos_limits=lambda **kw: torch.tensor([[[-2.0, 2.0]]]).expand(1, 3, 2),
        compute_batch_ik=Mock(
            side_effect=lambda pose, **kw: (
                torch.tensor([[True]]),
                pose[..., :3, 3].clone(),
            )
        ),
    )
    action = GenSimPickUp()
    action._planning_services = NS(
        robot=robot, device=robot.device, motion_generator=NS(robot=robot)
    )
    action.path_simulation = object()
    calls = []

    def solve(generator, targets, options, **kw):
        torch.rand(1)
        calls.append(
            (options.start_qpos.clone(), options.sample_count, options.interpolation_dt)
        )
        positions = torch.stack(
            (options.start_qpos, *(state.xpos[:, :3, 3] for state in targets)), 1
        )
        return NS(positions=positions, success=torch.tensor([True]))

    monkeypatch.setattr(candidate, "resolve_continuous_ik_path", solve)
    monkeypatch.setattr(
        candidate, "_resolve_object_pose", lambda sem, context: torch.eye(4)[None]
    )
    grasp = torch.eye(4)[None]
    grasp[:, 0, 3] = 0.2
    direction = torch.tensor([[0.0, 0.0, -1.0]])
    return action, request, context, grasp, direction, calls


def test_full_pick_continuous_trajectory_preserves_branch_timing_passive_and_hand(
    monkeypatch,
):
    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    exact, receipt = candidate.trajectory(action, request, context, grasp, direction)
    success, full, lengths = exact
    assert success.all() and receipt["timing_unchanged"]
    assert full.shape == (
        1,
        request.motion_policy.sample_count + request.skill_options.grasp_settle_steps,
        5,
    )
    assert torch.equal(
        full[:, :, 4], context.robot.qpos[:, 4:5].expand(-1, full.shape[1])
    )
    approach_end = lengths["approach"] - 1
    lift_start = lengths["approach"] + lengths["close"]
    assert torch.equal(full[:, approach_end, :3], full[:, lift_start, :3])
    assert torch.equal(calls[1][0], full[:, approach_end, :3])
    assert all(call[2] == context.require_control_dt() for call in calls)
    assert torch.equal(full[:, -1, :3], torch.tensor([[0.2, 0.0, 0.1]]))
    assert torch.equal(full[:, -1, 3], torch.tensor([0.1]))


@pytest.mark.parametrize(
    "damage", ["lift", "fk", "downstream_ik", "downstream_fk", "budget"]
)
def test_full_pick_rejects_missing_qualifications(monkeypatch, damage):
    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    if damage == "lift":
        solve = candidate.resolve_continuous_ik_path
        monkeypatch.setattr(
            candidate,
            "resolve_continuous_ik_path",
            lambda g, t, o, **kw: (
                None if "lift" in kw["operation"] else solve(g, t, o, **kw)
            ),
        )
    if damage == "fk":
        action.robot.compute_batch_fk = lambda qpos, **kw: torch.eye(4)[
            None, None
        ].expand(1, qpos.shape[1], 4, 4)
    if damage.startswith("downstream"):
        request.skill_options = replace(
            request.skill_options, downstream_object_target_poses=(torch.eye(4),)
        )
        if damage == "downstream_ik":
            action.robot.compute_batch_ik = Mock(
                return_value=(torch.tensor([[False]]), torch.zeros(1, 1, 3))
            )
        else:
            action.robot.compute_batch_ik = Mock(
                return_value=(torch.tensor([[True]]), torch.ones(1, 1, 3))
            )
    if damage == "budget":
        request.recovery_policy.action_timeout = 0.01
    exact, receipt = candidate.trajectory(action, request, context, grasp, direction)
    assert exact is None and receipt["valid"] is False


def test_downstream_without_timing_retains_endpoint_gate_without_path_claim(
    monkeypatch,
):
    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    request.skill_options = replace(
        request.skill_options, downstream_object_target_poses=(torch.eye(4),)
    )
    receipt = candidate.downstream_endpoints(
        action, request, context, grasp, torch.zeros(1, 3)
    )
    assert receipt == {
        "valid": True,
        "target_count": 1,
        "qualification_scope": "untimed_endpoint_ik_fk_only",
        "continuous_path_qualified": False,
    }
    assert action.robot.compute_batch_ik.call_args.kwargs["joint_seed"].shape == (
        1,
        1,
        3,
    )


def test_constrained_pick_keeps_raw_pool_when_vertical_lift_screen_rejects(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.grasp_filter import (
        TaskGraspPoseGenerator,
    )
    from embodichain.gen_sim.task_engine._task_program.pick_safety import (
        _PICK_GRASP_RULE,
    )

    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    request.goal.semantics.affordance.set_custom_config(
        _PICK_GRASP_RULE, ("item", "upright")
    )
    generator = Mock(spec=TaskGraspPoseGenerator)
    action.planning_services.grasp_pose_generator = Mock(return_value=generator)
    pool = _pool()
    monkeypatch.setattr(
        request.goal.semantics.affordance,
        "get_grasp_candidates",
        Mock(return_value=pool),
    )
    monkeypatch.setattr(
        action,
        "_select_feasible_grasp_variants",
        lambda *args: (pool.poses, torch.zeros_like(pool.valid)),
    )
    context.affordance_sampling = None
    selection = _PickSelection(request=request)
    token = _PICK_SELECTION.set(selection)
    try:
        sample = action._resolve_grasp_pose(
            request.goal.semantics,
            torch.eye(4)[None],
            torch.zeros(1, 3),
            request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            ),
            "hand",
            request.skill_options,
            direction,
            context,
            sample_key="same",
        )
    finally:
        _PICK_SELECTION.reset(token)
    assert not sample.success.any()
    assert selection.constrained and torch.equal(
        selection.raw_candidate_routes[0][1].valid, pool.valid
    )
    assert candidate.eligible(action, request, context, selection)
    generator.for_pick.assert_called_once()


def test_upright_calibration_and_call_local_filter_are_not_skipped(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.grasp_filter import (
        TaskGraspPoseGenerator,
    )
    from embodichain.gen_sim.task_engine._task_program.pick_safety import (
        _PICK_GRASP_RULE,
    )

    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    calibration = torch.eye(4)
    calibration[2, 3] = 0.03
    request.skill_options = replace(
        request.skill_options, rotate_upright=0.5, grasp_frame_to_eef=calibration
    )
    request.goal.semantics.affordance.set_custom_config(
        _PICK_GRASP_RULE, ("item", "upright")
    )
    generator = Mock(spec=TaskGraspPoseGenerator)
    scoped = generator.for_pick.return_value
    scoped.accepts_grasp_poses.return_value = torch.tensor([True, False, False, False])
    action.planning_services.grasp_pose_generator = Mock(return_value=generator)
    pool = _pool()
    result = candidate.geometric_variants(action, pool, request, context, direction)
    base = torch.stack((pool.poses, pool.poses.clone()), 2)
    base[:, :, 1, :3, :2] *= -1
    adjusted = action._upright_adjusted_grasp_poses(
        base, torch.eye(4)[None], request.skill_options
    )
    torch.testing.assert_close(result.poses, (adjusted @ calibration).flatten(1, 2))
    torch.testing.assert_close(
        scoped.accepts_grasp_poses.call_args.args[0], adjusted.reshape(-1, 4, 4)
    )
    assert result.valid.tolist() == [[True, False, False, False]]


def test_successful_original_pick_never_enters_raw_candidate_recovery(monkeypatch):
    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    first = NS(plan_success=torch.tensor([True]))
    monkeypatch.setattr(PickUp, "_plan", lambda *args: first)
    monkeypatch.setattr(
        action,
        "_recover_raw_pick_candidates",
        Mock(side_effect=AssertionError("must preserve original")),
    )
    assert action._plan(request, context) is first


def test_qualified_existing_constrained_roll_recovery_is_preserved(monkeypatch):
    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    first = Plan(torch.tensor([False]), PlannerDiagnostics(backend="mock"))
    existing = replace(first, plan_success=torch.tensor([True]))

    def old_pick(self, request, context):
        selection = _PICK_SELECTION.get()
        selection.constrained = True
        selection.raw_candidate_routes = [(0, _pool(), direction)]
        return first

    monkeypatch.setattr(PickUp, "_plan", old_pick)
    monkeypatch.setattr(
        action, "_recover_constrained_pick_roll", Mock(return_value=existing)
    )
    monkeypatch.setattr(
        action,
        "_recover_raw_pick_candidates",
        Mock(side_effect=AssertionError("must preserve qualified old recovery")),
    )
    assert action._plan(request, context) is existing


@pytest.mark.parametrize("gate", [None, "velocity", "scene", "preview"])
def test_empty_ik_pool_recovery_hands_the_same_trajectory_to_shared_effect_builder(
    monkeypatch, gate
):
    from embodichain.gen_sim.task_engine._task_program import (
        actions,
        articulation_collision,
        pick_continuation,
    )

    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    first = Plan(torch.tensor([False]), PlannerDiagnostics(backend="mock"))
    selection = _PickSelection(request=request, constrained=True)
    selection.raw_candidate_routes = [(0, _pool(), direction)]
    selection.candidates = replace(_pool(), valid=torch.zeros_like(_pool().valid))
    built = []
    original = request.skill_options
    request.skill_options = replace(original, rotate_upright=0.5)
    # Geometric qualification has already applied the original transform.
    monkeypatch.setattr(candidate, "geometric_variants", lambda *args: _pool())

    def build(self, selected, context):
        exact = _PICK_SELECTION.get().trajectory
        assert selected.skill_options.rotate_upright is None
        assert exact is not None
        built.append(exact)
        success, positions, lengths = exact
        return Plan(
            success,
            PlannerDiagnostics(backend="mock"),
            NS(positions=positions),
            (NS(name="approach", stop=lengths["approach"]),),
        )

    monkeypatch.setattr(PickUp, "_plan", build)
    monkeypatch.setattr(
        actions, "_inspect_pick_plan", lambda *args: {"valid": gate != "velocity"}
    )
    monkeypatch.setattr(
        articulation_collision,
        "check_free_motion",
        lambda *args: {"valid": gate != "scene"},
    )
    if gate == "preview":
        action.continuation_previews = {request.invocation_id: object()}
        monkeypatch.setattr(
            pick_continuation, "preview_place", lambda *args, **kw: {"valid": False}
        )
    token = _PICK_SELECTION.set(selection)
    rng = torch.random.get_rng_state().clone()
    try:
        result = action._recover_raw_pick_candidates(request, context, first, selection)
    finally:
        _PICK_SELECTION.reset(token)
    assert bool(result.plan_success.all()) is (gate is None)
    assert len(built) == 1
    if gate is None:
        assert torch.equal(built[0][1], result.joint_trajectory.positions)
    assert selection.trajectory is None and selection.request is request
    assert request.skill_options.rotate_upright == 0.5
    assert torch.equal(rng, torch.random.get_rng_state())


def test_raw_retry_shares_old_four_candidate_budget(monkeypatch):
    action, request, context, grasp, direction, calls = _fixture(monkeypatch)
    first = Plan(
        torch.tensor([False]),
        PlannerDiagnostics(
            backend="mock",
            metadata={"gen_sim_pick_candidate_retry": {"attempts": [{}, {}, {}, {}]}},
        ),
    )
    selection = _PickSelection(
        request=request, raw_candidate_routes=[(0, _pool(), direction)]
    )
    monkeypatch.setattr(
        candidate,
        "trajectory",
        Mock(side_effect=AssertionError("budget already exhausted")),
    )
    assert (
        action._recover_raw_pick_candidates(request, context, first, selection) is first
    )
