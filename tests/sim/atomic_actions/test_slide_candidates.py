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


"""Selected Slide grasps preserve identity, axis geometry, and row-local failures."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AtomicAction,
    JointPositionGoal,
    MotionPolicy,
    MoveJoints,
    ObjectSemantics,
    SceneEntityPose,
    Slide,
    SlideAffordance,
    SlideGoal,
    SlideOptions,
)
from embodichain.lab.sim.atomic_actions.primitives.slide import SlideCandidateBatch
from embodichain.toolkits.graspkit import GraspCandidateBatch

from .test_actions import ARM_DOF, NUM_ENVS, ROBOT_DOF
from .test_candidates import _setup as _pickup_setup


def _setup(*, direction="pull", hand_interp_steps=5):
    engine, _, context, raw = _pickup_setup()
    engine.register(Slide())
    semantics = ObjectSemantics(
        affordance=SlideAffordance(
            mesh_vertices=torch.tensor(
                ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0))
            ),
            mesh_triangles=torch.tensor(((0, 1, 2),)),
            translation_axis=torch.tensor((0.0, 1.0, 0.0)),
        ),
        geometry={},
        label="drawer",
        entity_id="target",
    )
    invocation = ActionInvocation(
        skill_id="slide",
        invocation_id="slide-1",
        goal=SlideGoal(semantics, SceneEntityPose("target")),
        binding=engine.bind_control_parts(
            "slide", {"primary": {"motion": "arm", "grasp": "hand"}}
        ),
        motion_policy=MotionPolicy(sample_count=80),
        skill_options=SlideOptions(
            direction=direction, hand_interp_steps=hand_interp_steps
        ),
    )
    sampler = engine.actions["slide"].planning_services.grasp_pose_generator("hand")
    sampler.get_best_grasp_poses = Mock(
        side_effect=AssertionError("selected grasp must never be reselected")
    )
    sampler.get_valid_grasp_poses = Mock(
        side_effect=AssertionError("supplied grasps must not be resampled")
    )
    return engine, invocation, context, raw


def test_slide_candidate_is_one_to_one_with_each_original_grasp():
    engine, invocation, context, raw = _setup()
    assert Slide.__bases__ == (AtomicAction,)
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    assert isinstance(batch, SlideCandidateBatch)
    assert batch.valid_mask.tolist() == [[True, True], [True, True]]
    assert batch.grasp_ids == raw.grasp_ids
    assert batch.qpos_anchors.shape == (NUM_ENVS, 2, 3, ARM_DOF)
    torch.testing.assert_close(batch.grasp_poses, raw.poses)
    expected_pre, expected_end = raw.poses.clone(), raw.poses.clone()
    expected_pre[..., 1, 3] -= invocation.skill_options.approach_distance
    expected_end[..., 1, 3] -= invocation.skill_options.translation_distance
    torch.testing.assert_close(batch.pre_grasp_poses, expected_pre)
    torch.testing.assert_close(batch.translated_poses, expected_end)


@pytest.mark.parametrize("direction", ("pull", "push"))
def test_selected_slide_keeps_exact_anchors_axis_and_initial_closed_hand(direction):
    engine, invocation, context, raw = _setup(direction=direction)
    initial = context.robot.qpos.clone()
    initial[:, ARM_DOF:] = 0.7
    context = context.project(qpos=initial, task=context.task)
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    indices = torch.tensor((0, 1))
    plan = engine.plan_candidate(invocation, batch.select(indices), context)
    assert plan.plan_success.all(), plan.diagnostics.metadata
    trajectory = plan.joint_trajectory.positions
    assert trajectory.shape == (
        NUM_ENVS,
        invocation.motion_policy.sample_count,
        ROBOT_DOF,
    )
    torch.testing.assert_close(trajectory[:, 0], initial, rtol=0, atol=0)
    torch.testing.assert_close(context.robot.qpos, initial, rtol=0, atol=0)
    names = ["approach", "reach", "close", direction, "open"]
    if direction == "push":
        names.append("return")
    assert [segment.name for segment in plan.segments] == names
    assert plan.scene_dependency_end_segment == "reach"
    assert not plan.expected_effects.held_object_updates
    anchors = batch.qpos_anchors[torch.arange(NUM_ENVS), indices]
    endpoints = (("approach", 0), ("reach", 1), (direction, 2))
    if direction == "push":
        endpoints += (("return", 0),)
    for name, index in endpoints:
        segment = plan.segment(name)
        torch.testing.assert_close(
            trajectory[:, segment.stop - 1, :ARM_DOF], anchors[:, index]
        )
    for name in ("reach", direction, *(("return",) if direction == "push" else ())):
        segment = plan.segment(name)
        poses = engine.robot.compute_batch_fk(
            qpos=trajectory[:, segment.start : segment.stop, :ARM_DOF]
        )
        selected = raw.poses[torch.arange(NUM_ENVS), indices]
        torch.testing.assert_close(
            poses[..., 0, 3], selected[:, None, 0, 3].expand_as(poses[..., 0, 3])
        )
        torch.testing.assert_close(
            poses[..., 2, 3], selected[:, None, 2, 3].expand_as(poses[..., 2, 3])
        )
        torch.testing.assert_close(
            poses[..., :3, :3], selected[:, None, :3, :3].expand_as(poses[..., :3, :3])
        )
    torch.testing.assert_close(
        trajectory[:, plan.segment("close").stop - 1, ARM_DOF:],
        torch.ones(NUM_ENVS, ROBOT_DOF - ARM_DOF),
    )
    torch.testing.assert_close(
        trajectory[:, -1, ARM_DOF:], torch.zeros(NUM_ENVS, ROBOT_DOF - ARM_DOF)
    )
    assert plan.diagnostics.metadata["candidate_ids"] == (
        batch.candidate_ids[0][0],
        batch.candidate_ids[1][1],
    )


@pytest.mark.parametrize("stage_index", (0, 1, 2))
@pytest.mark.parametrize(
    "failure", ("IK_NOT_FOUND", "IK_INVALID_RESULT", "FK_MISMATCH", "JOINT_LIMIT")
)
def test_every_anchor_stage_rejects_only_its_bad_candidate(stage_index, failure):
    engine, invocation, context, raw = _setup()
    solve = engine.robot.compute_batch_ik.side_effect
    calls = []

    def failing(**kwargs):
        assert torch.isfinite(kwargs["joint_seed"]).all()
        calls.append(kwargs["joint_seed"].clone())
        flags, qpos = solve(**kwargs)
        if len(calls) == stage_index + 1:
            if failure == "IK_NOT_FOUND":
                flags[0, 0] = False
            elif failure == "IK_INVALID_RESULT":
                qpos[0, 0] = torch.nan
            elif failure == "FK_MISMATCH":
                qpos[0, 0, 0] += 0.02
            else:
                qpos[0, 0, 0] = 5.0
        return flags, qpos

    engine.robot.compute_batch_ik.side_effect = failing
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    assert batch.valid_mask.tolist() == [[False, True], [True, True]]
    assert (
        batch.failure_stages[0][0] == ("pre_grasp", "grasp", "translated")[stage_index]
    )
    assert batch.failure_reasons[0][0] == failure
    assert torch.isfinite(batch.qpos_anchors).all()
    plan = engine.plan_candidate(
        invocation, batch.select(torch.tensor((0, 1))), context
    )
    assert plan.plan_success.tolist() == [False, True]
    torch.testing.assert_close(
        plan.joint_trajectory.positions[0],
        context.robot.qpos[0].expand_as(plan.joint_trajectory.positions[0]),
    )
    assert failure in plan.diagnostics.metadata["candidate_failure_reasons"][0]


@pytest.mark.parametrize("stage", ("reach", "push", "return"))
def test_strict_cartesian_checks_cover_push_return_and_hold_failed_rows(stage):
    engine, invocation, context, raw = _setup(direction="push")
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    lengths = Slide._motion_segment_lengths(80, 5, direction="push")
    stage_index = ("reach", "push", "return").index(stage)
    fault_at = 1 + sum(lengths[index + 1] - 2 for index in range(stage_index))
    solve = engine.robot.compute_batch_ik.side_effect
    count = 0

    def failing(**kwargs):
        nonlocal count
        assert torch.isfinite(kwargs["joint_seed"]).all()
        flags, qpos = solve(**kwargs)
        count += 1
        if count == fault_at:
            qpos[0, 0] = torch.nan
        return flags, qpos

    engine.robot.compute_batch_ik.side_effect = failing
    plan = engine.plan_candidate(
        invocation, batch.select(torch.tensor((0, 1))), context
    )
    assert plan.plan_success.tolist() == [False, True]
    assert plan.diagnostics.metadata["candidate_failure_reasons"] == (
        f"{stage}:IK_INVALID_RESULT",
        None,
    )
    torch.testing.assert_close(
        plan.joint_trajectory.positions[0],
        context.robot.qpos[0].expand_as(plan.joint_trajectory.positions[0]),
    )


def test_continuity_rejects_a_valid_fk_solution_on_another_joint_branch():
    engine, invocation, context, raw = _setup()
    fk = engine.robot.compute_batch_fk.side_effect

    def redundant_fk(**kwargs):
        values = kwargs["qpos"].clone()
        values[..., 5] = 0.0
        return fk(**{**kwargs, "qpos": values})

    engine.robot.compute_batch_fk.side_effect = redundant_fk
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    solve = engine.robot.compute_batch_ik.side_effect

    def discontinuous(**kwargs):
        flags, qpos = solve(**kwargs)
        qpos[0, 0, 5] = 1.0  # Valid FK and limits, but an unacceptable seed jump.
        return flags, qpos

    engine.robot.compute_batch_ik.side_effect = discontinuous
    plan = engine.plan_candidate(
        invocation, batch.select(torch.tensor((0, 1))), context
    )
    assert plan.plan_success.tolist() == [False, True]
    assert (
        plan.diagnostics.metadata["candidate_failure_reasons"][0]
        == "reach:BRANCH_DISCONTINUITY"
    )


@pytest.mark.parametrize("empty", (False, True))
def test_empty_or_inactive_candidates_never_invoke_ik(empty):
    engine, invocation, context, raw = _setup()
    active = torch.zeros(NUM_ENVS, dtype=torch.bool)
    if empty:
        raw = GraspCandidateBatch(
            raw.poses[:, :0], raw.costs[:, :0], raw.valid_mask[:, :0]
        )
        active[:] = True
    batch = engine.enumerate_candidates(
        invocation, context, grasp_candidates=raw, active_mask=active
    )
    engine.robot.compute_batch_ik.assert_not_called()
    plan = engine.plan_candidate(
        invocation, batch.select(torch.tensor((-1, -1))), context
    )
    assert not plan.plan_success.any()
    assert not plan.expected_effects.held_object_updates


def test_sampling_only_active_rows_uses_current_link_axis_and_no_winner():
    engine, invocation, context, raw = _setup()
    sampler = engine.actions["slide"].planning_services.grasp_pose_generator("hand")
    sampler.get_valid_grasp_poses = Mock(return_value=[(raw.poses[0], raw.costs[0])])
    active = torch.tensor((True, False))
    batch = engine.enumerate_candidates(invocation, context, active_mask=active)
    assert batch.valid_mask.tolist() == [[True, True], [False, False]]
    assert sampler.get_valid_grasp_poses.call_args.kwargs["obj_poses"].shape == (
        1,
        4,
        4,
    )
    torch.testing.assert_close(
        sampler.get_valid_grasp_poses.call_args.kwargs["approach_direction"],
        torch.tensor(((0.0, 1.0, 0.0),)),
    )
    sampler.get_best_grasp_poses.assert_not_called()


@pytest.mark.parametrize("changed", ("qpos", "scene", "options"))
def test_selected_slide_rejects_stale_certificates(changed):
    engine, invocation, context, raw = _setup()
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    if changed == "qpos":
        context = context.project(qpos=context.robot.qpos + 0.01, task=context.task)
    elif changed == "scene":
        context = replace(context, scene=replace(context.scene, version=1))
    else:
        invocation = replace(
            invocation,
            skill_options=replace(invocation.skill_options, translation_distance=0.2),
        )
    with pytest.raises(ValueError, match="stale"):
        engine.plan_candidate(invocation, batch.select(torch.tensor((0, 1))), context)


def test_selected_slide_rejects_mutated_payload_and_unsupported_backend():
    engine, invocation, context, raw = _setup()
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    batch.grasp_poses[0, 0, 0, 3] += 0.01
    with pytest.raises(ValueError, match="modified"):
        batch.select(torch.tensor((0, 1)))
    invocation = replace(
        invocation,
        motion_policy=replace(invocation.motion_policy, strategy="motion_gen"),
    )
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    with pytest.raises(NotImplementedError, match="ik_interp"):
        engine.plan_candidate(invocation, batch.select(torch.tensor((0, 1))), context)


def test_compile_selects_slide_at_projected_start_and_never_reactivates_idle_rows():
    engine, invocation, context, raw = _setup()
    engine.register(MoveJoints())
    prefix = ActionInvocation(
        skill_id="move_joints",
        invocation_id="prefix",
        goal=JointPositionGoal(torch.full((NUM_ENVS, ARM_DOF), 0.01)),
        binding=engine.bind_control_parts(
            "move_joints", {"primary": {"motion": "arm"}}
        ),
        motion_policy=MotionPolicy(sample_count=2),
    )
    starts = []

    def select(request, projected, active):
        starts.append(projected.robot.qpos.clone())
        assert request.skill_id == "slide"
        assert active.tolist() == [True, False]
        batch = engine.enumerate_candidates(
            invocation, projected, grasp_candidates=raw, active_mask=active
        )
        return batch.select(torch.tensor((1, -1)))

    result = engine.compile(
        (prefix, invocation),
        context,
        candidate_selections={"slide-1": select},
        eligible_mask=torch.tensor((True, False)),
    )
    assert result.plan_success.tolist() == [True, False]
    torch.testing.assert_close(starts[0][0, :ARM_DOF], torch.full((ARM_DOF,), 0.01))
    torch.testing.assert_close(
        result.trajectory.positions[1],
        context.robot.qpos[1].expand_as(result.trajectory.positions[1]),
    )


def test_slide_axis_is_resolved_from_each_current_link_pose():
    engine, invocation, context, raw = _setup()
    link_pose = torch.eye(4).repeat(NUM_ENVS, 1, 1)
    link_pose[0, :3, :3] = torch.tensor(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    invocation = replace(
        invocation, goal=replace(invocation.goal, target_pose=link_pose)
    )
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    displacement = batch.pre_grasp_poses[..., :3, 3] - batch.grasp_poses[..., :3, 3]
    torch.testing.assert_close(displacement[0], torch.tensor(((0.1, 0.0, 0.0),) * 2))
    torch.testing.assert_close(displacement[1], torch.tensor(((0.0, -0.1, 0.0),) * 2))


def test_full_joint_limits_filter_the_hand_without_poisoning_peer_rows():
    engine, invocation, context, raw = _setup()
    limits = engine.robot.get_qpos_limits.side_effect

    def limited(**kwargs):
        result = limits(**kwargs)
        if kwargs.get("joint_ids") is None:
            result[0, ARM_DOF:, 1] = 0.5
        return result

    engine.robot.get_qpos_limits.side_effect = limited
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    assert batch.valid_mask.all()
    plan = engine.plan_candidate(
        invocation, batch.select(torch.tensor((0, 1))), context
    )
    assert plan.plan_success.tolist() == [False, True]
    assert (
        plan.diagnostics.metadata["candidate_failure_reasons"][0]
        == "trajectory:JOINT_LIMIT"
    )


def test_single_sample_hand_phases_issue_both_terminal_commands():
    engine, invocation, context, raw = _setup(hand_interp_steps=1)
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    plan = engine.plan_candidate(
        invocation, batch.select(torch.tensor((0, 1))), context
    )
    assert plan.plan_success.all()
    trajectory = plan.joint_trajectory.positions
    assert (trajectory[:, plan.segment("close").stop - 1, ARM_DOF:] == 1.0).all()
    assert (trajectory[:, plan.segment("open").stop - 1, ARM_DOF:] == 0.0).all()


def test_invalid_raw_geometry_is_masked_before_candidate_ik():
    engine, invocation, context, raw = _setup()
    poses = raw.poses.clone()
    poses[0, 0, 0, 0] = torch.nan
    raw = GraspCandidateBatch(poses, raw.costs, raw.valid_mask)
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    assert batch.valid_mask.tolist() == [[False, True], [True, True]]
    assert batch.failure_stages[0][0] == "grasp"
    assert torch.isfinite(batch.qpos_anchors).all()
