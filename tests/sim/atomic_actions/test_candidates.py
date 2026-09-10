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

"""Candidate choices preserve row state, Cartesian geometry, and IK branches."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    GraspGoal,
    JointPositionGoal,
    JointPositionTarget,
    MotionPolicy,
    MoveJoints,
    PickUp,
    PickUpOptions,
)
from embodichain.toolkits.graspkit.candidates import GraspCandidateBatch
from embodichain.utils.math import (
    axis_angle_from_quat,
    axis_angle_to_rotation_matrix,
    quat_from_matrix,
)
from .test_actions import (
    ARM_DOF,
    NUM_ENVS,
    ROBOT_DOF,
    _ACTION_ENGINES,
    _bind_action,
    _binding,
    _context,
    _motion_generator,
    _semantics,
    _target_scene,
    _torch_interpolation,
)


def _setup(*, options: PickUpOptions | None = None):
    """Use a reversible XYZ/rotation-vector robot to test real geometric checks."""
    generator = _motion_generator()
    robot = generator.robot

    def fk(*, qpos, name=None, env_ids=None, to_matrix=True):
        del name, env_ids, to_matrix
        result = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        result[..., :3, 3] = qpos[..., :3]
        result[..., :3, :3] = axis_angle_to_rotation_matrix(
            qpos[..., 3:6].reshape(-1, 3)
        ).reshape(*qpos.shape[:-1], 3, 3)
        return result

    def ik(*, pose, joint_seed, name=None, env_ids=None):
        del name, env_ids
        qpos = joint_seed.clone()
        qpos[..., :3] = pose[..., :3, 3]
        qpos[..., 3:6] = axis_angle_from_quat(quat_from_matrix(pose[..., :3, :3]))
        return torch.ones(pose.shape[:2], dtype=torch.bool), qpos

    def limits(*, joint_ids=None, env_ids=None):
        rows = NUM_ENVS if env_ids is None else len(env_ids)
        dof = ROBOT_DOF if joint_ids is None else len(joint_ids)
        return torch.tensor((-4.0, 4.0)).expand(rows, dof, 2).clone()

    robot.compute_batch_fk.side_effect = fk
    robot.compute_fk.side_effect = fk
    robot.compute_batch_ik.side_effect = ik
    robot.get_qpos_limits.side_effect = limits
    action = _bind_action(generator, PickUp())
    engine = _ACTION_ENGINES[id(action)]
    invocation = ActionInvocation(
        skill_id="pick_up",
        invocation_id="pick",
        goal=GraspGoal(_semantics(entity_id="target")),
        binding=_binding(action),
        motion_policy=MotionPolicy(sample_count=80),
        skill_options=options
        or PickUpOptions(
            pre_grasp_distance=0.1, lift_height=0.15, grasp_settle_steps=2
        ),
    )
    context = _context(
        scene=_target_scene(
            torch.eye(4).repeat(NUM_ENVS, 1, 1), timestamp=0.0, version=0
        )
    )
    poses = torch.eye(4).repeat(NUM_ENVS, 2, 1, 1)
    poses[..., 0, 3] = torch.tensor((0.1, 0.2))
    batch = GraspCandidateBatch(
        poses=poses,
        costs=torch.tensor(((1.0, 2.0), (1.0, 2.0))),
        valid_mask=torch.ones(NUM_ENVS, 2, dtype=torch.bool),
        grasp_ids=(("a", "b"), ("a", "b")),
    )
    return engine, invocation, context, batch


def test_enumeration_retains_all_grasps_rolls_and_joint_anchors() -> None:
    engine, invocation, context, raw = _setup()
    result = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    assert result.valid_mask.tolist() == [[True] * 4] * NUM_ENVS
    assert result.variant_ids.tolist() == [[0, 1, 0, 1]] * NUM_ENVS
    assert result.grasp_ids == (("a", "a", "b", "b"),) * NUM_ENVS
    assert result.qpos_anchors.shape == (NUM_ENVS, 4, 3, ARM_DOF)
    torch.testing.assert_close(
        result.qpos_anchors[..., 0, 2], torch.full((NUM_ENVS, 4), 0.1)
    )
    torch.testing.assert_close(
        result.qpos_anchors[..., 2, 2], torch.full((NUM_ENVS, 4), 0.15)
    )


def test_selected_materialization_preserves_cartesian_segments_and_anchors() -> None:
    engine, invocation, context, raw = _setup()
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    selection = batch.select(torch.tensor((0, 3)))
    plan = engine.plan_candidate(invocation, selection, context)
    assert plan.plan_success.all(), plan.diagnostics.metadata
    assert [segment.name for segment in plan.segments] == [
        "transit",
        "approach",
        "close",
        "lift",
    ]
    trajectory = plan.joint_trajectory.positions
    torch.testing.assert_close(trajectory[:, 0], context.robot.qpos)
    chosen = batch.qpos_anchors[torch.arange(NUM_ENVS), selection.indices]
    for segment_name, anchor_index in (("transit", 0), ("approach", 1), ("lift", 2)):
        segment = plan.segment(segment_name)
        torch.testing.assert_close(
            trajectory[:, segment.stop - 1, :ARM_DOF], chosen[:, anchor_index]
        )
    for name in ("approach", "lift"):
        segment = plan.segment(name)
        arm = trajectory[:, segment.start : segment.stop, :ARM_DOF]
        poses = engine.robot.compute_batch_fk(qpos=arm)
        torch.testing.assert_close(
            poses[..., :2, 3], poses[:, :1, :2, 3].expand(-1, poses.shape[1], -1)
        )
        torch.testing.assert_close(
            poses[..., :3, :3], poses[:, :1, :3, :3].expand_as(poses[..., :3, :3])
        )


def test_calibration_and_upright_are_applied_once() -> None:
    calibration = torch.eye(4)
    calibration[0, 3] = 0.025
    options = PickUpOptions(grasp_frame_to_eef=calibration, rotate_upright=0.2)
    engine, invocation, context, raw = _setup(options=options)
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    plan = engine.plan_candidate(
        invocation, batch.select(torch.tensor((0, 0))), context
    )
    assert plan.plan_success.all(), plan.diagnostics.metadata
    held = plan.expected_effects.held_object_updates["arm"]
    torch.testing.assert_close(held.grasp_xpos, batch.grasp_poses[:, 0])
    torch.testing.assert_close(held.grasp_xpos[:, 0, 3], torch.full((NUM_ENVS,), 0.125))


@pytest.mark.parametrize(
    "failure", ("IK_NOT_FOUND", "IK_INVALID_RESULT", "FK_MISMATCH", "JOINT_LIMIT")
)
def test_candidate_failure_is_local_and_never_poisons_next_seed(failure: str) -> None:
    engine, invocation, context, raw = _setup()
    solve = engine.robot.compute_batch_ik.side_effect
    call_count = 0

    def failing(**kwargs):
        nonlocal call_count
        assert torch.isfinite(kwargs["joint_seed"]).all()
        flags, qpos = solve(**kwargs)
        call_count += 1
        if call_count == 1:
            if failure == "IK_NOT_FOUND":
                flags[0, 0] = False
            elif failure == "IK_INVALID_RESULT":
                qpos[0, 0] = torch.nan
            elif failure == "FK_MISMATCH":
                qpos[0, 0, 0] += 0.03
            else:
                qpos[0, 0, 0] = 5.0
        return flags, qpos

    engine.robot.compute_batch_ik.side_effect = failing
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    assert not batch.valid_mask[0, 0]
    assert batch.valid_mask[0, 1:].all() and batch.valid_mask[1].all()
    assert batch.failure_stages[0][0] == "pre_grasp"
    assert batch.failure_reasons[0][0] == failure
    assert torch.isfinite(batch.qpos_anchors).all()


def test_idle_and_empty_rows_never_invoke_ik_or_create_effects() -> None:
    engine, invocation, context, raw = _setup()
    active = torch.zeros(NUM_ENVS, dtype=torch.bool)
    batch = engine.enumerate_candidates(
        invocation, context, grasp_candidates=raw, active_mask=active
    )
    engine.robot.compute_batch_ik.assert_not_called()
    plan = engine.plan_candidate(
        invocation, batch.select(torch.tensor((-1, -1))), context
    )
    assert not plan.plan_success.any()
    assert not plan.expected_effects.held_object_updates
    empty = replace(
        raw,
        poses=raw.poses[:, :0],
        costs=raw.costs[:, :0],
        valid_mask=raw.valid_mask[:, :0],
        grasp_ids=((), ()),
        rejection_reasons=(),
    )
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=empty)
    assert batch.valid_mask.shape == (NUM_ENVS, 0)
    engine.robot.compute_batch_ik.assert_not_called()


@pytest.mark.parametrize("change", ("qpos", "scene", "options"))
def test_stale_evaluation_cannot_be_materialized(change: str) -> None:
    engine, invocation, context, raw = _setup()
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    if change == "qpos":
        context = context.project(qpos=context.robot.qpos + 0.01, task=context.task)
    elif change == "scene":
        context = replace(context, scene=replace(context.scene, version=1))
    else:
        invocation = replace(
            invocation, skill_options=replace(invocation.skill_options, lift_height=0.2)
        )
    with pytest.raises(ValueError, match="stale"):
        engine.plan_candidate(invocation, batch.select(torch.tensor((0, 0))), context)


def test_compile_callback_uses_projected_start_and_masks_idle_effects() -> None:
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
    seen = []

    def choose(request, projected, active):
        seen.append(projected.robot.qpos.clone())
        assert request.invocation_id == "pick"
        assert active.tolist() == [True, False]
        batch = engine.enumerate_candidates(
            invocation, projected, grasp_candidates=raw, active_mask=active
        )
        return batch.select(torch.tensor((0, -1)))

    result = engine.compile(
        (prefix, invocation),
        context,
        candidate_selections={"pick": choose},
        eligible_mask=torch.tensor((True, False)),
    )
    assert result.plan_success.tolist() == [True, False]
    torch.testing.assert_close(seen[0][0, :ARM_DOF], torch.full((ARM_DOF,), 0.01))
    torch.testing.assert_close(seen[0][1], context.robot.qpos[1])
    held = result.projected_context.get_held_object("arm")
    assert held.env_mask.tolist() == [True, False]


def test_unsupported_branch_backend_is_explicit() -> None:
    engine, invocation, context, raw = _setup()
    invocation = replace(
        invocation,
        motion_policy=replace(invocation.motion_policy, strategy="motion_gen"),
    )
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    with pytest.raises(NotImplementedError, match="ik_interp"):
        engine.plan_candidate(invocation, batch.select(torch.tensor((0, 0))), context)


def test_modified_anchor_payload_cannot_reuse_evaluation_certificate() -> None:
    engine, invocation, context, raw = _setup()
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    selection = batch.select(torch.tensor((0, 0)))
    selection.candidates.qpos_anchors[0, 0, 1, 0] += 0.01
    with pytest.raises(ValueError, match="payload was modified"):
        engine.plan_candidate(invocation, selection, context)


def test_compile_all_idle_never_calls_a_skill() -> None:
    engine, invocation, context, raw = _setup()
    result = engine.compile(
        (invocation,), context, eligible_mask=torch.zeros(NUM_ENVS, dtype=torch.bool)
    )
    assert result.trajectory.positions.shape == (NUM_ENVS, 0, ROBOT_DOF)
    assert not result.plan_success.any()
    assert not result.action_plans
    engine.robot.compute_batch_ik.assert_not_called()


@pytest.mark.parametrize(
    ("invalid_output", "message"),
    (
        ("flags_nan", "success flags"),
        ("flags_two", "success flags"),
        ("qpos_integer", "floating dtype"),
        ("qpos_double", "floating dtype"),
        ("flags_device", "share the seed device"),
        ("qpos_device", "share the seed device"),
        ("flags_shape", "incompatible batch dimensions"),
        ("qpos_shape", "incompatible batch dimensions"),
    ),
)
def test_backend_contract_errors_are_not_cast_into_candidate_success(
    invalid_output: str, message: str
) -> None:
    engine, invocation, context, raw = _setup()
    solve = engine.robot.compute_batch_ik.side_effect

    def malformed(**kwargs):
        flags, qpos = solve(**kwargs)
        if invalid_output == "flags_nan":
            flags = flags.to(torch.float32)
            flags[0, 0] = torch.nan
        elif invalid_output == "flags_two":
            flags = flags.to(torch.int32)
            flags[0, 0] = 2
        elif invalid_output == "qpos_integer":
            qpos = qpos.to(torch.long)
        elif invalid_output == "qpos_double":
            qpos = qpos.to(torch.float64)
        elif invalid_output == "flags_device":
            flags = flags.to("meta")
        elif invalid_output == "qpos_device":
            qpos = qpos.to("meta")
        elif invalid_output == "flags_shape":
            flags = flags[:, :1]
        else:
            qpos = qpos[..., :-1]
        return flags, qpos

    engine.robot.compute_batch_ik.side_effect = malformed
    with pytest.raises(ValueError, match=message):
        engine.enumerate_candidates(invocation, context, grasp_candidates=raw)


@pytest.mark.parametrize("bad_seed", ("nonfinite", "outside_limits"))
def test_unsafe_candidate_seed_is_rejected_before_fk_or_ik(bad_seed: str) -> None:
    engine, invocation, context, raw = _setup()
    action = engine.actions["pick_up"]
    manipulator = invocation.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    seed = context.robot.qpos[:, None, :ARM_DOF].clone()
    seed[0, 0, 0] = torch.nan if bad_seed == "nonfinite" else 5.0
    with pytest.raises(ValueError, match="finite|outside joint limits"):
        action._checked_candidate_ik(
            raw.poses[:, :1],
            seed,
            manipulator,
            context,
            torch.ones(NUM_ENVS, 1, dtype=torch.bool),
        )
    engine.robot.compute_batch_fk.assert_not_called()
    engine.robot.compute_batch_ik.assert_not_called()


def test_binary_integer_backend_flags_remain_supported() -> None:
    engine, invocation, context, raw = _setup()
    solve = engine.robot.compute_batch_ik.side_effect

    def numeric_flags(**kwargs):
        flags, qpos = solve(**kwargs)
        return flags.to(torch.int32), qpos

    engine.robot.compute_batch_ik.side_effect = numeric_flags
    batch = engine.enumerate_candidates(invocation, context, grasp_candidates=raw)
    assert batch.valid_mask.all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_candidate_fingerprint_accepts_strided_single_joint_commands(dtype) -> None:
    from embodichain.lab.sim.atomic_actions.candidates import _value_fingerprint

    command = torch.arange(8).to(dtype)[::2][1:2]
    assert command.stride() == (2,)
    assert command.is_contiguous()  # Singleton axes allow a non-unit stride.
    packed = torch.tensor([2], dtype=dtype)
    assert _value_fingerprint(command) == _value_fingerprint(packed)
    assert _value_fingerprint(command) != _value_fingerprint(packed + 1)
