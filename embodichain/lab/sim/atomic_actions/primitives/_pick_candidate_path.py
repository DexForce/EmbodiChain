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

"""Complete candidate contact paths for the bounded ordinary Pick search."""

from __future__ import annotations

from typing import Any
import torch

from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import interpolate_xpos_batched
from embodichain.lab.sim.motion.ik import (
    sample_cartesian_targets,
    resolve_continuous_ik_path,
)
from embodichain.lab.sim.motion.solvers.tolerances import pose_validation_tolerances
from embodichain.utils.math import pose_inv
from ..bindings import JointPositionTarget
from ..affordance_sampling import AffordancePoseCandidates
from ..goals import _resolve_object_pose, resolve_pose_goal
from ..trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    split_three_segments,
)
from ._pick_path import _pose_path_valid

__all__: list[str] = []


def geometric_variants(action, pool, request, context, direction):
    mirrored = pool.poses.clone()
    mirrored[..., :3, :2] *= -1
    variants = torch.stack((pool.poses, mirrored), dim=2)
    object_pose = _resolve_object_pose(request.goal.semantics, context).to(variants)
    variants = action._upright_adjusted_grasp_poses(
        variants, object_pose, request.skill_options
    )
    valid = (pool.valid & torch.isfinite(pool.costs))[..., None].expand(-1, -1, 2)
    variants = variants @ request.skill_options.grasp_frame_to_eef.to(variants)
    valid = valid & torch.isfinite(variants).all(dim=(-1, -2))
    return AffordancePoseCandidates(
        variants.flatten(1, 2),
        pool.costs[..., None].expand(-1, -1, 2).flatten(1, 2),
        valid.flatten(1, 2),
    )


def trajectory(
    action: Any,
    request: Any,
    context: Any,
    grasp: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[tuple | None, dict[str, Any]]:
    """Resolve complete Pick samples once, preserving phase timing and endpoints."""
    options = request.skill_options
    arm = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    hand_endpoint = request.binding.endpoint("primary", "grasp")
    hand = hand_endpoint.require_target(JointPositionTarget)
    arm_ids, hand_ids = list(arm.joint_ids), list(hand.joint_ids)
    dt = context.require_control_dt()
    n_approach, n_close, n_lift = split_three_segments(
        request.motion_policy.sample_count,
        options.hand_interp_steps,
        first_segment_name="approach",
        third_segment_name="lift",
    )
    total = n_approach + n_close + options.grasp_settle_steps + n_lift
    budget = request.recovery_policy.action_timeout
    if budget is not None and (total - 1) * dt > budget:
        return None, {"valid": False, "reason": "action_duration_budget"}
    seed = context.robot.qpos[:, arm_ids]
    start = action.robot.compute_fk(qpos=seed, name=arm.control_part, to_matrix=True)
    pre = grasp.clone()
    pre[:, :3, 3] -= direction * options.pre_grasp_distance
    lift = grasp.clone()
    lift[:, 2, 3] += options.lift_height
    targets = sample_cartesian_targets(
        start, build_pose_plan_states(torch.stack((pre, grasp), 1)), n_approach
    )
    motion = MotionGenOptions(
        strategy="ik_interp",
        # The Cartesian grid is already explicit. Retain the declared solver
        # port: source sampled contracts are sequential, shared contracts batched.
        interpolation_mode=(
            "fixed_grid"
            if request.motion_policy.ik_seed_mode == "sequential"
            else "adaptive"
        ),
        ik_seed_mode=request.motion_policy.ik_seed_mode,
        acceleration_limit=None,
        start_qpos=seed,
        control_part=arm.control_part,
        sample_count=n_approach,
        interpolation_dt=dt,
        preserve_cartesian_samples=True,
        is_linear=True,
    )
    approach = resolve_continuous_ik_path(
        action.motion_generator, targets, motion, operation="Pick candidate approach"
    )
    if approach is None:
        return None, {"valid": False, "reason": "approach_continuity_rejected"}
    desired_approach = torch.stack((start, *(target.xpos for target in targets)), 1)
    desired_lift = interpolate_xpos_batched(grasp, lift, n_lift)
    motion.start_qpos = approach.positions[:, -1]
    motion.sample_count = n_lift
    raised = resolve_continuous_ik_path(
        action.motion_generator,
        build_pose_plan_states(desired_lift[:, 1:]),
        motion,
        operation="Pick candidate lift",
    )
    if raised is None:
        return None, {"valid": False, "reason": "lift_continuity_rejected"}
    tolerances = pose_validation_tolerances(action.robot.get_solver(arm.control_part))
    for actual, desired in (
        (approach.positions, desired_approach),
        (raised.positions, desired_lift),
    ):
        fk = action.robot.compute_batch_fk(
            qpos=actual, name=arm.control_part, to_matrix=True
        )
        if not _pose_path_valid(fk, desired, *tolerances):
            return None, {"valid": False, "reason": "cartesian_samples_rejected"}
    if not torch.equal(approach.positions[:, -1], raised.positions[:, 0]):
        return None, {"valid": False, "reason": "grasp_branch_changed"}
    receipt = downstream_endpoints(
        action, request, context, grasp, raised.positions[:, -1]
    )
    if not receipt["valid"]:
        return None, receipt
    opened, closed = (
        hand_endpoint.joint_positions(
            name, num_envs=1, device=seed.device, dtype=seed.dtype
        )
        for name in ("open", "grasp")
    )
    full = context.robot.qpos[:, None].expand(-1, total, -1).clone()
    close_stop = n_approach + n_close + options.grasp_settle_steps
    full[:, :n_approach, arm_ids] = approach.positions
    full[:, :n_approach, hand_ids] = opened[:, None]
    full[:, n_approach:close_stop, arm_ids] = approach.positions[:, -1:]
    full[:, n_approach : n_approach + n_close, hand_ids] = interpolate_hand_qpos(
        opened, closed, n_waypoints=n_close
    )
    full[:, n_approach + n_close :, hand_ids] = closed[:, None]
    full[:, close_stop:, arm_ids] = raised.positions
    lengths = {
        "approach": n_approach,
        "close": n_close + options.grasp_settle_steps,
        "lift": n_lift,
    }
    return (torch.ones(1, dtype=torch.bool, device=seed.device), full, lengths), {
        "valid": True,
        "downstream": receipt,
        "phase_samples": lengths,
        "timing_unchanged": True,
        "qualification_scope": "continuous_pick_cartesian_samples_not_physical_acquisition",
    }


def downstream_endpoints(
    action: Any, request: Any, context: Any, grasp: torch.Tensor, seed: torch.Tensor
) -> dict[str, Any]:
    """Keep untimed look-ahead as endpoint IK/FK, never certify a future path."""
    arm = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    object_pose = _resolve_object_pose(request.goal.semantics, context)
    object_to_eef = pose_inv(object_pose.to(grasp)) @ grasp
    bounds = action.robot.get_qpos_limits(
        name=arm.control_part, env_ids=context.env_ids
    ).to(seed)
    if (
        bounds.shape != (*seed.shape, 2)
        or torch.isnan(bounds).any()
        or (bounds[..., 0] > bounds[..., 1]).any()
    ):
        return {"valid": False, "reason": "downstream_limits_unavailable"}
    tolerances = pose_validation_tolerances(action.robot.get_solver(arm.control_part))
    count = 0
    for index, value in enumerate(request.skill_options.downstream_object_target_poses):
        target = resolve_pose_goal(
            value, context, name=f"downstream_object_target_poses[{index}]"
        ).to(grasp)
        if target.shape == (4, 4):
            target = target[None]
        if target.shape != grasp.shape:
            raise ValueError("Pick downstream target must match the environment batch.")
        desired = target @ object_to_eef
        ok, qpos = action.robot.compute_batch_ik(
            pose=desired[:, None], name=arm.control_part, joint_seed=seed[:, None]
        )
        qpos = qpos.reshape_as(seed)
        if not bool(
            ok.all()
            and torch.isfinite(qpos).all()
            and (qpos >= bounds[..., 0]).all()
            and (qpos <= bounds[..., 1]).all()
        ):
            return {
                "valid": False,
                "reason": "downstream_endpoint_ik_rejected",
                "target_index": index,
            }
        fk = action.robot.compute_fk(qpos=qpos, name=arm.control_part, to_matrix=True)
        if not _pose_path_valid(fk, desired, *tolerances):
            return {
                "valid": False,
                "reason": "downstream_endpoint_fk_rejected",
                "target_index": index,
            }
        seed, count = qpos, count + 1
    return {
        "valid": True,
        "target_count": count,
        "qualification_scope": "untimed_endpoint_ik_fk_only",
        "continuous_path_qualified": False,
    }
