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

"""Continuous qualification for one selected GenSim grasp or a legal raw pool."""

from __future__ import annotations

import math
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import AntipodalAffordance, JointPositionTarget
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
)
from embodichain.lab.sim.atomic_actions.goals import (
    _resolve_object_pose,
    resolve_pose_goal,
)
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    split_three_segments,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import interpolate_xpos_batched
from embodichain.utils.math import pose_inv

from .adaptive_grasp import ADAPTIVE_GRASP
from .motion import _cartesian_samples, resolve_continuous_ik_path
from .pick_path import _pose_path_valid
from .pick_safety import _PICK_GRASP_RULE

__all__: list[str] = []


def owns_fixed(action: Any, request: Any, context: Any) -> bool:
    """Own only an ordinary fixed grasp, never another call's grasp contract."""
    if not action.is_bound:
        return False
    options, goal = request.skill_options, request.goal
    affordance = goal.semantics.affordance
    invocation_id = getattr(request, "invocation_id", None)
    return bool(
        options.fixed_object_to_eef is not None
        and (getattr(action, "pick_purposes", None) or {}).get(invocation_id)
        == "ordinary"
        and invocation_id not in getattr(action, "path_excluded_invocations", ())
        and context.batch_size == 1
        and context.env_ids.tolist() == [0]
        and goal.object_pose is None
        and goal.grasp_xpos is None
        and isinstance(affordance, AntipodalAffordance)
        and affordance.get_custom_config(_PICK_GRASP_RULE) is None
        and affordance.get_custom_config(ADAPTIVE_GRASP) in {None, "ordinary"}
        and options.rotate_upright is None
        and options.obj_upright_direction is None
        and options.pick_object_part == "center"
        and request.motion_policy.strategy == "ik_interp"
        and request.motion_policy.dynamic_collision_mode == "off"
        and not getattr(action.motion_generator, "direct_transport_planning", False)
    )


def eligible(action: Any, request: Any, context: Any, selection: Any) -> bool:
    """Select by the current call's geometry and motion contract, not E family."""
    options, goal = request.skill_options, request.goal
    return bool(
        getattr(action, "path_simulation", None) is not None
        and context.batch_size == 1
        and context.env_ids.tolist() == [0]
        and selection.raw_candidate_routes
        and getattr(selection, "pour_preview", None) is None
        and goal.object_pose is None
        and goal.grasp_xpos is None
        and options.fixed_object_to_eef is None
        and options.pick_object_part == "center"
        and request.motion_policy.strategy == "ik_interp"
        and request.motion_policy.dynamic_collision_mode == "off"
        and not getattr(action.motion_generator, "direct_transport_planning", False)
    )


def geometric_variants(
    action: Any,
    pool: AffordancePoseCandidates,
    request: Any,
    context: Any,
    direction: torch.Tensor,
    *,
    qualify: bool = True,
) -> AffordancePoseCandidates:
    """Retain source geometry validity and calibrated, aligned symmetric rolls."""
    mirrored = pool.poses.clone()
    mirrored[..., :3, :2] *= -1
    variants = torch.stack((pool.poses, mirrored), dim=2)
    options = request.skill_options
    object_pose = _resolve_object_pose(request.goal.semantics, context).to(variants)
    variants = action._upright_adjusted_grasp_poses(variants, object_pose, options)
    valid = (pool.valid & torch.isfinite(pool.costs))[..., None].expand(-1, -1, 2)
    rule = request.goal.semantics.affordance.get_custom_config(_PICK_GRASP_RULE)
    if rule is not None:
        from .grasp_filter import TaskGraspPoseGenerator

        hand = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        generator = action.planning_services.grasp_pose_generator(hand.target_id)
        if not isinstance(generator, TaskGraspPoseGenerator):
            raise ValueError("Constrained Pick requires its original geometry filter.")
        affordance = request.goal.semantics.affordance
        scoped = generator.for_pick(
            *rule, affordance.mesh_vertices, affordance.mesh_triangles
        )
        valid = valid & scoped.accepts_grasp_poses(
            variants.reshape(-1, 4, 4),
            object_pose[:, None, None].expand_as(variants).reshape(-1, 4, 4),
            affordance.mesh_vertices,
            affordance.mesh_triangles,
        ).reshape_as(valid)
    variants = variants @ options.grasp_frame_to_eef.to(variants)
    valid = valid & torch.isfinite(variants).all(dim=(-1, -2))
    if qualify:
        valid = valid & action._approach_alignment_mask(
            variants, options, direction[:, None, None]
        )
    return AffordancePoseCandidates(
        poses=variants.flatten(1, 2),
        costs=pool.costs[..., None].expand(-1, -1, 2).flatten(1, 2),
        valid=valid.flatten(1, 2),
    )


def _tolerances(robot: Any, part: str) -> tuple[float, float]:
    parameters = robot.get_solver(part).get_iteration_params()
    values = parameters["pos_eps"], parameters["rot_eps"]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Pick IK pose tolerances are unavailable.")
    return values


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
    targets = _cartesian_samples(
        start, build_pose_plan_states(torch.stack((pre, grasp), 1)), n_approach
    )
    motion = MotionGenOptions(
        strategy="ik_interp",
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
    tolerances = _tolerances(action.robot, arm.control_part)
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
    tolerances = _tolerances(action.robot, arm.control_part)
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
