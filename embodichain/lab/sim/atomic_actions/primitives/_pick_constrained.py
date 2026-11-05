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

"""Explicit constrained-grasp safety and bounded roll/opening recovery."""

from __future__ import annotations
from dataclasses import replace
from typing import Any
import math
import torch
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.toolkits.graspkit.pg_grasp.constraints import (
    ConstrainedGraspPoseGenerator,
)
from embodichain.utils.math import pose_inv
from ..bindings import JointPositionTarget
from ..plans import PlanningFailure
from ._manipulation_checks import inspect_manipulation_plan, inspect_robot_motion
from ._pick_adaptive import current_pick_state

__all__: list[str] = []


def owns_constrained_pick(request, context, generator):
    goal, options = request.goal, request.skill_options
    return (
        request.skill_id == "pick_up"
        and options.constrained_grasp
        and goal.grasp_constraint is not None
        and context.batch_size == 1
        and context.env_ids.tolist() == [0]
        and goal.grasp_xpos is None
        and options.fixed_object_to_eef is None
        and options.rotate_upright is None
        and options.obj_upright_direction is None
        and options.pick_object_part == "center"
        and not options.downstream_object_target_poses
        and request.motion_policy.strategy == "ik_interp"
        and request.motion_policy.dynamic_collision_mode == "off"
    )


def prepend_pick_opening(
    action: Any, request: Any, context: Any, plan: Any
) -> tuple[Any | None, dict[str, Any]]:
    """Repair only a qualified free-hand opening jump with real timed commands."""
    from embodichain.compute.trajectory import differentiate_positions
    from embodichain.lab.sim.atomic_actions.plans import TimedTrajectory
    from embodichain.lab.sim.atomic_actions.trajectory_ops import interpolate_hand_qpos

    report: dict[str, Any] = {"accepted": False}
    try:
        if (
            not owns_constrained_pick(request, context, action.motion_generator)
            or not plan.plan_success.all()
        ):
            return None, {**report, "reason": "unowned_or_failed_plan"}
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        arm, hand = (
            endpoint.require_target(JointPositionTarget) for endpoint in (motion, grasp)
        )
        if context.task.get_held_object(motion.task_state_key) is not None or any(
            motion.task_state_key in keys
            and (held.env_mask is None or held.env_mask.any())
            for keys, held in context.task.coordinated_held_objects.items()
        ):
            return None, {**report, "reason": "hand_is_not_free"}
        strict = inspect_manipulation_plan(action.robot, request, context, plan)
        if strict.get("reason") != "initial_velocity_rejected":
            return None, {
                **report,
                "reason": "not_hand_opening_boundary",
                "original_certificate": strict,
            }
        trajectory = plan.joint_trajectory
        positions, dt = trajectory.positions, trajectory.dt
        ids = list(hand.joint_ids)
        opened = grasp.joint_positions(
            "open", num_envs=1, device=positions.device, dtype=positions.dtype
        )
        control_dt = context.require_control_dt()
        if (
            tuple(action.robot.get_joint_ids(hand.control_part))
            != tuple(hand.joint_ids)
            or not torch.equal(positions[:, 0, ids], opened)
            or not torch.equal(dt[:, :1], dt.new_zeros((1, 1)))
            or not torch.equal(dt[:, 1:], dt.new_full(dt[:, 1:].shape, control_dt))
            or trajectory.accelerations is not None
            or not plan.segments
            or plan.segments[0].name != "approach"
        ):
            return None, {**report, "reason": "opening_target_or_timing_unavailable"}
        original_collision = inspect_robot_motion(
            action.robot, positions[0], arm.control_part
        )
        if not original_collision["valid"]:
            return None, {
                **report,
                "reason": "original_collision_rejected",
                "collision": original_collision,
            }
        observed_collision = inspect_robot_motion(
            action.robot, context.robot.qpos, arm.control_part
        )
        if not observed_collision["valid"]:
            return None, {
                **report,
                "reason": "observed_start_collision_rejected",
                "collision": observed_collision,
            }
        limits = authored_joint_velocity_limits(action.robot, None).to(positions)[
            context.env_ids
        ][:, ids]
        if not torch.isfinite(limits).all() or (limits <= 0).any():
            return None, {**report, "reason": "opening_velocity_limits_unavailable"}
        observed = context.robot.qpos[:, ids].to(positions)
        intervals = math.ceil(
            float(((observed - opened).abs() / (limits * control_dt)).amax())
        )
        frames = intervals + 1
        report.update(
            intervals=intervals,
            prefix_frame_count=frames,
            duration=intervals * control_dt,
            frame_budget=request.skill_options.hand_interp_steps,
            joint_ids=ids,
            observed_hand_qpos=observed.tolist(),
            open_qpos=opened.tolist(),
            velocity_limits=limits.tolist(),
        )
        if intervals < 1 or frames > request.skill_options.hand_interp_steps:
            return None, {**report, "reason": "opening_prefix_budget_exceeded"}
        prefix_positions = (
            context.robot.qpos.to(positions)[:, None].expand(-1, frames, -1).clone()
        )
        prefix_positions[:, :, ids] = interpolate_hand_qpos(
            observed, opened, n_waypoints=frames
        )
        bounds = action.robot.get_qpos_limits(
            name=hand.control_part, env_ids=context.env_ids
        ).to(positions)
        if (
            bounds.shape != (1, len(ids), 2)
            or not torch.isfinite(bounds).all()
            or (bounds[..., 0] > bounds[..., 1]).any()
        ):
            return None, {**report, "reason": "opening_position_limits_unavailable"}
        first_desired = observed.clamp(bounds[..., 0], bounds[..., 1])
        report.update(
            position_limits_source="Robot.get_qpos_limits(name=hand, env_ids=context.env_ids)",
            effective_position_limits=bounds.tolist(),
            first_desired_hand_qpos=first_desired.tolist(),
        )
        boundary = torch.stack((observed, first_desired), dim=1)
        boundary_dt = dt.new_tensor([[0.0, control_dt]])
        if not joint_velocity_validity(boundary, boundary_dt, limits).all():
            return None, {**report, "reason": "first_projection_velocity_rejected"}
        # Project only the first desired command. Keep actual observations and
        # every later observed-to-open interpolation sample unchanged.
        prefix_positions[:, 0, ids] = first_desired
        prefix = TimedTrajectory.from_uniform_step(
            prefix_positions, env_ids=context.env_ids, step_dt=control_dt
        )
        if trajectory.velocities is not None:
            velocities = (
                torch.zeros_like(prefix_positions)
                if request.motion_policy.velocity_targets == "zero"
                else differentiate_positions(prefix_positions, prefix.dt)
            )
            velocities[:, -1] = 0
            prefix = replace(prefix, velocities=velocities)
        combined = TimedTrajectory.concatenate((prefix, trajectory))
        commands = action._joint_command_sequence(
            request, combined, active_mask=plan.plan_success
        )
        candidate = replace(
            plan,
            joint_trajectory=combined,
            commands=commands,
            tracking=action._tracking_sequence(request, commands),
            segments=tuple(
                replace(
                    segment,
                    start=segment.start + frames if index else 0,
                    stop=segment.stop + frames,
                )
                for index, segment in enumerate(plan.segments)
            ),
            scene_dependency_monitor_until={
                entity: bound + frames if bound else 0
                for entity, bound in plan.scene_dependency_monitor_until.items()
            },
        )
        certificate = inspect_manipulation_plan(
            action.robot, request, context, candidate
        )
        report.update(accepted=certificate["valid"], command_certificate=certificate)
        if not certificate["valid"]:
            return None, report
        return (
            replace(
                candidate,
                diagnostics=replace(
                    candidate.diagnostics,
                    metadata={
                        **candidate.diagnostics.metadata,
                        "gen_sim_pick_opening": report,
                    },
                ),
            ),
            report,
        )
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
        OverflowError,
    ) as error:
        return None, {**report, "unavailable": str(error)}


def recover_constrained_pick(self, request, context, first):
    """Keep failure-only roll recovery; reject robot-unsafe successful Pick paths."""
    goal, options = request.goal, request.skill_options
    affordance = goal.semantics.affordance
    if not owns_constrained_pick(request, context, self.motion_generator):
        return first
    endpoint = request.binding.endpoint("primary", "motion")
    held = first.expected_effects.held_object_updates.get(endpoint.task_state_key)
    if held is None or held.semantics.entity_id != goal.semantics.entity_id:
        return (
            replace(
                first,
                plan_success=torch.zeros_like(first.plan_success),
                diagnostics=replace(
                    first.diagnostics,
                    failure=PlanningFailure(
                        "pick_attachment_unavailable", retryable=False
                    ),
                ),
            )
            if first.plan_success.any()
            else first
        )

    calibration = options.grasp_frame_to_eef.to(held.grasp_xpos)
    original_grasp = held.grasp_xpos @ pose_inv(calibration)
    grasp_pose = original_grasp.clone()
    grasp_pose[..., :3, :2] *= -1
    choices = [(grasp_pose, None, "grasp_z_roll_180")]
    object_pose = held.grasp_xpos @ pose_inv(held.object_to_eef)
    hand = request.binding.endpoint("primary", "grasp").require_target(
        JointPositionTarget
    )
    original_success = bool(first.plan_success.all())
    attempts = []

    def finish(
        plan, reason, alternate=None, candidate_id=None, variant="grasp_z_roll_180"
    ):
        accepted = reason == "accepted"
        metadata = dict(plan.diagnostics.metadata)
        if accepted and alternate is not None:
            samples = dict(first.diagnostics.metadata.get("affordance_sample", {}))
            samples["grasp"] = {
                **samples.get("grasp", {}),
                "selected_poses": alternate.detach().cpu().tolist(),
            }
            if candidate_id is not None:
                samples["grasp"]["candidate_ids"] = [candidate_id]
            metadata["affordance_sample"] = samples
        details = {
            "accepted": accepted,
            "reason": reason,
            "variant": variant,
            "original_plan_success": first.plan_success.tolist(),
        }
        metadata[
            (
                "gen_sim_pick_safety"
                if original_success
                else "gen_sim_pick_roll_recovery"
            )
        ] = {**details, "attempts": attempts}
        return replace(
            plan,
            plan_success=(
                plan.plan_success if accepted else torch.zeros_like(plan.plan_success)
            ),
            diagnostics=replace(
                plan.diagnostics,
                metadata=metadata,
                **(
                    {
                        "failure": PlanningFailure(
                            "pick_robot_safety_failed", retryable=False
                        )
                    }
                    if original_success and not accepted
                    else {}
                ),
            ),
        )

    arm = endpoint.require_target(JointPositionTarget)
    start = inspect_robot_motion(self.robot, context.robot.qpos, arm.control_part)
    if not start["valid"]:
        attempts.append({"variant": "observed_start", **start})
        return finish(first, "start_collision_or_unavailable")
    if original_success:
        certificate = inspect_manipulation_plan(self.robot, request, context, first)
        if certificate["valid"]:
            return first
        attempts.append({"variant": "original", **certificate})
        if "unavailable" in certificate:
            return finish(first, "certificate_unavailable")
        if certificate.get("reason") == "initial_velocity_rejected":

            opened, opening_report = prepend_pick_opening(self, request, context, first)
            attempts.append({"variant": "observed_hand_opening", **opening_report})
            if opened is not None:
                return finish(opened, "accepted", variant="observed_hand_opening")
        pool = current_pick_state().candidates
        if pool is not None:
            indices = torch.argsort(
                pool.costs[0].masked_fill(~pool.valid[0], torch.inf), stable=True
            )
            for index in indices[pool.valid[0, indices]][:4].tolist():
                pose = pool.poses[:, index] @ pose_inv(calibration)
                mirror = pose.clone()
                mirror[..., :3, :2] *= -1
                choices.extend(
                    (candidate, index, f"candidate_{index}_{roll}")
                    for candidate, roll in (
                        (pose, "original"),
                        (mirror, "z_roll_180"),
                    )
                    if not any(
                        torch.equal(candidate, previous)
                        for previous in [
                            original_grasp,
                            *(choice[0] for choice in choices),
                        ]
                    )
                )
    # Every alternate is a complete shared PickUp plan; keep its proposed
    # attachment and emitted commands together and restore all extra draws.
    state = PlanningRandomState.capture((self.device, self.robot.device))
    try:
        generator = ConstrainedGraspPoseGenerator(
            self.planning_services.grasp_pose_generator(hand.target_id),
            goal.grasp_constraint,
        )
        for grasp_pose, candidate_id, variant in choices:
            alternate = grasp_pose @ calibration
            reason = "grasp_filter_rejected"
            if generator.accepts_grasp_poses(
                grasp_pose,
                object_pose,
                affordance.mesh_vertices,
                affordance.mesh_triangles,
            ).all():
                direction = options.approach_direction.to(alternate)
                direction = direction / torch.linalg.vector_norm(direction)
                reason = "approach_alignment_rejected"
                if self._approach_alignment_mask(
                    alternate[:, None, None], options, direction
                ).all():
                    selected = replace(
                        request, goal=replace(goal, grasp_xpos=alternate)
                    )
                    second = self._finish_native_plan(
                        selected, context, self._plan_pick(selected, context)
                    )
                    certificate = (
                        inspect_manipulation_plan(self.robot, request, context, second)
                        if second.plan_success.all()
                        else {"valid": False, "reason": "planning_failed"}
                    )
                    attempts.append({"variant": variant, **certificate})
                    if certificate["valid"]:
                        return finish(
                            second, "accepted", alternate, candidate_id, variant
                        )
                    reason = certificate.get("reason", "robot_collision_rejected")
                    if "unavailable" in certificate:
                        return finish(first, "certificate_unavailable")
            if not original_success:
                return finish(first, reason)
        return finish(first, "safe_candidates_exhausted")
    finally:
        state.restore()
