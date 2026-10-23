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

"""Robot-proxy qualification and bounded opening for owned constrained Pick."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.runtime_commands import JointPositionPayload

from . import articulation_collision
from .held_alignment_recovery import _RandomState, _cuda_devices

__all__: list[str] = []

_PICK_GRASP_RULE = "gen_sim.pick_grasp_rule"
PICK_OPENING_RECOVERY_REVISION = 3


def _owns_constrained_pick(request: Any, context: Any, generator: Any) -> bool:
    """Keep action screening and post-transform qualification in one owned scope."""
    if getattr(request, "skill_id", "pick_up") != "pick_up":
        return False
    goal, options = request.goal, request.skill_options
    affordance = goal.semantics.affordance
    return (
        (context is None or context.batch_size == 1)
        and goal.grasp_xpos is None
        and options.fixed_object_to_eef is None
        and options.rotate_upright is None
        and options.obj_upright_direction is None
        and options.pick_object_part == "center"
        and not options.downstream_object_target_poses
        and isinstance(affordance, AntipodalAffordance)
        and affordance.get_custom_config(_PICK_GRASP_RULE) is not None
        and request.motion_policy.strategy == "ik_interp"
        and request.motion_policy.dynamic_collision_mode == "off"
        and not getattr(generator, "direct_transport_planning", False)
    )


def _inspect_pick_motion(
    robot: Any, positions: torch.Tensor, part: str
) -> dict[str, Any]:
    state = _RandomState.capture(_cuda_devices(robot.device, positions.device))
    try:
        certificate = articulation_collision.check_robot_motion(robot, positions, part)
        if type(certificate) is not dict or type(certificate.get("valid")) is not bool:
            raise ValueError("Pick robot-proxy collision certificate is unavailable.")
        return certificate
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        return {"valid": False, "unavailable": str(error)}
    finally:
        state.restore()


def _inspect_pick_plan(
    robot: Any,
    request: Any,
    context: Any,
    plan: Any,
    *,
    _initial_command_qpos: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Certify full emitted approach/close/lift, not only grasp candidate IK."""
    from .motion import (
        _joint_velocity_limits,
        _velocity_diagnostics,
        _velocity_validity,
    )

    try:
        trajectory = plan.joint_trajectory
        positions, dt = trajectory.positions, trajectory.dt
        endpoints = tuple(
            request.binding.endpoint("primary", name) for name in ("motion", "grasp")
        )
        targets = tuple(
            endpoint.require_target(JointPositionTarget) for endpoint in endpoints
        )
        active = [index for target in targets for index in target.joint_ids]
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or positions.shape[1] == 0
            or positions.shape[-1] != context.robot.qpos.shape[-1]
            or dt.shape != positions.shape[:2]
            or context.env_ids.tolist() != [0]
            or not torch.isfinite(positions).all()
            or not torch.isfinite(dt).all()
            or (dt < 0).any()
            or len(set(active)) != len(active)
            or not active
            or min(active) < 0
            or max(active) >= positions.shape[-1]
            or endpoints[0].task_state_key != endpoints[1].task_state_key
        ):
            raise ValueError(
                "Pick full-robot trajectory or endpoint scope is unavailable."
            )
        passive = [index for index in range(positions.shape[-1]) if index not in active]
        if passive and not torch.equal(
            positions[:, :, passive],
            context.robot.qpos[:, None, passive].expand(-1, positions.shape[1], -1),
        ):
            raise ValueError("Pick changed an uncommanded joint.")
        if not torch.equal(
            positions[:, 0, list(targets[0].joint_ids)],
            context.robot.qpos[:, list(targets[0].joint_ids)],
        ):
            raise ValueError("Pick changed its zero-time arm start.")
        limits = _joint_velocity_limits(robot, None).to(positions)[context.env_ids]
        if not _velocity_validity(positions, dt, limits).all():
            velocity = _velocity_diagnostics(positions, dt, limits)
            for row in velocity:
                row["phase"] = next(
                    (s.name for s in plan.segments if s.start <= row["frame"] < s.stop),
                    None,
                )
            return {
                "valid": False,
                "reason": "final_velocity_rejected",
                "velocity": velocity,
            }
        for target in targets:
            bounds = robot.get_qpos_limits(
                name=target.control_part, env_ids=context.env_ids
            ).to(positions)
            values = positions[:, :, list(target.joint_ids)]
            if (
                bounds.shape != (1, len(target.joint_ids), 2)
                or torch.isnan(bounds).any()
                or (bounds[..., 0] > bounds[..., 1]).any()
            ):
                raise ValueError("Pick endpoint joint limits are unavailable.")
            if (
                (values < bounds[:, None, :, 0]) | (values > bounds[:, None, :, 1])
            ).any():
                return {"valid": False, "reason": "joint_limits_rejected"}
        if (
            plan.commands.frame_count != positions.shape[1]
            or not torch.equal(plan.commands.env_ids, context.env_ids)
            or not torch.equal(trajectory.env_ids, context.env_ids)
        ):
            raise ValueError("Pick command/trajectory scope differs.")
        for index, frame in enumerate(plan.commands.frames):
            if (
                len(frame.commands) != len(targets)
                or not torch.equal(frame.env_ids, context.env_ids)
                or not torch.equal(frame.active_mask, plan.plan_success)
                or not torch.equal(
                    frame.hold_duration, dt[:, min(index + 1, positions.shape[1] - 1)]
                )
            ):
                raise ValueError("Pick emitted a different command schedule.")
            seen = set()
            for command in frame.commands:
                if (
                    type(command.target) is not JointPositionTarget
                    or command.target not in targets
                    or command.target.control_part in seen
                    or type(command.payload) is not JointPositionPayload
                ):
                    raise ValueError("Pick emitted an unexpected command target.")
                seen.add(command.target.control_part)
                ids = list(command.target.joint_ids)
                if not torch.equal(command.payload.positions, positions[:, index, ids]):
                    raise ValueError(
                        "Pick command positions differ from its trajectory."
                    )
                velocities = command.payload.velocities
                if (velocities is None) != (trajectory.velocities is None) or (
                    velocities is not None
                    and (
                        not torch.equal(
                            velocities, trajectory.velocities[:, index, ids]
                        )
                        or not torch.isfinite(velocities).all()
                        or (velocities.abs() > limits[:, ids] + 1e-5).any()
                    )
                ):
                    raise ValueError("Pick command velocities are unqualified.")
        first_hold = plan.commands.frames[0].hold_duration
        boundary_start = context.robot.qpos.to(positions)
        if _initial_command_qpos is not None:
            hand_ids = list(targets[1].joint_ids)
            observed_ids = [
                index for index in range(positions.shape[-1]) if index not in hand_ids
            ]
            if (
                _initial_command_qpos.shape != boundary_start.shape
                or not torch.isfinite(_initial_command_qpos).all()
                or not torch.equal(
                    _initial_command_qpos[:, observed_ids],
                    boundary_start[:, observed_ids],
                )
                or not torch.equal(
                    positions[:, :, hand_ids],
                    _initial_command_qpos[:, None, hand_ids].expand(
                        -1, positions.shape[1], -1
                    ),
                )
            ):
                raise ValueError(
                    "Retained-hand boundary cannot replace robot observations."
                )
            boundary_start = _initial_command_qpos.to(positions)
        boundary_positions = torch.cat(
            (boundary_start[:, None], positions[:, :1]), dim=1
        )
        boundary_dt = torch.stack((torch.zeros_like(first_hold), first_hold), dim=1)
        if not _velocity_validity(boundary_positions, boundary_dt, limits).all():
            return {"valid": False, "reason": "initial_velocity_rejected"}
        return _inspect_pick_motion(robot, positions[0], targets[0].control_part)
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        return {"valid": False, "unavailable": str(error)}


def _prepend_pick_opening(
    action: Any, request: Any, context: Any, plan: Any
) -> tuple[Any | None, dict[str, Any]]:
    """Repair only a qualified free-hand opening jump with real timed commands."""
    from embodichain.compute.trajectory import differentiate_positions
    from embodichain.lab.sim.atomic_actions.plans import TimedTrajectory
    from embodichain.lab.sim.atomic_actions.trajectory_ops import interpolate_hand_qpos
    from .motion import _joint_velocity_limits, _velocity_validity

    report: dict[str, Any] = {"accepted": False}
    try:
        if (
            not _owns_constrained_pick(request, context, action.motion_generator)
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
        strict = _inspect_pick_plan(action.robot, request, context, plan)
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
        original_collision = _inspect_pick_motion(
            action.robot, positions[0], arm.control_part
        )
        if not original_collision["valid"]:
            return None, {
                **report,
                "reason": "original_collision_rejected",
                "collision": original_collision,
            }
        observed_collision = _inspect_pick_motion(
            action.robot, context.robot.qpos, arm.control_part
        )
        if not observed_collision["valid"]:
            return None, {
                **report,
                "reason": "observed_start_collision_rejected",
                "collision": observed_collision,
            }
        limits = _joint_velocity_limits(action.robot, None).to(positions)[
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
        if not _velocity_validity(boundary, boundary_dt, limits).all():
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
        certificate = _inspect_pick_plan(action.robot, request, context, candidate)
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
