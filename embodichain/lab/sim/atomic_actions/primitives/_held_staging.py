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

"""Declared rigid staging hints and retained-command planning qualification."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
import torch
from embodichain.compute.geometry.pose_checks import as_rigid_matrix as _pose
from embodichain.compute.geometry.staging import distal_staging_variants
from embodichain.lab.sim.scene.robot_distal import RobotDistalMeshSnapshot
from embodichain.lab.sim.evidence import RobotCommandObservation
from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from ..bindings import JointPositionTarget
from ..planning_probes import HeldStagingBinding, RetainedCommandAuthority
from .move_held_object import HeldObjectPoseGoal
from ._manipulation_checks import inspect_manipulation_plan, inspect_robot_motion

__all__: list[str] = []
_UNAVAILABLE = (
    ImportError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    TypeError,
)


def inspect_staging_plan(
    action: Any,
    request: Any,
    context: Any,
    plan: Any,
    *,
    require_retained_command: bool = False,
) -> dict[str, Any]:
    """Distinguish a new hand command from an unchanged native retained target."""
    robot = action.robot
    strict = inspect_manipulation_plan(robot, request, context, plan)
    if (strict["valid"] and not require_retained_command) or (
        not strict["valid"] and strict.get("reason") != "initial_velocity_rejected"
    ):
        return strict
    provider = action.planning_services.retained_command_provider
    authority = None if provider is None else provider.authority(request)
    if authority is None:
        return (
            {
                "valid": False,
                "reason": "retained_previous_command_authority_unavailable",
            }
            if require_retained_command
            else strict
        )
    if type(authority) is not RetainedCommandAuthority:
        raise TypeError("Retained command providers must return typed authority.")
    retention_source = authority.metadata()
    proof = {
        **retention_source,
        "accepted": False,
        "scope": "prior_applied_retained_hand_command",
        "source_api": "Robot.get_qpos(target=True)/get_qvel(target=True)",
    }
    try:
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        arm, hand = (
            endpoint.require_target(JointPositionTarget) for endpoint in (motion, grasp)
        )
        held = context.task.get_held_object(motion.task_state_key)
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or held is None
            or held.semantics.entity_id != retention_source["entity_id"]
            or getattr(motion, "resource_id", motion.task_state_key)
            != retention_source["resource_id"]
            or motion.task_state_key != grasp.task_state_key
            or not context.task.exclusive_held_object_mask(motion.task_state_key).all()
            or tuple(robot.get_joint_ids(hand.control_part)) != tuple(hand.joint_ids)
        ):
            raise ValueError("retained_hand_ownership_unavailable")
        hand_ids = list(hand.joint_ids)

        commands = provider.capture(request, context)
        if (
            type(commands) is not RobotCommandObservation
            or commands.robot_uid != robot.uid
            or commands.joint_names != tuple(robot.joint_names)
            or commands.timestamp != context.robot.timestamp
            or commands.source_stamp != context.world_source_stamp
        ):
            raise ValueError("retained_native_command_snapshot_unavailable")
        observed, velocity, desired, desired_velocity = (
            commands.qpos,
            commands.qvel,
            commands.target_qpos,
            commands.target_qvel,
        )
        proof.update(
            control_part=hand.control_part,
            joint_ids=hand_ids,
            actual_qpos=observed[:, hand_ids].tolist(),
            actual_qvel=velocity[:, hand_ids].tolist(),
            prior_target_qpos=desired[:, hand_ids].tolist(),
            prior_target_qvel=desired_velocity[:, hand_ids].tolist(),
            motion_control_part=arm.control_part,
            motion_joint_ids=list(arm.joint_ids),
            task_state_key=motion.task_state_key,
        )
        if not torch.equal(observed, context.robot.qpos) or not torch.equal(
            velocity, context.robot.qvel
        ):
            raise ValueError("retained_observation_changed")
        limits = authored_joint_velocity_limits(robot, None).to(velocity)[
            context.env_ids
        ]
        proof["velocity_limits"] = limits[:, hand_ids].tolist()
        within_limits = not bool(
            (velocity[:, hand_ids].abs() > limits[:, hand_ids]).any()
        )
        if not within_limits:
            raise ValueError("retained_actual_hand_velocity_rejected")
        command = grasp.joint_positions(
            "grasp", num_envs=1, device=desired.device, dtype=desired.dtype
        )
        positions = plan.joint_trajectory.positions
        if not torch.equal(desired[:, hand_ids], command) or not torch.equal(
            positions[:, :, hand_ids],
            desired[:, None, hand_ids].expand(-1, positions.shape[1], -1),
        ):
            raise ValueError("retained_hand_target_changed")
        for frame in plan.commands.frames:
            payload = next(
                item.payload for item in frame.commands if item.target == hand
            )
            payload_velocity = (
                torch.zeros_like(payload.positions)
                if payload.velocities is None
                else payload.velocities
            )
            if not torch.equal(payload_velocity, desired_velocity[:, hand_ids]):
                raise ValueError("retained_hand_target_velocity_changed")
        initial = inspect_robot_motion(robot, observed, arm.control_part)
        proof["observed_start_collision"] = initial
        if not initial["valid"]:
            raise ValueError("retained_observed_start_collision_or_unavailable")
        boundary = observed.clone()
        boundary[:, hand_ids] = desired[:, hand_ids]
        certificate = inspect_manipulation_plan(
            robot, request, context, plan, _initial_command_qpos=boundary
        )
        provider.verify_current(commands)
        proof["accepted"] = certificate["valid"]
        return {**certificate, "retained_hand_boundary": proof}
    except _UNAVAILABLE as error:
        return {"valid": False, "reason": str(error), "retained_hand_boundary": proof}


@dataclass(frozen=True, slots=True)
class StagingPreparation:
    before_original: PlanningRandomState
    binding: HeldStagingBinding


def prepare_staging(
    action: Any, request: Any, context: Any
) -> StagingPreparation | None:
    """Capture source RNG only after pure staging ownership has been established."""
    binding = staging_binding(action, request, context)
    return (
        None
        if binding is None
        else StagingPreparation(
            PlanningRandomState.capture(
                (action.device, action.robot.device, context.robot.qpos.device)
            ),
            binding,
        )
    )


def staging_binding(
    action: Any, request: Any, context: Any
) -> HeldStagingBinding | None:
    """Decide ownership before planning without geometry work or random draws."""
    provider = action.planning_services.held_staging_provider
    if provider is None:
        raise ValueError("Declared staging recovery requires its binding provider.")
    binding = provider.binding(request)
    if binding is None:
        return None
    if type(binding) is not HeldStagingBinding:
        raise TypeError("Staging providers must return HeldStagingBinding.")
    try:
        if (
            type(request.goal) is not HeldObjectPoseGoal
            or request.goal.world_yaw_free is not True
            or not isinstance(request.goal.object_target_pose, torch.Tensor)
            or request.goal.object_target_pose.shape != (1, 2, 4, 4)
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.robot.qpos.ndim != 2
            or request.motion_policy.strategy != "ik_interp"
            or request.motion_policy.dynamic_collision_mode != "off"
        ):
            return None
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        arm, hand = (
            endpoint.require_target(JointPositionTarget) for endpoint in (motion, grasp)
        )
        ids = (*arm.joint_ids, *hand.joint_ids)
        held = context.task.get_held_object(motion.task_state_key)
        eligible = (
            motion.task_state_key == grasp.task_state_key
            and bool(arm.joint_ids)
            and bool(hand.joint_ids)
            and len(set(ids)) == len(ids)
            and min(ids) >= 0
            and max(ids) < context.robot.qpos.shape[-1]
            and held is not None
            and held.semantics.entity_id == binding.rigid_entity_id
            and [
                key
                for key in context.task.held_objects
                if context.task.held_object_mask(key).any()
            ]
            == [motion.task_state_key]
            and not any(
                state.env_mask is None or state.env_mask.any()
                for state in context.task.coordinated_held_objects.values()
            )
            and bool(
                context.task.exclusive_held_object_mask(motion.task_state_key).all()
            )
        )
        return binding if eligible else None
    except _UNAVAILABLE:
        return None


def build_staging_variants(
    action: Any, request: Any, context: Any
) -> list[tuple[str, Any, dict[str, Any]]]:
    robot = action.robot
    targets = request.goal.object_target_pose
    motion = request.binding.endpoint("primary", "motion")
    arm = motion.require_target(JointPositionTarget)
    if tuple(robot.get_joint_ids(arm.control_part)) != tuple(arm.joint_ids):
        raise ValueError("staging_active_joint_ownership_changed")
    held = context.task.get_held_object(motion.task_state_key)
    observed = context.scene.entities.get(held.semantics.entity_id)
    if observed is None or observed.confidence <= 0:
        raise ValueError("staging_object_observation_unavailable")
    object_pose = observed.pose.to(targets).reshape(1, 4, 4)
    attachment = _pose(held.object_to_eef)
    for pose in (*targets[0], object_pose[0]):
        _pose(pose)
    if (
        not torch.allclose(
            targets[:, 0, :3, :3], object_pose[:, :3, :3], atol=1e-6, rtol=0
        )
        or (targets[:, 0, 2, 3] < object_pose[:, 2, 3]).any()
    ):
        raise ValueError("staging_changed_acquisition_orientation_or_height")
    observed_robot = (
        context.observed_robot if context.observed_robot is not None else context.robot
    )
    if not torch.equal(context.robot.qpos, observed_robot.qpos):
        raise ValueError("staging_observed_qpos_changed")
    snapshot = action.planning_services.held_staging_provider.capture_distal_geometry(
        request, context
    )
    if (
        type(snapshot) is not RobotDistalMeshSnapshot
        or snapshot.active != arm.control_part
    ):
        raise ValueError("staging_distal_geometry_ownership_unavailable")
    return [
        (
            label,
            (
                None
                if poses is None
                else replace(
                    request, goal=replace(request.goal, object_target_pose=poses)
                )
            ),
            report,
        )
        for label, poses, report in distal_staging_variants(
            targets,
            object_pose,
            attachment,
            active=snapshot.active,
            passive=snapshot.passive,
            tcp=snapshot.tcp,
            roots=snapshot.roots,
            points=snapshot.points,
            hand_owner_fk_error=snapshot.hand_owner_fk_error,
        )
    ]


def recover_staging(
    action: Any,
    original: Any,
    request: Any,
    context: Any,
    scope: Any,
    before_original: PlanningRandomState | None,
    build: Callable[[Any], Any],
) -> Any:
    robot = action.robot
    if (
        before_original is None
        or original.plan_success.any()
        or scope is None
        or not scope.rejected_raw_success
        or len(scope.calls) != 8
        or not scope.fallback_attempted
    ):
        return original
    # Exit the original query lifetime/checkpoint just as the source scope did.
    # The outer owner may lazily open a separate final certificate afterwards.
    scope.close()
    scope.query = None
    scope.articulation_query = None
    after_legacy = PlanningRandomState.capture(
        (robot.device, context.robot.qpos.device)
    )
    attempts = []
    selected = original
    selected_boundary = None
    try:
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        initial = inspect_robot_motion(robot, context.robot.qpos, arm.control_part)
        if not initial["valid"]:
            attempts.append({"variant": "observed_start", **initial})
        else:
            for label, candidate_request, report in build_staging_variants(
                action, request, context
            ):
                if candidate_request is None:
                    attempts.append({**report, "accepted": False})
                    continue
                before_original.restore()
                candidate = build(candidate_request)
                # The canonical candidate builder already performs this final
                # alignment check; do not duplicate the source query/draw order.
                certificate = (
                    inspect_staging_plan(
                        action,
                        candidate_request,
                        context,
                        candidate,
                    )
                    if candidate.plan_success.all()
                    else {"valid": False, "reason": "planning_or_alignment_gate_failed"}
                )
                attempts.append(
                    {
                        **report,
                        "accepted": certificate["valid"],
                        "command_certificate": certificate,
                    }
                )
                if certificate["valid"]:
                    selected = candidate
                    selected_boundary = certificate.get("retained_hand_boundary")
                    break
                if "unavailable" in certificate:
                    break
    except _UNAVAILABLE as error:
        attempts.append({"accepted": False, "unavailable": str(error)})
    finally:
        after_legacy.restore()
    return replace(
        selected,
        diagnostics=replace(
            selected.diagnostics,
            metadata={
                **selected.diagnostics.metadata,
                "staging_recovery": {
                    "accepted": selected is not original,
                    "attempts": attempts,
                    **(
                        {"retained_hand_boundary": selected_boundary}
                        if selected_boundary is not None
                        else {}
                    ),
                },
            },
        ),
    )
