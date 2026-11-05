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

"""Action-owned bounded free-yaw selection qualified against canonical Place."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
import math
from typing import Any
import torch
from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.mesh_collision import robot_collision_query_scope
from ..bindings import JointPositionTarget
from ..plans import PlanningFailure
from ..place_continuation import PlaceContinuationSession
from .move_held_object import HeldObjectPoseGoal
from ._held_alignment import qualify_alignment_plan
from ._held_staging import inspect_staging_plan, staging_binding
from ._manipulation_checks import inspect_manipulation_plan, inspect_robot_motion

__all__: list[str] = []
_OFFSETS = tuple(
    math.pi * value for value in (0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1)
)
_UNAVAILABLE = (
    ImportError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    TypeError,
)


def effective_sample_count(plan, request):
    return plan.diagnostics.metadata.get("motion_sample_retries", {}).get(
        "sample_count", request.motion_policy.sample_count
    )


def eligible(action, request, context, preview):
    if preview is None or context.is_preview:
        return False
    try:
        provider = action.planning_services.retained_command_provider
        return bool(
            request.invocation_id == preview.route.source_invocation_id
            and request.motion_policy.sample_count == 260
            and staging_binding(action, request, context) is not None
            and provider is not None
            and provider.authority(request) is not None
        )
    except _UNAVAILABLE:
        return False


def _same(before: Any, after: Any) -> bool:
    if isinstance(before, torch.Tensor):
        return (
            isinstance(after, torch.Tensor)
            and before.dtype == after.dtype
            and before.device == after.device
            and torch.equal(before, after)
        )
    if type(before) is not type(after):
        return False
    if hasattr(before, "equivalent_to"):
        return before.equivalent_to(after)
    if is_dataclass(before) and not isinstance(before, type):
        return all(
            _same(getattr(before, field.name), getattr(after, field.name))
            for field in fields(before)
        )
    if isinstance(before, Mapping):
        return before.keys() == after.keys() and all(
            _same(value, after[key]) for key, value in before.items()
        )
    if isinstance(before, (tuple, list)):
        return len(before) == len(after) and all(
            _same(a, b) for a, b in zip(before, after)
        )
    return bool(before == after)


def exact_requests(request: Any) -> tuple[tuple[float, Any], ...]:
    result = []
    for angle in _OFFSETS:
        poses = request.goal.object_target_pose.clone()
        cosine, sine = math.cos(angle), math.sin(angle)
        yaw = poses.new_tensor([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
        poses[:, -1, :3, :3] = yaw @ poses[:, -1, :3, :3]
        result.append(
            (
                angle,
                replace(
                    request,
                    goal=replace(
                        request.goal, object_target_pose=poses, world_yaw_free=False
                    ),
                ),
            )
        )
    return tuple(result)


def _velocity_payload_valid(robot: Any, context: Any, plan: Any) -> bool:
    trajectory = plan.joint_trajectory
    if not torch.equal(
        trajectory.dt[:, :1], torch.zeros_like(trajectory.dt[:, :1])
    ) or not torch.equal(
        trajectory.dt[:, 1:], torch.full_like(trajectory.dt[:, 1:], context.control_dt)
    ):
        return False
    if trajectory.velocities is None:
        return True
    velocities = trajectory.velocities
    limits = authored_joint_velocity_limits(robot, None).to(velocities)[context.env_ids]
    return bool(
        velocities.shape == trajectory.positions.shape
        and torch.isfinite(velocities).all()
        and (velocities.abs() <= limits[:, None] + 1e-5).all()
    )


def inspect_alignment(
    action: Any, original: Any, selected: Any, context: Any, plan: Any
) -> dict[str, Any]:
    """The original factory authority survives only exact legal yaw selection."""
    try:
        controls = (
            "skill_id",
            "invocation_id",
            "revision",
            "binding",
            "motion_policy",
            "tracking_policy",
            "recovery_policy",
            "skill_options",
            "phase_effect_gates",
        )
        expected, actual = (
            original.goal.object_target_pose,
            selected.goal.object_target_pose,
        )
        if (
            type(selected.goal) is not HeldObjectPoseGoal
            or any(
                not _same(getattr(original, name), getattr(selected, name))
                for name in controls
            )
            or actual.shape != expected.shape
            or not torch.equal(actual[:, :-1], expected[:, :-1])
            or not torch.equal(actual[..., :3, 3], expected[..., :3, 3])
            or not torch.equal(actual[..., 3, :], expected[..., 3, :])
            or not any(
                torch.allclose(
                    actual[:, -1, :3, :3],
                    expected.new_tensor(
                        [
                            [math.cos(angle), -math.sin(angle), 0],
                            [math.sin(angle), math.cos(angle), 0],
                            [0, 0, 1],
                        ]
                    )
                    @ expected[:, -1, :3, :3],
                    atol=1e-6,
                    rtol=0,
                )
                for angle in _OFFSETS
            )
            or plan.invocation_id != original.invocation_id
            or plan.skill_id != original.skill_id
            or not bool(plan.plan_success.all())
            or effective_sample_count(plan, selected) != 260
        ):
            raise ValueError("alignment_controls_or_heading_changed")
        checked = qualify_alignment_plan(selected, context, plan, action.robot, None)
        if not bool(checked.plan_success.all()):
            return {"valid": False, "reason": "alignment_final_commands_rejected"}
        arm = selected.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        observed = inspect_robot_motion(
            action.robot, context.robot.qpos, arm.control_part
        )
        if not observed["valid"]:
            return {"valid": False, "reason": "alignment_observed_start_rejected"}
        acceptance = inspect_staging_plan(
            action,
            selected,
            context,
            plan,
        )
        if not acceptance["valid"] or not _velocity_payload_valid(
            action.robot, context, plan
        ):
            return {
                "valid": False,
                "reason": acceptance.get(
                    "reason", "alignment_velocity_payload_rejected"
                ),
            }
        key = selected.binding.endpoint("primary", "motion").task_state_key
        held = context.task.get_held_object(key)
        updates = dict(plan.expected_effects.held_object_updates)
        if (
            set(updates) != {key}
            or updates[key] is None
            or updates[key].semantics.entity_id != held.semantics.entity_id
            or not torch.equal(updates[key].object_to_eef, held.object_to_eef)
            or not _same(updates[key].grasp_xpos, held.grasp_xpos)
            or plan.expected_effects.coordinated_held_object_updates
            or plan.expected_effects.articulation_joint_updates
        ):
            raise ValueError("alignment_retention_effect_changed")
        return {
            "valid": True,
            "retained_hand_boundary": acceptance.get("retained_hand_boundary", {}),
        }
    except _UNAVAILABLE as error:
        return {"valid": False, "reason": str(error)}


def inspect_place(
    action: Any,
    request: Any,
    context: Any,
    plan: Any,
    preview: PlaceContinuationSession,
) -> dict[str, Any]:
    """A private endpoint prediction is never a measured hold or release effect."""
    try:
        route = preview.route
        preview.validate_source(context)
        motion = request.binding.endpoint("primary", "motion")
        arm = motion.require_target(JointPositionTarget)
        held = context.task.get_held_object(motion.task_state_key)
        if (
            held.semantics.entity_id != route.object_id
            or not context.task.exclusive_held_object_mask(motion.task_state_key).all()
        ):
            raise ValueError("continuation_hold_identity_changed")
        final = plan.joint_trajectory.positions[:, -1].clone()
        eef = action.robot.compute_fk(
            qpos=final[:, list(arm.joint_ids)], name=arm.control_part, to_matrix=True
        ).to(final)
        pose = eef @ torch.linalg.inv(held.object_to_eef.to(final))
        if pose.shape != (1, 4, 4) or not torch.isfinite(pose).all():
            raise ValueError("continuation_endpoint_fk_unavailable")
        entities = dict(context.scene.entities)
        observed = entities[held.semantics.entity_id]
        if observed.confidence <= 0:
            raise ValueError("continuation_object_observation_unavailable")
        entities[held.semantics.entity_id] = replace(
            observed, pose=pose[0] if observed.pose.shape == (4, 4) else pose
        )
        predicted = replace(
            context.project(qpos=final, task=context.task),
            scene=replace(context.scene, entities=entities),
            is_preview=True,
        )
        place_request = preview.resolve_place(predicted)
        if (
            place_request.skill_id != "place"
            or place_request.motion_policy.sample_count != 260
        ):
            raise ValueError("canonical_place_policy_changed")
        for name in ("motion", "grasp"):
            before, after = (
                value.binding.endpoint("primary", name)
                for value in (request, place_request)
            )
            if (
                before.task_state_key != after.task_state_key
                or before.target != after.target
            ):
                raise ValueError("canonical_place_endpoint_changed")
        place = preview.plan(place_request, predicted)
        if not bool(place.plan_success.all()):
            return {"valid": False, "reason": "canonical_place_planning_failed"}
        certificate = inspect_manipulation_plan(
            action.robot, place_request, predicted, place
        )
        if not certificate["valid"] or not _velocity_payload_valid(
            action.robot, predicted, place
        ):
            return {
                "valid": False,
                "reason": certificate.get("reason", "place_velocity_payload_rejected"),
            }
        hand_endpoint = place_request.binding.endpoint("primary", "grasp")
        hand = hand_endpoint.require_target(JointPositionTarget)
        opened = hand_endpoint.joint_positions(
            "open", num_envs=1, device=final.device, dtype=final.dtype
        )
        if (
            place.invocation_id != route.place_invocation_id
            or place.skill_id != "place"
            or effective_sample_count(place, place_request) != 260
            or not torch.equal(
                place.joint_trajectory.positions[:, -1, list(hand.joint_ids)], opened
            )
            or dict(place.expected_effects.held_object_updates)
            != {hand_endpoint.task_state_key: None}
            or place.expected_effects.coordinated_held_object_updates
            or place.expected_effects.articulation_joint_updates
        ):
            raise ValueError("canonical_place_open_or_release_changed")
        return {
            "valid": True,
            "conditional_on_future_measured_state": True,
            "physical_success": False,
        }
    except _UNAVAILABLE as error:
        return {"valid": False, "reason": str(error)}


def _annotate(plan: Any, report: dict[str, Any], *, failed: bool = False) -> Any:
    return replace(
        plan,
        plan_success=(
            torch.zeros_like(plan.plan_success) if failed else plan.plan_success
        ),
        diagnostics=replace(
            plan.diagnostics,
            metadata={
                **plan.diagnostics.metadata,
                "gen_sim_alignment_continuation": report,
            },
            **(
                {
                    "failure": PlanningFailure(
                        "alignment_continuation_failed", retryable=False
                    )
                }
                if failed
                else {}
            ),
        ),
    )


def recover(
    action: Any,
    request: Any,
    context: Any,
    original: Any,
    *,
    preview: PlaceContinuationSession | None,
    build: Callable[[Any], Any],
    before_original: PlanningRandomState | None = None,
    planned_request: Any = None,
) -> Any:
    """Return only an Align plan; leave actual Place to the unchanged executor."""
    if not eligible(action, request, context, preview) or not bool(
        original.plan_success.all()
    ):
        return original
    stop = PlanningRandomState.capture((action.robot.device, context.robot.qpos.device))
    report = {
        "accepted": False,
        "conditional_on_future_measured_state": True,
        "physical_success": False,
        "place_invocation_id": preview.route.place_invocation_id,
        "attempts": [],
    }
    try:
        with robot_collision_query_scope():
            selected_original = request if planned_request is None else planned_request
            certificate = inspect_alignment(
                action, request, selected_original, context, original
            )
            if not certificate["valid"]:
                return _annotate(
                    original, {**report, "reason": certificate["reason"]}, failed=True
                )
            stop.restore()
            place = inspect_place(action, selected_original, context, original, preview)
            if place["valid"]:
                return original
            report["original_place_reason"] = place.get("reason")
            for angle, selected in exact_requests(request):
                (before_original or stop).restore()
                entry = {"yaw_offset_rad": angle}
                report["attempts"].append(entry)
                try:
                    candidate = build(selected)
                    certificate = inspect_alignment(
                        action, request, selected, context, candidate
                    )
                    entry.update(alignment_valid=certificate["valid"])
                    if not certificate["valid"]:
                        entry["reason"] = certificate.get("reason")
                        continue
                    stop.restore()
                    place = inspect_place(action, selected, context, candidate, preview)
                    entry.update(place)
                    if place["valid"]:
                        report.update(
                            accepted=True,
                            yaw_offset_rad=angle,
                            retained_hand_boundary=certificate.get(
                                "retained_hand_boundary", {}
                            ),
                        )
                        return _annotate(candidate, report)
                except _UNAVAILABLE as error:
                    entry.update(valid=False, reason=str(error))
            return _annotate(original, report, failed=True)
    finally:
        stop.restore()
