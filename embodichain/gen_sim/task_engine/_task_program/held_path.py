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

"""Discrete carried-mesh qualification and bounded GenSim transport recovery."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.plans import PlanningFailure
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    resolve_pose_target,
    to_full_robot_trajectory,
)
from embodichain.utils.math import pose_inv, quat_error_magnitude, quat_from_matrix

from .held_alignment_recovery import _RandomState, _cuda_devices
from .motion import (
    _CARTESIAN_APPROACH,
    _cartesian_samples,
    resolve_continuous_ik_path,
    _velocity_retry_samples,
)
from .staging_recovery import inspect_staging_plan
from .scene_geometry import scene_components, world_bounds

__all__: list[str] = []

_CLEARANCE = 0.002
_UNAVAILABLE = (
    ImportError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    TypeError,
)


def _geometry(action: Any, request: Any, context: Any) -> tuple[Any, Any, Any]:
    endpoint = request.binding.endpoint("primary", "motion")
    arm = endpoint.require_target(JointPositionTarget)
    held = context.get_held_object(endpoint.task_state_key)
    meshes = action.held_path_scene_vertices
    if getattr(meshes, "unavailable", None):
        raise ValueError(meshes.unavailable)
    if (
        context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or held is None
        or not meshes
        or held.semantics.entity_id not in meshes
        or "table" not in meshes
        or not context.task.exclusive_held_object_mask(endpoint.task_state_key).all()
    ):
        raise ValueError("held_path_geometry_or_exclusive_resource_unavailable")
    return arm, held, meshes


def _world_bounds(
    mesh: torch.Tensor, pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return world_bounds(mesh, pose)


def _inspect_object_poses(
    action: Any, request: Any, context: Any, poses: torch.Tensor
) -> dict[str, Any]:
    """Apply the same bound-rigid geometry contract to targets and emitted FK."""
    try:
        _, held, meshes = _geometry(action, request, context)
        if (
            poses.ndim != 4
            or poses.shape[0] != 1
            or not poses.shape[1]
            or poses.shape[-2:] != (4, 4)
            or not torch.isfinite(poses).all()
        ):
            raise ValueError("held_path_object_pose_samples_unavailable")
        vertices = meshes[held.semantics.entity_id].to(poses)
        _world_bounds(vertices, poses[:, 0])
        world = (
            torch.einsum("btij,bvj->btvi", poses[:, :, :3, :3], vertices)
            + poses[:, :, None, :3, 3]
        )
        lower, upper = world.amin(2), world.amax(2)
        rejected, minimum_gap = [], None
        for uid, component, mesh, pose in scene_components(
            meshes, context, excluded_owner=held.semantics.entity_id
        ):
            other_lower, other_upper = _world_bounds(mesh, pose.to(poses))
            overlap = (
                (upper > other_lower[:, None] - _CLEARANCE)
                & (lower < other_upper[:, None] + _CLEARANCE)
            ).all(-1)
            if overlap.any():
                entry = {"entity": uid, "first_sample": int(overlap[0].nonzero()[0, 0])}
                if component != uid:
                    entry["component"] = component
                rejected.append(entry)
            if uid == "table":
                minimum_gap = float((lower[:, :, 2] - other_upper[:, None, 2]).min())
        return {
            "valid": not rejected,
            "scope": "discrete_carried_mesh_bound_scene_component_envelopes",
            "geometry_sources": "legacy_rigid_visual_vertices_and_articulation_collision_components",
            "not_certified": "continuous_collision_contact_robot_scene_or_payload_robot_clearance",
            "sample_count": poses.shape[1],
            "minimum_table_gap_m": minimum_gap,
            "rejected": rejected,
            **({"reason": "held_path_scene_envelope_rejected"} if rejected else {}),
        }
    except _UNAVAILABLE as error:
        return {"valid": False, "unavailable": str(error)}


def inspect_payload(
    action: Any, request: Any, context: Any, positions: torch.Tensor
) -> dict[str, Any]:
    """Measure every commanded carried mesh; endpoints alone are insufficient."""
    try:
        arm, held, _ = _geometry(action, request, context)
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or not positions.shape[1]
            or positions.shape[-1] != context.robot.qpos.shape[-1]
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("held_path_full_robot_samples_unavailable")
        fk = action.robot.compute_batch_fk(
            qpos=positions[:, :, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )
        if fk.shape != (*positions.shape[:2], 4, 4) or not torch.isfinite(fk).all():
            raise ValueError("held_path_fk_unavailable")
        attachment = held.object_to_eef.to(fk).reshape(1, 4, 4)
        return _inspect_object_poses(
            action, request, context, fk @ pose_inv(attachment)[:, None]
        )
    except _UNAVAILABLE as error:
        return {"valid": False, "unavailable": str(error)}


def inspect(action: Any, request: Any, context: Any, plan: Any) -> dict[str, Any]:
    if not plan.plan_success.all() or plan.joint_trajectory is None:
        return {"valid": False, "reason": "held_path_planning_failed"}
    payload = inspect_payload(action, request, context, plan.joint_trajectory.positions)
    if not payload["valid"]:
        return payload
    source = getattr(action, "held_command_sources", {}).get(request.invocation_id)
    revision = 1
    if source is not None:
        revision = source.get("command_continuity_revision", 1)
    else:
        source = action.staging_contract.retention_source(request.invocation_id)
    commands = inspect_staging_plan(
        action.robot,
        request,
        context,
        plan,
        retention_source=source,
        _contract_revision=revision,
    )
    return {**payload, "valid": commands["valid"], "commands": commands}


def _targets(action: Any, request: Any, context: Any) -> torch.Tensor:
    # Resolve with the canonical ordering, including world_orientation before relative_pose.
    poses = resolve_pose_target(
        resolve_pose_goal(
            request.goal.object_target_pose, context, name="object_target_pose"
        ),
        num_envs=1,
        device=action.robot.device,
    )
    return poses[:, None] if poses.ndim == 3 else poses


def phase_scope(action: Any, request: Any, context: Any) -> dict[str, Any]:
    """Own only geometrically separated exact endpoints, never contact targets.

    Legacy support-contact or unavailable phases retain their original route.
    This is target-owned scope, not a family/target-name inference or a new
    permitted-contact certificate.
    """
    try:
        _geometry(action, request, context)
        certificate = _inspect_object_poses(
            action, request, context, _targets(action, request, context)
        )
    except _UNAVAILABLE as error:
        certificate = {"valid": False, "unavailable": str(error)}
    owned = certificate["valid"]
    return {
        "owned": owned,
        "domain": "free_space" if owned else "not_migrated",
        "reason": (
            "required_endpoints_bound_rigid_separated"
            if owned
            else (
                "required_endpoint_geometry_unavailable"
                if "unavailable" in certificate
                else "required_endpoint_not_free_space"
            )
        ),
        "endpoint_check": certificate,
    }


def _lifted_targets(
    action: Any, request: Any, context: Any, targets: torch.Tensor, extra: float
) -> torch.Tensor:
    arm, held, meshes = _geometry(action, request, context)
    start = action.robot.compute_fk(
        qpos=context.robot.qpos[:, list(arm.joint_ids)],
        name=arm.control_part,
        to_matrix=True,
    )
    attachment = held.object_to_eef.to(targets).reshape(1, 4, 4)
    inverse = pose_inv(attachment)
    relative = meshes[held.semantics.entity_id].to(start)
    relative = relative @ inverse[:, :3, :3].transpose(-1, -2) + inverse[:, None, :3, 3]
    radius = torch.linalg.vector_norm(relative, dim=-1).amax()
    table_pose = context.scene.entities["table"].pose.to(start)
    _, table_upper = _world_bounds(meshes["table"], table_pose)
    eef_targets = targets @ attachment[:, None]
    height = torch.maximum(
        torch.maximum(start[:, 2, 3], eef_targets[:, :, 2, 3].amax(1)) + extra,
        table_upper[:, 2] + radius + _CLEARANCE,
    )
    result, previous = [], start
    for target in eef_targets.unbind(1):
        lifted, above = previous.clone(), target.clone()
        lifted[:, 2, 3], above[:, 2, 3] = height, height
        result.extend((lifted @ inverse, above @ inverse, target @ inverse))
        previous = target
    return torch.stack(result, dim=1)


def _cartesian_plan(
    action: Any, request: Any, context: Any, poses: torch.Tensor, count: int
) -> Any | None:
    arm, held, _ = _geometry(action, request, context)
    seed = context.robot.qpos[:, list(arm.joint_ids)]
    start = action.robot.compute_fk(qpos=seed, name=arm.control_part, to_matrix=True)
    attachment = held.object_to_eef.to(poses).reshape(1, 4, 4)
    options = request.motion_policy.to_motion_gen_options(
        start_qpos=seed,
        control_part=arm.control_part,
        interpolation_dt=context.control_dt,
    )
    options = replace(options, sample_count=count, preserve_cartesian_samples=True)
    targets = _cartesian_samples(
        start, build_pose_plan_states(poses @ attachment[:, None]), count
    )

    def payload_valid(qpos: torch.Tensor) -> bool:
        full = context.robot.qpos.clone()
        full[:, list(arm.joint_ids)] = qpos
        return inspect_payload(action, request, context, full[:, None])["valid"]

    result = resolve_continuous_ik_path(
        action.motion_generator,
        targets,
        options,
        candidate_predicate=payload_valid,
        operation="held transport",
    )
    if result is None:
        return None
    actual = action.robot.compute_batch_fk(
        qpos=result.positions[:, 1:], name=arm.control_part, to_matrix=True
    )
    desired = torch.stack([target.xpos for target in targets], dim=1)
    tolerances = action.robot.get_solver(arm.control_part).get_iteration_params()
    position_tolerance, rotation_tolerance = (
        tolerances["pos_eps"],
        tolerances["rot_eps"],
    )
    if any(
        not math.isfinite(value) or value <= 0
        for value in (position_tolerance, rotation_tolerance)
    ):
        raise ValueError("held_path_ik_tolerances_unavailable")
    if (
        actual.shape != desired.shape
        or not torch.isfinite(actual).all()
        or (
            torch.linalg.vector_norm(actual[..., :3, 3] - desired[..., :3, 3], dim=-1)
            > position_tolerance
        ).any()
        or (
            quat_error_magnitude(
                quat_from_matrix(actual[..., :3, :3]),
                quat_from_matrix(desired[..., :3, :3]),
            )
            > rotation_tolerance
        ).any()
    ):
        return None
    grasp = request.binding.endpoint("primary", "grasp")
    hand = grasp.require_target(JointPositionTarget)
    base = context.last_qpos.clone()
    base[:, list(hand.joint_ids)] = grasp.joint_positions(
        "grasp", num_envs=1, device=base.device, dtype=base.dtype
    )
    _, timed = to_full_robot_trajectory(
        result,
        base_qpos=base,
        joint_ids=list(arm.joint_ids),
        env_ids=context.env_ids,
        control_dt=context.require_control_dt(),
    )
    return action.build_plan(
        request,
        context,
        success=result.success,
        trajectory=timed,
        segment_lengths={"transport": timed.waypoint_count},
    )


def qualify_or_recover(action: Any, request: Any, context: Any, original: Any) -> Any:
    """Retain qualified originals; recover exact targets only on the owned route."""
    phase = phase_scope(action, request, context)
    if not phase["owned"]:
        return replace(
            original,
            diagnostics=replace(
                original.diagnostics,
                metadata={
                    **original.diagnostics.metadata,
                    "gen_sim_held_path": {
                        "phase_scope": phase,
                        "not_migrated": True,
                        "not_migrated_reason": phase["reason"],
                        "accepted_recovery": False,
                        "attempts": [],
                    },
                },
            ),
        )
    acceptance = inspect(action, request, context, original)
    if acceptance["valid"]:
        return replace(
            original,
            diagnostics=replace(
                original.diagnostics,
                metadata={
                    **original.diagnostics.metadata,
                    "gen_sim_held_path": {
                        "phase_scope": phase,
                        "accepted_recovery": False,
                        "original": acceptance,
                        "attempts": [],
                    },
                },
            ),
        )
    selected, attempts = None, []
    state = _RandomState.capture(
        _cuda_devices(action.robot.device, context.robot.qpos.device)
    )
    try:
        _geometry(action, request, context)
        if (
            request.motion_policy.strategy == "ik_interp"
            and request.motion_policy.dynamic_collision_mode == "off"
            and not _CARTESIAN_APPROACH.get()
            and "unavailable" not in acceptance
        ):
            targets = _targets(action, request, context)
            count = request.motion_policy.sample_count
            if count is None:
                raise ValueError("held_path_motion_sample_budget_unavailable")
            variants = [("exact_cartesian", targets, count)]
            budgets = (count, *_velocity_retry_samples(count))
            for extra, budget in zip((0.06, 0.12), budgets[-2:]):
                variants.append(
                    (
                        "lifted_cartesian",
                        _lifted_targets(action, request, context, targets, extra),
                        budget,
                    )
                )
            for label, poses, budget in variants:
                candidate = _cartesian_plan(action, request, context, poses, budget)
                measured = (
                    inspect(action, request, context, candidate)
                    if candidate is not None
                    else {"valid": False, "reason": "continuous_ik_path_unavailable"}
                )
                attempts.append({"variant": label, "sample_budget": budget, **measured})
                if measured["valid"]:
                    selected = candidate
                    break
    except _UNAVAILABLE as error:
        attempts.append({"valid": False, "unavailable": str(error)})
    finally:
        state.restore()
    plan = selected if selected is not None else original
    return replace(
        plan,
        plan_success=(
            plan.plan_success
            if selected is not None
            else torch.zeros_like(plan.plan_success)
        ),
        diagnostics=replace(
            plan.diagnostics,
            failure=(
                None
                if selected is not None
                else PlanningFailure("held_transport_path_unqualified", retryable=False)
            ),
            metadata={
                **plan.diagnostics.metadata,
                "gen_sim_held_path": {
                    "phase_scope": phase,
                    "accepted_recovery": selected is not None,
                    "original": acceptance,
                    "attempts": attempts,
                },
            },
        ),
    )
