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

"""Failure-only reuse of shared waypoint IK, without dropping required poses."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.utils.math import pose_inv
from .motion import _CARTESIAN_APPROACH, _WAYPOINT_KINEMATICS
from .pick_safety import _inspect_pick_motion
from .planning_probe import _validate_final_plan_velocity
from .held_alignment_recovery import _RandomState, _cuda_devices
from .adaptive_grasp import ADAPTIVE_GRASP

__all__: list[str] = []


def scene_vertices(simulation: Any, bindings: Any) -> dict[str, torch.Tensor]:
    """Capture immutable native, scaled geometry; live poses remain context-owned."""
    result = {}
    if not callable(getattr(simulation, "get_rigid_object", None)):
        return result
    if getattr(bindings, "articulations", ()) or getattr(
        bindings, "rigidized_articulations", ()
    ):
        return result
    for binding in getattr(bindings, "rigid_objects", ()):
        native = simulation.get_rigid_object(binding.simulation_uid)
        if native is None or not callable(getattr(native, "get_vertices", None)):
            return {}
        vertices = native.get_vertices(scale=True)
        if (
            isinstance(vertices, torch.Tensor)
            and vertices.ndim == 3
            and vertices.shape[-1] == 3
            and vertices.shape[1]
            and torch.isfinite(vertices).all()
        ):
            result[binding.entity_id] = vertices.detach().clone()
        else:
            return {}
    return result


def inspect(action: Any, request: Any, context: Any, plan: Any) -> dict[str, Any]:
    """Check emitted limits, robot meshes and held-object scene envelopes."""
    if (
        not plan.plan_success.all()
        or not _validate_final_plan_velocity(plan, action.robot)["valid_mask"].all()
    ):
        return {"valid": False, "reason": "plan_or_final_velocity_failed"}
    endpoint = request.binding.endpoint("primary", "motion")
    arm = endpoint.require_target(JointPositionTarget)
    bounds = action.robot.get_qpos_limits(name=arm.control_part).to(context.last_qpos)
    positions = plan.joint_trajectory.positions
    values = positions[:, :, list(arm.joint_ids)]
    if (
        bounds.shape != (1, len(arm.joint_ids), 2)
        or torch.isnan(bounds).any()
        or (bounds[..., 0] > bounds[..., 1]).any()
    ):
        return {"valid": False, "reason": "joint_limits_unavailable"}
    if (
        not torch.isfinite(values).all()
        or (values < bounds[:, None, :, 0]).any()
        or (values > bounds[:, None, :, 1]).any()
    ):
        return {"valid": False, "reason": "joint_limit_rejected"}
    collision = _inspect_pick_motion(action.robot, positions[0], arm.control_part)
    if not collision["valid"]:
        return {
            "valid": False,
            "reason": "robot_proxy_rejected",
            "collision": collision,
        }
    held = context.get_held_object(endpoint.task_state_key)
    meshes = action.recovery_scene_vertices or {}
    if held is None or held.semantics.entity_id not in meshes or "table" not in meshes:
        return {"valid": False, "reason": "held_object_scene_geometry_unavailable"}
    end = next(s.stop for s in plan.segments if s.name == "approach")
    fk = action.robot.compute_batch_fk(
        qpos=values[:, :end], name=arm.control_part, to_matrix=True
    )
    poses = fk @ pose_inv(held.object_to_eef)[:, None]
    vertices = meshes[held.semantics.entity_id].to(poses)
    world = (
        torch.einsum("btij,bvj->btvi", poses[:, :, :3, :3], vertices)
        + poses[:, :, None, :3, 3]
    )
    lower, upper = world.amin(2), world.amax(2)
    minimum_gap = None
    for uid, mesh in meshes.items():
        if uid == held.semantics.entity_id:
            continue
        if uid not in context.scene.entities:
            return {"valid": False, "reason": "scene_pose_unavailable", "entity": uid}
        pose = context.scene.entities[uid].pose.to(poses)
        other = (
            mesh.to(poses) @ pose[:, :3, :3].transpose(-1, -2) + pose[:, None, :3, 3]
        )
        other_lower, other_upper = other.amin(1), other.amax(1)
        overlap = (
            (upper > other_lower[:, None] - 0.002)
            & (lower < other_upper[:, None] + 0.002)
        ).all(-1)
        if overlap.any():
            return {
                "valid": False,
                "reason": "held_object_scene_envelope_rejected",
                "entity": uid,
            }
        if uid == "table":
            minimum_gap = float((lower[:, :, 2] - other_upper[:, None, 2]).min())
    return {
        "valid": True,
        "scope": "discrete_robot_self_meshes_and_bound_rigid_envelopes",
        "minimum_table_gap_m": minimum_gap,
    }


def recover(action: Any, request: Any, context: Any, first: Any, build: Any) -> Any:
    """Keep every successful path and all explicitly Cartesian contracts intact."""
    if (
        first.plan_success.any()
        or context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or _CARTESIAN_APPROACH.get()
    ):
        return first
    if (
        request.motion_policy.strategy != "ik_interp"
        or request.motion_policy.dynamic_collision_mode != "off"
        or not request.skill_options.preserve_current_object_orientation
        or request.skill_options.max_approach_retract_z is not None
    ):
        return first
    held = context.get_held_object(
        request.binding.endpoint("primary", "motion").task_state_key
    )
    affordance = getattr(getattr(held, "semantics", None), "affordance", None)
    if affordance is not None and affordance.get_custom_config(ADAPTIVE_GRASP) in {
        "pour",
        "ends",
    }:
        return first
    state = _RandomState.capture(
        _cuda_devices(action.robot.device, context.robot.qpos.device)
    )
    evidence: list[torch.Tensor] = []
    token = _WAYPOINT_KINEMATICS.set(evidence)
    try:
        alternative = build(request, context)
        acceptance = inspect(action, request, context, alternative)
        if not acceptance["valid"]:
            if not evidence:
                return first
            return replace(
                first,
                diagnostics=replace(
                    first.diagnostics,
                    metadata={
                        **first.diagnostics.metadata,
                        "gen_sim_place_waypoint_probe": {
                            "kinematic_valid": bool(evidence[0].all()),
                            "acceptance": acceptance,
                        },
                    },
                ),
            )
        return replace(
            alternative,
            diagnostics=replace(
                alternative.diagnostics,
                metadata={
                    **alternative.diagnostics.metadata,
                    "gen_sim_place_waypoint_recovery": acceptance,
                },
            ),
        )
    finally:
        _WAYPOINT_KINEMATICS.reset(token)
        state.restore()
