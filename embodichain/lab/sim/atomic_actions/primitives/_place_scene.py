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

"""Source-preserving qualification of planned Place envelopes, not physical effects."""

from __future__ import annotations
from typing import Any
import torch
from embodichain.compute.geometry.mesh_bounds import transform_vertex_bounds
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.solvers.tolerances import pose_validation_tolerances
from embodichain.lab.sim.scene.mesh_snapshot import iter_scene_components
from embodichain.lab.sim.scene.convex_clearance import convex_input_clearance
from embodichain.utils.math import pose_inv, quat_error_magnitude, quat_from_matrix
from ..bindings import JointPositionTarget
from ..planning_probes import PlaceSceneInputs
from ._manipulation_checks import inspect_robot_motion

__all__: list[str] = []
_SCENE_CLEARANCE_M = 0.002
_DISTANCE_TOLERANCE_M = 1e-6


def inspect_place_plan(
    action: Any, request: Any, context: Any, plan: Any, inputs: PlaceSceneInputs
) -> dict[str, Any]:
    """Check emitted limits, robot meshes and held-object scene envelopes."""
    if (
        not plan.plan_success.all()
        or not joint_velocity_validity(
            plan.joint_trajectory.positions,
            plan.joint_trajectory.dt,
            authored_joint_velocity_limits(action.robot, None).to(
                plan.joint_trajectory.positions
            ),
        ).all()
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
    collision = inspect_robot_motion(action.robot, positions[0], arm.control_part)
    if not collision["valid"]:
        return {
            "valid": False,
            "reason": "robot_proxy_rejected",
            "collision": collision,
        }
    held = context.get_held_object(endpoint.task_state_key)
    meshes = inputs.geometry
    if (
        held is None
        or held.semantics.entity_id not in meshes
        or inputs.floor_entity_id not in meshes
    ):
        return {"valid": False, "reason": "held_object_scene_geometry_unavailable"}
    end = next(s.stop for s in plan.segments if s.name == "approach")
    fk = action.robot.compute_batch_fk(
        qpos=values[:, :end], name=arm.control_part, to_matrix=True
    )
    rotations = quat_from_matrix(fk[:, :, :3, :3]).reshape(-1, 4)
    reference = rotations[:1].expand_as(rotations)
    rotation_max = float(quat_error_magnitude(rotations, reference).max())
    rotation_path = float(quat_error_magnitude(rotations[1:], rotations[:-1]).sum())
    _, rot_eps = pose_validation_tolerances(action.robot.get_solver(arm.control_part))
    poses = fk @ pose_inv(held.object_to_eef)[:, None]
    vertices = meshes[held.semantics.entity_id].to(poses)
    if not torch.isfinite(poses).all() or not torch.isfinite(vertices).all():
        return {"valid": False, "reason": "held_object_scene_geometry_unavailable"}
    world = (
        torch.einsum("btij,bvj->btvi", poses[:, :, :3, :3], vertices)
        + poses[:, :, None, :3, 3]
    )
    lower, upper = world.amin(2), world.amax(2)
    minimum_gap = None
    convex_checks = []
    support_owner = inputs.support_owner_id
    deferred_supports = []
    try:
        observed = {
            identity: entity.pose
            for identity, entity in context.scene.entities.items()
            if entity.confidence > 0
        }
        for uid, component, mesh, pose in iter_scene_components(
            meshes, observed, excluded_owner=held.semantics.entity_id
        ):
            other_lower, other_upper = transform_vertex_bounds(mesh, pose.to(poses))
            if uid == support_owner:
                # Landing on this owner requires contact, not free-space clearance.
                # Native release/support observations remain the physical evidence.
                deferred_supports.append(uid)
                if uid == inputs.floor_entity_id:
                    minimum_gap = float(
                        (lower[:, :, 2] - other_upper[:, None, 2]).min()
                    )
                continue
            margin = _SCENE_CLEARANCE_M + _DISTANCE_TOLERANCE_M
            overlap = (
                (upper >= other_lower[:, None] - margin)
                & (lower <= other_upper[:, None] + margin)
            ).all(-1)
            if overlap.any():
                checked = convex_input_clearance(
                    vertices,
                    poses,
                    mesh,
                    pose,
                    overlap[0].nonzero().flatten(),
                    clearance_m=_SCENE_CLEARANCE_M,
                    numerical_allowance_m=_DISTANCE_TOLERANCE_M,
                )
                convex_checks.append({"entity": uid, "component": component, **checked})
                if not checked["valid"]:
                    return {
                        "valid": False,
                        "reason": "held_object_scene_envelope_rejected",
                        "entity": uid,
                        "component": component,
                        "convex_clearance": convex_checks,
                    }
            if uid == inputs.floor_entity_id:
                minimum_gap = float((lower[:, :, 2] - other_upper[:, None, 2]).min())
    except (ImportError, ValueError, TypeError, AttributeError, RuntimeError) as error:
        return {
            "valid": False,
            "reason": "scene_geometry_unavailable",
            "unavailable": str(error),
        }
    return {
        "valid": True,
        "scope": (
            "discrete_robot_self_meshes_and_noncontact_scene_component_envelopes"
            if deferred_supports
            else "discrete_robot_self_meshes_and_bound_scene_component_envelopes"
        ),
        "geometry_sources": "legacy_rigid_visual_vertices_and_articulation_collision_components",
        "not_certified": "continuous_collision_contact_or_robot_scene_clearance",
        "minimum_table_gap_m": minimum_gap,
        "rotation_max_rad": rotation_max,
        "rotation_path_rad": rotation_path,
        "orientation_preserved": rotation_max <= rot_eps,
        "convex_clearance": convex_checks,
        **(
            {
                "intended_support_contact_deferred": sorted(set(deferred_supports)),
                "physical_evidence": False,
            }
            if deferred_supports
            else {}
        ),
    }
