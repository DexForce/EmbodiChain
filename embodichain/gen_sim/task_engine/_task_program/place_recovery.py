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
import math
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.utils.math import pose_inv, quat_error_magnitude, quat_from_matrix
from .motion import _CARTESIAN_APPROACH, _WAYPOINT_KINEMATICS
from .pick_safety import _inspect_pick_motion
from .planning_probe import _validate_final_plan_velocity
from .held_alignment_recovery import _RandomState, _cuda_devices
from .adaptive_grasp import ADAPTIVE_GRASP
from .scene_geometry import capture_scene_vertices, scene_components, world_bounds
from .solver_tolerances import _pose_validation_parameters

__all__: list[str] = []

_SCENE_CLEARANCE_M = 0.002
_DISTANCE_TOLERANCE_M = 1e-6


def _maximum_cover_violation(points: Any, normals: Any, offsets: Any) -> float:
    """Retain every vertex/plane pair with bounded multiplication scratch."""
    import numpy as np

    maximum = -np.inf
    for start in range(0, len(points), 256):
        values = points[start : start + 256] @ normals.T
        values -= offsets
        maximum = np.maximum(maximum, np.max(values))
        del values
    return float(maximum)


def _convex_clearance(
    vertices: torch.Tensor,
    poses: torch.Tensor,
    other_vertices: torch.Tensor,
    other_pose: torch.Tensor,
    frames: torch.Tensor,
) -> dict[str, Any]:
    """Refine overlapping AABBs with conservative solid covers of the same inputs."""
    import fcl
    import numpy as np
    import trimesh

    def solid(points: torch.Tensor) -> Any:
        if (
            points.ndim != 3
            or points.shape[0] != 1
            or points.shape[-1] != 3
            or not torch.isfinite(points).all()
        ):
            raise ValueError("Convex clearance requires complete finite vertices.")
        points = points[0].detach().cpu().double().numpy()
        if len(points) < 4 or np.linalg.matrix_rank(points - points[0]) != 3:
            raise ValueError("Convex clearance requires a nondegenerate solid.")
        hull = trimesh.convex.convex_hull(points)
        if not hull.is_volume or not hull.is_convex:
            raise ValueError("Convex clearance requires a nondegenerate solid.")
        offsets = np.einsum("ij,ij->i", hull.face_normals, hull.triangles[:, 0])
        if _maximum_cover_violation(points, hull.face_normals, offsets) > 1e-8:
            raise ValueError("Convex cover does not contain every source vertex.")
        faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).ravel()
        return fcl.Convex(hull.vertices, len(hull.faces), faces)

    def transform(pose: torch.Tensor) -> Any:
        pose = pose.detach().cpu().double().numpy()
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-7, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=2e-5)
        ):
            raise ValueError("Convex clearance requires rigid observed poses.")
        return fcl.Transform(pose[:3, :3], pose[:3, 3])

    payload = solid(vertices)
    obstacle = fcl.CollisionObject(solid(other_vertices), transform(other_pose[0]))
    result: dict[str, Any] = {
        "valid": True,
        "scope": "discrete_source_convex_solids_not_native_contact",
        "required_clearance_m": _SCENE_CLEARANCE_M,
        "numerical_allowance_m": _DISTANCE_TOLERANCE_M,
        "minimum_distance_m": None,
        "checked_frames": 0,
    }
    for frame in frames.tolist():
        moving = fcl.CollisionObject(payload, transform(poses[0, frame]))
        # Convex solids, unlike triangle soups, also reject complete containment.
        distance = (
            0.0
            if fcl.collide(moving, obstacle)
            else float(fcl.distance(moving, obstacle))
        )
        if not math.isfinite(distance):
            raise ValueError("Convex solid distance is unavailable.")
        previous = result["minimum_distance_m"]
        result["minimum_distance_m"] = (
            distance if previous is None else min(previous, distance)
        )
        result["checked_frames"] += 1
        if distance <= _SCENE_CLEARANCE_M + _DISTANCE_TOLERANCE_M:
            return {**result, "valid": False, "frame": frame}
    return result


def scene_vertices(simulation: Any, bindings: Any) -> dict[str, torch.Tensor]:
    """Capture immutable native, scaled geometry; live poses remain context-owned."""
    return capture_scene_vertices(simulation, bindings)


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
    rotations = quat_from_matrix(fk[:, :, :3, :3]).reshape(-1, 4)
    reference = rotations[:1].expand_as(rotations)
    rotation_max = float(quat_error_magnitude(rotations, reference).max())
    rotation_path = float(quat_error_magnitude(rotations[1:], rotations[:-1]).sum())
    rot_eps = _pose_validation_parameters(action.robot.get_solver(arm.control_part))[
        "rot_eps"
    ]
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
    support_owner = None
    if request.invocation_id in getattr(action, "ordinary_on_invocations", ()):
        reference = getattr(request.goal.xpos, "entity_id", None)
        support_owner = (getattr(action, "support_surface_owners", None) or {}).get(
            reference, reference
        )
    deferred_supports = []
    try:
        for uid, component, mesh, pose in scene_components(
            meshes, context, excluded_owner=held.semantics.entity_id
        ):
            other_lower, other_upper = world_bounds(mesh, pose.to(poses))
            if uid == support_owner:
                # Landing on this owner requires contact, not free-space clearance.
                # Native release/support observations remain the physical evidence.
                deferred_supports.append(uid)
                if uid == "table":
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
                checked = _convex_clearance(
                    vertices, poses, mesh, pose, overlap[0].nonzero().flatten()
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
            if uid == "table":
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


def eligible(action: Any, request: Any, context: Any) -> bool:
    """Limit the cheap waypoint probe to its existing ordinary Place boundary."""
    if (
        context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or _CARTESIAN_APPROACH.get()
    ):
        return False
    if (
        request.motion_policy.strategy != "ik_interp"
        or request.motion_policy.dynamic_collision_mode != "off"
        or not request.skill_options.preserve_current_object_orientation
        or request.skill_options.max_approach_retract_z is not None
    ):
        return False
    held = context.get_held_object(
        request.binding.endpoint("primary", "motion").task_state_key
    )
    affordance = getattr(getattr(held, "semantics", None), "affordance", None)
    if affordance is not None and affordance.get_custom_config(ADAPTIVE_GRASP) in {
        "pour",
        "ends",
    }:
        return False
    return True


def preflight(action: Any, request: Any, context: Any) -> dict[str, Any] | None:
    """Screen ordinary required poses before expanding an expensive path.

    This is a finite IK qualification, not proof of global unreachability. It
    neither changes a target nor consumes the caller's random stream.
    """
    from embodichain.lab.sim.atomic_actions.primitives.place import Place, PlaceGoal
    from .pick_path import _pose_path_valid
    from ._search_cost import measure_search

    if (
        not eligible(action, request, context)
        or not isinstance(getattr(request, "goal", None), PlaceGoal)
        or request.goal.tcp_symmetry != "none"
    ):
        return None
    endpoint = request.binding.endpoint("primary", "motion")
    arm = endpoint.require_target(JointPositionTarget)
    target, _ = action._resolve_place_xpos(
        request.goal, context, endpoint.task_state_key, request=request
    )
    if target.ndim == 4:
        if target.shape[1] != 1:
            return None
        target = target[:, 0]
    if target.shape != (1, 4, 4):
        return None
    start_qpos = context.robot.qpos[:, list(arm.joint_ids)]
    start = action.robot.compute_fk(
        qpos=start_qpos, name=arm.control_part, to_matrix=True
    )
    lifted = Place._lifted_pose(action, target, request.skill_options)
    poses = torch.cat(
        (
            Place._translation_keyframes(
                action, start, torch.stack((lifted, target), 1), request.skill_options
            ),
            Place._translation_keyframes(
                action, target, lifted[:, None], request.skill_options
            ),
        ),
        1,
    )
    state = _RandomState.capture(
        _cuda_devices(action.robot.device, context.robot.qpos.device)
    )
    record: dict[str, Any] = {
        "valid": False,
        "scope": "finite_required_waypoint_ik_not_global_unreachability",
        "physical_evidence": False,
        "required_keyframes": poses.shape[1],
        "full_plans": 0,
        "required_poses": poses.detach().cpu().tolist(),
    }
    try:
        with measure_search("place_keyframe_preflight"):
            ok, qpos = action.robot.compute_batch_ik(
                pose=poses,
                name=arm.control_part,
                joint_seed=start_qpos[:, None].expand(-1, poses.shape[1], -1).clone(),
            )
            bounds = action.robot.get_qpos_limits(name=arm.control_part).to(qpos)
            record["ik_valid"] = bool(ok.all())
            record["joint_limits_valid"] = bool(
                torch.isfinite(qpos).all()
                and (qpos >= bounds[:, None, :, 0]).all()
                and (qpos <= bounds[:, None, :, 1]).all()
            )
            if record["ik_valid"] and record["joint_limits_valid"]:
                fk = action.robot.compute_batch_fk(
                    qpos=qpos, name=arm.control_part, to_matrix=True
                )
                params = _pose_validation_parameters(
                    action.robot.get_solver(arm.control_part)
                )
                record["fk_valid"] = _pose_path_valid(
                    fk, poses, params["pos_eps"], params["rot_eps"]
                )
                record["valid"] = record["fk_valid"]
    finally:
        state.restore()
    return record


def recover(action: Any, request: Any, context: Any, first: Any, build: Any) -> Any:
    """Keep every successful path and all explicitly Cartesian contracts intact."""
    if first.plan_success.any() or not eligible(action, request, context):
        return first
    state = _RandomState.capture(
        _cuda_devices(action.robot.device, context.robot.qpos.device)
    )
    evidence: list[torch.Tensor] = []
    token = _WAYPOINT_KINEMATICS.set(evidence)
    try:
        chosen, acceptance = None, None
        attempts = []
        for _ in range(3):
            alternative = build(request, context)
            measured = inspect(action, request, context, alternative)
            attempts.append(measured)
            if not measured["valid"]:
                continue
            score = (measured["rotation_max_rad"], measured["rotation_path_rad"])
            if acceptance is None or score < (
                acceptance["rotation_max_rad"],
                acceptance["rotation_path_rad"],
            ):
                chosen, acceptance = alternative, measured
            if measured["orientation_preserved"]:
                break
        if chosen is None:
            if not evidence:
                return first
            return replace(
                first,
                diagnostics=replace(
                    first.diagnostics,
                    metadata={
                        **first.diagnostics.metadata,
                        "gen_sim_place_waypoint_probe": {
                            "kinematic_valid": any(bool(v.all()) for v in evidence),
                            "acceptance": attempts[-1],
                            "attempts": attempts,
                        },
                    },
                ),
            )
        return replace(
            chosen,
            diagnostics=replace(
                chosen.diagnostics,
                metadata={
                    **chosen.diagnostics.metadata,
                    "gen_sim_place_waypoint_recovery": {
                        **acceptance,
                        "attempts": attempts,
                    },
                },
            ),
        )
    finally:
        _WAYPOINT_KINEMATICS.reset(token)
        state.restore()
