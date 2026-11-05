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

"""Complete measured preshape and inside planning owned by Atomic Skills."""

from __future__ import annotations
from dataclasses import replace
import math, time
from typing import Any
import numpy as np
import torch
from embodichain.compute.geometry.measured_manipulation import (
    PadProjectionStroke,
    fit_projected_preshape,
    project_floor_contents,
)
from embodichain.compute.trajectory import differentiate_positions
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.mesh_collision import robot_collision_query_scope
from embodichain.lab.sim.motion.search import measure_search, PlanningBudgetExceeded
from embodichain.lab.sim.scene.native_motion_scene import (
    NativeRestCollisionPart,
    native_path_link_poses,
)
from ..bindings import JointPositionTarget
from ..control import JointPositionCommand
from ..goals import SceneEntityPose, resolve_pose_goal
from ..trajectory_ops import interpolate_hand_qpos
from ._manipulation_checks import inspect_manipulation_plan, inspect_robot_motion
from ._pick_path import _free_motion
from ._pick_closing import inspect_closing_plan

__all__: list[str] = []


def plan_measured_rigid(action, request, context, *, build, external_nominal):
    if external_nominal:
        raise ValueError(
            "Measured rigid planning does not support external nominal motion."
        )
    if context.is_preview:
        return build(request)
    if (
        request.recovery_policy.planning_budget is None
        or request.recovery_policy.planning_budget.wall_seconds != 90.0
    ):
        raise ValueError(
            "Measured rigid planning requires its explicit source 90-second planning budget."
        )
    provider = action.planning_services.measured_rigid_provider
    if provider is None or provider.robot is not action.robot:
        raise ValueError(
            "Measured rigid planning requires its exact installed fact provider."
        )
    session = None
    started = time.perf_counter()
    try:
        session = provider.prepare(request, context)
        if session is None:
            raise ValueError("Measured rigid planning occurrence is unavailable.")
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or session.physics_backend != "default"
            or request.motion_policy.strategy != "ik_interp"
            or request.motion_policy.dynamic_collision_mode != "off"
        ):
            return action.failed_plan(
                request,
                context,
                message="Measured rigid policy requires its single-environment Default IK scope.",
                failure_code="measured_rigid_scope_unavailable",
                retryable=False,
            )
        session.validate_observation()
        with measure_search("measured_rigid_policy"):
            if request.skill_id == "pick_up":
                plan = _pick(action, request, context, build, session)
            elif request.skill_id == "place":
                plan = _place(action, request, context, build, session)
            else:
                raise ValueError("Measured rigid policy received a mismatched skill.")
            if time.perf_counter() - started > 90.0:
                raise ValueError(
                    "Measured rigid planning exceeded its 90-second budget."
                )
        session.validate_observation()
        return plan
    except PlanningBudgetExceeded:
        raise
    except (ValueError, RuntimeError, AttributeError, TypeError, KeyError) as error:
        return action.failed_plan(
            request,
            context,
            message=str(error),
            failure_code="measured_rigid_unqualified",
            retryable=False,
        )
    finally:
        if session is not None:
            session.close()


def _closing(session, request, context, semantics):
    query = session.closing_query(request, context, semantics)
    if (
        query.closed
        or query.revision != 4
        or query.steps != 65
        or query.scope.observation_timestamp != context.robot.timestamp
        or query.scope.scene_version != context.scene.version
        or query.scope.env_id != 0
    ):
        raise ValueError(
            "Measured hand facts require the exact source native-four 65-sample observation."
        )
    return query


def _velocity_mask(action, plan):
    trajectory = plan.joint_trajectory
    limits = authored_joint_velocity_limits(action.robot, None).to(trajectory.positions)
    return joint_velocity_validity(trajectory.positions, trajectory.dt, limits)


def _pick(action, request, context, build, session):
    subject = session.subject_id
    if session.container_id is not None or subject != request.goal.semantics.entity_id:
        raise ValueError("Measured Pick source differs from its bound occurrence.")
    if (
        request.goal.grasp_xpos is not None
        or request.goal.object_pose is not None
        or request.skill_options.fixed_object_to_eef is not None
        or request.skill_options.pick_object_part != "center"
    ):
        raise ValueError(
            "Measured preshape cannot replace an explicitly constrained Pick."
        )
    original = build(request)
    if not original.plan_success.all():
        return original
    endpoint = request.binding.endpoint("primary", "motion")
    held = original.expected_effects.held_object_updates[endpoint.task_state_key]
    screen = _closing(session, request, context, request.goal.semantics)
    scene = session.scene_query()
    target = next(body for body in scene.bodies if body.part.name == session.source_uid)
    grasp = held.grasp_xpos[0].detach().cpu().double().numpy()
    relative = np.linalg.inv(grasp) @ target.pose
    points = np.vstack(
        [
            mesh.vertices @ relative[:3, :3].T + relative[:3, 3]
            for mesh in target.part.meshes
        ]
    )
    pads = [part for part in screen.parts if part.pad]
    if len(pads) != 2:
        raise ValueError("Measured preshape requires two native pads.")
    fitted = fit_projected_preshape(
        points,
        *(
            PadProjectionStroke(
                tuple(mesh.vertices for mesh in part.meshes), part.tcp_poses
            )
            for part in pads
        ),
        pad_rest_offsets=tuple(part.rest_offset for part in pads),
        target_rest_offset=screen.target_rest_offset,
    )
    fraction = fitted.fraction
    hand_endpoint = request.binding.endpoint("primary", "grasp")
    opened, closed = [
        hand_endpoint.joint_positions(
            command,
            num_envs=1,
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        for command in ("open", "grasp")
    ]
    narrowed = torch.lerp(opened, closed, fraction)
    selected = replace(
        request,
        binding=request.binding.with_command_overrides(
            {("primary", "grasp"): {"open": JointPositionCommand(narrowed)}}
        ),
    )
    positions = original.joint_trajectory.positions.clone()
    hand = hand_endpoint.require_target(JointPositionTarget)
    motion = endpoint.require_target(JointPositionTarget)
    ids = list(hand.joint_ids)
    segments = {segment.name: segment for segment in original.segments}
    dt = context.require_control_dt()
    limits = authored_joint_velocity_limits(action.robot, None).to(positions)[
        context.env_ids
    ][:, ids]
    actual = context.robot.qpos[:, ids]
    count = (
        max(1, math.ceil(float(((narrowed - actual).abs() / (limits * dt)).amax()))) + 1
    )
    if count > segments["approach"].stop:
        raise ValueError("Measured preshape does not fit the unchanged approach time.")
    positions[:, : segments["approach"].stop, ids] = narrowed[:, None]
    positions[:, :count, ids] = interpolate_hand_qpos(
        actual, narrowed, n_waypoints=count
    )
    close = segments["close"]
    n = request.skill_options.hand_interp_steps
    if close.stop - close.start != n + request.skill_options.grasp_settle_steps:
        raise ValueError("Measured preshape requires the canonical close grid.")
    positions[:, close.start : close.start + n, ids] = interpolate_hand_qpos(
        narrowed, closed, n_waypoints=n
    )
    velocities = original.joint_trajectory.velocities
    if velocities is not None:
        computed = (
            torch.zeros_like(positions)
            if request.motion_policy.velocity_targets == "zero"
            else differentiate_positions(positions, original.joint_trajectory.dt)
        )
        stationary = positions[:, 1:] == positions[:, :-1]
        computed[:, :-1] = computed[:, :-1].masked_fill(stationary, 0.0)
        computed[:, -1] = 0.0
        velocities = velocities.clone()
        velocities[:, :, ids] = computed[:, :, ids]
    trajectory = replace(
        original.joint_trajectory, positions=positions, velocities=velocities
    )
    commands = action._joint_command_sequence(
        selected, trajectory, active_mask=original.plan_success
    )
    candidate = replace(
        original,
        joint_trajectory=trajectory,
        commands=commands,
        tracking=action._tracking_sequence(selected, commands),
    )
    selected_screen = _closing(session, selected, context, request.goal.semantics)
    closure = inspect_closing_plan(
        action, selected, context, candidate, selected_screen
    ).diagnostics.metadata["pick_contact_geometry"]
    with robot_collision_query_scope():
        software = inspect_manipulation_plan(action.robot, selected, context, candidate)
        free = _free_motion(
            action, positions[0, : segments["approach"].stop], motion.control_part
        )
    native = scene.inspect(
        selected_screen.parts,
        native_path_link_poses(action.robot, positions[0], selected_screen.parts),
        session.source_uid,
        source_contact_from=close.start,
    )
    if not all(checked["valid"] for checked in (closure, software, free, native)):
        raise ValueError(
            "Measured preshape failed unchanged closure, motion or native qualification."
        )
    report = dict(
        accepted=True,
        preshape_fraction=fraction,
        target_width_m=fitted.target_width_m,
        extra_side_reserve_m=0.002,
        unchanged_arm_and_timing=True,
        closure=closure,
        command_qualification=software,
        approach=free,
        native=native,
    )
    return replace(
        candidate,
        diagnostics=replace(
            candidate.diagnostics,
            metadata={**candidate.diagnostics.metadata, "measured_rigid": report},
        ),
    )


def _landing(request, context, held, inputs):
    subject = held.semantics.entity_id
    reference = (
        resolve_pose_goal(
            SceneEntityPose(inputs.parent_id), context, name="measured_parent"
        )[0]
        .cpu()
        .numpy()
    )
    vertices, floors = inputs.geometry.source_geometry()
    floor, height = floors[0]
    contents = []
    for source, points in inputs.content_vertices:
        pose = (
            resolve_pose_goal(
                SceneEntityPose(source), context, name="measured_content"
            )[0]
            .cpu()
            .numpy()
        )
        relative = np.linalg.solve(reference, pose)
        values = points.cpu().numpy()
        contents.append((source, values @ relative[:3, :3].T + relative[:3, 3]))
    occupancy = project_floor_contents(floor, height, contents)
    if not occupancy.contents:
        return request, {"observed_contents": []}
    release = resolve_pose_goal(request.goal.xpos, context, name="measured_release")
    object_release = release @ torch.linalg.inv(held.object_to_eef.to(release))
    relative = np.linalg.solve(reference, object_release[0].cpu().numpy())
    rotated = vertices[0].numpy() @ relative[:3, :3].T
    proposal = occupancy.propose(rotated, relative[:3, 3])
    if proposal.displacement_parent is None:
        return request, proposal.metadata()
    delta = release.new_tensor(
        reference[:3, :3] @ np.array(proposal.displacement_parent)
    )
    if request.goal.xpos.world_displacement is not None:
        delta += request.goal.xpos.world_displacement.to(delta)
    selected = replace(
        request,
        goal=replace(
            request.goal, xpos=replace(request.goal.xpos, world_displacement=delta)
        ),
    )
    return selected, proposal.metadata()


def _place_check(action, request, context, plan, screen, scene, arm, source_uid):
    if not plan.plan_success.all():
        return {"valid": False, "planning_success": False}
    robot = action.robot
    q = plan.joint_trajectory.positions[0]
    stop = next(segment.stop for segment in plan.segments if segment.name == "approach")
    endpoint = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    hand = request.binding.endpoint("primary", "grasp").require_target(
        JointPositionTarget
    )
    hand_ids = list(hand.joint_ids)
    held_q = q[:stop].clone()
    held_q[:, hand_ids] = context.robot.qpos[0, hand_ids]
    observed = scene.inspect(
        screen.parts, native_path_link_poses(robot, held_q, screen.parts), source_uid
    )
    emitted = scene.inspect(
        screen.parts, native_path_link_poses(robot, q, screen.parts), source_uid
    )
    arm_scene = scene.inspect(arm, native_path_link_poses(robot, q, arm), source_uid)
    tcp = (
        robot.compute_batch_fk(
            qpos=q[None, :stop, list(endpoint.joint_ids)],
            name=endpoint.control_part,
            to_matrix=True,
        )[0]
        .detach()
        .cpu()
        .double()
        .numpy()
    )
    actual = (
        robot.compute_batch_fk(
            qpos=context.robot.qpos[:, None, list(endpoint.joint_ids)],
            name=endpoint.control_part,
            to_matrix=True,
        )[0, 0]
        .detach()
        .cpu()
        .double()
        .numpy()
    )
    source = next(body for body in scene.bodies if body.part.name == source_uid)
    relative = np.linalg.inv(actual) @ source.pose
    cargo = NativeRestCollisionPart(
        source_uid, source.part.meshes, source.part.rest_offset
    )
    carried = scene.inspect((cargo,), (tcp @ relative)[:, None], source_uid)
    own = inspect_robot_motion(robot, q, endpoint.control_part)
    velocity = bool(_velocity_mask(action, plan).all())
    return dict(
        valid=all(
            checked["valid"] for checked in (observed, emitted, arm_scene, carried, own)
        )
        and velocity,
        observed_hold_hand=observed,
        emitted_hand=emitted,
        operated_arm_scene=arm_scene,
        rigid_cargo_approach=carried,
        robot_self_proxy=own,
        full_robot_velocity_valid=velocity,
    )


def _place(action, request, context, build, session):
    subject = session.subject_id
    if (
        not isinstance(request.goal.xpos, SceneEntityPose)
        or request.goal.xpos.entity_id != session.container_id
    ):
        raise ValueError("Measured Place target differs from its bound occurrence.")
    held = context.get_held_object(
        request.binding.endpoint("primary", "motion").task_state_key
    )
    if held is None or held.semantics.entity_id != subject:
        raise ValueError("Measured Place requires its verified input hold.")
    inputs = session.inside_inputs(context)
    if (
        inputs.release_clearance != 0.12
        or request.goal.xpos.world_orientation is not None
        or request.skill_options.max_approach_retract_z is not None
    ):
        raise ValueError(
            "Measured inside policy cannot change an explicit constrained release."
        )
    selected, landing = _landing(request, context, held, inputs)
    screen = _closing(session, request, context, held.semantics)
    scene = session.scene_query()
    part = (
        request.binding.endpoint("primary", "motion")
        .require_target(JointPositionTarget)
        .control_part
    )
    arm = session.arm_parts(part, screen.parts)
    attempts = []
    for extra in (0.0, 0.05, 0.10, 0.15):
        candidate = replace(
            selected,
            skill_options=replace(
                selected.skill_options,
                lift_height=selected.skill_options.lift_height + extra,
            ),
        )
        plan = build(candidate)
        with measure_search("measured_inside_clearance"):
            qualification = _place_check(
                action, candidate, context, plan, screen, scene, arm, session.source_uid
            )
        attempts.append(
            dict(
                lift_height_m=candidate.skill_options.lift_height,
                qualification=qualification,
            )
        )
        if qualification["valid"]:
            return replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "measured_rigid": dict(
                            accepted=True,
                            landing=landing,
                            candidates=attempts,
                            release_clearance_m_unchanged=inputs.release_clearance,
                            scope="discrete_observed_geometry_not_continuous_or_physical_success",
                        ),
                    },
                ),
            )
    raise ValueError("Bounded measured inside transport candidates are unqualified.")
