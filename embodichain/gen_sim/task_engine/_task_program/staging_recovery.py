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

"""Bounded, invocation-owned staging hints; never a collision certificate."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
)

from .articulation_collision import _descendants
from .held_alignment_recovery import _RandomState, _cuda_devices, qualify_alignment_plan
from .pick_safety import _inspect_pick_motion, _inspect_pick_plan

__all__: list[str] = []

STAGING_RECOVERY_REVISION = 2
_FK_ATOL = 2e-4
_UNAVAILABLE = (
    ImportError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    TypeError,
)


def bind_staging_calls(program: Any, rigid_ids: frozenset[str]) -> tuple[str, ...]:
    from .align_held import ALIGN_HELD_CALL
    from .invocation_policy import _invocation_ids

    ids = _invocation_ids(program)
    result = []
    for segment in program.iter_segments():
        call = segment.calls[0].call
        if getattr(call, "call_id", None) == ALIGN_HELD_CALL:
            args = dict(call.arguments)
            if (
                args["object"] in rigid_ids
                and args["preserve_yaw"] is False
                and args["target"] != "current_object_pose"
            ):
                result.append(ids[segment.name])
    return tuple(result)


def verified_retention_sources(
    profile: Any,
    preset_id: str | None,
    sources: tuple[tuple[str, str, str, str, str], ...],
    *,
    retention_call_id: str = "gen_sim.align_held",
) -> dict[str, dict[str, Any]]:
    from embodichain.lab.task_program.semantics import (
        EffectAssurance,
        EffectMonitorRef,
        SkillPolicyPreset,
    )

    presets = getattr(profile, "presets", {})
    if isinstance(presets, Mapping):
        preset = presets.get(preset_id)
    else:
        preset = next(
            (
                value
                for value in presets
                if getattr(value, "preset_id", None) == preset_id
            ),
            None,
        )
    if (
        type(preset) is not SkillPolicyPreset
        or preset.effect_assurance is not EffectAssurance.VERIFIED
    ):
        return {}
    result = {}
    for (
        invocation_id,
        acquisition_invocation_id,
        acquisition_call_id,
        entity_id,
        resource_id,
    ) in sources:
        monitors = tuple(
            preset.effect_monitors.get(call_id)
            for call_id in (acquisition_call_id, retention_call_id)
        )
        if any(
            type(ref) is not EffectMonitorRef
            or ref.monitor_id != "builtin.composite_effect"
            or ref.revision != "1"
            for ref in monitors
        ):
            continue
        result[invocation_id] = {
            "preset_id": preset.preset_id,
            "predecessor_call_id": acquisition_call_id,
            "predecessor_invocation_id": acquisition_invocation_id,
            "entity_id": entity_id,
            "resource_id": resource_id,
            **{
                name: {"monitor_id": ref.monitor_id, "revision": ref.revision}
                for name, ref in zip(
                    ("attachment_monitor", "retention_monitor"), monitors
                )
            },
        }
    return result


def inspect_staging_plan(
    robot: Any,
    request: Any,
    context: Any,
    plan: Any,
    *,
    retention_source: dict[str, Any] | None = None,
    _require_retained_command: bool = False,
    _contract_revision: int = 1,
) -> dict[str, Any]:
    """Distinguish a new hand command from an unchanged native retained target."""
    if type(_contract_revision) is not int or _contract_revision not in (1, 2):
        return {"valid": False, "reason": "retained_command_revision_unavailable"}
    require_retained = _require_retained_command or _contract_revision == 2
    strict = _inspect_pick_plan(robot, request, context, plan)
    if (strict["valid"] and not require_retained) or (
        not strict["valid"] and strict.get("reason") != "initial_velocity_rejected"
    ):
        return strict
    if retention_source is None:
        return (
            {
                "valid": False,
                "reason": "retained_previous_command_authority_unavailable",
            }
            if require_retained
            else strict
        )
    if _contract_revision == 2 and (
        type(retention_source.get("command_continuity_revision")) is not int
        or retention_source["command_continuity_revision"] != 2
    ):
        return {"valid": False, "reason": "retained_command_revision_unavailable"}
    proof = {
        **retention_source,
        "accepted": False,
        "scope": "prior_applied_retained_hand_command",
        "source_api": "Robot.get_qpos(target=True)/get_qvel(target=True)",
    }
    try:
        from .motion import _joint_velocity_limits

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

        def snapshot() -> tuple[torch.Tensor, ...]:
            values = []
            for reader, target in (
                (robot.get_qpos, False),
                (robot.get_qvel, False),
                (robot.get_qpos, True),
                (robot.get_qvel, True),
            ):
                value = reader(target=target)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != context.robot.qpos.shape
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("retained_native_command_snapshot_unavailable")
                values.append(value.detach().clone())
            return tuple(values)

        observed, velocity, desired, desired_velocity = snapshot()
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
        limits = _joint_velocity_limits(robot, None).to(velocity)[context.env_ids]
        proof["velocity_limits"] = limits[:, hand_ids].tolist()
        within_limits = not bool(
            (velocity[:, hand_ids].abs() > limits[:, hand_ids]).any()
        )
        if _contract_revision == 2:
            proof.update(
                actual_hand_velocity_within_limits=within_limits,
                actual_hand_velocity_policy="diagnostic_only",
            )
        elif not within_limits:
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
        initial = _inspect_pick_motion(robot, observed, arm.control_part)
        proof["observed_start_collision"] = initial
        if not initial["valid"]:
            raise ValueError("retained_observed_start_collision_or_unavailable")
        boundary = observed.clone()
        boundary[:, hand_ids] = desired[:, hand_ids]
        certificate = _inspect_pick_plan(
            robot, request, context, plan, _initial_command_qpos=boundary
        )
        if any(
            not torch.equal(previous, current)
            for previous, current in zip(
                (observed, velocity, desired, desired_velocity), snapshot()
            )
        ):
            raise ValueError("retained_command_changed_during_check")
        proof["accepted"] = certificate["valid"]
        return {**certificate, "retained_hand_boundary": proof}
    except _UNAVAILABLE as error:
        return {"valid": False, "reason": str(error), "retained_hand_boundary": proof}


def owns_staging(
    request: Any,
    context: Any,
    rigid_ids: frozenset[str],
    invocation_ids: frozenset[str],
    generator: Any,
) -> bool:
    """Decide ownership before planning without geometry work or random draws."""
    try:
        if (
            request.invocation_id not in invocation_ids
            or type(request.goal) is not HeldObjectPoseGoal
            or request.goal.world_yaw_free is not True
            or not isinstance(request.goal.object_target_pose, torch.Tensor)
            or request.goal.object_target_pose.shape != (1, 2, 4, 4)
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.robot.qpos.ndim != 2
            or request.motion_policy.strategy != "ik_interp"
            or request.motion_policy.dynamic_collision_mode != "off"
            or getattr(generator, "direct_transport_planning", False)
        ):
            return False
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        arm, hand = (
            endpoint.require_target(JointPositionTarget) for endpoint in (motion, grasp)
        )
        ids = (*arm.joint_ids, *hand.joint_ids)
        held = context.task.get_held_object(motion.task_state_key)
        return (
            motion.task_state_key == grasp.task_state_key
            and bool(arm.joint_ids)
            and bool(hand.joint_ids)
            and len(set(ids)) == len(ids)
            and min(ids) >= 0
            and max(ids) < context.robot.qpos.shape[-1]
            and held is not None
            and held.semantics.entity_id in rigid_ids
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
    except _UNAVAILABLE:
        return False


def _pose(value: Any) -> np.ndarray:
    value = (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )
    if value.shape == (1, 4, 4):
        value = value[0]
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(
            value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5, rtol=0
        )
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("non_rigid_or_missing_staging_pose")
    return value


def _calibrate(actual: Any, predicted: Any) -> float:
    error = float(np.max(np.abs(_pose(actual) - _pose(predicted))))
    if error > _FK_ATOL:
        raise ValueError("staging_arm_fk_calibration_failed")
    return error


def _proxy_snapshot(robot: Any, context: Any, active: str) -> dict[str, Any]:
    """Measure owner poses; arm calibration is strict, hand error is diagnostic."""
    import yourdfpy

    if not torch.equal(robot.get_qpos()[0], context.robot.qpos[0]):
        raise ValueError("staging_observed_qpos_changed")
    model = yourdfpy.URDF.load(
        robot.cfg.fpath,
        load_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    children: dict[str, set[str]] = {}
    parents = {}
    for joint in model.robot.joints:
        children.setdefault(joint.parent, set()).add(joint.child)
        if joint.child in parents:
            raise ValueError("staging_ambiguous_arm_chain")
        parents[joint.child] = joint
    if active not in robot.control_parts or robot.get_solver(active) is None:
        raise ValueError("staging_missing_active_solver")
    active_links = _descendants(children, robot.get_solver(active).root_link_name)
    alternatives = [
        (part, robot.get_solver(part))
        for part in robot.control_parts
        if part != active and robot.get_solver(part) is not None
    ]
    if len(alternatives) != 1 or not _descendants(
        children, alternatives[0][1].root_link_name
    ).isdisjoint(active_links):
        raise ValueError("staging_ambiguous_passive_arm")
    passive, passive_solver = alternatives[0]
    if not set(robot.get_joint_ids(active)).isdisjoint(robot.get_joint_ids(passive)):
        raise ValueError("staging_overlapping_arm_joints")
    world_root = _pose(robot.get_local_pose(to_matrix=True)).copy()
    if context.robot.root_pose is not None:
        _calibrate(context.robot.root_pose, world_root)
    model.update_cfg(
        {
            name: float(value)
            for name, value in zip(robot.joint_names, context.robot.qpos[0])
            if name in model.joint_map
        }
    )
    tcp, roots, distal, chains = {}, {}, {}, {}
    for part, solver in ((active, robot.get_solver(active)), (passive, passive_solver)):
        cursor, moving, chain = solver.end_link_name, [], {solver.root_link_name}
        while cursor != solver.root_link_name:
            if cursor in chain or cursor not in parents:
                raise ValueError("staging_missing_arm_chain")
            chain.add(cursor)
            joint = parents[cursor]
            if joint.type != "fixed":
                moving.append(cursor)
            cursor = joint.parent
        if len(moving) < 2:
            raise ValueError("staging_missing_distal_arm_chain")
        distal[part], chains[part] = _descendants(children, moving[1]), chain
        for link in chain:
            _calibrate(
                robot.get_link_pose(link, env_ids=[0], to_matrix=True),
                world_root @ model.get_transform(link),
            )
        roots[part] = _pose(
            robot.get_link_pose(solver.root_link_name, env_ids=[0], to_matrix=True)
        )
        joint_ids = tuple(robot.get_joint_ids(part))
        if not joint_ids:
            raise ValueError("staging_missing_arm_joints")
        tcp[part] = _pose(
            robot.compute_fk(
                context.robot.qpos[:, list(joint_ids)],
                name=part,
                env_ids=[0],
                to_matrix=True,
            )
        )
        _calibrate(
            tcp[part],
            world_root
            @ model.get_transform(solver.end_link_name)
            @ _pose(solver.get_tcp()),
        )
    points: dict[str, list[np.ndarray]] = {active: [], passive: []}
    hand_error: dict[str, float] = {active: 0.0, passive: 0.0}
    graph = model.collision_scene.graph
    for node in graph.nodes_geometry:
        owner, visited = node, set()
        while owner not in model.link_map:
            if owner in visited or owner not in graph.transforms.parents:
                raise ValueError("staging_missing_mesh_owner")
            visited.add(owner)
            owner = graph.transforms.parents[owner]
        _, mesh_name = graph[node]
        vertices = np.asarray(model.collision_scene.geometry[mesh_name].vertices)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("staging_invalid_collision_mesh")
        for part in (active, passive):
            if owner not in distal[part]:
                continue
            observed = _pose(robot.get_link_pose(owner, env_ids=[0], to_matrix=True))
            predicted = _pose(world_root @ model.get_transform(owner))
            if owner in chains[part]:
                _calibrate(observed, predicted)
            else:
                hand_error[part] = max(
                    hand_error[part], float(np.max(np.abs(observed - predicted)))
                )
            transform = observed @ _pose(
                model.get_transform(node, frame_from=owner, collision_geometry=True)
            )
            points[part].append(vertices @ transform[:3, :3].T + transform[:3, 3])
    if any(not points[part] for part in (active, passive)):
        raise ValueError("staging_missing_distal_meshes")
    if not torch.equal(
        robot.get_qpos()[0], context.robot.qpos[0]
    ) or not np.array_equal(_pose(robot.get_local_pose(to_matrix=True)), world_root):
        raise ValueError("staging_observed_robot_changed")
    return dict(
        active=active,
        passive=passive,
        tcp=tcp,
        roots=roots,
        points={part: np.concatenate(values) for part, values in points.items()},
        hand_owner_fk_error=hand_error,
    )


def build_staging_variants(
    request: Any, context: Any, robot: Any
) -> list[tuple[str, Any, dict[str, Any]]]:
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
    snapshot = _proxy_snapshot(robot, context, arm.control_part)
    active, passive = snapshot["active"], snapshot["passive"]
    variants = []
    for label, anchors in (("B", snapshot["tcp"]), ("C", snapshot["roots"])):
        report = {
            "variant": label,
            "scope": "measured_distal_mesh_hint_only",
            "hand_owner_fk_error": snapshot["hand_owner_fk_error"],
        }
        direction = anchors[active][:2, 3] - anchors[passive][:2, 3]
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length <= np.finfo(np.float32).eps:
            variants.append(
                (label, None, {**report, "reason": "degenerate_xy_direction"})
            )
            continue
        direction = direction / length
        active_projection = snapshot["points"][active][:, :2] @ direction
        passive_projection = snapshot["points"][passive][:, :2] @ direction
        active_span, passive_span = float(np.ptp(active_projection)), float(
            np.ptp(passive_projection)
        )
        stage_delta = (_pose(targets[:, 0]) @ attachment)[:2, 3] - snapshot["tcp"][
            active
        ][:2, 3]
        required = max(
            float(passive_projection.max())
            - float(active_projection.min())
            - min(float(stage_delta @ direction), 0),
            0,
        )
        bound = active_span + passive_span
        shift = max(required + _FK_ATOL, active_span / 2)
        if min(active_span, passive_span) <= np.finfo(np.float32).eps or shift > bound:
            variants.append(
                (label, None, {**report, "reason": "distal_mesh_bound_exceeded"})
            )
            continue
        offset = targets.new_tensor([direction[0] * shift, direction[1] * shift, 0.0])
        outward, elevated = object_pose.clone(), targets[:, 0].clone()
        outward[:, :3, 3] += offset
        elevated[:, :3, 3] += offset
        poses = torch.stack((outward, elevated, targets[:, -1]), dim=1)
        variants.append(
            (
                label,
                replace(request, goal=replace(request.goal, object_target_pose=poses)),
                {
                    **report,
                    "direction_xy": direction.tolist(),
                    "shift_m": shift,
                    "shift_bound_m": bound,
                },
            )
        )
    return variants


def recover_staging(
    original: Any,
    request: Any,
    context: Any,
    robot: Any,
    scope: Any,
    before_original: _RandomState | None,
    build: Callable[[Any], Any],
    *,
    retention_source: dict[str, Any] | None = None,
) -> Any:
    if (
        before_original is None
        or original.plan_success.any()
        or scope is None
        or not scope.rejected_raw_success
        or len(scope.calls) != 8
        or not scope.fallback_attempted
    ):
        return original
    after_legacy = _RandomState.capture(
        _cuda_devices(robot.device, context.robot.qpos.device)
    )
    attempts = []
    selected = original
    selected_boundary = None
    try:
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        initial = _inspect_pick_motion(robot, context.robot.qpos, arm.control_part)
        if not initial["valid"]:
            attempts.append({"variant": "observed_start", **initial})
        else:
            for label, candidate_request, report in build_staging_variants(
                request, context, robot
            ):
                if candidate_request is None:
                    attempts.append({**report, "accepted": False})
                    continue
                before_original.restore()
                candidate = build(candidate_request)
                candidate = qualify_alignment_plan(
                    candidate_request, context, candidate, robot
                )
                certificate = (
                    inspect_staging_plan(
                        robot,
                        candidate_request,
                        context,
                        candidate,
                        retention_source=retention_source,
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
                "gen_sim_staging_recovery": {
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
