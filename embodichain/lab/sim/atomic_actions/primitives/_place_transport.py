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

"""Atomic-owned held-body free transport and mandatory Place contact phases."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import torch

from embodichain.compute.geometry.mesh_bounds import aabb_sphere_cover
from embodichain.compute.trajectory import retime_to_control_grid
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.goal_seeded import (
    qualify_ik_waypoints,
    solve_seeded_pose_path,
)
from embodichain.lab.sim.motion.ik import sample_cartesian_targets
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import CuroboPlanOptions
from embodichain.lab.sim.motion.planners.curobo.payload import CuroboPayload
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanState
from embodichain.utils.math import pose_inv
from ..bindings import JointPositionTarget
from ..goals import resolve_pose_goal
from ..plans import ActionPlan
from ..trajectory_ops import split_three_segments
from . import _place_branch as branch

if TYPE_CHECKING:
    from ._place_branch import PlaceBranchBuild

__all__: list[str] = []


@dataclass
class _ContactPhases:
    desired: tuple[torch.Tensor, torch.Tensor]
    qposes: torch.Tensor
    positions: torch.Tensor
    monitor_until: dict[str, int] = field(default_factory=dict)


def prepare_contact(
    scope: PlaceBranchBuild,
    poses: torch.Tensor,
    qposes: torch.Tensor,
    arm: JointPositionTarget,
) -> tuple[torch.Tensor, _ContactPhases | None]:
    """Distinguish an internal free midpoint from caller-owned contact targets."""
    options = scope.request.skill_options
    if options.cartesian_waypoint_count != 2:
        return qposes, None
    goal = resolve_pose_goal(
        scope.request.goal.xpos, scope.context, name="release_goal"
    )
    if goal.shape != (1, 4, 4) or [g.shape[1] for g in scope.groups] != [4, 2]:
        raise ValueError(
            "Free transport requires one release goal and shared Place phases."
        )
    start = scope.action.robot.compute_fk(
        qpos=scope.context.robot.qpos[:, list(arm.joint_ids)],
        name=arm.control_part,
        to_matrix=True,
    )
    expected = (start[:, :3, 3] + poses[:, 1, :3, 3]) / 2
    if not torch.allclose(poses[:, 0, :3, 3], expected, atol=1e-6, rtol=0):
        raise ValueError("Free transport cannot remove a caller-owned waypoint.")
    n_down, _, _ = split_three_segments(
        scope.request.motion_policy.sample_count, options.hand_interp_steps
    )
    count = 2 * max(1, (n_down - 1) // 4) + 1
    targets = sample_cartesian_targets(
        poses[:, 1],
        [PlanState(move_type=MoveType.EEF_MOVE, xpos=poses[:, i]) for i in (2, 3)],
        count,
    )
    robot = scope.action.robot
    bounds = robot.get_qpos_limits(name=arm.control_part).to(qposes)
    points = solve_seeded_pose_path(
        robot,
        arm.control_part,
        torch.stack([target.xpos[0] for target in targets]),
        qposes[:, 1],
        control_dt=scope.context.require_control_dt(),
    )
    if points is None:
        raise ValueError("Place Cartesian contact has no qualified continuous IK path.")
    contact = points[None]
    desired = torch.stack([poses[:, 1], *(target.xpos for target in targets)], 1)
    actual = robot.compute_batch_fk(qpos=contact, name=arm.control_part, to_matrix=True)
    if not branch._pose_path_valid(
        actual, desired, *branch._tolerances(robot, arm.control_part)
    ):
        raise ValueError("Place Cartesian contact FK rejected.")
    result = qposes.clone()
    midpoint = (count - 1) // 2
    result[:, 2], result[:, 3] = contact[:, midpoint], contact[:, -1]
    result[:, 4], result[:, 5] = contact[:, midpoint], contact[:, 0]
    evidence = qualify_ik_waypoints(
        robot,
        arm.control_part,
        result,
        poses,
        torch.ones(poses.shape[:2], dtype=torch.bool, device=poses.device),
        bounds,
    )
    if not evidence["valid"]:
        raise ValueError("Place contact changed a required release or retreat pose.")
    scope.receipt["waypoints"] = evidence
    scope.receipt["superseded_free_midpoint"] = poses[:, 0].detach().cpu().tolist()
    return result, _ContactPhases(
        (scope.groups[0][:, 1:], scope.groups[1]), result[:, 1:], contact
    )


def _free_prefix(scope: PlaceBranchBuild, arm: JointPositionTarget) -> torch.Tensor:
    context, action = scope.context, scope.action
    held = context.get_held_object(
        scope.request.binding.endpoint("primary", "motion").task_state_key
    )
    subject = held.semantics.entity_id
    vertices = scope.inputs.geometry[subject]
    if (
        vertices.ndim != 3
        or vertices.shape[0] != 1
        or not torch.isfinite(vertices).all()
    ):
        raise ValueError(
            "Place payload requires the complete single-row live geometry."
        )
    if scope.preserved_transport:
        pose = action.robot.compute_fk(
            qpos=context.robot.qpos[:, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )
        pose = pose @ pose_inv(held.object_to_eef.to(pose))
        if pose.shape != (1, 4, 4) or not torch.isfinite(pose).all():
            raise ValueError("Place payload requires a finite projected held pose.")
    else:
        entity = context.scene.entities.get(subject)
        if entity is None or entity.confidence <= 0:
            raise ValueError("Place payload requires its same-observation object pose.")
        pose = entity.pose
    services = action.planning_services.curobo_queries
    if services is None or services.robot is not action.robot:
        raise ValueError(
            "Place payload transport requires its declared observed cuRobo query service."
        )
    payload = CuroboPayload(
        object_id=subject,
        control_part=arm.control_part,
        spheres=aabb_sphere_cover(vertices[0].clone()),
        object_pose=pose,
    )
    query = services.open(
        context=context,
        control_part=arm.control_part,
        full_qpos=context.robot.qpos,
        payload=payload,
    )
    dt = context.require_control_dt()
    record: dict[str, Any] = {
        "provider": "curobo_payload_direct",
        "payload": subject,
        "simulator_attachment": False,
        **(
            {"payload_pose_source": "projected_held_context"}
            if scope.preserved_transport
            else {}
        ),
        "single_release_goal_preserved": True,
        "superseded_sample_source": "Place._translation_keyframes transport midpoint",
    }
    scope.receipt["free_transport"] = record
    start, target = (
        context.robot.qpos[:, list(arm.joint_ids)],
        scope.contact.qposes[:, 0],
    )
    device = services.planning_device()
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    device_scope = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with device_scope, torch.random.fork_rng(devices=devices):
        with query as generator:
            info = generator.collision_world_info
            if info is None:
                raise ValueError("Place payload query has no declared collision world.")
            record["world_ids"] = list(info.entity_ids)
            result = generator.generate(
                [PlanState(move_type=MoveType.JOINT_MOVE, qpos=target)],
                MotionGenOptions(
                    strategy="motion_gen",
                    start_qpos=start,
                    control_part=arm.control_part,
                    sample_count=None,
                    velocity_limit=None,
                    acceleration_limit=None,
                    plan_opts=CuroboPlanOptions(),
                ),
            )
            record["native_plan_success"] = result.success.cpu().tolist()
            if (
                not result.success.all()
                or result.positions is None
                or result.dt is None
            ):
                raise ValueError("Place payload-aware free transport failed.")
            prefix, _, _, _ = retime_to_control_grid(
                result.positions.to(start), result.dt.to(start), dt
            )
            error = max(
                float((prefix[:, 0] - start).abs().max()),
                float((prefix[:, -1] - target).abs().max()),
            )
            record["endpoint_joint_error"] = error
            if error > 1e-6:
                raise ValueError("Place free transport changed exact joint endpoints.")
            prefix[:, 0], prefix[:, -1] = start, target
            return prefix


def compose(
    scope: PlaceBranchBuild, original: ActionPlan, arm: JointPositionTarget
) -> tuple[torch.Tensor, dict[str, int], torch.Tensor]:
    """Compose explicit free/contact/release/retract phases before final checks."""
    approach, release, retract = branch._phases(original)
    contact = scope.contact
    if contact is None:
        raise ValueError("Place transport requires a qualified contact phase.")
    old, ids = original.joint_trajectory.positions, list(arm.joint_ids)
    dt = scope.context.require_control_dt()
    times = original.joint_trajectory.dt
    if (times[:, 0] != 0).any() or not (times[:, 1:] == dt).all():
        raise ValueError("Place transport requires the original uniform control grid.")
    passive = [i for i in range(old.shape[-1]) if i not in ids]
    if not torch.equal(
        old[:, : approach.stop, passive],
        old[:, :1, passive].expand(-1, approach.stop, -1),
    ):
        raise ValueError("Free transport cannot replace moving passive commands.")
    if not torch.equal(
        old[:, release.start : release.stop, ids],
        contact.positions[:, -1:].expand(-1, release.stop - release.start, -1),
    ):
        raise ValueError("Free transport cannot alter release arm commands.")
    prefix = _free_prefix(scope, arm)
    down = torch.cat((prefix, contact.positions[:, 1:]), 1)
    count = max(approach.stop, down.shape[1])
    down = torch.cat((down[:, :1].expand(-1, count - down.shape[1], -1), down), 1)
    back = contact.positions.flip(1)
    n_back = retract.stop - retract.start
    if back.shape[1] > n_back:
        raise ValueError("Place Cartesian retract exceeds its original duration.")
    back = torch.cat((back[:, :1].expand(-1, n_back - back.shape[1], -1), back), 1)
    for values in (down, back):
        times = values.new_tensor([[0.0, *([dt] * (values.shape[1] - 1))]])
        if not joint_velocity_validity(
            values,
            times,
            authored_joint_velocity_limits(scope.action.robot, arm.control_part).to(
                values
            ),
        ).all():
            raise ValueError(
                "Place transport/contact violates original velocity limits."
            )
    full_prefix = old[:, :1].expand(-1, count, -1).clone()
    full_prefix[:, :, ids] = down
    full = torch.cat(
        (
            full_prefix,
            old[:, release.start : release.stop],
            old[:, retract.start : retract.stop],
        ),
        1,
    )
    full[:, count + release.stop - release.start :, ids] = back
    timeout = scope.request.recovery_policy.action_timeout
    if timeout is None or (full.shape[1] - 1) * dt > timeout:
        raise ValueError("Place free transport exceeds the original action timeout.")
    for key, boundary in original.scene_dependency_monitor_until.items():
        if boundary not in (0, approach.stop):
            raise ValueError(
                "Free transport cannot remap a non-phase monitor boundary."
            )
        contact.monitor_until[key] = 0 if boundary == 0 else count
    scope.receipt["free_transport"].update(
        original_approach_samples=approach.stop,
        approach_samples=count,
        native_prefix_samples=prefix.shape[1],
        contact_samples=contact.positions.shape[1],
        total_seconds=(full.shape[1] - 1) * dt,
        action_timeout=timeout,
    )
    scope.receipt["timing_unchanged"] = count == approach.stop
    lengths = {
        "approach": count,
        "release": release.stop - release.start,
        "retract": n_back,
    }
    times = full.new_full(full.shape[:2], dt)
    times[:, 0] = 0
    return full, lengths, times
