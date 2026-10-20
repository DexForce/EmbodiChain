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

"""Call-local free transport followed by the unchanged Cartesian Place contact."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

import torch

from embodichain.compute.trajectory import retime_to_control_grid
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.sim.atomic_actions.trajectory_ops import split_three_segments
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanState
from . import motion, place_branch as branch
from .drawer_curobo import (
    CuroboPlannerCfg,
    CuroboWorldCfg,
    CuroboPlanOptions,
    _PayloadMotionGenerator,
    world_sources,
)

if TYPE_CHECKING:
    from .place_branch import _Scope

__all__: list[str] = []


@dataclass
class _ContactPhases:
    desired: tuple[torch.Tensor, torch.Tensor]
    qposes: torch.Tensor
    positions: torch.Tensor
    monitor_until: dict[str, int] = field(default_factory=dict)


def prepare_contact(
    scope: _Scope, poses: torch.Tensor, qposes: torch.Tensor, arm: JointPositionTarget
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
    targets = motion._cartesian_samples(
        poses[:, 1],
        [PlanState(move_type=MoveType.EEF_MOVE, xpos=poses[:, i]) for i in (2, 3)],
        count,
    )
    robot = scope.action.robot
    bounds = robot.get_qpos_limits(name=arm.control_part).to(qposes)
    points = motion.resolve_seeded_ik_path(
        targets,
        qposes[:, 1],
        solve=lambda target, seed: robot.compute_ik(
            pose=target.xpos, name=arm.control_part, joint_seed=seed
        ),
        lower=bounds[..., 0],
        upper=bounds[..., 1],
        limits=motion._joint_velocity_limits(robot, arm.control_part).to(qposes),
        dt=scope.context.require_control_dt(),
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
    evidence = branch._qualification(
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


def _free_prefix(scope: _Scope, arm: JointPositionTarget) -> torch.Tensor:
    context, action = scope.context, scope.action
    held = context.get_held_object(
        scope.request.binding.endpoint("primary", "motion").task_state_key
    )
    simulation = SimulationManager.get_instance(
        action.motion_generator.planner.cfg.sim_instance_id
    )
    subject = held.semantics.entity_id
    vertices = action.recovery_scene_vertices[subject]
    if (
        vertices.ndim != 3
        or vertices.shape[0] != 1
        or not torch.isfinite(vertices).all()
    ):
        raise ValueError(
            "Place payload requires the complete single-row live geometry."
        )
    observation = SimpleNamespace(
        route=SimpleNamespace(object_id=subject),
        obj=simulation.get_rigid_object(subject),
        vertices=vertices[0].clone(),
    )
    sources = world_sources(simulation, subject)
    dt = context.require_control_dt()
    cfg = CuroboPlannerCfg(
        robot_uid=action.robot.uid,
        sim_instance_id=action.motion_generator.planner.cfg.sim_instance_id,
        world=CuroboWorldCfg(rigid_objects=sources, max_voxel_count=4_000_000),
        use_cuda_graph=False,
        warmup_iterations=0,
        interpolation_dt=dt,
        preserve_plan_samples=True,
    )
    record: dict[str, Any] = {
        "provider": "curobo_payload_direct",
        "payload": subject,
        "world_ids": list(sources),
        "simulator_attachment": False,
        "single_release_goal_preserved": True,
        "superseded_sample_source": "Place._translation_keyframes transport midpoint",
    }
    scope.receipt["free_transport"] = record
    start, target = (
        context.robot.qpos[:, list(arm.joint_ids)],
        scope.contact.qposes[:, 0],
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        generator = _PayloadMotionGenerator(
            cfg, observation=observation, motion=arm, qpos=context.robot.qpos
        )
        try:
            result = generator.generate(
                [PlanState(move_type=MoveType.JOINT_MOVE, qpos=target)],
                MotionGenOptions(
                    strategy="motion_gen",
                    start_qpos=start,
                    control_part=arm.control_part,
                    sample_count=None,
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
        finally:
            generator.planner.close()


def compose(
    scope: _Scope, original: ActionPlan, arm: JointPositionTarget
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
        if not motion._velocity_validity(
            values,
            times,
            motion._joint_velocity_limits(scope.action.robot, arm.control_part).to(
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
