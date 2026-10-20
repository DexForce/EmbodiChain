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

"""GenSim ordinary pickup with continuous contact and qualified free-space entry."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import torch

from embodichain.compute.trajectory import retime_to_control_grid
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.plans import PlanningFailure, TimedTrajectory
from embodichain.lab.sim.atomic_actions.trajectory_ops import build_pose_plan_states
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanState
from embodichain.lab.sim.motion.planners.utils import interpolate_xpos_batched
from embodichain.utils import logger
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

from .articulation_collision import check_free_motion
from .held_alignment_recovery import _RandomState, _cuda_devices
from .motion import _joint_velocity_limits, resolve_continuous_ik_path
from .pick_safety import _inspect_pick_plan
from .solver_tolerances import _pose_validation_parameters

__all__: list[str] = []


def eligible(action: Any, request: Any, context: Any, selection: Any) -> bool:
    """Do not change unmarked, explicit, constrained, drawer or pour contracts."""
    if getattr(action, "path_simulation", None) is None:
        return False
    options, goal = request.skill_options, request.goal
    return bool(
        (getattr(action, "pick_purposes", None) or {}).get(request.invocation_id)
        == "ordinary"
        and request.invocation_id
        not in getattr(action, "path_excluded_invocations", ())
        and context.batch_size == 1
        and context.env_ids.tolist() == [0]
        and not selection.constrained
        and selection.direction is not None
        and isinstance(goal.semantics.affordance, AntipodalAffordance)
        and goal.object_pose is None
        and goal.grasp_xpos is None
        and options.fixed_object_to_eef is None
        and options.rotate_upright is None
        and options.obj_upright_direction is None
        and options.pick_object_part == "center"
        and request.motion_policy.strategy == "ik_interp"
        and request.motion_policy.dynamic_collision_mode == "off"
    )


def _pose_path_valid(
    actual: torch.Tensor,
    desired: torch.Tensor,
    position_tolerance: float,
    rotation_tolerance: float,
) -> bool:
    """Check every Cartesian sample, including intermediates and both endpoints."""
    if actual.shape != desired.shape or not torch.isfinite(actual).all():
        return False
    position = torch.linalg.vector_norm(
        actual[..., :3, 3] - desired[..., :3, 3], dim=-1
    )
    rotation = quat_error_magnitude(
        quat_from_matrix(actual[..., :3, :3]), quat_from_matrix(desired[..., :3, :3])
    )
    return bool(
        (position <= position_tolerance + 1e-6).all()
        and (rotation <= rotation_tolerance * 2 + 1e-5).all()
    )


def _compose_approach(
    original: torch.Tensor,
    approach_stop: int,
    arm_ids: list[int],
    prefix: torch.Tensor,
    contact: torch.Tensor,
) -> torch.Tensor:
    """Keep the exact acquired joint endpoint, close/lift suffix and passive holds."""
    if not torch.equal(prefix[:, -1], contact[:, 0]):
        raise ValueError(
            "Free-space and contact paths end on different joint branches."
        )
    if not torch.equal(contact[:, -1], original[:, approach_stop - 1, arm_ids]):
        raise ValueError("Contact path changed the selected grasp joint endpoint.")
    arm = torch.cat((prefix, contact[:, 1:]), dim=1)
    approach = original[:, :1].expand(-1, arm.shape[1], -1).clone()
    approach[:, :, arm_ids] = arm
    return torch.cat((approach, original[:, approach_stop:]), dim=1)


def _curobo_prefix(
    action: Any,
    context: Any,
    part: str,
    start: torch.Tensor,
    target: torch.Tensor,
    opened_full: torch.Tensor,
) -> Any:
    """Use the existing planner with a fresh world and exact joint-space goal."""
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenCfg,
        MotionGenerator,
    )
    from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
        CuroboPlanner,
        CuroboPlannerCfg,
        CuroboWorldCfg,
        CuroboPlanOptions,
    )
    from .drawer_curobo import snapshot_in_base, world_sources

    class SnapshotPlanner(CuroboPlanner):
        def _load_runtime_robot_config(self, path: str) -> dict:
            config = super()._load_runtime_robot_config(path)
            joints = dict(
                zip(self.robot.joint_names, opened_full[0].tolist(), strict=True)
            )
            kinematics = config["robot_cfg"]["kinematics"]
            kinematics["lock_joints"] = {
                name: joints[name] for name in kinematics.get("lock_joints", {})
            }
            return config

        def _materialize_profile(self, control_part: str) -> Any:
            profile = super()._materialize_profile(control_part)
            self.base_pose = self.robot.get_link_pose(
                profile.sim_base_link_name, to_matrix=True
            ).clone()
            return profile

        def _auto_generate_world_scene(self, cfg: Any) -> Any:
            return super()._auto_generate_world_scene(
                replace(
                    cfg,
                    rigid_objects=snapshot_in_base(cfg.rigid_objects, self.base_pose),
                )
            )

    class SnapshotGenerator(MotionGenerator):
        def _create_planner(self, cfg: Any) -> Any:
            return SnapshotPlanner(cfg)

    cfg = CuroboPlannerCfg(
        robot_uid=action.robot.uid,
        sim_instance_id=action.motion_generator.planner.cfg.sim_instance_id,
        world=CuroboWorldCfg(
            rigid_objects=world_sources(action.path_simulation, ""),
            max_voxel_count=4_000_000,
        ),
        use_cuda_graph=False,
        warmup_iterations=0,
        interpolation_dt=context.require_control_dt(),
        preserve_plan_samples=True,
    )
    # cuRobo may use CUDA even when the simulator/IK input is on CPU.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        generator = SnapshotGenerator(MotionGenCfg(planner_cfg=cfg))
        try:
            return generator.generate(
                [PlanState(move_type=MoveType.JOINT_MOVE, qpos=target)],
                options=MotionGenOptions(
                    strategy="motion_gen",
                    start_qpos=start,
                    control_part=part,
                    sample_count=None,
                    plan_opts=CuroboPlanOptions(),
                ),
            )
        finally:
            generator.planner.close()


def rebuild(action: Any, request: Any, context: Any, first: Any, selection: Any) -> Any:
    """Preserve a qualified original; rebuild a curved contact leg without new grasps."""
    if not first.plan_success.all():
        return first
    state = _RandomState.capture(
        _cuda_devices(action.robot.device, context.robot.qpos.device)
    )
    attempts: list[dict[str, Any]] = []
    try:
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        ids = list(arm.joint_ids)
        segments = {segment.name: segment for segment in first.segments}
        if set(segments) != {"approach", "close", "lift"}:
            raise ValueError(
                "Ordinary Pick requires approach, close and lift boundaries."
            )
        stop = segments["approach"].stop
        original = first.joint_trajectory.positions
        held = first.expected_effects.held_object_updates.get(
            request.binding.endpoint("primary", "motion").task_state_key
        )
        if held is None or held.semantics.entity_id != request.goal.semantics.entity_id:
            raise ValueError("Pick requires its exact selected attachment proposal.")
        grasp = held.grasp_xpos.to(original)
        direction = selection.direction.to(grasp)
        pre = grasp.clone()
        pre[:, :3, 3] -= direction * request.skill_options.pre_grasp_distance
        fk = action.robot.compute_batch_fk(
            qpos=original[:, :stop, ids], name=arm.control_part, to_matrix=True
        )
        boundary = int(
            torch.linalg.vector_norm(fk[0, :, :3, 3] - pre[0, :3, 3], dim=-1).argmin()
        )
        solver = action.robot.get_solver(arm.control_part)
        parameters = _pose_validation_parameters(solver)
        pos_eps, rot_eps = parameters["pos_eps"], parameters["rot_eps"]
        if any(not math.isfinite(v) or v <= 0 for v in (pos_eps, rot_eps)):
            raise ValueError("Pick IK pose tolerances are unavailable.")
        points = fk[:, boundary:, :3, 3]
        delta = grasp[:, :3, 3] - pre[:, :3, 3]
        fraction = ((points - pre[:, None, :3, 3]) * delta[:, None]).sum(-1)
        fraction = (fraction / delta.square().sum(-1)[:, None].clamp_min(1e-12)).clamp(
            0, 1
        )
        desired = grasp[:, None].expand_as(fk[:, boundary:]).clone()
        desired[:, :, :3, 3] = (
            pre[:, None, :3, 3] + fraction[..., None] * delta[:, None]
        )
        original_safe = _inspect_pick_plan(action.robot, request, context, first)
        if original_safe["valid"] and _pose_path_valid(
            fk[:, boundary:], desired, pos_eps, rot_eps
        ):
            scene = check_free_motion(
                action.robot,
                action.path_simulation,
                original[0, :stop],
                arm.control_part,
            )
            if scene["valid"]:
                return first
        start_scene = check_free_motion(
            action.robot,
            action.path_simulation,
            context.robot.qpos[0:1],
            arm.control_part,
        )
        if not start_scene["valid"]:
            raise ValueError("Observed Pick start already intersects the scene.")
        dt = context.require_control_dt()
        start = context.robot.qpos[:, ids].to(original)
        qgrasp = original[:, stop - 1, ids]
        base_count = max(2, stop - boundary)
        for count in (base_count, base_count * 2 - 1):
            state.restore()
            contact_targets = build_pose_plan_states(
                interpolate_xpos_batched(grasp, pre, count)[:, 1:]
            )
            contact_options = MotionGenOptions(
                strategy="ik_interp",
                start_qpos=qgrasp,
                control_part=arm.control_part,
                sample_count=count,
                interpolation_dt=dt,
                preserve_cartesian_samples=True,
                is_linear=True,
            )
            backward = action.motion_generator._generate_checked(
                contact_targets,
                options=contact_options,
            )
            note: dict[str, Any] = {
                "contact_samples": count,
                "contact_ik_valid": bool(backward.success.all()),
            }
            attempts.append(note)
            if not backward.success.all():
                recovered = resolve_continuous_ik_path(
                    action.motion_generator,
                    contact_targets,
                    contact_options,
                    operation="Pick contact",
                )
                note["contact_local_ik_valid"] = recovered is not None
                if recovered is None:
                    continue
                backward = recovered
            contact = backward.positions.flip(1)
            poses = action.robot.compute_batch_fk(
                qpos=contact, name=arm.control_part, to_matrix=True
            )
            desired = interpolate_xpos_batched(pre, grasp, count)
            if not _pose_path_valid(poses, desired, pos_eps, rot_eps):
                note["pose_valid"] = False
                continue
            target = contact[:, 0]
            limits = _joint_velocity_limits(action.robot, arm.control_part).to(start)
            if not torch.isfinite(limits).all() or (limits <= 0).any():
                raise ValueError(
                    "Pick requires finite positive active joint velocity limits."
                )
            required = (
                int(torch.ceil(((target - start).abs() / (limits * dt)).max()).item())
                + 1
            )
            prefix_count = max(2, boundary + 1, required)
            budget = request.recovery_policy.action_timeout
            minimum_samples = prefix_count + count - 1 + original.shape[1] - stop
            if budget is not None and (minimum_samples - 1) * dt > budget:
                note["reason"] = "action_duration_budget"
                continue
            direct = action.motion_generator._generate_checked(
                [PlanState(move_type=MoveType.JOINT_MOVE, qpos=target)],
                options=MotionGenOptions(
                    strategy="ik_interp",
                    start_qpos=start,
                    control_part=arm.control_part,
                    sample_count=prefix_count,
                    interpolation_dt=dt,
                ),
            )
            for provider in ("joint_interp", "curobo"):
                if provider == "joint_interp":
                    result = direct
                else:
                    state.restore()
                    result = _curobo_prefix(
                        action, context, arm.control_part, start, target, original[:, 0]
                    )
                if (
                    not result.success.all()
                    or result.positions is None
                    or result.dt is None
                ):
                    note[provider] = {
                        "valid": False,
                        "reason": "prefix_planning_failed",
                    }
                    continue
                prefix, _, _, _ = retime_to_control_grid(
                    result.positions, result.dt, dt
                )
                error = max(
                    float((prefix[:, 0] - start).abs().max()),
                    float((prefix[:, -1] - target).abs().max()),
                )
                if error > 1e-6:
                    raise ValueError(
                        "Pick prefix endpoints differ beyond native rounding."
                    )
                prefix[:, 0], prefix[:, -1] = start, target
                full = _compose_approach(original, stop, ids, prefix, contact)
                lengths = {
                    segment.name: segment.stop - segment.start
                    for segment in first.segments
                }
                lengths["approach"] = prefix.shape[1] + contact.shape[1] - 1
                if budget is not None and (full.shape[1] - 1) * dt > budget:
                    note[provider] = {
                        "valid": False,
                        "reason": "action_duration_budget",
                    }
                    continue
                monitor = dict(first.scene_dependency_monitor_until)
                for uid in monitor:
                    monitor[uid] = max(
                        1,
                        math.ceil(
                            lengths["approach"]
                            * request.skill_options.grasp_commit_fraction
                        ),
                    )
                plan = action.build_plan(
                    request,
                    context,
                    success=first.plan_success,
                    trajectory=TimedTrajectory.from_uniform_step(
                        full, env_ids=context.env_ids, step_dt=dt
                    ),
                    expected_effects=first.expected_effects,
                    effect_candidates=first.effect_candidates,
                    effect_verification=first.effect_verification,
                    replannable=first.replannable,
                    diagnostics=replace(
                        first.diagnostics,
                        metadata={
                            **first.diagnostics.metadata,
                            "gen_sim_pick_path": {
                                "provider": provider,
                                "approach_samples": lengths["approach"],
                                "contact_samples": count,
                                "selected_grasp_unchanged": True,
                                "close_lift_unchanged": True,
                                "attempts": attempts,
                            },
                        },
                    ),
                    segment_lengths=lengths,
                    scene_dependency_monitor_until=monitor,
                    scene_dependency_end_segment=first.scene_dependency_end_segment,
                )
                certificate = _inspect_pick_plan(action.robot, request, context, plan)
                note[provider] = certificate
                if not certificate["valid"]:
                    continue
                scene = check_free_motion(
                    action.robot,
                    action.path_simulation,
                    full[0, : lengths["approach"]],
                    arm.control_part,
                )
                note[provider] = {**certificate, "approach_scene": scene}
                if scene["valid"]:
                    logger.log_info(
                        f"GenSim Pick path: {arm.control_part}, {provider}, {lengths['approach']} approach samples."
                    )
                    return replace(
                        plan,
                        diagnostics=replace(
                            plan.diagnostics,
                            metadata={
                                **plan.diagnostics.metadata,
                                "gen_sim_pick_path": {
                                    **plan.diagnostics.metadata["gen_sim_pick_path"],
                                    "accepted": True,
                                    "attempts": attempts,
                                    "qualification_scope": "full_robot_and_approach_scene_discrete_meshes",
                                },
                            },
                        ),
                    )
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        attempts.append({"unavailable": str(error)})
    finally:
        state.restore()
    failed = action.failed_plan(
        request, context, message="No qualified continuous ordinary Pick path."
    )
    return replace(
        failed,
        diagnostics=replace(
            failed.diagnostics,
            failure=PlanningFailure("gen_sim_pick_path_unqualified", retryable=False),
            metadata={
                **first.diagnostics.metadata,
                "gen_sim_pick_path": {"attempts": attempts},
            },
        ),
    )
