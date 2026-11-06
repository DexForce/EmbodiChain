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

"""Planning of explicit ordered-pad Pour goal variants."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from embodichain.compute.geometry.pouring import jaw_pour_frames, pour_frames
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.ik import resolve_continuous_ik_path
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.mesh_collision import RobotMeshCollisionSnapshot
from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.utils.math import pose_inv

from ..affordance import AxisAlignAffordance
from ..bindings import EndpointBinding, JointPositionTarget
from ..control import GRASP_COMMAND
from ..effects import StateDelta
from ..invocation import ResolvedActionRequest
from ..plans import ActionPlan, PlannerDiagnostics
from ..state import HeldObjectState, PlanningContext
from ..trajectory_ops import build_pose_plan_states, to_full_robot_trajectory
from ._helpers import arm_qpos_from_state, require_shared_task_state_key

if TYPE_CHECKING:
    from .pour import JawPourGoal, Pour, PourOptions, RetainedPourGoal

__all__: list[str] = []


def _scene_pose(context: PlanningContext, entity_id: str) -> torch.Tensor:
    entity = context.scene.entities.get(entity_id)
    if entity is None or entity.pose is None:
        raise ValueError(f"Pour entity {entity_id!r} is not observed.")
    pose = entity.pose
    return pose[None].expand(context.batch_size, -1, -1) if pose.ndim == 2 else pose


def _ordered_pad_line(
    action: Pour, motion: JointPositionTarget, grasp: JointPositionTarget
) -> torch.Tensor:
    from embodichain.lab.sim.utility.grasp_geometry import ordered_pad_line_world

    cfg = action.planning_services.hand_geometry_cfg(grasp.target_id)
    if cfg.motion_part != motion.control_part or cfg.hand_part != grasp.control_part:
        raise ValueError("Pour hand calibration differs from its bound endpoints.")
    return ordered_pad_line_world(action.robot, cfg=cfg)


def plan_pad_pour(
    action: Pour,
    request: ResolvedActionRequest[JawPourGoal | RetainedPourGoal, PourOptions],
    context: PlanningContext,
) -> ActionPlan:
    """Resolve explicit observations and dispatch one of the two source arcs."""
    from .pour import RetainedPourGoal

    goal = request.goal
    motion = request.binding.endpoint("primary", "motion")
    grasp = request.binding.endpoint("primary", "grasp")
    motion_target = motion.require_target(JointPositionTarget)
    hand_target = grasp.require_target(JointPositionTarget)
    resource = require_shared_task_state_key(
        motion, grasp, participant="Pour primary participant"
    )
    held = context.get_held_object(resource)
    if held is None or not context.task.exclusive_held_object_mask(resource).all():
        raise ValueError("Ordered-pad Pour requires an exclusive held object.")
    arm_qpos = arm_qpos_from_state(context, list(motion_target.joint_ids))
    eef = action.robot.compute_fk(
        qpos=arm_qpos, name=motion_target.control_part, to_matrix=True
    )
    object_to_eef = held.object_to_eef.to(action.device, torch.float32)
    object_pose = eef @ pose_inv(object_to_eef)
    jaw_world = _ordered_pad_line(action, motion_target, hand_target)
    jaw_local = torch.bmm(
        object_pose[:, :3, :3].transpose(1, 2), jaw_world[..., None]
    ).squeeze(-1)
    retained = isinstance(goal, RetainedPourGoal)
    if retained:
        if context.batch_size != 1 or held.semantics.entity_id != goal.source_entity_id:
            raise ValueError("Retained Pour requires its exact B=1 held source.")
        receiver_pose = _scene_pose(context, goal.receiver_entity_id)
        frames = pour_frames(
            goal.source_vertices.detach().cpu().numpy(),
            goal.receiver_vertices.detach().cpu().numpy(),
            object_pose[0].detach().cpu().numpy(),
            receiver_pose[0].detach().cpu().numpy(),
            jaw_local[0].detach().cpu().numpy(),
            float(request.skill_options.rotate_angle),
            goal.return_pose.detach().cpu().numpy(),
            sample_count=request.motion_policy.sample_count,
            gravity_world=goal.gravity_world,
        )
        object_poses = torch.as_tensor(
            frames["object_arc"], device=action.device, dtype=object_pose.dtype
        )[None]
        metadata = {
            "pad_line_local": jaw_local.detach().cpu().tolist(),
            "receiver_entity_id": goal.receiver_entity_id,
            "tilt_axis_local": [frames["tilt_axis_object"].tolist()],
            "pour_geometry": {
                "revision": 2 if goal.proxy_position_tolerance_m is not None else 1,
                **(
                    {"proxy_position_tolerance_m": goal.proxy_position_tolerance_m}
                    if goal.proxy_position_tolerance_m is not None
                    else {}
                ),
                "source_entity_id": goal.source_entity_id,
                "receiver_entity_id": goal.receiver_entity_id,
                "physical_evidence": False,
                "fluid_transfer": "unmodeled",
                "planning_pose_source": "held_fk_and_existing_object_to_eef",
                "observed_source_entry_pose": _scene_pose(
                    context, goal.source_entity_id
                )[0]
                .detach()
                .cpu()
                .tolist(),
                "entry_pose": frames["object_arc"][0].tolist(),
                "peak_pose": frames["object_arc"][frames["peak_index"]].tolist(),
                "restore_pose": frames["object_arc"][-1].tolist(),
                "pivot_object": frames["pivot_object"].tolist(),
                "pivot_world": frames["pivot_world"].tolist(),
                "receiver_pose": receiver_pose[0].detach().cpu().tolist(),
                "gravity_world": list(goal.gravity_world),
                "target_roles": list(frames["target_roles"]),
                "receiver_bearing_m": frames["receiver_bearing_m"],
                "receiver_bearing_tie": frames["receiver_bearing_tie"],
                "tilt_direction_policy": frames["tilt_direction_policy"],
                "effective_angle_rad": frames["effective_angle_rad"],
                "peak_index": frames["peak_index"],
                "sample_count": request.motion_policy.sample_count,
            },
        }
    else:
        affordance = held.semantics.affordance
        if not isinstance(affordance, AxisAlignAffordance):
            raise ValueError("Jaw Pour requires an axis-aware grasp affordance.")
        receiver_pose = (
            None
            if goal.receiver_entity_id is None
            else _scene_pose(context, goal.receiver_entity_id).to(
                device=action.device, dtype=object_pose.dtype
            )
        )
        legacy = jaw_pour_frames(
            object_pose,
            jaw_local,
            affordance.internal_axis.to(action.device, torch.float32),
            getattr(affordance, "mesh_vertices", None),
            receiver_pose,
            angle_rad=float(request.skill_options.rotate_angle),
            sample_count=request.motion_policy.sample_count,
        )
        object_poses = legacy["object_poses"]
        metadata = {
            "pad_line_local": jaw_local.detach().cpu().tolist(),
            "receiver_entity_id": goal.receiver_entity_id,
            "tilt_axis_local": legacy["tilt_axis_object"].detach().cpu().tolist(),
            "tilt_direction_local": legacy["tilt_direction_object"]
            .detach()
            .cpu()
            .tolist(),
        }
    return _build_arc_plan(
        action,
        request,
        context,
        resource,
        held,
        arm_qpos,
        motion_target,
        grasp,
        object_to_eef,
        object_poses,
        metadata,
        retained=retained,
    )


def _build_arc_plan(
    action: Pour,
    request: ResolvedActionRequest,
    context: PlanningContext,
    resource: str,
    held: HeldObjectState,
    arm_qpos: torch.Tensor,
    motion_target: JointPositionTarget,
    grasp: EndpointBinding,
    object_to_eef: torch.Tensor,
    object_poses: torch.Tensor,
    metadata: dict[str, Any],
    *,
    retained: bool,
) -> ActionPlan:
    eef_targets = (object_poses @ object_to_eef[:, None])[:, 1:]
    options = request.motion_policy.to_motion_gen_options(
        start_qpos=arm_qpos,
        control_part=motion_target.control_part,
        interpolation_dt=context.require_control_dt(),
    )
    if options.strategy == "ik_interp":
        options.preserve_cartesian_samples = True
    before_ik_rng = torch.get_rng_state().tolist() if retained else None
    targets = build_pose_plan_states(eef_targets)
    result = action.motion_generator.generate(targets, options=options)
    recovered = False
    if (
        retained
        and options.strategy == "ik_interp"
        and context.batch_size == 1
        and result.positions is not None
        and result.dt is not None
        and not result.success.all()
    ):
        limits = authored_joint_velocity_limits(
            action.robot, motion_target.control_part
        ).to(result.positions)
        rejected = result.rejected_candidate
        # Failed executable rows hold. Retry qualification must use the actual
        # rejected candidate, never infer valid speed from that zero-speed hold.
        candidate_positions = (
            rejected.positions
            if rejected is not None and rejected.env_mask[0]
            else result.positions
        )
        candidate_dt = (
            rejected.dt if rejected is not None and rejected.env_mask[0] else result.dt
        )
        if (
            torch.isfinite(candidate_positions).all()
            and not joint_velocity_validity(
                candidate_positions, candidate_dt, limits
            ).all()
        ):
            retry = resolve_continuous_ik_path(
                action.motion_generator, targets, options, operation="retained Pour"
            )
            if retry is not None:
                expected = (1, object_poses.shape[1], arm_qpos.shape[1])
                if (
                    retry.positions.shape != expected
                    or not torch.equal(retry.positions[:, 0], arm_qpos)
                    or not torch.equal(retry.dt, result.dt)
                    or not retry.success.all()
                ):
                    raise ValueError(
                        "Continuous Pour recovery changed its start or time grid."
                    )
                result, recovered = retry, True
    if result.positions is None or result.dt is None or not result.success.all():
        failed = action.failed_plan(
            request, context, message="Pour tilt arc has no feasible IK path."
        )
        if not retained:
            return failed
        from dataclasses import replace

        return replace(
            failed,
            diagnostics=replace(
                failed.diagnostics,
                metadata={
                    **failed.diagnostics.metadata,
                    **metadata,
                    "pour_failed_kinematics": {
                        "scope": "planning_inputs_not_execution_or_ik_certificate",
                        "entry_arm_qpos": arm_qpos.detach().cpu().tolist(),
                        "torch_cpu_rng_state_before_generate": before_ik_rng,
                        "torch_intraop_threads": torch.get_num_threads(),
                        "object_to_eef": object_to_eef.detach().cpu().tolist(),
                        "required_eef_targets": eef_targets.detach().cpu().tolist(),
                        "per_waypoint_status": "not_exposed_by_shared_result",
                    },
                },
            ),
        )
    hand = grasp.require_target(JointPositionTarget)
    hand_qpos = grasp.joint_positions(
        GRASP_COMMAND,
        num_envs=context.batch_size,
        device=action.device,
        dtype=context.robot.qpos.dtype,
    )
    base = context.last_qpos.clone()
    base[:, list(hand.joint_ids)] = hand_qpos
    _, timed = to_full_robot_trajectory(
        result,
        base_qpos=base,
        joint_ids=list(motion_target.joint_ids),
        env_ids=context.env_ids,
        control_dt=context.require_control_dt(),
    )
    if recovered:
        random_state = PlanningRandomState.capture(
            (action.robot.device, timed.positions.device)
        )
        try:
            with RobotMeshCollisionSnapshot(
                action.robot, control_part=motion_target.control_part
            ) as query:
                certificate = query.inspect(timed.positions[0]).metadata()
        except (
            ImportError,
            ValueError,
            RuntimeError,
            AttributeError,
            KeyError,
            TypeError,
        ) as error:
            certificate = {"valid": False, "unavailable": str(error)}
        finally:
            random_state.restore()
        metadata = {
            **metadata,
            "pour_continuous_ik": {
                "unchanged_targets_and_timing": True,
                "robot_proxy": certificate,
            },
        }
        if not certificate["valid"]:
            return action.failed_plan(
                request,
                context,
                message="Continuous Pour recovery failed final robot-proxy qualification.",
                retryable=False,
            )
    return action.build_plan(
        request,
        context,
        success=result.success & context.task.exclusive_held_object_mask(resource),
        trajectory=timed,
        diagnostics=PlannerDiagnostics(
            backend=action.planning_services.planner_name, metadata=metadata
        ),
        segment_lengths=(
            {"entry_hold": 1, "pour": timed.waypoint_count - 1}
            if retained
            else {"pour": timed.waypoint_count}
        ),
        expected_effects=(
            StateDelta(held_object_updates={resource: held}) if retained else None
        ),
    )
