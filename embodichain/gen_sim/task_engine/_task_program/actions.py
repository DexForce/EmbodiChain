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

"""GenSim-only action wrappers; the shared Atomic Action implementations stay unchanged."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.affordance import AxisAlignAffordance
from embodichain.lab.sim.atomic_actions.core import ObjectSemantics
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    require_shared_task_state_key,
)
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
    MoveHeldObject,
)
from embodichain.lab.sim.atomic_actions.primitives.pour import Pour
from embodichain.lab.sim.atomic_actions.primitives.place import Place
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    to_full_robot_trajectory,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

__all__ = ["GenSimMoveHeldObject", "GenSimPlace", "GenSimPour"]


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-6)


def jaw_tilt_axis(
    jaw_line: torch.Tensor, upright: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the projected jaw axis, visible tilt direction, and validity."""
    up = _unit(upright)
    raw = jaw_line - (jaw_line * up).sum(-1, keepdim=True) * up
    valid = torch.isfinite(raw).all(-1) & (torch.linalg.vector_norm(raw, dim=-1) > 1e-5)
    axis = _unit(raw)
    direction = _unit(torch.linalg.cross(axis, up))
    return axis, direction, valid


class GenSimMoveHeldObject(MoveHeldObject):
    """GenSim transport with task-owned staging and retention evidence."""

    skill_id = MoveHeldObject.skill_id
    GoalType = MoveHeldObject.GoalType
    OptionsType = MoveHeldObject.OptionsType
    binding_contract = MoveHeldObject.binding_contract

    def _plan(self, request, context):
        goal = request.goal
        target = goal.object_target_pose
        if isinstance(target, torch.Tensor) and target.ndim == 3:
            motion = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            task_key = require_shared_task_state_key(
                request.binding.endpoint("primary", "motion"),
                request.binding.endpoint("primary", "grasp"),
                participant="GenSim transport",
            )
            held = context.get_held_object(task_key)
            if held is not None:
                arm_qpos = arm_qpos_from_state(context, list(motion.joint_ids))
                start = self.robot.compute_fk(
                    qpos=arm_qpos, name=motion.control_part, to_matrix=True
                )
                target_eef = target @ held.object_to_eef
                lifted = start.clone()
                lifted[..., 2, 3] = (
                    torch.maximum(start[..., 2, 3], target_eef[..., 2, 3]) + 0.06
                )
                above_eef = target_eef.clone()
                above_eef[..., 2, 3] = lifted[..., 2, 3]
                above = above_eef @ pose_inv(held.object_to_eef)
                goal = replace(
                    goal,
                    object_target_pose=torch.stack(
                        (lifted @ pose_inv(held.object_to_eef), above, target), dim=1
                    ),
                )
                request = replace(request, goal=goal)
        plan = super()._plan(request, context)
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        task_key = require_shared_task_state_key(
            motion, grasp, participant="GenSim transport"
        )
        held = context.get_held_object(task_key)
        if held is not None and plan.plan_success.any():
            plan = replace(
                plan,
                expected_effects=StateDelta(held_object_updates={task_key: held}),
            )
        return plan


class GenSimPlace(Place):
    """Keep ordinary Place unchanged unless its Cartesian IK plan fails."""

    skill_id = Place.skill_id
    GoalType = Place.GoalType
    OptionsType = Place.OptionsType
    binding_contract = Place.binding_contract

    def _plan(self, request, context):
        from .motion import (
            _joint_velocity_limits,
            _velocity_validity,
            place_ik_recovery,
        )

        with place_ik_recovery() as recovery:
            plan = super()._plan(request, context)
        if recovery.used and plan.plan_success.any():
            trajectory = plan.joint_trajectory
            limits = _joint_velocity_limits(self.robot, None).to(trajectory.positions)
            if not _velocity_validity(
                trajectory.positions, trajectory.dt, limits
            ).all():
                return self.failed_plan(
                    request,
                    context,
                    message="Recovered Place exceeds final resampled joint velocity limits.",
                )
        return plan


class GenSimPour(Pour):
    """GenSim Pour using measured pad-line geometry, not TCP-column assumptions."""

    skill_id = Pour.skill_id
    GoalType = Pour.GoalType
    OptionsType = Pour.OptionsType
    binding_contract = Pour.binding_contract

    def __init__(self, pour_receivers: dict[str, str] | None = None) -> None:
        super().__init__()
        self._pour_receivers = dict(pour_receivers or {})

    def _plan(self, request, context):
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        motion_target = motion.require_target(JointPositionTarget)
        task_key = require_shared_task_state_key(
            motion, grasp, participant="GenSim Pour"
        )
        held = context.get_held_object(task_key)
        if held is None:
            raise ValueError("GenSim Pour requires an exclusively held object.")
        arm_qpos = arm_qpos_from_state(context, list(motion_target.joint_ids))
        eef = self.robot.compute_fk(
            qpos=arm_qpos, name=motion_target.control_part, to_matrix=True
        )
        object_to_eef = held.object_to_eef.to(self.device, torch.float32)
        object_pose = eef @ pose_inv(object_to_eef)
        prefix = motion_target.control_part.split("_arm")[0]
        names = set(self.robot.link_names)
        pad_names = (
            ["left_inner_finger_pad", "left_right_inner_finger_pad"]
            if prefix == "left"
            else ["right_left_inner_finger_pad", "right_inner_finger_pad"]
        )
        if not all(name in names for name in pad_names):
            raise ValueError(
                f"GenSim Pour cannot resolve both pad links for {prefix!r}."
            )
        pad_poses = [
            self.robot.get_link_pose(name, to_matrix=True) for name in pad_names
        ]
        jaw_world = pad_poses[1][:, :3, 3] - pad_poses[0][:, :3, 3]
        jaw_local = torch.bmm(
            object_pose[:, :3, :3].transpose(1, 2), jaw_world[..., None]
        ).squeeze(-1)
        affordance = deepcopy(held.semantics.affordance)
        if not isinstance(affordance, AxisAlignAffordance):
            raise ValueError("GenSim Pour requires an axis-aware grasp affordance.")
        upright = _unit(affordance.internal_axis.to(self.device, torch.float32))
        axis, direction, valid = jaw_tilt_axis(jaw_local, upright.expand_as(jaw_local))
        if not valid.all():
            raise ValueError(
                "GenSim Pour found a jaw line parallel to the vessel upright axis."
            )
        if not torch.allclose(axis, axis[0:1].expand_as(axis), atol=2e-2, rtol=0):
            raise ValueError(
                "GenSim Pour requires one stable jaw-line axis across environments."
            )
        receiver_id = self._pour_receivers.get(held.semantics.entity_id)
        if receiver_id is not None:
            receiver = context.scene.entities.get(receiver_id)
            if receiver is None or receiver.pose is None:
                raise ValueError(
                    f"GenSim Pour receiver {receiver_id!r} is not observed."
                )
            receiver_pose = receiver.pose.to(
                device=self.device, dtype=object_pose.dtype
            )
            if receiver_pose.ndim == 2:
                receiver_pose = receiver_pose.unsqueeze(0).expand(
                    context.batch_size, -1, -1
                )
            receiver_direction = receiver_pose[:, :3, 3] - object_pose[:, :3, 3]
            upright_world = object_pose[:, :3, :3] @ upright.expand_as(axis)[..., None]
            axis_world = object_pose[:, :3, :3] @ axis[..., None]
            receiver_direction = receiver_direction - (
                receiver_direction * upright_world.squeeze(-1)
            ).sum(-1, keepdim=True) * upright_world.squeeze(-1)
            receiver_direction = receiver_direction - (
                receiver_direction * axis_world.squeeze(-1)
            ).sum(-1, keepdim=True) * axis_world.squeeze(-1)
            visible_direction = torch.linalg.cross(
                axis_world.squeeze(-1), upright_world.squeeze(-1)
            )
            signed_visible = visible_direction * float(
                request.skill_options.rotate_angle
            )
            if torch.linalg.vector_norm(receiver_direction, dim=-1).min() <= 1e-5:
                raise ValueError("GenSim Pour receiver direction is degenerate.")
            flip = (signed_visible * receiver_direction).sum(-1, keepdim=True) < 0
            axis = torch.where(flip, -axis, axis)
        if not torch.allclose(axis, axis[0:1].expand_as(axis), atol=2e-2, rtol=0):
            raise ValueError(
                "GenSim Pour receiver direction selects inconsistent tilt signs across environments."
            )
        affordance.internal_axis = axis[0].detach().clone()
        semantics = ObjectSemantics(
            affordance=affordance,
            geometry=deepcopy(held.semantics.geometry),
            entity_id=held.semantics.entity_id,
            properties=deepcopy(held.semantics.properties),
            label=held.semantics.label,
        )
        count = request.motion_policy.sample_count
        if count < 3:
            raise ValueError("GenSim Pour needs at least three trajectory samples.")
        first = count // 2
        angles = torch.cat(
            (
                torch.linspace(0.0, 1.0, first + 1, device=self.device)[1:],
                torch.linspace(1.0, 0.0, count - first + 1, device=self.device)[1:],
            )
        ) * abs(float(request.skill_options.rotate_angle))
        try:
            rotations = axis_angle_to_rotation_matrix(
                axis[:, None] * angles[None, :, None]
            )
            object_poses = object_pose[:, None].expand(-1, count, -1, -1).clone()
            pivot = torch.zeros(3, device=self.device, dtype=object_pose.dtype)
            vertices = getattr(affordance, "mesh_vertices", None)
            if isinstance(vertices, torch.Tensor) and vertices.numel():
                vertices = vertices.to(device=self.device, dtype=object_pose.dtype)
                center = (vertices.amin(0) + vertices.amax(0)) * 0.5
                up0 = upright.reshape(-1, 3)[0]
                top = (vertices @ up0).amax()
                pivot = center + up0 * (top - (center * up0).sum())
            anchor = (
                object_pose[:, None, :3, 3]
                + torch.einsum("bij,j->bi", object_pose[:, :3, :3], pivot)[:, None]
            )
            object_poses[:, :, :3, :3] = object_pose[:, None, :3, :3] @ rotations
            object_poses[:, :, :3, 3] = anchor - torch.einsum(
                "bnij,j->bni", object_poses[:, :, :3, :3], pivot
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "GenSim Pour geometry shape failure: "
                f"object_pose={tuple(object_pose.shape)}, axis={tuple(axis.shape)}, "
                f"angles={tuple(angles.shape)}, rotations={tuple(rotations.shape) if 'rotations' in locals() else None}, "
                f"pivot={tuple(pivot.shape) if 'pivot' in locals() else None}, "
                f"vertices={tuple(vertices.shape) if 'vertices' in locals() and isinstance(vertices, torch.Tensor) else None}"
            ) from exc
        # ik_interp prepends the observed start qpos; pass the remaining
        # count-1 Cartesian keyframes so every output sample is grounded.
        eef_targets = (object_poses @ object_to_eef[:, None])[:, 1:]
        options = request.motion_policy.to_motion_gen_options(
            start_qpos=arm_qpos,
            control_part=motion_target.control_part,
            interpolation_dt=context.require_control_dt(),
        )
        if options.strategy == "ik_interp":
            options.preserve_cartesian_samples = True
        result = self.motion_generator.generate(
            build_pose_plan_states(eef_targets), options=options
        )
        if result.positions is None or result.dt is None or not result.success.all():
            return self.failed_plan(
                request,
                context,
                message="GenSim Pour tilt arc has no feasible IK path.",
            )
        grasp_target = grasp.require_target(JointPositionTarget)
        hand_ids = list(grasp_target.joint_ids)
        hand_qpos = grasp.joint_positions(
            "grasp",
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        base = context.last_qpos.clone()
        base[:, hand_ids] = hand_qpos
        _, timed = to_full_robot_trajectory(
            result,
            base_qpos=base,
            joint_ids=list(motion_target.joint_ids),
            env_ids=context.env_ids,
            control_dt=context.require_control_dt(),
        )
        previous = None
        metadata = {
            "pad_line_local": jaw_local.detach().cpu().tolist(),
            "tilt_axis_local": axis.detach().cpu().tolist(),
            "tilt_direction_local": direction.detach().cpu().tolist(),
        }
        return self.build_plan(
            request,
            context,
            success=result.success & context.task.exclusive_held_object_mask(task_key),
            trajectory=timed,
            diagnostics=PlannerDiagnostics(
                backend=self.planning_services.planner_name, metadata=metadata
            ),
            segment_lengths={"pour": timed.waypoint_count},
        )
