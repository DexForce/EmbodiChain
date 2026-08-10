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

"""MoveHeldObject atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import logger
from embodichain.utils.math import axis_angle_to_rotation_matrix, get_relative_rotation

from ._helpers import arm_qpos_from_state, resolve_object_target
from ..control import GRASP_COMMAND
from ..core import AtomicAction
from ..goals import PoseGoalValue, resolve_pose_goal, validate_pose_goal
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan
from ..state import PlanningContext


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectPoseGoal:
    """Desired pose for the object held by this action's control part."""

    goal_kind: ClassVar[str] = "held_object_pose"

    object_target_pose: PoseGoalValue
    """Target object pose, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    def __post_init__(self) -> None:
        validate_pose_goal(
            self.object_target_pose,
            "object_target_pose",
            allow_waypoints=False,
        )


@dataclass(frozen=True, slots=True, eq=False)
class MoveHeldObjectOptions(ActionOptions):
    """Per-invocation held-object transport behavior."""

    obj_upright_direction: torch.Tensor | None = None
    """Optional object-local direction to align with world up while moving."""

    pick_rotate_upright: float | None = None
    """Optional rotation in radians used by the legacy upright transport mode."""

    def __post_init__(self) -> None:
        if self.obj_upright_direction is not None:
            if (
                self.obj_upright_direction.shape != (3,)
                or not torch.isfinite(self.obj_upright_direction).all()
            ):
                raise ValueError(
                    "obj_upright_direction must be a finite tensor with shape (3,)."
                )
            object.__setattr__(
                self, "obj_upright_direction", self.obj_upright_direction.clone()
            )


class MoveHeldObject(AtomicAction[HeldObjectPoseGoal, MoveHeldObjectOptions]):
    """Move the held object to a target object pose; keep the gripper closed."""

    skill_id: ClassVar[str] = "move_held_object"
    GoalType: ClassVar[type] = HeldObjectPoseGoal
    OptionsType: ClassVar[type] = MoveHeldObjectOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)

    def __init__(
        self,
        default_options: MoveHeldObjectOptions | None = None,
    ) -> None:
        super().__init__(default_options)

    def _on_bind(self) -> None:
        """Resolve engine-wide resources from the owning engine."""
        self.n_envs = self.robot.get_qpos().shape[0]
        self.robot_dof = self.robot.dof

    def _plan(
        self,
        request: ResolvedActionRequest[HeldObjectPoseGoal, MoveHeldObjectOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan held-object transport without changing the attachment relation."""
        target = self.require_goal(request)
        options = request.skill_options
        binding = request.binding
        manipulator = binding.manipulator()
        end_effector = binding.end_effector()
        control_part = manipulator.name
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        hand_grasp_qpos = end_effector.joint_positions(
            GRASP_COMMAND,
            n_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        state = context
        held_object = state.get_held_object(control_part)
        if held_object is None:
            logger.log_error(
                "MoveHeldObject requires an object held by control part "
                f"{control_part!r} - run PickUp first.",
                ValueError,
            )
        object_target_pose = resolve_object_target(
            resolve_pose_goal(
                target.object_target_pose,
                context,
                name="object_target_pose",
            ),
            n_envs=self.n_envs,
            device=self.device,
        )
        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=manipulator.dof,
            control_part=control_part,
        )
        end_arm_xpos = self.robot.compute_fk(
            start_arm_qpos, name=control_part, to_matrix=True
        )
        if options.pick_rotate_upright is not None:
            self._apply_configured_upright_rotation(
                object_target_pose,
                end_arm_xpos,
                held_object.semantics.entity.get_local_pose(to_matrix=True),
                options,
            )
        object_to_eef = held_object.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.n_envs, 1, 1)
        move_eef_xpos = torch.bmm(object_target_pose, object_to_eef)

        if options.pick_rotate_upright is None:
            self._apply_automatic_transport_rotation(move_eef_xpos, end_arm_xpos)

        target_states_list = [
            [PlanState(xpos=move_eef_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        success, arm_traj = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            request.motion_policy.sample_count,
            control_part=control_part,
            arm_dof=manipulator.dof,
            cfg=request.motion_policy,
        )

        full = torch.empty(
            (self.n_envs, arm_traj.shape[1], self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :, arm_joint_ids] = arm_traj
        full[:, :, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=full,
            phase_name="transport",
        )

    def _apply_configured_upright_rotation(
        self,
        object_target_pose: torch.Tensor,
        end_arm_xpos: torch.Tensor,
        held_object_xpos: torch.Tensor,
        options: MoveHeldObjectOptions,
    ) -> None:
        if options.obj_upright_direction is None:
            upright_direction = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
            )
        else:
            upright_direction = options.obj_upright_direction.to(
                device=self.device, dtype=torch.float32
            )
        object_upright = torch.matmul(held_object_xpos[:, :3, :3], upright_direction)
        dot_result = torch.sum(end_arm_xpos[:, :3, 1] * object_upright, dim=-1)
        revert_flag = torch.where(dot_result < 0, 1.0, -1.0)
        axis_angle = (
            -float(options.pick_rotate_upright)
            * revert_flag.unsqueeze(-1)
            * end_arm_xpos[:, :3, 0]
        )
        rotation_offset = axis_angle_to_rotation_matrix(axis_angle)
        object_target_pose[:, :3, :3] = torch.bmm(
            rotation_offset, held_object_xpos[:, :3, :3]
        )

    def _apply_automatic_transport_rotation(
        self,
        move_eef_xpos: torch.Tensor,
        end_arm_xpos: torch.Tensor,
    ) -> None:
        down_z = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        arm_dot_angle = torch.acos(
            torch.clamp(torch.sum(end_arm_xpos[:, :3, 2] * down_z, dim=-1), -1.0, 1.0)
        )
        adjust_mask = arm_dot_angle > torch.pi * 0.25
        if not adjust_mask.any():
            return

        revert_flag = torch.where(end_arm_xpos[:, 2, 1] > 0, 1.0, -1.0)
        rotation_axis = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        ).repeat(self.n_envs, 1)
        axis_angle = (
            (torch.pi * 0.5 - arm_dot_angle).unsqueeze(-1)
            * rotation_axis
            * revert_flag.unsqueeze(-1)
        )
        rotation_offset = axis_angle_to_rotation_matrix(axis_angle)
        template_rotation_a = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.n_envs, 1, 1)
        template_rotation_b = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.n_envs, 1, 1)
        target_rotation_a = torch.bmm(template_rotation_a, rotation_offset)
        target_rotation_b = torch.bmm(template_rotation_b, rotation_offset)
        relative_rotation_a = get_relative_rotation(
            target_rotation_a, end_arm_xpos[:, :3, :3]
        )
        relative_rotation_b = get_relative_rotation(
            target_rotation_b, end_arm_xpos[:, :3, :3]
        )
        target_rotation = torch.where(
            (relative_rotation_a < relative_rotation_b)[:, None, None],
            target_rotation_a,
            target_rotation_b,
        )
        move_eef_xpos[:, :3, :3] = torch.where(
            adjust_mask[:, None, None],
            target_rotation,
            move_eef_xpos[:, :3, :3],
        )


__all__ = [
    "HeldObjectPoseGoal",
    "MoveHeldObject",
    "MoveHeldObjectOptions",
]
