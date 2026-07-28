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

from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger
from embodichain.utils.math import axis_angle_to_rotation_matrix, get_relative_rotation

from ._helpers import arm_qpos_from_state, resolve_object_target
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    HeldObjectPoseTarget,
    WorldState,
)
from ..trajectory import TrajectoryBuilder


@configclass
class MoveHeldObjectCfg(ActionCfg):
    name: str = "move_held_object"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 50
    """Number of waypoints in the planned trajectory."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""

    obj_upright_direction: torch.Tensor | None = None
    """Optional object-local direction to align with world up while moving."""

    pick_rotate_upright: float | None = None
    """Optional rotation in radians used by the legacy upright transport mode."""


class MoveHeldObject(AtomicAction):
    """Move the held object to a target object pose; keep the gripper closed."""

    TargetType: ClassVar[type] = HeldObjectPoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: MoveHeldObjectCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveHeldObjectCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in MoveHeldObjectCfg", ValueError
            )
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

    def execute(self, target: HeldObjectPoseTarget, state: WorldState) -> ActionResult:
        if state.held_object is None:
            logger.log_error(
                "MoveHeldObject requires WorldState.held_object - run PickUp first.",
                ValueError,
            )
        object_target_pose = resolve_object_target(
            target.object_target_pose, n_envs=self.n_envs, device=self.device
        )
        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        end_arm_xpos = self.robot.compute_fk(
            start_arm_qpos, name=self.cfg.control_part, to_matrix=True
        )
        if self.cfg.pick_rotate_upright is not None:
            self._apply_configured_upright_rotation(
                object_target_pose,
                end_arm_xpos,
                state.held_object.semantics.entity.get_local_pose(to_matrix=True),
            )
        object_to_eef = state.held_object.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.n_envs, 1, 1)
        move_eef_xpos = torch.bmm(object_target_pose, object_to_eef)

        if self.cfg.pick_rotate_upright is None:
            self._apply_automatic_transport_rotation(move_eef_xpos, end_arm_xpos)

        target_states_list = [
            [PlanState(xpos=move_eef_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        success, arm_traj = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=self.cfg,
        )

        full = torch.empty(
            (self.n_envs, arm_traj.shape[1], self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :, self.arm_joint_ids] = arm_traj
        full[:, :, self.hand_joint_ids] = self.hand_close_qpos

        return ActionResult(
            success=success,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,
                coordinated_held_object=state.coordinated_held_object,
            ),
        )

    def _apply_configured_upright_rotation(
        self,
        object_target_pose: torch.Tensor,
        end_arm_xpos: torch.Tensor,
        held_object_xpos: torch.Tensor,
    ) -> None:
        if self.cfg.obj_upright_direction is None:
            upright_direction = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
            )
        else:
            upright_direction = self.cfg.obj_upright_direction.to(
                device=self.device, dtype=torch.float32
            )
        object_upright = torch.matmul(held_object_xpos[:, :3, :3], upright_direction)
        dot_result = torch.sum(end_arm_xpos[:, :3, 1] * object_upright, dim=-1)
        revert_flag = torch.where(dot_result < 0, 1.0, -1.0)
        axis_angle = (
            -float(self.cfg.pick_rotate_upright)
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

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=torch.zeros(self.n_envs, dtype=torch.bool, device=self.device),
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )


__all__ = ["MoveHeldObject", "MoveHeldObjectCfg"]
