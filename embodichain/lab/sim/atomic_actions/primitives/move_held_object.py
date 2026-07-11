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

from ._helpers import arm_qpos_from_state, resolve_object_target
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    HeldObjectPoseTarget,
    WorldState,
)
from ..trajectory import TrajectoryBuilder

from embodichain.utils.math import (
    axis_angle_to_rotation_matrix
)

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
        end_arm_xpos = self.robot.compute_fk(start_arm_qpos, name=self.cfg.control_part, to_matrix=True)
        end_arm_rx = end_arm_xpos[:, :3, 0]
        down_z = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.float32).repeat(self.n_envs, 1)
        fake_y = torch.cross(down_z, end_arm_rx, dim=-1)
        fake_y_normalized = fake_y / torch.norm(fake_y, dim=-1, keepdim=True)
        fake_z = torch.cross(end_arm_rx, fake_y_normalized, dim=-1)
        fake_z_normalized = fake_z / torch.norm(fake_z, dim=-1, keepdim=True)
        end_arm_rz = end_arm_xpos[:, :3, 2]
        # arm dot angle
        arm_dot_angle = torch.acos(torch.clamp(torch.sum(end_arm_rz * down_z, dim=-1), -1.0, 1.0))
        rota_axis_angle = (torch.pi - arm_dot_angle) * end_arm_rx
        rota_offset = axis_angle_to_rotation_matrix(
            rota_axis_angle.reshape(-1, 3)
        ).reshape(*rota_axis_angle.shape[:-1], 3, 3)
        eef_rotation = torch.bmm(rota_offset, end_arm_xpos[:, :3, :3])

        object_to_eef = state.held_object.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.n_envs, 1, 1)
        move_eef_xpos = torch.bmm(object_target_pose, object_to_eef)


        end_arm_xpos = self.robot.compute_fk(start_arm_qpos, name=self.cfg.control_part, to_matrix=True)
        end_arm_rx = end_arm_xpos[:, :3, 0]
        end_arm_rz = end_arm_xpos[:, :3, 2]
        down_z = torch.tensor([[0.0, 0.0, -1.0]], device=self.device, dtype=torch.float32).repeat(self.n_envs, 1)
        angle_diff = torch.acos(torch.clamp(torch.sum(end_arm_rz * down_z, dim=-1), -1.0, 1.0))
        cross_z = torch.cross(end_arm_rz, down_z, dim=-1)
        dot_z = torch.sum(end_arm_rx * cross_z, dim=-1)
        revert_flag = torch.where(dot_z < 0, -1.0, 1.0)
        if (angle_diff > 0.5).any():
            rota_axis_angle = revert_flag * torch.pi * 0.5 * end_arm_rx
            rota_offset = axis_angle_to_rotation_matrix(rota_axis_angle)
            eef_rotation = torch.bmm(rota_offset, end_arm_xpos[:, :3, :3])
            move_eef_xpos[:, :3, :3] = eef_rotation
        print(f"move_eef_xpos: {move_eef_xpos}")


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
