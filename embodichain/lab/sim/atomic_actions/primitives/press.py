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

"""Press atomic action implementation."""

from __future__ import annotations

from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger

from ._helpers import arm_qpos_from_state
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    EndEffectorPoseTarget,
    WorldState,
)
from ..trajectory import TrajectoryBuilder


@configclass
class PressCfg(ActionCfg):
    name: str = "press"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (hand close + down + back)."""

    hand_interp_steps: int = 5
    """Number of waypoints for closing the gripper before pressing."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""


class Press(AtomicAction):
    """Close the gripper, press down to a target pose, then return."""

    TargetType: ClassVar[type] = EndEffectorPoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: PressCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PressCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.hand_dof = len(self.hand_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PressCfg", ValueError
            )
        self.hand_close_qpos = self.builder.expand_hand_qpos(
            self.cfg.hand_close_qpos,
            n_envs=self.n_envs,
            hand_dof=self.hand_dof,
        )

    def execute(self, target: EndEffectorPoseTarget, state: WorldState) -> ActionResult:
        press_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        start_hand_qpos = state.last_qpos[:, self.hand_joint_ids]

        n_close, n_down, n_back = self._compute_phase_waypoints()

        hand_close_path = self.builder.interpolate_hand_qpos(
            start_hand_qpos,
            self.hand_close_qpos,
            n_waypoints=n_close,
        )

        target_states_list = [
            [PlanState(xpos=press_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        down_success, down_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_down,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=self.cfg,
        )

        press_arm_qpos = down_arm[:, -1, :]
        back_arm = self.builder.plan_joint_traj(press_arm_qpos, start_arm_qpos, n_back)
        success = down_success

        full = torch.empty(
            (self.n_envs, n_close + n_down + n_back, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_close, self.arm_joint_ids] = start_arm_qpos.unsqueeze(1)
        full[:, :n_close, self.hand_joint_ids] = hand_close_path
        full[:, n_close : n_close + n_down, self.arm_joint_ids] = down_arm
        full[:, n_close : n_close + n_down, self.hand_joint_ids] = (
            self.hand_close_qpos.unsqueeze(1)
        )
        full[:, n_close + n_down :, self.arm_joint_ids] = back_arm
        full[:, n_close + n_down :, self.hand_joint_ids] = (
            self.hand_close_qpos.unsqueeze(1)
        )

        return ActionResult(
            success=success,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,
                coordinated_held_object=state.coordinated_held_object,
            ),
        )

    def _compute_phase_waypoints(self) -> tuple[int, int, int]:
        n_close = self.cfg.hand_interp_steps
        if n_close < 1:
            logger.log_error(
                "hand_interp_steps must be at least 1 for PressCfg.", ValueError
            )

        motion_waypoints = self.cfg.sample_interval - n_close
        n_down = motion_waypoints // 2
        n_back = motion_waypoints - n_down
        if n_down < 2 or n_back < 2:
            logger.log_error(
                "Not enough waypoints for press trajectory. Increase "
                "sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        return n_close, n_down, n_back

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


__all__ = ["Press", "PressCfg"]
