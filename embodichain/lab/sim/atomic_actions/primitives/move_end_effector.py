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

"""MoveEndEffector atomic action implementation."""

from __future__ import annotations

from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass

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
class MoveEndEffectorCfg(ActionCfg):
    name: str = "move_end_effector"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 50
    """Number of waypoints in the planned trajectory."""


class MoveEndEffector(AtomicAction):
    """Plan a free-space end-effector move to a target pose.

    The :class:`EndEffectorPoseTarget` may carry either a single waypoint
    ``(n_envs, 4, 4)`` (or a broadcastable ``(4, 4)``) or a multi-waypoint
    trajectory ``(n_envs, n_waypoint, 4, 4)``. In the multi-waypoint case the
    action plans a single trajectory that visits every waypoint in order,
    starting from the inherited ``WorldState.last_qpos``; IK is solved for each
    waypoint with the previous waypoint's solution as the seed.
    """

    TargetType: ClassVar[type] = EndEffectorPoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: MoveEndEffectorCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveEndEffectorCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

    def execute(self, target: EndEffectorPoseTarget, state: WorldState) -> ActionResult:
        move_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        start_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        target_states_list = self._build_target_states(move_xpos)
        ok, arm_traj = self.builder.plan_arm_traj(
            target_states_list,
            start_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            return self._fail(state)
        full = self._embed(arm_traj, state.last_qpos)
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(), held_object=state.held_object
            ),
        )

    def _build_target_states(self, move_xpos: torch.Tensor) -> list[list[PlanState]]:
        """Build per-env PlanState lists from a single- or multi-waypoint target."""
        if move_xpos.dim() == 3:
            move_xpos = move_xpos.unsqueeze(1)
        n_waypoint = move_xpos.shape[1]
        return [
            [
                PlanState(xpos=move_xpos[i, j], move_type=MoveType.EEF_MOVE)
                for j in range(n_waypoint)
            ]
            for i in range(self.n_envs)
        ]

    def _embed(
        self, arm_traj: torch.Tensor, last_full_qpos: torch.Tensor
    ) -> torch.Tensor:
        n_wp = arm_traj.shape[1]
        full = torch.empty(
            (self.n_envs, n_wp, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = last_full_qpos.unsqueeze(1)
        full[:, :, self.arm_joint_ids] = arm_traj
        return full

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=False,
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )


__all__ = ["MoveEndEffector", "MoveEndEffectorCfg"]
