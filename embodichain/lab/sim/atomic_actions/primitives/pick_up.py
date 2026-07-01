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

"""PickUp atomic action implementation."""

from __future__ import annotations

from typing import ClassVar

import torch

from embodichain.lab.sim.planners import MoveType, PlanState
from embodichain.utils import configclass, logger
from embodichain.utils.math import pose_inv

from ._helpers import arm_qpos_from_state
from ..affordance import AntipodalAffordance
from ..core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    GraspTarget,
    HeldObjectState,
    ObjectSemantics,
    WorldState,
)
from ..trajectory import TrajectoryBuilder


@configclass
class PickUpCfg(ActionCfg):
    name: str = "pick_up"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (approach + hand + lift)."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper close interpolation phase."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    hand_open_qpos: torch.Tensor | None = None
    """Joint positions for the open hand state, shape ``[hand_dof,]``."""

    hand_close_qpos: torch.Tensor | None = None
    """Joint positions for the closed hand state, shape ``[hand_dof,]``."""

    lift_height: float = 0.1
    """Height (m) to lift the end-effector after closing the gripper."""

    pre_grasp_distance: float = 0.15
    """Distance to offset back from the grasp pose along the approach direction."""

    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)
    """Approach direction in the object local frame."""


class PickUp(AtomicAction):
    """Approach a grasp pose, close the gripper, lift."""

    TargetType: ClassVar[type] = GraspTarget

    def __init__(
        self,
        motion_generator,
        cfg: PickUpCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PickUpCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_open_qpos is None:
            logger.log_error(
                "hand_open_qpos must be specified in PickUpCfg", ValueError
            )
        if self.cfg.hand_close_qpos is None:
            logger.log_error(
                "hand_close_qpos must be specified in PickUpCfg", ValueError
            )
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)
        self.approach_direction = self.cfg.approach_direction.to(self.device)

    def execute(self, target: GraspTarget, state: WorldState) -> ActionResult:
        sem = target.semantics
        if not isinstance(sem.affordance, AntipodalAffordance):
            logger.log_error(
                "PickUp requires an AntipodalAffordance on the target semantics.",
                ValueError,
            )
        if sem.entity is None:
            logger.log_error(
                "PickUp requires an entity on the target semantics.", ValueError
            )

        is_success, grasp_xpos = self._resolve_grasp_pose(sem)
        if not self.builder.all_envs_success(is_success):
            logger.log_warning("PickUp failed to resolve a grasp pose.")
            return self._fail(state)

        grasp_z = grasp_xpos[:, :3, 2]
        pre_grasp_xpos = self.builder.apply_local_offset(
            grasp_xpos, -grasp_z * self.cfg.pre_grasp_distance
        )

        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )

        n_approach, n_close, n_lift = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="lift",
        )

        target_states_list = [
            [
                PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        ok, approach_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_approach,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("PickUp failed to plan the approach trajectory.")
            return self._fail(state)

        grasp_arm_qpos = approach_arm[:, -1, :]
        lift_xpos = self.builder.apply_local_offset(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, lift_arm = self.builder.plan_arm_traj(
            target_states_list,
            grasp_arm_qpos,
            n_lift,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("PickUp failed to plan the lift trajectory.")
            return self._fail(state)

        hand_close_path = self.builder.interpolate_hand_qpos(
            self.hand_open_qpos, self.hand_close_qpos, n_waypoints=n_close
        )

        full = torch.empty(
            (self.n_envs, n_approach + n_close + n_lift, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_approach, self.arm_joint_ids] = approach_arm
        full[:, :n_approach, self.hand_joint_ids] = self.hand_open_qpos
        full[:, n_approach : n_approach + n_close, self.arm_joint_ids] = (
            grasp_arm_qpos.unsqueeze(1)
        )
        full[:, n_approach : n_approach + n_close, self.hand_joint_ids] = (
            hand_close_path
        )
        full[:, n_approach + n_close :, self.arm_joint_ids] = lift_arm
        full[:, n_approach + n_close :, self.hand_joint_ids] = self.hand_close_qpos

        obj_poses = sem.entity.get_local_pose(to_matrix=True)
        object_to_eef = torch.bmm(pose_inv(obj_poses), grasp_xpos)
        held = HeldObjectState(
            semantics=sem, object_to_eef=object_to_eef, grasp_xpos=grasp_xpos
        )
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(last_qpos=full[:, -1, :].clone(), held_object=held),
        )

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

    def _resolve_grasp_pose(
        self, semantics: ObjectSemantics
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=obj_poses, approach_direction=self.approach_direction
        )
        n_envs = obj_poses.shape[0]
        init_qpos = self.robot.get_qpos(name=self.cfg.control_part)
        n_max_pose = max(r[0].shape[0] for r in grasp_poses_result)
        grasp_xpos_padding = torch.zeros(
            (n_envs, n_max_pose, 4, 4), dtype=torch.float32, device=self.device
        )
        grasp_cost_padding = torch.full(
            (n_envs, n_max_pose),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(n_envs):
            n_pose = grasp_poses_result[i][0].shape[0]
            grasp_xpos_padding[i, :n_pose] = grasp_poses_result[i][0]
            grasp_cost_padding[i, :n_pose] = grasp_poses_result[i][1]
            grasp_xpos_padding[i, n_pose:] = grasp_poses_result[i][0][0]
            grasp_cost_padding[i, n_pose:] = grasp_poses_result[i][1][0]
        init_qpos_repeat = init_qpos[:, None, :].repeat(1, n_max_pose, 1)
        ik_success, _ = self.robot.compute_batch_ik(
            pose=grasp_xpos_padding,
            name=self.cfg.control_part,
            joint_seed=init_qpos_repeat,
        )
        grasp_cost_masked = torch.where(ik_success, grasp_cost_padding, 10000.0)
        best_cost, best_idx = grasp_cost_masked.min(dim=1)
        is_success = best_cost < 9999.0
        best_grasp_xpos = grasp_xpos_padding[
            torch.arange(n_envs, device=self.device), best_idx
        ]
        return is_success, best_grasp_xpos


__all__ = ["PickUp", "PickUpCfg"]
