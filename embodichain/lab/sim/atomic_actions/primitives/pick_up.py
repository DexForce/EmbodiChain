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
from embodichain.utils.math import (
    axis_angle_to_rotation_matrix,
    pose_inv,
    quat_error_magnitude,
    quat_from_matrix,
)

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

    obj_upright_direction: torch.Tensor | None = None
    """Optional object local direction used to choose the upright grasp rotation."""

    rotate_upright: float | None = None
    """Optional rotation (radians) about the grasp x-axis to apply after grasp selection."""


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
        start_arm_qpos = self.builder.resolve_start_qpos(
            arm_qpos_from_state(state, self.arm_joint_ids),
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        is_success, grasp_xpos = self._resolve_grasp_pose(sem, start_arm_qpos)

        if not self.builder.all_envs_success(is_success):
            logger.log_warning("PickUp failed to resolve a grasp pose.")
            return self._fail(state)
        self._apply_upright_rotation(sem, grasp_xpos)

        grasp_z = grasp_xpos[:, :3, 2]
        pre_grasp_xpos = self.builder.apply_local_offset(
            grasp_xpos, -grasp_z * self.cfg.pre_grasp_distance
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
        approach_success, approach_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_approach,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=self.cfg,
        )

        grasp_arm_qpos = approach_arm[:, -1, :]
        lift_xpos = self.builder.apply_local_offset(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        lift_success, lift_arm = self.builder.plan_arm_traj(
            target_states_list,
            grasp_arm_qpos,
            n_lift,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
            cfg=self.cfg,
        )
        success = approach_success & lift_success

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
            success=success,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=held,
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

    def _resolve_grasp_pose(
        self, semantics: ObjectSemantics, start_qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=obj_poses, approach_direction=self.approach_direction
        )
        n_envs = obj_poses.shape[0]
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
            grasp_poses = grasp_poses_result[i][0].to(
                device=self.device, dtype=torch.float32
            )
            grasp_costs = grasp_poses_result[i][1].to(
                device=self.device, dtype=torch.float32
            )
            grasp_xpos_padding[i, :n_pose] = grasp_poses
            grasp_cost_padding[i, :n_pose] = grasp_costs
            grasp_xpos_padding[i, n_pose:] = grasp_poses[0]
            grasp_cost_padding[i, n_pose:] = grasp_costs[0]
        grasp_xpos_padding, ik_success = self._select_symmetric_grasp_variants(
            grasp_xpos_padding, start_qpos
        )
        grasp_cost_masked = torch.where(ik_success, grasp_cost_padding, 10000.0)
        best_cost, best_idx = grasp_cost_masked.min(dim=1)
        is_success = best_cost < 9999.0
        best_grasp_xpos = grasp_xpos_padding[
            torch.arange(n_envs, device=self.device), best_idx
        ]
        return is_success, best_grasp_xpos

    def _select_symmetric_grasp_variants(
        self, grasp_xpos: torch.Tensor, start_qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose the closest TCP z-roll variant, then validate reachability."""
        n_envs, n_pose = grasp_xpos.shape[:2]
        mirrored_grasp_xpos = grasp_xpos.clone()
        mirrored_grasp_xpos[..., :3, 0] = -mirrored_grasp_xpos[..., :3, 0]
        mirrored_grasp_xpos[..., :3, 1] = -mirrored_grasp_xpos[..., :3, 1]
        grasp_variants = torch.stack([grasp_xpos, mirrored_grasp_xpos], dim=2)

        start_xpos = self.robot.compute_fk(
            qpos=start_qpos,
            name=self.cfg.control_part,
            to_matrix=True,
        )
        start_quat = quat_from_matrix(start_xpos[:, :3, :3])
        variant_quat = quat_from_matrix(grasp_variants[..., :3, :3])
        start_quat = start_quat[:, None, None, :].expand_as(variant_quat)
        rotation_error = quat_error_magnitude(
            variant_quat.reshape(-1, 4),
            start_quat.reshape(-1, 4),
        ).reshape(n_envs, n_pose, 2)
        best_variant_idx = rotation_error.argmin(dim=2)

        env_idx = torch.arange(n_envs, device=self.device)[:, None]
        pose_idx = torch.arange(n_pose, device=self.device)[None, :]
        selected_grasp_xpos = grasp_variants[env_idx, pose_idx, best_variant_idx]
        start_qpos_repeat = start_qpos[:, None, :].repeat(1, n_pose, 1)
        ik_success, _ = self.robot.compute_batch_ik(
            pose=selected_grasp_xpos,
            name=self.cfg.control_part,
            joint_seed=start_qpos_repeat,
        )
        return selected_grasp_xpos, ik_success

    def _apply_upright_rotation(
        self, semantics: ObjectSemantics, grasp_xpos: torch.Tensor
    ) -> None:
        """Apply the WAIC upright-in-place grasp roll adjustment in place."""
        if self.cfg.rotate_upright is None:
            return

        if self.cfg.obj_upright_direction is None:
            upright_direction = torch.tensor(
                [0, 0, 1], dtype=torch.float32, device=self.device
            )
        else:
            upright_direction = self.cfg.obj_upright_direction.to(
                device=self.device, dtype=torch.float32
            )
        obj_pose = semantics.entity.get_local_pose(to_matrix=True)
        obj_upright = torch.matmul(obj_pose[:, :3, :3], upright_direction)
        grasp_ry = grasp_xpos[:, :3, 1]
        dot_result = (grasp_ry * obj_upright).sum(axis=1)
        revert_flag = torch.where(dot_result < 0, 1.0, -1.0)
        grasp_rx = grasp_xpos[:, :3, 0]
        rota_axis_angle = self.cfg.rotate_upright * revert_flag[:, None] * grasp_rx
        rota_offset = axis_angle_to_rotation_matrix(rota_axis_angle)
        grasp_xpos[:, :3, :3] = torch.bmm(grasp_xpos[:, :3, :3], rota_offset)


__all__ = ["PickUp", "PickUpCfg"]
