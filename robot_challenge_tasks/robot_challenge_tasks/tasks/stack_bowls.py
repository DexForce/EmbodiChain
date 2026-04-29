# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.utility.action_utils import interpolate_with_nums
from embodichain.utils import logger
from scipy.spatial.transform import Rotation as R

__all__ = ["StackBowlsEnv"]


@register_env("StackBowls-v1")
class StackBowlsEnv(EmbodiedEnv):

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        interp_times = self.cfg.extensions.get("interp_times", [15] * 16)
        self.interp_times = torch.tensor(
            interp_times,
            dtype=torch.int32,
            device=self.device,
        )

        # Define some constants for the demo trajectory generation.
        downward_r = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        down_rota_np = (
            R.from_euler("xyz", [20, -30, 0], degrees=True).as_matrix() @ downward_r
        )

        self.right_down_rota = torch.tensor(down_rota_np, device=self.device)

        left_down_rota_np = (
            R.from_euler("xyz", [-20, -30, 0], degrees=True).as_matrix() @ downward_r
        )
        self.left_down_rota = torch.tensor(left_down_rota_np, device=self.device)

        self.right_arm_ids = self.robot.get_joint_ids("right_arm")
        self.right_eef_ids = self.robot.get_joint_ids("right_eef")
        self.left_arm_ids = self.robot.get_joint_ids("left_arm")
        self.left_eef_ids = self.robot.get_joint_ids("left_eef")

        self.right_pick_offset = torch.tensor([0.00, -0.06, 0.01], device=self.device)
        self.left_pick_offset = torch.tensor([0.00, 0.06, 0.01], device=self.device)
        self.bowl_mid_pre_pick_offset = torch.tensor(
            [0.0, 0.0, 0.12], device=self.device
        )
        self.bowl_min_pre_pick_offset = torch.tensor(
            [0.0, 0.0, 0.16], device=self.device
        )

        self.right_bowl_mid_place_offset = torch.tensor(
            [0.00, -0.05, 0.12], device=self.device
        )
        self.left_bowl_mid_place_offset = torch.tensor(
            [0.00, 0.05, 0.12], device=self.device
        )
        self.right_bowl_min_place_offset = torch.tensor(
            [0.00, -0.045, 0.16], device=self.device
        )
        self.left_bowl_min_place_offset = torch.tensor(
            [0.00, 0.045, 0.16], device=self.device
        )

        init_qpos = self.robot.get_qpos()
        init_right_arm_qpos = init_qpos[:, self.right_arm_ids]
        self.init_right_arm_xpos = self.robot.compute_fk(
            qpos=init_right_arm_qpos, name="right_arm", to_matrix=True
        )
        init_left_arm_qpos = init_qpos[:, self.left_arm_ids]
        self.init_left_arm_xpos = self.robot.compute_fk(
            qpos=init_left_arm_qpos, name="left_arm", to_matrix=True
        )

    def choice_arm_by_distance(self, target_pos) -> str:
        """Choose the arm that is closer to the target position."""
        left_arm_xpos = self.init_left_arm_xpos[:, :3, 3]
        right_arm_xpos = self.init_right_arm_xpos[:, :3, 3]
        left_dist = torch.norm(left_arm_xpos - target_pos, dim=-1)
        right_dist = torch.norm(right_arm_xpos - target_pos, dim=-1)
        if left_dist < right_dist:
            return "left_arm"
        else:
            return "right_arm"

    def create_demo_action_list(self, *args, **kwargs) -> Optional[List[torch.Tensor]]:
        """Create a scripted stacking demo for stacking the top bowl into the bottom bowl.

        This produces a sequence of joint-space actions for the right arm and right gripper
        which (1) grasps the top bowl at its rim, (2) lifts it up, and
        (3) places it on top of the bottom bowl.
        """

        if self.num_envs > 1:
            logger.log_error(
                f"Demo trajectory generation is only supported for single environment. Got num_envs={self.num_envs}."
            )

        eef_open = self.robot.get_qpos_limits(name="left_eef")[:, :, 1]
        eef_close = self.robot.get_qpos_limits(name="left_eef")[:, :, 0]

        bowl_max = self.sim.get_rigid_object("bowl_max")
        bowl_mid = self.sim.get_rigid_object("bowl_mid")
        bowl_min = self.sim.get_rigid_object("bowl_min")
        bowl_min_pose = bowl_min.get_local_pose(to_matrix=True)
        bowl_mid_pose = bowl_mid.get_local_pose(to_matrix=True)
        bowl_max_pose = bowl_max.get_local_pose(to_matrix=True)

        arm_to_use_mid = self.choice_arm_by_distance(bowl_mid_pose[:, :3, 3])
        pick_offset = (
            self.left_pick_offset
            if arm_to_use_mid == "left_arm"
            else self.right_pick_offset
        )
        down_rota = (
            self.left_down_rota
            if arm_to_use_mid == "left_arm"
            else self.right_down_rota
        )
        bowl_mid_pick_pose = bowl_mid_pose.clone()
        bowl_mid_pick_pose[:, :3, :3] = down_rota
        bowl_mid_pick_pose[:, :3, 3] = bowl_mid_pick_pose[:, :3, 3] + pick_offset

        bowl_mid_pre_pick_pose = bowl_mid_pick_pose.clone()
        bowl_mid_pre_pick_pose[:, :3, 3] = (
            bowl_mid_pre_pick_pose[:, :3, 3] + self.bowl_mid_pre_pick_offset
        )
        arm_to_use_min = self.choice_arm_by_distance(bowl_min_pose[:, :3, 3])
        pick_offset = (
            self.left_pick_offset
            if arm_to_use_min == "left_arm"
            else self.right_pick_offset
        )
        down_rota = (
            self.left_down_rota
            if arm_to_use_min == "left_arm"
            else self.right_down_rota
        )

        bowl_min_pick_pose = bowl_min_pose.clone()
        bowl_min_pick_pose[:, :3, :3] = down_rota
        bowl_min_pick_pose[:, :3, 3] = bowl_min_pick_pose[:, :3, 3] + pick_offset

        # Check whether arm is changed.
        return_pose = None
        if arm_to_use_mid != arm_to_use_min:
            return_pose = (
                self.init_left_arm_xpos
                if arm_to_use_mid == "left_arm"
                else self.init_right_arm_xpos
            )

        bowl_min_pre_pick_pose = bowl_min_pick_pose.clone()
        bowl_min_pre_pick_pose[:, :3, 3] = (
            bowl_min_pre_pick_pose[:, :3, 3] + self.bowl_min_pre_pick_offset
        )

        bowl_mid_place_offset = (
            self.left_bowl_mid_place_offset
            if arm_to_use_mid == "left_arm"
            else self.right_bowl_mid_place_offset
        )
        down_rota = (
            self.left_down_rota
            if arm_to_use_mid == "left_arm"
            else self.right_down_rota
        )
        bowl_mid_place_pose = bowl_max_pose.clone()
        bowl_mid_place_pose[:, :3, :3] = down_rota
        bowl_mid_place_pose[:, :3, 3] = (
            bowl_mid_place_pose[:, :3, 3] + bowl_mid_place_offset
        )

        bowl_min_place_offset = (
            self.left_bowl_min_place_offset
            if arm_to_use_min == "left_arm"
            else self.right_bowl_min_place_offset
        )
        down_rota = (
            self.left_down_rota
            if arm_to_use_min == "left_arm"
            else self.right_down_rota
        )
        bowl_min_place_pose = bowl_max_pose.clone()
        bowl_min_place_pose[:, :3, :3] = down_rota
        bowl_min_place_pose[:, :3, 3] = (
            bowl_min_place_pose[:, :3, 3] + bowl_min_place_offset
        )

        bowl_mid_mid_pose = (bowl_mid_pre_pick_pose + bowl_mid_place_pose) / 2.0
        bowl_min_mid_pose = (bowl_min_pre_pick_pose + bowl_min_place_pose) / 2.0

        mid_critical_pose = (
            bowl_min_mid_pose if arm_to_use_mid == arm_to_use_min else return_pose
        )

        start_pose = (
            self.init_left_arm_xpos
            if arm_to_use_mid == "left_arm"
            else self.init_right_arm_xpos
        )
        final_return_pose = (
            self.init_left_arm_xpos
            if arm_to_use_min == "left_arm"
            else self.init_right_arm_xpos
        )

        xpos_list = [
            start_pose,
            bowl_mid_pre_pick_pose,
            bowl_mid_pick_pose,
            bowl_mid_pick_pose,
            bowl_mid_pre_pick_pose,
            bowl_mid_mid_pose,
            bowl_mid_place_pose,
            bowl_mid_place_pose,
            mid_critical_pose,
            bowl_min_pre_pick_pose,
            bowl_min_pick_pose,
            bowl_min_pick_pose,
            bowl_min_pre_pick_pose,
            bowl_min_mid_pose,
            bowl_min_place_pose,
            bowl_min_place_pose,
            final_return_pose,
        ]

        eef_list = [
            eef_close,
            eef_open,
            eef_open,
            eef_close,
            eef_close,
            eef_close,
            eef_close,
            eef_open,
            eef_open,
            eef_open,
            eef_open,
            eef_close,
            eef_close,
            eef_close,
            eef_close,
            eef_open,
            eef_close,
        ]

        arm_type_list = [
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_mid == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
            "left_arm" if arm_to_use_min == "left_arm" else "right_arm",
        ]
        if return_pose is not None:
            arm_type_list[8] = (
                "right_arm" if arm_type_list[0] == "right_arm" else "left_arm"
            )

        # # [n_env, n_waypoints, dof]
        pack_traj = self._pack_trajectory(xpos_list, eef_list, arm_type_list)
        if pack_traj is None:
            return None

        pack_traj = pack_traj.permute(
            1, 0, 2
        )  # switch to (n_waypoints, n_env, dof) -> (n_env, n_waypoints, dof)
        interp_traj = interpolate_with_nums(
            pack_traj, interp_nums=self.interp_times, device=self.device
        )
        interp_traj = interp_traj.permute(
            1, 0, 2
        )  # switch back to (n_env, n_waypoints, dof) -> (n_waypoints, n_env, dof)

        return interp_traj[:, :, self.active_joint_ids]

    def _pack_trajectory(self, xpos_list, eef_list, arm_type_list):
        assert len(xpos_list) == len(
            eef_list
        ), "xpos_list and eef_list must have the same length."

        init_qpos = self.robot.get_qpos()
        n_waypoints = len(xpos_list)
        dof = init_qpos.shape[-1]
        n_env = init_qpos.shape[0]
        traj = torch.zeros(
            size=(n_waypoints, n_env, dof), dtype=torch.float32, device=self.device
        )
        traj[:] = init_qpos
        for i, (xpos, eef, arm_type) in enumerate(
            zip(xpos_list, eef_list, arm_type_list)
        ):
            arm_ids = (
                self.left_arm_ids if arm_type == "left_arm" else self.right_arm_ids
            )
            eef_ids = (
                self.left_eef_ids if arm_type == "left_arm" else self.right_eef_ids
            )

            is_success, qpos = self.robot.compute_ik(pose=xpos, name=arm_type)
            if not is_success:
                logger.log_warning(f"IK failed for {i}-th waypoint. Returning None.")
                return None

            traj[i, :, arm_ids] = qpos
            traj[i, :, eef_ids] = eef
        return traj

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Check if bowl_mid and bowl_min are successfully stacked on bowl_max.

        This checks the x/y position and rotation error of both bowl_mid and bowl_min relative to bowl_max.
        """
        bowl_max = self.sim.get_rigid_object("bowl_max")
        bowl_mid = self.sim.get_rigid_object("bowl_mid")
        bowl_min = self.sim.get_rigid_object("bowl_min")

        bowl_max_pose = bowl_max.get_local_pose(to_matrix=True)
        bowl_mid_pose = bowl_mid.get_local_pose(to_matrix=True)
        bowl_min_pose = bowl_min.get_local_pose(to_matrix=True)

        # Compute x/y position difference relative to bowl_max
        mid_xy_diff = torch.norm(
            bowl_mid_pose[:, :2, 3] - bowl_max_pose[:, :2, 3], dim=-1
        )
        min_xy_diff = torch.norm(
            bowl_min_pose[:, :2, 3] - bowl_max_pose[:, :2, 3], dim=-1
        )
        xy_threshold = getattr(self.cfg.extensions, "success_xy_tol", 0.03)

        mid_success = mid_xy_diff < xy_threshold
        min_success = min_xy_diff < xy_threshold
        success = mid_success & min_success
        return success