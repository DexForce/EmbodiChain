# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import os
import torch
import numpy as np
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from copy import deepcopy
from embodichain.lab.gym.utils.misc import mul_linear_expand
from embodichain.lab.sim.utility.action_utils import (
    compute_pose_offset_related_to_first,
    get_trajectory_object_offset_qpos,
    warp_trajectory_qpos,
)


@register_env("WarpTrajectory", max_episode_steps=600)
class WarpTrajectory(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        self.affordance_datas = {}

    def create_demo_action_list(self, *args, **kwargs):
        # wait util stable
        self.sim.update(step=200)
        (
            first_left_arm_qposes,  # origin trajectory, torch.tensor [waypoint_num, dof]
            key_indices,  # key frame waypoint indices, torch.tensor [key_frame_num,]
            key_obj_indices,  # key frame belong to which object index, torch.tensor [key_frame_num,]
        ) = self._create_first_trajectory()
        cup_pose = self.sim.get_rigid_object("paper_cup").get_local_pose(to_matrix=True)
        scoop_pose = self.sim.get_rigid_object("scoop").get_local_pose(to_matrix=True)
        cup_offset = compute_pose_offset_related_to_first(cup_pose)
        scoop_offset = compute_pose_offset_related_to_first(scoop_pose)
        obj_offset = torch.concatenate(
            [
                cup_offset[None, :, :, :],
                scoop_offset[None, :, :, :],
            ]
        )
        left_arm_solver = self.robot.get_solver("left_arm")
        left_arm_base_pose = self.robot.get_link_pose(
            left_arm_solver.root_link_name, to_matrix=True
        )
        # [n_batch, waypoint_num, dof]
        is_success, key_qpos_offset = get_trajectory_object_offset_qpos(
            trajectory=first_left_arm_qposes,
            key_indices=key_indices,
            key_obj_indices=key_obj_indices,
            obj_offset=obj_offset,
            solver=left_arm_solver,
            base_xpos=left_arm_base_pose[0],
            device=self.device,
        )
        left_arm_warp_trajectory = warp_trajectory_qpos(
            first_left_arm_qposes, key_indices, key_qpos_offset, device=self.device
        )

        # pack action list
        demo_action_list = []
        left_arm_ids = self.robot.get_joint_ids("left_arm")
        current_qpos = self.robot.get_qpos()
        n_waypoint = left_arm_warp_trajectory.shape[1]
        for i in range(n_waypoint):
            left_arm_qpos = left_arm_warp_trajectory[:, i]
            action = current_qpos.clone()
            action[:, left_arm_ids] = left_arm_qpos
            demo_action_list.append(action)
        return demo_action_list

    def _create_first_trajectory(self):
        """
        generate a demo trajectory according to the first arena of dexsim
        """
        cup_pose = self.sim.get_rigid_object("paper_cup").get_local_pose(to_matrix=True)
        scoop_pose = self.sim.get_rigid_object("scoop").get_local_pose(to_matrix=True)
        first_cup_pose = cup_pose[0].to("cpu").numpy()
        first_scoop_pose = scoop_pose[0].to("cpu").numpy()

        left_arm_ids = self.robot.get_joint_ids("left_arm")
        current_qpos = self.robot.get_qpos()
        left_arm_current_qpos = current_qpos[:, left_arm_ids]
        left_arm_xpos = self.robot.compute_fk(
            qpos=left_arm_current_qpos, name="left_arm", to_matrix=True
        )
        left_arm_xpos = left_arm_xpos.to("cpu").numpy()

        cup_up_xpos = deepcopy(left_arm_xpos)
        cup_up_xpos[:, :3, 3] = first_cup_pose[:3, 3] + np.array([0, 0, 0.22])
        _, cup_up_qpos = self.robot.compute_ik(
            cup_up_xpos, joint_seed=left_arm_current_qpos, name="left_arm"
        )
        cup_up_qpos = cup_up_qpos[0].to("cpu").numpy()

        cup_down_xpos = deepcopy(left_arm_xpos)
        cup_down_xpos[:, :3, 3] = first_cup_pose[:3, 3] + np.array([0, 0, 0.18])
        _, cup_down_qpos = self.robot.compute_ik(
            cup_down_xpos, joint_seed=left_arm_current_qpos, name="left_arm"
        )
        cup_down_qpos = cup_down_qpos[0].to("cpu").numpy()

        scoop_up_xpos = deepcopy(left_arm_xpos)
        scoop_up_xpos[:, :3, 3] = first_scoop_pose[:3, 3] + np.array([0, 0, 0.2])
        _, scoop_up_qpos = self.robot.compute_ik(
            scoop_up_xpos, joint_seed=left_arm_current_qpos, name="left_arm"
        )
        scoop_up_qpos = scoop_up_qpos[0].to("cpu").numpy()

        scoop_down_xpos = deepcopy(left_arm_xpos)
        scoop_down_xpos[:, :3, 3] = first_scoop_pose[:3, 3] + np.array([0, 0, 0.16])
        _, scoop_down_qpos = self.robot.compute_ik(
            scoop_down_xpos, joint_seed=left_arm_current_qpos, name="left_arm"
        )
        scoop_down_qpos = scoop_down_qpos[0].to("cpu").numpy()
        left_arm_current_qpos = left_arm_current_qpos[0].to("cpu").numpy()
        left_arm_qposes = np.array(
            [
                left_arm_current_qpos,
                cup_up_qpos,
                cup_down_qpos,
                scoop_down_qpos,
                scoop_up_qpos,
                left_arm_current_qpos,
            ]
        )
        expand_left_arm_qposes = torch.tensor(
            mul_linear_expand(left_arm_qposes, 10),
            dtype=torch.float32,
            device=self.device,
        )
        key_indices = torch.tensor(
            [10, 20, 30, 40], dtype=torch.int32, device=self.device
        )
        key_obj_indices = torch.tensor(
            [0, 0, 1, 1], dtype=torch.int32, device=self.device
        )  # 0: cup, 1: scoop
        return expand_left_arm_qposes, key_indices, key_obj_indices
