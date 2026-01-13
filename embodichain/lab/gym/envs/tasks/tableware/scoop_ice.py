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

import os
import torch
import numpy as np
import pickle

from copy import deepcopy
from typing import Sequence
from scipy.spatial.transform import Rotation as R

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.data import get_data_path
from embodichain.utils import logger
from tqdm import tqdm


@register_env("ScoopIce-v1", max_episode_steps=600)
class ScoopIce(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        self.affordance_datas = {}

        # TODO: hardcode code, should be implemented as functor way.
        self.trajectory = pickle.load(
            open(
                get_data_path("ScoopIceNewEnv/pose_record_20250919_184544.pkl"),
                "rb",
            )
        )
        self.trajectory_sample_rate = 2

    def set_scoop_pose(self, xyzrxryrz):
        scoop = self.sim.get_rigid_object("scoop")
        pose = np.eye(4)
        pose[:3, 3] = xyzrxryrz[:3]
        pose[:3, :3] = R.from_euler("XYZ", xyzrxryrz[3:], degrees=True).as_matrix()
        n_env = self.sim.num_envs
        pose_t = torch.tensor(
            pose[None, :, :].repeat(n_env, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        scoop.set_local_pose(pose_t)

    def set_cup_pose(self, xyzrxryrz):
        cup = self.sim.get_rigid_object("paper_cup")
        pose = np.eye(4)
        pose[:3, 3] = xyzrxryrz[:3]
        pose[:3, :3] = R.from_euler("XYZ", xyzrxryrz[3:], degrees=True).as_matrix()
        n_env = self.sim.num_envs
        pose_t = torch.tensor(
            pose[None, :, :].repeat(n_env, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        cup.set_local_pose(pose_t)

    def add_xpos_offset(self, arm_qpos: np.ndarray, offset: np.ndarray, is_left: bool):
        """Add offset to arm qposes along end-effector x axis.

        Args:
            arm_qposes (np.ndarray): [waypoint_num, dof]
        """
        waypoint_num = arm_qpos.shape[0]
        dof = arm_qpos.shape[1]
        offset_t = torch.tensor(offset, dtype=torch.float32, device=self.device)
        control_part = "left_arm" if is_left else "right_arm"

        arm_qpos_batch = torch.tensor(
            arm_qpos[None, :, :], dtype=torch.float32, device=self.device
        )

        arm_xpos_batch = self.robot.compute_batch_fk(
            qpos=arm_qpos_batch, name=control_part, to_matrix=True
        )
        arm_xpos_batch[:, :, :3, 3] += offset_t
        ret, arm_qpos_offset_batch = self.robot.compute_batch_ik(
            pose=arm_xpos_batch,
            joint_seed=arm_qpos_batch,
            name=control_part,
        )
        return arm_qpos_offset_batch[0].to("cpu").numpy()

    def pack_qpos(self):
        left_arm_qpos = self.trajectory["left_arm"]  # [waypoint_num, dof]
        logger.log_info("Adding x and z offset to left arm trajectory...")
        left_arm_qpos = self.add_xpos_offset(
            arm_qpos=left_arm_qpos, offset=np.array([-0.018, 0.0, -0.01]), is_left=True
        )
        right_arm_qpos = self.trajectory["right_arm"]  # [waypoint_num, dof]
        # TODO: add z offset to right arm
        logger.log_info("Adding z offset to right arm trajectory...")
        right_arm_qpos = self.add_xpos_offset(
            arm_qpos=right_arm_qpos, offset=np.array([0.00, 0.0, 0.02]), is_left=False
        )
        left_eef_qpos = self.trajectory["left_eef"]  # [waypoint_num, hand_dof]
        right_eef_qpos = self.trajectory["right_eef"]
        torso_qpos = self.trajectory["torso"]
        # TODO: need head qpos.

        left_arm_qpos_expand = left_arm_qpos[None, :, :].repeat(self.num_envs, axis=0)
        right_arm_qpos_expand = right_arm_qpos[None, :, :].repeat(self.num_envs, axis=0)
        left_eef_qpos_expand = left_eef_qpos[None, :, :].repeat(self.num_envs, axis=0)
        right_eef_qpos_expand = right_eef_qpos[None, :, :].repeat(self.num_envs, axis=0)
        torso_qpos_expand = torso_qpos[None, :, :].repeat(self.num_envs, axis=0)
        all_qpos = np.concatenate(
            [
                left_arm_qpos_expand,
                right_arm_qpos_expand,
                left_eef_qpos_expand,
                right_eef_qpos_expand,
                torso_qpos_expand,
            ],
            axis=2,
        )
        return all_qpos

    def _initialize_episode(
        self, env_ids: Sequence[int] | None = None, **kwargs
    ) -> None:

        left_arm_ids = self.robot.get_joint_ids(name="left_arm")
        right_arm_ids = self.robot.get_joint_ids(name="right_arm")
        left_eef_ids = self.robot.get_joint_ids(name="left_eef")
        right_eef_ids = self.robot.get_joint_ids(name="right_eef")
        torso_ids = self.robot.get_joint_ids(name="torso")
        all_ids = np.hstack(
            [left_arm_ids, right_arm_ids, left_eef_ids, right_eef_ids, torso_ids]
        )

        # TODO: read xy random range from config
        xy_random_range = np.array([[-0.01, -0.01], [0.01, 0.01]])
        xy_random_offset = np.zeros(shape=(self.num_envs, 2))
        for arena_id in range(self.num_envs):
            xy_random_offset[arena_id] = np.random.uniform(
                low=xy_random_range[0], high=xy_random_range[1], size=(2,)
            )
        # TODO: apply warping to container pose

        all_qpos = self.pack_qpos()
        all_qpos_t = torch.tensor(all_qpos, dtype=torch.float32, device=self.device)

        # to initial qpos
        left_open_qpos = np.array([0.06, 1.5, 0.2, 0.2, 0.2, 0.2])
        left_close_qpos = np.array([0.13, 1.5, 0.5, 0.5, 0.5, 0.5])
        right_open_qpos = np.array([0.3, 1.5, 0.3, 0.3, 0.3, 0.3])
        right_close_qpos = np.array([0.6, 1.5, 0.7, 0.5, 0.7, 0.6])

        all_qpos_t[:, :, 14:20] = torch.tensor(
            left_close_qpos, dtype=torch.float32, device=self.device
        )
        all_qpos_t[:, :, 20:26] = torch.tensor(
            right_close_qpos, dtype=torch.float32, device=self.device
        )

        first_close_qpos = all_qpos_t[:, 0, :].to("cpu").numpy()
        first_open_qpos = deepcopy(first_close_qpos)

        # to first open pose
        first_open_qpos[:, 14:20] = left_open_qpos
        first_open_qpos[:, 20:26] = right_open_qpos
        self.robot.set_qpos(
            torch.tensor(first_open_qpos, dtype=torch.float32, device=self.device),
            joint_ids=all_ids,
        )
        self.sim.update(step=200)
        # save warp trajectory as demo action list
        waypoint_num = self.trajectory["left_arm"].shape[0]
        current_qpos = self.robot.get_qpos()
        self.demo_action_list = []
        for waypoint_idx in range(waypoint_num):
            action = current_qpos.clone()
            action[:, all_ids] = all_qpos_t[:, waypoint_idx, :]
            # TODO: sample in trajectory
            self.demo_action_list.append(action)

            # TODO: tricky implementation. Hold the first joint state for a while.
            if waypoint_idx == 0:
                for _ in range(20):
                    self.demo_action_list.append(action)

        self.sim.update(step=100)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                self.event_manager.apply(mode="reset", env_ids=env_ids)

    def create_demo_action_list(self, *args, **kwargs):
        logger.log_info(
            f"The original demo action list length: {len(self.demo_action_list)}"
        )
        logger.log_info(
            f"Downsample the demo action list by self.trajectory_sample_rate5 times."
        )
        return self.demo_action_list[:: self.trajectory_sample_rate]
