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
import pytest
import numpy as np
import gymnasium as gym

from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.cfg import (
    RobotCfg,
    JointDrivePropertiesCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.data import get_data_path

NUM_ENVS = 10


@register_env("RandomReach-v1", max_episode_steps=100, override=True)
class RandomReachEnv(BaseEnv):

    robot_init_qpos = np.array(
        [1.57079, -1.57079, 1.57079, -1.57079, -1.57079, -3.14159]
    )

    def __init__(
        self,
        num_envs=1,
        drive_type="force",
        headless=False,
        device="cpu",
        **kwargs,
    ):
        self.drive_type = drive_type

        env_cfg = EnvCfg(
            sim_cfg=SimulationManagerCfg(
                headless=headless, arena_space=2.0, sim_device=device
            ),
            num_envs=num_envs,
        )

        super().__init__(
            cfg=env_cfg,
            **kwargs,
        )

    def _setup_robot(self, **kwargs):
        file_path = get_data_path("UniversalRobots/UR10/UR10.urdf")

        robot: Robot = self.sim.add_robot(
            cfg=RobotCfg(
                uid="UR10",
                fpath=file_path,
                init_pos=(0, 0, 1),
                init_qpos=self.robot_init_qpos,
                drive_pros=JointDrivePropertiesCfg(drive_type=self.drive_type),
            )
        )

        qpos_limits = robot.body_data.qpos_limits[0].cpu().numpy()
        self.single_action_space = gym.spaces.Box(
            low=qpos_limits[:, 0], high=qpos_limits[:, 1], dtype=np.float32
        )

        return robot

    def _prepare_scene(self, **kwargs):
        size = 0.03
        # Create a kinematic cube object without collision.
        # Currently, we use this workaround for visualization purposes.
        self.cube: RigidObject = self.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[size, size, size]),
                attrs=RigidBodyAttributesCfg(enable_collision=False),
                init_pos=(0.0, 0.0, 0.5),
                body_type="kinematic",
            ),
        )

    def _update_sim_state(self, **kwargs):
        pose = torch.eye(4, device=self.device)
        pose = pose.unsqueeze_(0).repeat(self.num_envs, 1, 1)
        pose[:, :3, 3] += torch.rand(self.num_envs, 3, device=self.device) * 0.5 - 0.25
        self.cube.set_local_pose(pose=pose)

    def _step_action(self, action):
        self.robot.set_qpos(qpos=action)
        return action

    def _extend_obs(self, obs, **kwargs):
        obs["cube_position"] = self.cube.get_local_pose()[:, :3]
        return obs


class BaseEnvTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        self.env = gym.make(
            "RandomReach-v1",
            num_envs=NUM_ENVS,
            headless=True,
            device=sim_device,
        )

    def test_env_rollout(self):
        """Test environment rollout."""
        for episode in range(2):
            print("Episode:", episode)
            obs, info = self.env.reset()

            for i in range(2):
                action = self.env.action_space.sample()
                action = torch.as_tensor(
                    action, dtype=torch.float32, device=self.env.device
                )

                init_pose = self.env.robot_init_qpos
                init_pose = (
                    torch.as_tensor(
                        init_pose, dtype=torch.float32, device=self.env.device
                    )
                    .unsqueeze_(0)
                    .repeat(self.env.num_envs, 1)
                )
                action = (
                    init_pose
                    + torch.rand_like(
                        action, dtype=torch.float32, device=self.env.device
                    )
                    * 0.2
                    - 0.1
                )

                obs, reward, done, truncated, info = self.env.step(action)

        assert reward.shape == (
            self.env.num_envs,
        ), f"Expected reward shape ({self.env.num_envs},), got {reward.shape}"
        assert done.shape == (
            self.env.num_envs,
        ), f"Expected done shape ({self.env.num_envs},), got {done.shape}"
        assert truncated.shape == (
            self.env.num_envs,
        ), f"Expected truncated shape ({self.env.num_envs},), got {truncated.shape}"
        assert (
            obs.get("cube_position") is not None
        ), "Expected 'cube_position' in the obs dict"
        assert obs.get("robot") is not None, "Expected 'robot' in the obs dict"

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.env.close()


class TestBaseEnvCPU(BaseEnvTest):
    def setup_method(self):
        self.setup_simulation("cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestBaseEnvCUDA(BaseEnvTest):
    def setup_method(self):
        self.setup_simulation("cuda")
