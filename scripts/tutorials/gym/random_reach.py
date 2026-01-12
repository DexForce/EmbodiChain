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

import torch
import numpy as np
import gymnasium as gym

from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.types import EnvAction, EnvObs
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.cfg import (
    RobotCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.gym.utils.registration import register_env


@register_env("RandomReach-v1", max_episode_steps=100, override=True)
class RandomReachEnv(BaseEnv):

    robot_init_qpos = np.array(
        [1.57079, -1.57079, 1.57079, -1.57079, -1.57079, -3.14159]
    )

    def __init__(
        self,
        num_envs=1,
        headless=False,
        device="cpu",
        **kwargs,
    ):
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

    def _setup_robot(self, **kwargs) -> Robot:
        from embodichain.data import get_data_path

        file_path = get_data_path("UniversalRobots/UR10/UR10.urdf")

        robot: Robot = self.sim.add_robot(
            cfg=RobotCfg(
                uid="ur10",
                fpath=file_path,
                init_pos=(0, 0, 1),
                init_qpos=self.robot_init_qpos,
            )
        )

        qpos_limits = robot.body_data.qpos_limits[0].cpu().numpy()
        self.single_action_space = gym.spaces.Box(
            low=qpos_limits[:, 0], high=qpos_limits[:, 1], dtype=np.float32
        )

        return robot

    def _prepare_scene(self, **kwargs) -> None:
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

    def _update_sim_state(self, **kwargs) -> None:
        pose = torch.eye(4, device=self.device)
        pose = pose.unsqueeze_(0).repeat(self.num_envs, 1, 1)
        pose[:, :3, 3] += torch.rand(self.num_envs, 3, device=self.device) * 0.5 - 0.25
        self.cube.set_local_pose(pose=pose)

    def _step_action(self, action: EnvAction) -> EnvAction:
        self.robot.set_qpos(qpos=action)
        return action

    def _extend_obs(self, obs: EnvObs, **kwargs) -> EnvObs:
        # You can also use `cube = self.sim.get_rigid_object("cube")` to access obj.
        # obs["cube_position"] = self.cube.get_local_pose()[:, :3]
        return obs


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Demo for running a random reach environment."
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="number of environments to run"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run the environment on, e.g., 'cpu' or 'cuda'",
    )
    parser.add_argument("--headless", action="store_true", help="run in headless mode")
    args = parser.parse_args()

    env = gym.make(
        "RandomReach-v1",
        num_envs=args.num_envs,
        headless=args.headless,
        device=args.device,
    )

    for episode in range(10):
        print("Episode:", episode)
        env.reset()
        start_time = time.time()
        total_steps = 0

        for i in range(100):
            action = env.action_space.sample()
            action = torch.as_tensor(action, dtype=torch.float32, device=env.device)

            init_pose = env.unwrapped.robot_init_qpos
            init_pose = (
                torch.as_tensor(init_pose, dtype=torch.float32, device=env.device)
                .unsqueeze_(0)
                .repeat(env.num_envs, 1)
            )
            action = (
                init_pose
                + torch.rand_like(action, dtype=torch.float32, device=env.device) * 0.2
                - 0.1
            )

            obs, reward, done, truncated, info = env.step(action)
            total_steps += env.num_envs

        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 0:
            fps = total_steps / elapsed_time
            print(f"Total steps: {total_steps}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")
            print(f"FPS: {fps:.2f}")
        else:
            print("Elapsed time is too short to calculate FPS.")
