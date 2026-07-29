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

"""Minimal Gym environment with random joint-space reach commands."""

from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.types import EnvAction, EnvObs
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.cfg import (
    RenderCfg,
    RobotCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.gym.utils.registration import register_env


@register_env("RandomReach-v1", override=True)
class RandomReachEnv(BaseEnv):
    """Environment that moves a UR10 toward random joint targets."""

    robot_init_qpos = np.array(
        [1.57079, -1.57079, 1.57079, -1.57079, -1.57079, -3.14159]
    )

    def __init__(
        self,
        num_envs: int = 1,
        headless: bool = False,
        device: str = "cpu",
        renderer: str = "hybrid",
        gpu_id: int = 0,
        arena_space: float = 2.0,
        **kwargs,
    ) -> None:
        env_cfg = EnvCfg(
            sim_cfg=SimulationManagerCfg(
                headless=headless,
                arena_space=arena_space,
                sim_device=device,
                render_cfg=RenderCfg(renderer=renderer),
                gpu_id=gpu_id,
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


def main() -> None:
    """Run a short random-reach throughput demo."""
    from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

    parser = argparse.ArgumentParser(
        description="Demo for running a random reach environment."
    )
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()

    env = gym.make(
        "RandomReach-v1",
        num_envs=args.num_envs,
        headless=args.headless,
        device=args.device,
        renderer=args.renderer,
        gpu_id=args.gpu_id,
        arena_space=args.arena_space,
    )

    try:
        device = env.get_wrapper_attr("device")
        num_envs = env.get_wrapper_attr("num_envs")
        init_pose = (
            torch.as_tensor(
                env.unwrapped.robot_init_qpos,
                dtype=torch.float32,
                device=device,
            )
            .unsqueeze(0)
            .repeat(num_envs, 1)
        )

        for episode in range(10):
            print("Episode:", episode)
            env.reset()
            started_at = time.perf_counter()

            for _ in range(100):
                action = init_pose + torch.rand_like(init_pose) * 0.2 - 0.1
                env.step(action)

            elapsed = time.perf_counter() - started_at
            total_steps = 100 * num_envs
            if elapsed > 0:
                print(f"Total steps: {total_steps}")
                print(f"Elapsed time: {elapsed:.2f} seconds")
                print(f"FPS: {total_steps / elapsed:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
