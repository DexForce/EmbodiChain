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
from typing import Dict, Any, Tuple

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs.rl_env import RLEnv
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.sim.types import EnvObs


@register_env("CartPoleRL", max_episode_steps=50, override=True)
class CartPoleEnv(RLEnv):
    """
    CartPole balancing task for reinforcement learning.

    The agent controls a cart (robot hand joint) to keep a pole balanced near the upright
    position by regulating its angle and angular velocity. Episodes are considered
    successful when the pole remains close to vertical with low velocity, and they
    terminate either when a maximum number of steps is reached or when the pole falls
    beyond an allowed tilt threshold.
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

    def get_reward(self, obs, action, info):
        """Get the reward for the current step (pole upward reward).

        Each SimulationManager env must implement its own get_reward function to define the reward function for the task, If the
        env is considered for RL/IL training.

        Args:
            obs: The observation from the environment.
            action: The action applied to the robot agent.
            info: The info dictionary.

        Returns:
            The reward for the current step.
        """
        pole_qpos = self.robot.get_qpos(name="hand").reshape(-1)  # [num_envs, ]

        normalized_upward = torch.abs(pole_qpos) / torch.pi
        reward = 1.0 - normalized_upward
        return reward

    def compute_task_state(
        self, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        qpos = self.robot.get_qpos(name="hand").reshape(-1)  # [num_envs, ]
        qvel = self.robot.get_qvel(name="hand").reshape(-1)  # [num_envs, ]
        upward_distance = torch.abs(qpos)
        is_success = torch.logical_and(upward_distance < 0.05, torch.abs(qvel) < 0.1)
        is_fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        metrics = {"distance_to_goal": upward_distance}
        return is_success, is_fail, metrics

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        is_timeout = self._elapsed_steps >= self.episode_length
        pole_qpos = self.robot.get_qpos(name="hand").reshape(-1)
        is_fallen = torch.abs(pole_qpos) > torch.pi * 0.5
        return is_timeout | is_fallen
