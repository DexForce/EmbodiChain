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
from typing import Dict, Any, Optional, Sequence
from gymnasium import spaces

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.types import EnvObs, EnvAction
from embodichain.utils import logger


@register_env("CartPoleRL", max_episode_steps=50, override=True)
class CartPoleEnv(EmbodiedEnv):
    """Push cube task for reinforcement learning.

    The task involves pushing a cube to a target goal position using a robotic arm.
    The reward consists of reaching reward, placing reward, action penalty, and success bonus.
    """

    def __init__(self, cfg=None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        cfg.sim_cfg.arena_space = 35.0
        extensions = getattr(cfg, "extensions", {}) or {}

        # cfg.sim_cfg.enable_rt = True

        defaults = {
            "qvel_weight": 0.01,
            "action_scale": 0.02,
            "episode_length": 100,
            "slider_limit": 15,
            "pole_limit": 6.283185307,
            "veloctity_weight": 0.01,
        }
        for name, default in defaults.items():
            value = extensions.get(name, getattr(cfg, name, default))
            setattr(cfg, name, value)
            setattr(self, name, getattr(cfg, name))

        self.last_cube_goal_dist = None

        super().__init__(cfg, **kwargs)

    def _init_sim_state(self, **kwargs):
        super()._init_sim_state(**kwargs)
        self.single_action_space = spaces.Box(
            low=-self.slider_limit,
            high=self.slider_limit,
            shape=(1,),
            dtype=np.float32,
        )
        if self.obs_mode == "state":
            self.single_observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )

    def _initialize_episode(
        self, env_ids: Optional[Sequence[int]] = None, **kwargs
    ) -> None:
        super()._initialize_episode(env_ids=env_ids, **kwargs)

    def _step_action(self, action: EnvAction) -> EnvAction:

        if self.obs_mode == "state":
            delta_qpos = action[:, 0]
        scaled_action = delta_qpos * self.action_scale
        scaled_action = torch.clamp(
            scaled_action, -self.slider_limit, self.slider_limit
        )
        current_qpos = self.robot.get_qpos(name="arm")
        target_qpos = current_qpos + scaled_action[:, None]
        self.robot.set_qpos(qpos=target_qpos, name="arm")
        return scaled_action

    def get_obs(self, **kwargs) -> EnvObs:
        qpos = self.robot.get_qpos(name="hand")
        qvel = self.robot.get_qvel(name="hand")
        if self.obs_mode == "state":
            return torch.cat([qpos, qvel], dim=1)
        return {
            "robot": {"qpos": qpos, "qvel": qvel},
        }

    def get_reward(
        self, obs: EnvObs, action: EnvAction, info: Dict[str, Any]
    ) -> torch.Tensor:
        if self.obs_mode == "state":
            qpos = obs[:, 0:1].reshape(-1)
            qvel = obs[:, 1:2].reshape(-1)
        else:
            qpos = obs["robot"]["qpos"].reshape(-1)
            qvel = obs["robot"]["qvel"].reshape(-1)
        upward_reward = 1 - torch.abs(qpos) / self.pole_limit
        velocity_reward = 1 - torch.tanh(torch.abs(qvel))
        reward = upward_reward + self.veloctity_weight * velocity_reward
        if "rewards" not in info:
            info["rewards"] = {}
        info["rewards"]["upward_reward"] = upward_reward
        info["rewards"]["velocity_reward"] = velocity_reward
        return reward

    def get_info(self, **kwargs) -> Dict[str, Any]:
        pole_qpos = self.robot.get_qpos(name="hand").reshape(-1)
        pole_qvel = self.robot.get_qvel(name="hand").reshape(-1)
        upward_distance = torch.abs(pole_qpos)
        is_success = torch.logical_and(
            upward_distance < 0.05, torch.abs(pole_qvel) < 0.1
        )

        info = {
            "success": is_success,
            "fail": torch.zeros(
                self.cfg.num_envs, device=self.device, dtype=torch.bool
            ),
            "elapsed_steps": self._elapsed_steps,
        }
        info["metrics"] = {
            "upward_distance": upward_distance,
        }
        return info

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        is_timeout = self._elapsed_steps >= self.episode_length
        pole_qpos = self.robot.get_qpos(name="hand").reshape(-1)
        is_fallen = torch.abs(pole_qpos) > torch.pi * 0.25
        return is_timeout | is_fallen

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        info = self.get_info(**kwargs)
        return {
            "success": info["success"][0].item(),
            "upward_distance": info["upward_distance"],
        }
