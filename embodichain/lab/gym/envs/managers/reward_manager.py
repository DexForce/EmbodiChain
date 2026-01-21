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

"""Reward manager for orchestrating reward computation in reinforcement learning tasks."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Union

from embodichain.utils import logger
from .manager_base import ManagerBase
from .cfg import RewardCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


class RewardManager(ManagerBase):
    """Manager for orchestrating reward computation in reinforcement learning tasks.

    The reward manager computes rewards based on the current state of the environment and actions.
    It supports multiple reward terms that can be combined through weighted summation.

    The reward manager offers two modes of operation:
        - `add`: This mode computes a reward term and adds it to the total reward (weighted by the term's weight).
        - `replace`: This mode replaces the total reward with the computed value (useful for single reward functions).

    Note: The config key is used as the unique identifier and display name for each reward functor.
    """

    _env: EmbodiedEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: EmbodiedEnv):
        """Initialize the reward manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, RewardCfg]``).
            env: An environment object.
        """

        self._mode_functor_names: dict[str, list[str]] = dict()
        self._mode_functor_cfgs: dict[str, list[RewardCfg]] = dict()
        self._mode_class_functor_cfgs: dict[str, list[RewardCfg]] = dict()

        # call the base class (this will parse the functors config)
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        functor_num = sum(len(v) for v in self._mode_functor_names.values())
        msg = f"<RewardManager> contains {functor_num} active reward terms.\n"

        # add info on each mode
        for mode in self._mode_functor_names:
            # create table for functor information
            table = PrettyTable()
            table.title = f"Active Reward Terms in Mode: '{mode}'"

            table.field_names = ["Index", "Name", "Weight"]
            table.align["Name"] = "l"
            for index, name in enumerate(self._mode_functor_names[mode]):
                functor_cfg = self._mode_functor_cfgs[mode][index]
                weight = getattr(functor_cfg, "weight", 1.0)
                table.add_row([index, name, f"{weight:.3f}"])

            # convert table to string
            msg += table.get_string()
            msg += "\n"
        return msg

    @property
    def active_functors(self) -> dict[str, list[str]]:
        """Name of active reward functors.

        The keys are the modes of reward computation and the values are the names of the reward functors.
        """
        return self._mode_functor_names

    def reset(self, env_ids: Union[Sequence[int], None] = None) -> dict[str, float]:
        """Reset reward terms that are stateful (implemented as classes).

        Args:
            env_ids: The environment indices to reset. If None, all environments are reset.

        Returns:
            An empty dictionary (no logging needed for reset).
        """
        # call all functors that are classes
        for mode_cfg in self._mode_class_functor_cfgs.values():
            for functor_cfg in mode_cfg:
                functor_cfg.func.reset(env_ids=env_ids)

        # nothing to log here
        return {}

    def compute(
        self,
        obs: "EnvObs",
        action: "EnvAction",
        info: dict,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total reward by calling each reward functor.

        This function iterates over all the reward functors and calls them to compute individual
        reward terms. The terms are then combined according to their mode and weight.

        Args:
            obs: The observation from the environment.
            action: The action applied to the robot.
            info: Additional information dictionary.

        Returns:
            A tuple containing:
                - total_reward: The total reward for each environment (shape: [num_envs]).
                - reward_info: A dictionary mapping reward term names to their values for logging.

        Raises:
            ValueError: If the mode is not supported.
        """
        # initialize total reward
        total_reward = torch.zeros(self._env.num_envs, device=self._env.device)
        reward_info = {}

        # iterate over all the reward functors
        for mode, functor_cfgs in self._mode_functor_cfgs.items():
            for functor_name, functor_cfg in zip(
                self._mode_functor_names[mode], functor_cfgs
            ):
                functor_cfg: RewardCfg

                # compute reward term
                reward_term = functor_cfg.func(
                    self._env, obs=obs, action=action, info=info, **functor_cfg.params
                )

                # ensure reward is a tensor
                if not isinstance(reward_term, torch.Tensor):
                    reward_term = torch.tensor(
                        reward_term, device=self._env.device, dtype=torch.float32
                    )

                # apply weight from config
                weighted_reward = reward_term * functor_cfg.weight

                # combine reward based on mode
                if mode == "add":
                    total_reward += weighted_reward
                elif mode == "replace":
                    total_reward = weighted_reward
                else:
                    logger.log_error(f"Unsupported reward mode '{mode}'.")

                # store for logging (use unweighted value for clarity)
                reward_info[functor_name] = reward_term

        return total_reward, reward_info

    def get_functor_cfg(self, functor_name: str) -> RewardCfg:
        """Gets the configuration for the specified functor.

        The method finds the functor by name by searching through all the modes.
        It then returns the configuration of the functor with the first matching name.

        Args:
            functor_name: The name of the reward functor.

        Returns:
            The configuration of the reward functor.

        Raises:
            ValueError: If the functor name is not found.
        """
        for mode, functors in self._mode_functor_names.items():
            if functor_name in functors:
                return self._mode_functor_cfgs[mode][functors.index(functor_name)]
        logger.log_error(f"Reward functor '{functor_name}' not found.")

    def _prepare_functors(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the functors
        for functor_name, functor_cfg in cfg_items:
            # check for non config
            if functor_cfg is None:
                continue
            # check for valid config type
            if not isinstance(functor_cfg, RewardCfg):
                raise TypeError(
                    f"Configuration for the functor '{functor_name}' is not of type RewardCfg."
                    f" Received: '{type(functor_cfg)}'."
                )

            # resolve common parameters
            self._resolve_common_functor_cfg(functor_name, functor_cfg, min_argc=4)

            # check if mode is a new mode
            if functor_cfg.mode not in self._mode_functor_names:
                # add new mode
                self._mode_functor_names[functor_cfg.mode] = list()
                self._mode_functor_cfgs[functor_cfg.mode] = list()
                self._mode_class_functor_cfgs[functor_cfg.mode] = list()
            # add functor name and parameters
            self._mode_functor_names[functor_cfg.mode].append(functor_name)
            self._mode_functor_cfgs[functor_cfg.mode].append(functor_cfg)

            # check if the functor is a class
            if inspect.isclass(functor_cfg.func):
                self._mode_class_functor_cfgs[functor_cfg.mode].append(functor_cfg)
