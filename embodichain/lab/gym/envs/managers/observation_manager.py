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

"""Observation manager for orchestrating operations based on different simulation observations."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Union

from embodichain.utils import logger
from embodichain.lab.sim.types import EnvObs
from embodichain.lab.gym.utils.gym_utils import (
    fetch_data_from_dict,
    assign_data_to_dict,
)
from .manager_base import ManagerBase
from .cfg import ObservationCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


class ObservationManager(ManagerBase):
    """Manager for orchestrating operations based on different simulation observations.

    The default observation space will contain two observation groups:
        - `robot`: Contains the default observations related to the robot.
            - `qpos`: The joint positions of the robot.
            - `qvel`: The joint velocities of the robot.
            - `qf`: The joint forces of the robot.
        - `sensor`: Contains the observations related to the sensors which are enabled in the environment.

    The observation manager offers two modes of operation:
        - `modify`: This mode perform data fetching and modification on existing observation data.
        - `add`: This mode perform new observation computation and add new observation data to the observation space.
    """

    _env: EmbodiedEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: EmbodiedEnv):
        """Initialize the observation manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, ObservationCfg]``).
            env: An environment object.
        """

        self._mode_functor_names: dict[str, list[str]] = dict()
        self._mode_functor_cfgs: dict[str, list[ObservationCfg]] = dict()
        self._mode_class_functor_cfgs: dict[str, list[ObservationCfg]] = dict()

        # call the base class (this will parse the functors config)
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for observation manager."""
        functor_num = sum(len(v) for v in self._mode_functor_names.values())
        msg = f"<ObservationManager> contains {functor_num} active functors.\n"

        # add info on each mode
        for mode in self._mode_functor_names:
            # create table for functor information
            table = PrettyTable()
            table.title = f"Active Observation Functors in Mode: '{mode}'"

            table.field_names = ["Index", "Name"]
            table.align["Name"] = "l"
            for index, name in enumerate(self._mode_functor_names[mode]):
                table.add_row([index, name])

            # convert table to string
            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_functors(self) -> dict[str, list[str]]:
        """Name of active observation functors.

        The keys are the modes of observation and the values are the names of the observation functors.
        """
        return self._mode_functor_names

    """
    Operations.
    """

    def reset(self, env_ids: Union[Sequence[int], None] = None) -> dict[str, float]:
        # call all functors that are classes
        for mode_cfg in self._mode_class_functor_cfgs.values():
            for functor_cfg in mode_cfg:
                functor_cfg.func.reset(env_ids=env_ids)

        # nothing to log here
        return {}

    def compute(
        self,
        obs: EnvObs,
    ) -> EnvObs:
        """Calls each observation functor in the specified mode.

        This function iterates over all the observation functors in the specified mode and calls the function
        corresponding to the functor. The function is called with the environment instance and the environment
        indices to apply the observation to.

        Args:
            obs: The observation data to apply the observation to.

        Returns:
            The modified observation data.

        Raises:
            ValueError: If the mode is not supported.
        """

        # iterate over all the observation functors
        for mode, functor_cfgs in self._mode_functor_cfgs.items():
            for functor_cfg in functor_cfgs:
                functor_cfg: ObservationCfg

                if mode == "modify":
                    data = fetch_data_from_dict(obs, functor_cfg.name)
                    data = functor_cfg.func(self._env, data, **functor_cfg.params)
                elif mode == "add":
                    data = functor_cfg.func(self._env, obs, **functor_cfg.params)
                    assign_data_to_dict(obs, functor_cfg.name, data)
                else:
                    logger.log_error(f"Unsupported observation mode '{mode}'.")

        return obs

    def get_functor_cfg(self, functor_name: str) -> ObservationCfg:
        """Gets the configuration for the specified functor.

        The method finds the functor by name by searching through all the modes.
        It then returns the configuration of the functor with the first matching name.

        Args:
            functor_name: The name of the observation functor.

        Returns:
            The configuration of the observation functor.

        Raises:
            ValueError: If the functor name is not found.
        """
        for mode, functors in self._mode_functor_names.items():
            if functor_name in functors:
                return self._mode_functor_cfgs[mode][functors.index(functor_name)]
        logger.log_error(f"observation functor '{functor_name}' not found.")

    """
    Helper functions.
    """

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
            if not isinstance(functor_cfg, ObservationCfg):
                raise TypeError(
                    f"Configuration for the functor '{functor_name}' is not of type ObservationCfg."
                    f" Received: '{type(functor_cfg)}'."
                )

            # resolve common parameters
            self._resolve_common_functor_cfg(functor_name, functor_cfg, min_argc=2)

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
