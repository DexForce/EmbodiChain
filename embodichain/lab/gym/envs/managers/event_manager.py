# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.

# All rights reserved.
#
# This file incorporates code from the Isaac Lab Project
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ----------------------------------------------------------------------------

"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Union

from embodichain.utils import logger
from .manager_base import ManagerBase
from .cfg import EventCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


class EventManager(ManagerBase):
    """Manager for orchestrating operations based on different simulation events.

    The event manager applies operations to the environment based on different simulation events. For example,
    changing the masses of objects or their friction coefficients during initialization/ reset, or applying random
    pushes to the robot at a fixed interval of steps. The user can specify several modes of events to fine-tune the
    behavior based on when to apply the event.

    The event functors are parsed from a config class containing the manager's settings and each functor's
    parameters. Each event functor should instantiate the :class:`EventCfg` class.

    Event functors can be grouped by their mode. The mode is a user-defined string that specifies when
    the event functor should be applied. This provides the user complete control over when event
    functors should be applied.

    For a typical training process, you may want to apply events in the following modes:

    - "prestartup": Event is applied once at the beginning of the training before the simulation starts.
      This is used to randomize USD-level properties of the simulation stage.
    - "startup": Event is applied once at the beginning of the training once simulation is started.
    - "reset": Event is applied at every reset.
    - "interval": Event is applied at pre-specified intervals of time.

    However, you can also define your own modes and use them in the training process as you see fit.
    For this you will need to add the triggering of that mode in the environment implementation as well.

    .. note::

        The triggering of operations corresponding to the mode ``"interval"`` are the only mode that are
        directly handled by the manager itself. The other modes are handled by the environment implementation.

    """

    _env: EmbodiedEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: EmbodiedEnv):
        """Initialize the event manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, EventCfg]``).
            env: An environment object.
        """
        # create buffers to parse and store functors
        self._mode_functor_names: dict[str, list[str]] = dict()
        self._mode_functor_cfgs: dict[str, list[EventCfg]] = dict()
        self._mode_class_functor_cfgs: dict[str, list[EventCfg]] = dict()

        # call the base class (this will parse the functors config)
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for event manager."""
        functor_num = sum(len(v) for v in self._mode_functor_names.values())
        msg = f"<EventManager> contains {functor_num} active functors.\n"

        # add info on each mode
        for mode in self._mode_functor_names:
            # create table for functor information
            table = PrettyTable()
            table.title = f"Active Event Functors in Mode: '{mode}'"
            # add table headers based on mode
            if mode == "interval":
                table.field_names = ["Index", "Name", "Interval step"]
                table.align["Name"] = "l"
                for index, (name, cfg) in enumerate(
                    zip(self._mode_functor_names[mode], self._mode_functor_cfgs[mode])
                ):
                    table.add_row([index, name, cfg.interval_step])
            else:
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
        """Name of active event functors.

        The keys are the modes of event and the values are the names of the event functors.
        """
        return self._mode_functor_names

    @property
    def available_modes(self) -> list[str]:
        """Modes of events."""
        return list(self._mode_functor_names.keys())

    """
    Operations.
    """

    def reset(self, env_ids: Union[Sequence[int], None] = None) -> dict[str, float]:
        # call all functors that are classes
        for mode_cfg in self._mode_class_functor_cfgs.values():
            for functor_cfg in mode_cfg:
                functor_cfg.func.reset(env_ids=env_ids)

        # resolve number of environments
        if env_ids is None:
            num_envs = self._env.num_envs
        else:
            num_envs = len(env_ids)

        # May be add more useful reset logic later.

        # nothing to log here
        return {}

    def apply(
        self,
        mode: str,
        env_ids: Union[Sequence[int], None] = None,
    ):
        """Calls each event functor in the specified mode.

        This function iterates over all the event functors in the specified mode and calls the function
        corresponding to the functor. The function is called with the environment instance and the environment
        indices to apply the event to.

        For the "interval" mode, the function is called when the time interval has passed. This requires
        specifying the time step of the environment.

        For the "reset" mode, the function is called when the mode is "reset" and the total number of environment
        steps that have happened since the last trigger of the function is equal to its configured parameter for
        the number of environment steps between resets.

        Args:
            mode: The mode of event.
            env_ids: The indices of the environments to apply the event to.
                Defaults to None, in which case the event is applied to all environments when applicable.

        Raises:
            ValueError: If the mode is ``"interval"`` and the environment indices are provided. This is an undefined
                behavior as the environment indices are computed based on the time left for each environment.
            ValueError: If the mode is ``"reset"`` and the total number of environment steps that have happened
                is not provided.
        """
        # check if mode is valid
        if mode not in self._mode_functor_names:
            logger.log_warning(f"Event mode '{mode}' is not defined. Skipping event.")
            return

        if mode == "interval" and env_ids is not None:
            logger.log_error(
                f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
                " as the environment indices are computed based on the time left for each environment."
            )

        # iterate over all the event functors
        for index, functor_cfg in enumerate(self._mode_functor_cfgs[mode]):
            functor_cfg: EventCfg
            if mode == "interval":
                self._interval_functor_step_count[index] += 1

                # check if the interval has passed and sample a new interval
                # note: we compare with a small value to handle floating point errors
                if (
                    functor_cfg.is_global
                    and self._interval_functor_step_count[index]
                    % functor_cfg.interval_step
                    == 0
                ):

                    # call the event functor (with None for env_ids)
                    functor_cfg.func(self._env, None, **functor_cfg.params)
                else:
                    valid_env_ids = (
                        (
                            self._interval_functor_step_count[index]
                            % functor_cfg.interval_step
                            == 0
                        )
                        .nonzero()
                        .flatten()
                    )
                    if len(valid_env_ids) > 0:
                        # call the event functor
                        functor_cfg.func(self._env, valid_env_ids, **functor_cfg.params)
            elif mode == "reset":
                # resolve the environment indices
                if env_ids is None:
                    env_ids = slice(None)

                functor_cfg.func(self._env, env_ids, **functor_cfg.params)
            else:
                # call the event functor
                functor_cfg.func(self._env, env_ids, **functor_cfg.params)

    """
    Operations - Functor settings.
    """

    def set_functor_cfg(self, functor_name: str, cfg: EventCfg):
        """Sets the configuration of the specified functor into the manager.

        The method finds the functor by name by searching through all the modes.
        It then updates the configuration of the functor with the first matching name.

        Args:
            functor_name: The name of the event functor.
            cfg: The configuration for the event functor.

        Raises:
            ValueError: If the functor name is not found.
        """
        functor_found = False
        for mode, functors in self._mode_functor_names.items():
            if functor_name in functors:
                self._mode_functor_cfgs[mode][functors.index(functor_name)] = cfg
                functor_found = True
                break
        if not functor_found:
            logger.log_error(f"Event functor '{functor_name}' not found.")

    def get_functor_cfg(self, functor_name: str) -> EventCfg:
        """Gets the configuration for the specified functor.

        The method finds the functor by name by searching through all the modes.
        It then returns the configuration of the functor with the first matching name.

        Args:
            functor_name: The name of the event functor.

        Returns:
            The configuration of the event functor.

        Raises:
            ValueError: If the functor name is not found.
        """
        for mode, functors in self._mode_functor_names.items():
            if functor_name in functors:
                return self._mode_functor_cfgs[mode][functors.index(functor_name)]
        logger.log_error(f"Event functor '{functor_name}' not found.")

    """
    Operations - Visit functor.
    """

    def get_functor(self, functor_name: str):
        """
        Retrieve a functor from the configuration by its name.

        Args:
            functor_name (str): The name of the functor to retrieve.

        Returns:
            The functor if it exists in the configuration, otherwise None.
        """
        if hasattr(self.cfg, functor_name):
            functor = getattr(self.cfg, functor_name).func
            return functor
        else:
            logger.log_warning(
                f"Got no functor {functor_name} in event_manager, please check again."
            )
            return None

    """
    Helper functions.
    """

    def _prepare_functors(self):
        # buffer to store the time left for "interval" mode
        # if interval is global, then it is a single value, otherwise it is per environment
        self._interval_functor_step_count: list[torch.Tensor] = list()

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
            if not isinstance(functor_cfg, EventCfg):
                raise TypeError(
                    f"Configuration for the functor '{functor_name}' is not of type EventCfg."
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

            # resolve the mode of the events
            # -- interval mode
            if functor_cfg.mode == "interval":
                # sample the time left for global
                if functor_cfg.is_global:
                    count = torch.zeros(1, dtype=torch.int32, device=self.device)
                    self._interval_functor_step_count.append(count)
                else:
                    count = torch.zeros(
                        self.num_envs, dtype=torch.int32, device=self.device
                    )
                    self._interval_functor_step_count.append(count)
