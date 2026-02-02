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

from __future__ import annotations

import copy
import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

from embodichain.utils.string import string_to_callable, resolve_matching_names
from embodichain.utils.utility import class_to_dict
from embodichain.utils import logger

from .cfg import FunctorCfg, SceneEntityCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


class Functor(ABC):
    """Base class for Functor.

    Functor implementations can be functions or classes. If the functor is a class, it should
    inherit from this base class and implement the required methods.

    Each manager is implemented as a class that inherits from the :class:`ManagerBase` class. Each manager
    class should also have a corresponding configuration class that defines the configuration functors for the
    manager. Each functor should the :class:`FunctorCfg` class or its subclass.

    Example pseudo-code for creating a manager:

    .. code-block:: python

        from embodichain.utils import configclass
        from embodichain.lab.gym.managers import ManagerBase
        from embodichain.lab.gym.managers FunctorCfg

        @configclass
        class MyManagerCfg:

            functor1: FunctorCfg = FunctorCfg(...)
            functor2: FunctorCfg = FunctorCfg(...)
            functor3: FunctorCfg = FunctorCfg(...)

        # define manager instance
        my_manager = ManagerBase(cfg=ManagerCfg(), env=env)

    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        """Initialize the functor.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # store the inputs
        self.cfg = cfg
        self._env = env

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._env.num_envs

    @property
    def device(self) -> str:
        """Device on which to perform computations."""
        return self._env.device

    @property
    def __name__(self) -> str:
        """Return the name of the class or subclass."""
        return self.__class__.__name__

    """
    Operations.
    """

    def reset(self, env_ids: Union[Sequence[int], None] = None) -> None:
        """Resets the functor.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        pass

    def serialize(self) -> dict:
        """General serialization call. Includes the configuration dict."""
        return {"cfg": class_to_dict(self.cfg)}

    def __call__(self, *args) -> Any:
        """Returns the value of the functor required by the manager.

        In case of a class implementation, this function is called by the manager
        to get the value of the functor. The arguments passed to this function are
        the ones specified in the functor configuration (see :attr:`FunctorCfg.params`).

        .. attention::
            To be consistent with memory-less implementation of functors with functions, it is
            recommended to ensure that the returned mutable quantities are cloned before
            returning them. For instance, if the functor returns a tensor, it is recommended
            to ensure that the returned tensor is a clone of the original tensor. This prevents
            the manager from storing references to the tensors and altering the original tensors.

        Args:
            *args: Variable length argument list.

        Returns:
            The value of the functor.
        """
        raise NotImplementedError(
            "The method '__call__' should be implemented by the subclass."
        )


class ManagerBase(ABC):
    """Base class for all managers."""

    def __init__(self, cfg: object, env: EmbodiedEnv):
        """Initialize the manager.

        This function is responsible for parsing the configuration object and creating the functors.

        If the simulation is not playing, the scene entities are not resolved immediately.
        Instead, the resolution is deferred until the simulation starts. This is done to ensure
        that the scene entities are resolved even if the manager is created after the simulation
        has already started.

        Args:
            cfg: The configuration object. If None, the manager is initialized without any functors.
            env: The environment instance.
        """
        # store the inputs
        self.cfg = copy.deepcopy(cfg)
        self._env = env

        # parse config to create functors information
        if self.cfg:
            self._prepare_functors()

    def __repr__(self) -> str:
        return self.__str__()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._env.num_envs

    @property
    def device(self) -> str:
        """Device on which to perform computations."""
        return self._env.device

    @property
    @abstractmethod
    def active_functors(self) -> Union[list[str], dict[str, list[str]]]:
        """Name of active functors."""
        raise NotImplementedError

    """
    Operations.
    """

    def reset(self, env_ids: Union[Sequence[int], None] = None) -> dict[str, float]:
        """Resets the manager and returns logging information for the current time-step.

        Args:
            env_ids: The environment ids for which to log data.
                Defaults None, which logs data for all environments.

        Returns:
            Dictionary containing the logging information.
        """
        return {}

    def find_functors(self, name_keys: Union[str, Sequence[str]]) -> list[str]:
        """Find functors in the manager based on the names.

        This function searches the manager for functors based on the names. The names can be
        specified as regular expressions or a list of regular expressions. The search is
        performed on the active functors in the manager.

        Please check the :meth:`~embodichain.utils.string.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the functor names.

        Returns:
            A list of functor names that match the input keys.
        """
        # resolve search keys
        if isinstance(self.active_functors, dict):
            list_of_strings = []
            for names in self.active_functors.values():
                list_of_strings.extend(names)
        else:
            list_of_strings = self.active_functors

        # return the matching names
        return resolve_matching_names(name_keys, list_of_strings)[1]

    def get_active_iterable_functors(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active functors as iterable sequence of tuples.

        The first element of the tuple is the name of the functor and the second element is the raw value(s) of the functor.

        Returns:
            The active functors.
        """
        raise NotImplementedError

    """
    Implementation specific.
    """

    @abstractmethod
    def _prepare_functors(self):
        """Prepare functors information from the configuration object."""
        raise NotImplementedError

    """
    Internal callbacks.
    """

    def _resolve_functors_callback(self, event):
        """Resolve configurations of functors once the simulation starts.

        Please check the :meth:`_process_functor_cfg_at_play` method for more information.
        """
        # check if scene entities have been resolved
        if self._is_scene_entities_resolved:
            return
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
            # process attributes at runtime
            # these properties are only resolvable once the simulation starts playing
            self._process_functor_cfg_at_play(functor_name, functor_cfg)

        # set the flag
        self._is_scene_entities_resolved = True

    """
    Internal functions.
    """

    def _resolve_common_functor_cfg(
        self, functor_name: str, functor_cfg: FunctorCfg, min_argc: int = 1
    ):
        """Resolve common attributes of the functor configuration.

        Usually, called by the :meth:`_prepare_functors` method to resolve common attributes of the functor
        configuration. These include:

        * Resolving the functor function and checking if it is callable.
        * Checking if the functor function's arguments are matched by the parameters.
        * Resolving special attributes of the functor configuration like ``entity_cfg``, ``sensor_cfg``, etc.
        * Initializing the functor if it is a class.

        By default, all functor functions are expected to have at least one argument, which is the
        environment object. Some other managers may expect functions to take more arguments, for
        instance, the environment indices as the second argument. In such cases, the
        ``min_argc`` argument can be used to specify the minimum number of arguments
        required by the functor function to be called correctly by the manager.

        Args:
            functor_name: The name of the functor.
            functor_cfg: The functor configuration.
            min_argc: The minimum number of arguments required by the functor function to be called correctly
                by the manager.

        Raises:
            TypeError: If the functor configuration is not of type :class:`FunctorCfg`.
            ValueError: If the scene entity defined in the functor configuration does not exist.
            AttributeError: If the functor function is not callable.
            ValueError: If the functor function's arguments are not matched by the parameters.
        """
        # check if the functor is a valid functor config
        if not isinstance(functor_cfg, FunctorCfg):
            raise TypeError(
                f"Configuration for the functor '{functor_name}' is not of type FunctorCfg."
                f" Received: '{type(functor_cfg)}'."
            )

        # get the corresponding function or functional class
        if isinstance(functor_cfg.func, str):
            functor_cfg.func = string_to_callable(functor_cfg.func)
        # check if function is callable
        if not callable(functor_cfg.func):
            raise AttributeError(
                f"The functor '{functor_name}' is not callable. Received: {functor_cfg.func}"
            )

        # check if the functor is a class of valid type
        if inspect.isclass(functor_cfg.func):
            if not issubclass(functor_cfg.func, Functor):
                raise TypeError(
                    f"Configuration for the functor '{functor_name}' is not of type ManagerTermBase."
                    f" Received: '{type(functor_cfg.func)}'."
                )
            func_static = functor_cfg.func.__call__
            min_argc += 1  # forward by 1 to account for 'self' argument
        else:
            func_static = functor_cfg.func
        # check if function is callable
        if not callable(func_static):
            raise AttributeError(
                f"The functor '{functor_name}' is not callable. Received: {functor_cfg.func}"
            )

        # check statically if the functor's arguments are matched by params
        functor_params = list(functor_cfg.params.keys())
        args = inspect.signature(func_static).parameters
        args_with_defaults = [
            arg for arg in args if args[arg].default is not inspect.Parameter.empty
        ]
        args_without_defaults = [
            arg for arg in args if args[arg].default is inspect.Parameter.empty
        ]
        args = args_without_defaults + args_with_defaults
        # ignore first two arguments for env and env_ids
        # Think: Check for cases when kwargs are set inside the function?
        if len(args) > min_argc:
            if set(args[min_argc:]) != set(functor_params + args_with_defaults):
                raise ValueError(
                    f"The functor '{functor_name}' expects mandatory parameters: {args_without_defaults[min_argc:]}"
                    f" and optional parameters: {args_with_defaults}, but received: {functor_params}."
                )

        # process attributes at runtime
        # these properties are only resolvable once the simulation starts playing
        self._process_functor_cfg_at_play(functor_name, functor_cfg)

    def _process_functor_cfg_at_play(self, functor_name: str, functor_cfg: FunctorCfg):
        """Process the functor configuration at runtime.

        This function is called when the simulation starts playing. It is used to process the functor
        configuration at runtime. This includes:

        * Resolving the scene entity configuration for the functor.
        * Initializing the functor if it is a class.

        Since the above steps rely on PhysX to parse over the simulation scene, they are deferred
        until the simulation starts playing.

        Args:
            functor_name: The name of the functor.
            functor_cfg: The functor configuration.
        """

        def _resolve_scene_entities(value, key_path: str):
            # Resolve SceneEntityCfg anywhere in nested params (dict/list/tuple).
            if isinstance(value, SceneEntityCfg):
                # load the entity
                try:
                    value.resolve(self._env.sim)
                except ValueError as e:
                    raise ValueError(
                        f"Error while parsing '{functor_name}:{key_path}'. {e}"
                    )
                # log the entity for checking later
                msg = f"[{functor_cfg.__class__.__name__}:{functor_name}] Found entity '{value.uid}'."
                if value.joint_ids is not None:
                    msg += f"\n\tJoint names: {value.joint_names} [{value.joint_ids}]"
                if value.body_ids is not None:
                    msg += f"\n\tBody names: {value.body_names} [{value.body_ids}]"
                # print the information
                print(f"[INFO]: {msg}")
                return value
            if isinstance(value, list):  # recursively resolve the list
                return [
                    _resolve_scene_entities(v, f"{key_path}[{i}]")
                    for i, v in enumerate(value)
                ]
            if isinstance(value, tuple):  # recursively resolve the tuple
                return tuple(
                    _resolve_scene_entities(v, f"{key_path}[{i}]")
                    for i, v in enumerate(value)
                )
            if isinstance(value, dict):  # recursively resolve the dict
                return {
                    k: _resolve_scene_entities(v, f"{key_path}.{k}")
                    for k, v in value.items()
                }
            return value

        for key, value in list(functor_cfg.params.items()):
            # store the entity
            functor_cfg.params[key] = _resolve_scene_entities(value, key)

        # initialize the functor if it is a class
        if inspect.isclass(functor_cfg.func):
            try:
                logger.log_info(
                    f"Initializing functor '{functor_name}' with class '{functor_cfg.func.__name__}'."
                )
                functor_cfg.func = functor_cfg.func(cfg=functor_cfg, env=self._env)
            except Exception as e:
                logger.log_error(f"Failed to initialize functor '{functor_name}': {e}")
