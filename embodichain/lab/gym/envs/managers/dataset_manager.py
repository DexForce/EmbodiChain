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

"""Dataset manager for orchestrating dataset collection functors."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from collections.abc import Sequence

import torch
from prettytable import PrettyTable

from embodichain.utils import logger
from embodichain.lab.sim.types import EnvObs, EnvAction
from .manager_base import ManagerBase
from .cfg import DatasetFunctorCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


class DatasetManager(ManagerBase):
    """Manager for orchestrating dataset collection and saving using functors.

    The dataset manager supports multiple dataset formats through a functor system:
    - LeRobot format (via LeRobotRecorder)
    - HDF5 format (via HDF5Recorder)
    - Zarr format (via ZarrRecorder)
    - Custom formats (via user-defined functors)

    Each functor's step() method is called once per environment step and handles:
    - Recording observation-action pairs
    - Detecting episode completion (dones=True)
    - Auto-saving completed episodes

    Example configuration:
        >>> from embodichain.lab.gym.envs.managers.cfg import DatasetFunctorCfg
        >>> from embodichain.lab.gym.envs.managers.datasets import LeRobotRecorder
        >>>
        >>> @configclass
        >>> class MyEnvCfg:
        >>>     dataset: dict = {
        >>>         "lerobot": DatasetFunctorCfg(
        >>>             func=LeRobotRecorder,
        >>>             params={
        >>>                 "robot_meta": {...},
        >>>                 "instruction": {"lang": "pick and place"},
        >>>                 "extra": {"scene_type": "kitchen"},
        >>>                 "save_path": "/data/datasets",
        >>>                 "export_success_only": True,
        >>>             }
        >>>         )
        >>>     }
    """

    _env: EmbodiedEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: EmbodiedEnv):
        """Initialize the dataset manager.

        Args:
            cfg: Configuration object containing dataset functor configurations.
            env: The environment instance.
        """
        # Store functors by mode (similar to EventManager)
        self._mode_functor_names: dict[str, list[str]] = {}
        self._mode_functor_cfgs: dict[str, list[DatasetFunctorCfg]] = {}
        self._mode_class_functor_cfgs: dict[str, list[DatasetFunctorCfg]] = {}

        # Call base class to parse functors
        super().__init__(cfg, env)

        ## TODO: fix configurable_action.py to avoid getting env.metadata['dataset']
        # Extract robot_meta from functor params or plain config and add to env.metadata for backward compatibility
        # This allows legacy code (like action_bank) to access robot_meta via env.metadata["dataset"]["robot_meta"]
        robot_meta_found = False

        # First, try to extract from functor params
        for mode_cfgs in self._mode_functor_cfgs.values():
            for functor_cfg in mode_cfgs:
                if "robot_meta" in functor_cfg.params:
                    if not hasattr(env, "metadata"):
                        env.metadata = {}
                    if "dataset" not in env.metadata:
                        env.metadata["dataset"] = {}
                    env.metadata["dataset"]["robot_meta"] = functor_cfg.params[
                        "robot_meta"
                    ]
                    logger.log_info(
                        "Added robot_meta to env.metadata for backward compatibility"
                    )
                    robot_meta_found = True
                    break
            if robot_meta_found:
                break

        # If not found in functor params, try to extract from plain config
        if not robot_meta_found:
            # Check if config is dict or object
            if isinstance(self.cfg, dict):
                cfg_items = self.cfg.items()
            else:
                cfg_items = self.cfg.__dict__.items()

            for config_name, config_value in cfg_items:
                if config_name == "robot_meta" and isinstance(config_value, dict):
                    if not hasattr(env, "metadata"):
                        env.metadata = {}
                    if "dataset" not in env.metadata:
                        env.metadata["dataset"] = {}
                    env.metadata["dataset"]["robot_meta"] = config_value
                    logger.log_info(
                        "Added robot_meta to env.metadata for backward compatibility (from plain config)"
                    )
                    break

        logger.log_info(
            f"DatasetManager initialized with {sum(len(v) for v in self._mode_functor_names.values())} functors"
        )

    def __str__(self) -> str:
        """Returns: A string representation for dataset manager."""
        msg = f"<DatasetManager> contains {len(self._functor_names)} active functors.\n"

        table = PrettyTable()
        table.title = "Active Dataset Functors"
        table.field_names = ["Index", "Name", "Type"]
        table.align["Name"] = "l"

        for index, name in enumerate(self._functor_names):
            functor_cfg = self._functor_cfgs[index]
            functor_type = (
                functor_cfg.func.__class__.__name__
                if hasattr(functor_cfg.func, "__class__")
                else str(functor_cfg.func)
            )
            table.add_row([index, name, functor_type])

        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_functors(self) -> dict[str, list[str]]:
        """Name of active dataset functors by mode.

        The keys are the modes and the values are the names of the dataset functors.
        """
        return self._mode_functor_names

    @property
    def available_modes(self) -> list[str]:
        """List of available modes for the dataset manager."""
        return list(self._mode_functor_names.keys())

    """
    Operations.
    """

    def reset(
        self, env_ids: Union[Sequence[int], torch.Tensor, None] = None
    ) -> dict[str, float]:
        """Reset all dataset functors.

        Args:
            env_ids: The environment ids. Defaults to None.

        Returns:
            Empty dict (no logging info).
        """
        # Call reset on all class functors across all modes
        for mode_cfgs in self._mode_class_functor_cfgs.values():
            for functor_cfg in mode_cfgs:
                functor_cfg.func.reset(env_ids=env_ids)

        return {}

    def apply(
        self,
        mode: str,
        env_ids: Union[Sequence[int], torch.Tensor, None] = None,
        obs: Optional[EnvObs] = None,
        action: Optional[EnvAction] = None,
        dones: Optional[torch.Tensor] = None,
        terminateds: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Apply dataset functors for the specified mode.

        This method follows the same pattern as EventManager.apply() for consistency.
        Currently only supports mode="save" which handles both recording and auto-saving.

        Args:
            mode: The mode to apply (currently only "save" is supported).
            env_ids: The indices of the environments to apply the functor to.
                Defaults to None, in which case the functor is applied to all environments.
            obs: Observation from the environment (batched for all envs).
            action: Action applied to the environment (batched for all envs).
            dones: Boolean tensor indicating which envs completed episodes.
            terminateds: Boolean tensor indicating termination (success/fail).
            info: Info dict containing success/fail information.
        """
        # check if mode is valid
        if mode not in self._mode_functor_names:
            logger.log_warning(
                f"Dataset mode '{mode}' is not defined. Skipping dataset operation."
            )
            return

        # iterate over all the dataset functors for this mode
        for functor_cfg in self._mode_functor_cfgs[mode]:
            functor_cfg.func(
                self._env,
                env_ids,
                obs,
                action,
                dones,
                terminateds,
                info,
                **functor_cfg.params,
            )

    def finalize(self) -> Optional[str]:
        """Finalize all dataset functors.

        Called when the environment is closed. Saves any remaining episodes
        and finalizes all datasets.

        Returns:
            Path to the first saved dataset, or None if failed.
        """
        dataset_paths = []

        # Call finalize on all class functors across all modes
        for mode_cfgs in self._mode_class_functor_cfgs.values():
            for functor_cfg in mode_cfgs:
                if hasattr(functor_cfg.func, "finalize"):
                    try:
                        path = functor_cfg.func.finalize()
                        if path:
                            dataset_paths.append(path)
                    except Exception as e:
                        logger.log_error(f"Failed to finalize functor: {e}")

        if dataset_paths:
            logger.log_info(f"Finalized {len(dataset_paths)} datasets")
            return dataset_paths[0]

        return None

    def get_cached_data(self) -> list[Dict[str, Any]]:
        """Get cached data from all dataset functors (for online training).

        Iterates through all functors and collects cached data from those
        that support online training mode (have get_cached_data method).

        Returns:
            List of cached data dictionaries from all functors.
        """
        all_cached_data = []

        # Iterate through all modes and functors
        for mode_cfgs in self._mode_functor_cfgs.values():
            for functor_cfg in mode_cfgs:
                if hasattr(functor_cfg.func, "get_cached_data"):
                    cached_data = functor_cfg.func.get_cached_data()
                    all_cached_data.extend(cached_data)

        return all_cached_data

    def clear_cache(self) -> int:
        """Clear cached data from all dataset functors (for online training).

        Iterates through all functors and clears their cache if they
        support online training mode (have clear_cache method).

        Returns:
            Total number of cached items cleared across all functors.
        """
        total_cleared = 0

        # Iterate through all modes and functors
        for mode_cfgs in self._mode_functor_cfgs.values():
            for functor_cfg in mode_cfgs:
                if hasattr(functor_cfg.func, "clear_cache"):
                    cleared = functor_cfg.func.clear_cache()
                    total_cleared += cleared

        return total_cleared

    def get_functor_cfg(self, functor_name: str) -> DatasetFunctorCfg:
        """Gets the configuration for the specified functor.

        Args:
            functor_name: The name of the dataset functor.

        Returns:
            The configuration of the dataset functor.

        Raises:
            ValueError: If the functor name is not found.
        """
        for mode, functors in self._mode_functor_names.items():
            if functor_name in functors:
                return self._mode_functor_cfgs[mode][functors.index(functor_name)]
        logger.log_error(f"Dataset functor '{functor_name}' not found.")

    def _prepare_functors(self):
        """Prepare dataset functors from configuration.

        This method parses the configuration and initializes all dataset functors,
        organizing them by mode (similar to EventManager).
        """
        # Check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        # Iterate over all the functors
        for functor_name, functor_cfg in cfg_items:
            # Check for non config
            if functor_cfg is None:
                continue

            # Skip non-functor configurations (e.g., robot_meta which is a plain dict)
            # Functor configurations must have a "func" field
            if isinstance(functor_cfg, dict) and "func" not in functor_cfg:
                # This is a plain configuration (not a functor), skip it
                continue

            # Convert dict to DatasetFunctorCfg if needed (for JSON configs)
            if isinstance(functor_cfg, dict):
                functor_cfg = DatasetFunctorCfg(**functor_cfg)

            # Check for valid config type
            if not isinstance(functor_cfg, DatasetFunctorCfg):
                raise TypeError(
                    f"Configuration for '{functor_name}' is not of type DatasetFunctorCfg."
                    f" Received: '{type(functor_cfg)}'."
                )

            # Resolve common parameters
            # min_argc=7 to skip: env, env_ids, obs, action, dones, terminateds, info
            # These are runtime positional arguments, not config parameters
            self._resolve_common_functor_cfg(functor_name, functor_cfg, min_argc=7)

            # Check if mode is a new mode
            if functor_cfg.mode not in self._mode_functor_names:
                # add new mode
                self._mode_functor_names[functor_cfg.mode] = []
                self._mode_functor_cfgs[functor_cfg.mode] = []
                self._mode_class_functor_cfgs[functor_cfg.mode] = []

            # Add functor name and parameters
            self._mode_functor_names[functor_cfg.mode].append(functor_name)
            self._mode_functor_cfgs[functor_cfg.mode].append(functor_cfg)

            # Check if the functor is a class
            if inspect.isclass(functor_cfg.func):
                self._mode_class_functor_cfgs[functor_cfg.mode].append(functor_cfg)
