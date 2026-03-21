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

"""Action manager for processing policy actions into robot control commands.

This module provides the :class:`ActionManager` class which handles the interpretation
and preprocessing of raw actions from the policy into the format expected by the robot.

The concrete action term implementations (e.g., :class:`QposTerm`, :class:`DeltaQposTerm`)
are available in :mod:`actions` module.
"""

from __future__ import annotations

import inspect
import torch
import numpy as np
import gymnasium as gym

from functools import cached_property
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal
from prettytable import PrettyTable
from tensordict import TensorDict

from embodichain.lab.sim.types import EnvAction
from embodichain.utils.string import string_to_callable
from embodichain.utils import logger

from .cfg import ActionTermCfg
from .manager_base import Functor, ManagerBase

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = ["ActionTerm", "ActionManager"]


class ActionTerm(Functor):
    """Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and converting them to the format expected by the robot (e.g., qpos, qvel, qf).
    """

    SUPPORTED_TYPES = ["qpos", "qvel", "qf", "eef_pose"]
    """The supported action types. Each term must specify one of these as its output type, which
    determines how the processed action is applied to the robot.
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        super().__init__(cfg, env)

    @property
    @abstractmethod
    def input_key(self) -> str:
        """The output type of the action term, which determines how the processed action is applied to the robot.

        Must be one of the supported types defined in SUPPORTED_TYPES.
        """
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term (policy output dimension)."""
        ...

    @abstractmethod
    def process_action(self, action: torch.Tensor) -> EnvAction | torch.Tensor:
        """Process raw action from policy into robot control format.

        Args:
            action: Raw action tensor from policy, shape (num_envs, action_dim).

        Returns:
            Processed action tensor ready for robot control, shape depends on input_key.
        """
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """Not used for ActionTerm; use process_action instead."""
        return self.process_action(*args, **kwargs)


class ActionManager(ManagerBase):
    """Manager for processing actions sent to the environment.

    The action manager handles the interpretation and preprocessing of raw actions
    from the policy into the format expected by the robot. It supports a single
    active action term per environment (matching current RL usage).
    """

    def __init__(self, cfg: object, env: EmbodiedEnv):
        """Initialize the action manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.
        """
        self._term_names: list[str] = []
        self._terms: dict[str, ActionTerm] = {}
        self._term_modes: dict[str, Literal["pre", "post"]] = {}
        self._mode_term_names: dict[Literal["pre", "post"], list[str]] = {
            "pre": [],
            "post": [],
        }
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active term(s).\n"
        table = PrettyTable()
        table.title = "Active Action Terms"
        table.field_names = ["Index", "Name", "Mode", "Dimension"]
        table.align["Name"] = "l"
        table.align["Mode"] = "c"
        table.align["Dimension"] = "r"
        for index, name in enumerate(self._term_names):
            term = self._terms[name]
            mode = self._term_modes.get(name, "pre")
            table.add_row([index, name, mode, term.action_dim])
        msg += table.get_string()
        msg += "\n"
        return msg

    @property
    def active_functors(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    def get_terms_by_mode(
        self, mode: Literal["pre", "post"]
    ) -> list[tuple[str, ActionTerm]]:
        """Get action terms filtered by mode.

        Args:
            mode: The mode to filter by ("pre" or "post").

        Returns:
            List of (name, term) tuples for terms with the specified mode.
        """
        return [(name, self._terms[name]) for name in self._mode_term_names[mode]]

    @cached_property
    def total_action_dim(self) -> int:
        """Total dimension of actions (sum of all term dimensions)."""
        terms = self.get_terms_by_mode("pre")
        return sum(term.action_dim for _, term in terms)

    @cached_property
    def single_action_space(self) -> torch.Tensor | gym.Space:
        terms = self.get_terms_by_mode("pre")
        if len(terms) == 0:
            qpos_limits = (
                self._env.robot.body_data.qpos_limits[0, self._env.active_joint_ids]
                .cpu()
                .numpy()
            )
            single_action_space = gym.spaces.Box(
                low=qpos_limits[:, 0], high=qpos_limits[:, 1], dtype=np.float32
            )
            return single_action_space
        else:
            # Create dict action space for multiple terms.
            spaces = {}
            for name, term in terms:
                if term.input_key == "qpos":
                    qpos_limits = (
                        self._env.robot.body_data.qpos_limits[
                            0, self._env.active_joint_ids
                        ]
                        .cpu()
                        .numpy()
                    )
                    spaces[term.input_key] = gym.spaces.Box(
                        low=qpos_limits[:, 0], high=qpos_limits[:, 1], dtype=np.float32
                    )
                elif term.input_key == "qvel":
                    qvel_limits = (
                        self._env.robot.body_data.qvel_limits[
                            0, self._env.active_joint_ids
                        ]
                        .cpu()
                        .numpy()
                    )
                    spaces[term.input_key] = gym.spaces.Box(
                        low=-qvel_limits, high=qvel_limits, dtype=np.float32
                    )
                elif term.input_key == "qf":
                    qf_limits = (
                        self._env.robot.body_data.qf_limits[
                            0, self._env.active_joint_ids
                        ]
                        .cpu()
                        .numpy()
                    )
                    spaces[term.input_key] = gym.spaces.Box(
                        low=-qf_limits, high=qf_limits, dtype=np.float32
                    )
                else:
                    spaces[term.input_key] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(term.action_dim,),
                        dtype=np.float32,
                    )
            if len(spaces) == 1 and "qpos" in spaces:
                return spaces["qpos"]
            else:
                return gym.spaces.Dict(spaces)

    def convert_policy_action_to_env_action(self, action: torch.Tensor) -> EnvAction:
        """Convert raw action from policy into robot control format.

        This is a convenience method for processing a raw action tensor through the active terms.
        It assumes the input action is ordered according to the active terms and concatenated into a single tensor.

        Args:
            action: Raw action tensor from policy, shape (num_envs, total_action_dim).

        Returns:
            Processed action tensor ready for robot control, shape depends on active terms.
        """
        terms = self.get_terms_by_mode("pre")
        if len(terms) == 0 or len(terms) == 1:
            return action
        else:
            action_dict = {}
            current_dim = 0
            for _, term in terms:
                term_action = action[:, current_dim : current_dim + term.action_dim]
                action_dict[term.input_key] = term_action
                current_dim += term.action_dim
            return TensorDict(
                action_dict, batch_size=[action.shape[0]], device=action.device
            )

    def get_action_dim_by_mode(self, mode: Literal["pre", "post"]) -> int:
        """Get total action dimension for terms of a specific mode.

        Args:
            mode: The mode to filter by ("pre" or "post").

        Returns:
            Sum of action dimensions for terms with the specified mode.
        """
        mode_terms = self.get_terms_by_mode(mode)
        return sum(term.action_dim for _, term in mode_terms)

    def process_action(
        self, action: EnvAction, mode: Literal["pre", "post"] = "pre"
    ) -> EnvAction:
        """Process raw action from policy into robot control format.

        Supports:
        1. Tensor input: Passed to the active (first) term of the specified mode.
        2. Dict/TensorDict input: Uses key matching term name; raises an error if no match.

        Args:
            action: Raw action from policy (tensor or dict).
            mode: The processing mode - "pre" for preprocessing (default) or "post"
                for postprocessing. When "post", only terms with mode="post" are applied.

        Returns:
            TensorDict action ready for robot control.
        """
        # Filter terms by mode
        mode_terms = self._mode_term_names[mode]

        if not mode_terms:
            return action

        if len(mode_terms) == 1:
            term_name = mode_terms[0]
            term = self._terms[term_name]
            return term.process_action(action)
        else:
            for name in mode_terms:
                term = self._terms[name]
                action[term.input_key] = term.process_action(action[term.input_key])
            return action

    def get_term(self, name: str) -> ActionTerm:
        """Get action term by name."""
        return self._terms[name]

    def _prepare_functors(self) -> None:
        """Parse config and create action terms.

        ActionTerm uses process_action(action) (a bound instance method) rather than
        __call__(env, env_ids, ...), so we skip the base class params signature check
        and resolve terms directly.
        """
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, ActionTermCfg):
                logger.log_error(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg. "
                    f"Received: '{type(term_cfg)}'.",
                    error_type=TypeError,
                )
            # Resolve string to callable (skip base class params check for ActionTerm)
            if isinstance(term_cfg.func, str):
                term_cfg.func = string_to_callable(term_cfg.func)
            if not callable(term_cfg.func):
                logger.log_error(
                    f"The action term '{term_name}' is not callable. "
                    f"Received: '{term_cfg.func}'",
                    error_type=TypeError,
                )
            if inspect.isclass(term_cfg.func) and not issubclass(
                term_cfg.func, ActionTerm
            ):
                logger.log_error(
                    f"Configuration for the term '{term_name}' must be a subclass of "
                    f"ActionTerm. Received: '{type(term_cfg.func)}'.",
                    error_type=TypeError,
                )
            self._process_functor_cfg_at_play(term_name, term_cfg)
            self._term_names.append(term_name)
            self._terms[term_name] = term_cfg.func
            self._term_modes[term_name] = term_cfg.mode
            self._mode_term_names[term_cfg.mode].append(term_name)
