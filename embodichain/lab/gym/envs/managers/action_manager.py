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
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import torch
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

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        super().__init__(cfg, env)

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term (policy output dimension)."""
        ...

    @abstractmethod
    def process_action(self, action: torch.Tensor) -> EnvAction:
        """Process raw action from policy into robot control format.

        Args:
            action: Raw action tensor from policy, shape (num_envs, action_dim).

        Returns:
            TensorDict with keys such as "qpos", "qvel", or "qf" ready for robot control.
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

    def get_functors_by_mode(
        self, mode: Literal["pre", "post"]
    ) -> list[tuple[str, ActionTerm]]:
        """Get action terms filtered by mode.

        Args:
            mode: The mode to filter by ("pre" or "post").

        Returns:
            List of (name, term) tuples for terms with the specified mode.
        """
        return [(name, self._terms[name]) for name in self._mode_term_names[mode]]

    @property
    def action_type(self) -> str:
        """The active action type (term name) for backward compatibility."""
        return self._term_names[0]

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions (sum of all term dimensions)."""
        return sum(t.action_dim for t in self._terms.values())

    def get_action_dim_by_mode(self, mode: Literal["pre", "post"]) -> int:
        """Get total action dimension for terms of a specific mode.

        Args:
            mode: The mode to filter by ("pre" or "post").

        Returns:
            Sum of action dimensions for terms with the specified mode.
        """
        mode_terms = self.get_functors_by_mode(mode)
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
            logger.log_error(
                f"No action terms found for mode '{mode}'. "
                f"Available terms: {self._term_names}",
                error_type=ValueError,
            )

        # TODO: We should refactor the action manager to support multiple active terms.
        if not isinstance(action, (dict, TensorDict)):
            return self._terms[mode_terms[0]].process_action(action)

        # Dict input: find matching term
        for term_name in mode_terms:
            if term_name in action:
                return self._terms[term_name].process_action(action[term_name])

        logger.log_error(
            f"No valid action keys. Expected one of: {mode_terms}",
            error_type=ValueError,
        )

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
