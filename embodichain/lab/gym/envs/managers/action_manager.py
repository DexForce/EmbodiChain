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

"""Ordered flat policy-action processing and application."""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
import inspect
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from prettytable import PrettyTable
import torch

from embodichain.utils import logger
from embodichain.utils.string import string_to_callable

from .action_types import ActionDescriptor, ActionTermDescriptor
from .cfg import ActionTermCfg
from .manager_base import Functor, ManagerBase

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = ["ActionManager", "ActionTerm"]


class ActionTerm(Functor):
    """Base class for one independently processed and applied action slice."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        """Initialize an action term.

        Args:
            cfg: Term configuration.
            env: Owning embodied environment.
        """
        super().__init__(cfg, env)

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Number of raw policy-action dimensions owned by the term."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Box:
        """Raw policy-action bounds for this term."""
        ...

    @property
    @abstractmethod
    def raw_actions(self) -> torch.Tensor:
        """Current raw policy-action rows."""
        ...

    @property
    @abstractmethod
    def processed_actions(self) -> torch.Tensor:
        """Current term-owned processed command rows."""
        ...

    @property
    @abstractmethod
    def command_type(self) -> str:
        """Controller command or resource type written by the term."""
        ...

    @property
    @abstractmethod
    def controlled_joint_ids(self) -> tuple[int, ...]:
        """Ordered robot joint IDs owned by the term."""
        ...

    @property
    @abstractmethod
    def descriptor(self) -> ActionTermDescriptor:
        """Describe the raw policy-action representation."""
        ...

    @abstractmethod
    def process_actions(self, actions: torch.Tensor) -> None:
        """Process one manager-owned raw action slice."""
        ...

    @abstractmethod
    def apply_actions(self) -> None:
        """Apply the latest processed command to the owned resource."""
        ...


class ActionManager(ManagerBase):
    """Split one flat policy action across independently applied terms."""

    def __init__(self, cfg: object, env: EmbodiedEnv):
        """Initialize action terms, their slices, and manager-owned history.

        Args:
            cfg: Mapping or component object containing `ActionTermCfg` values.
            env: Owning embodied environment.
        """
        self._term_names: list[str] = []
        self._terms: dict[str, ActionTerm] = {}
        self._slices: dict[str, slice] = {}
        self._descriptors: tuple[ActionDescriptor, ...] = ()
        super().__init__(cfg, env)
        self._action = torch.zeros(
            (env.num_envs, self.total_action_dim),
            dtype=torch.float32,
            device=env.device,
        )
        self._previous_action = torch.zeros_like(self._action)

    def __str__(self) -> str:
        """Return a table describing the ordered flat action layout."""
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Slice", "Dimension", "Command"]
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        for index, name in enumerate(self._term_names):
            term = self._terms[name]
            term_slice = self._slices[name]
            table.add_row(
                [
                    index,
                    name,
                    f"[{term_slice.start}:{term_slice.stop}]",
                    term.action_dim,
                    term.command_type,
                ]
            )
        return f"<ActionManager> contains {len(self._term_names)} active term(s).\n{table}\n"

    @property
    def active_functors(self) -> list[str]:
        """Return active term names in flat policy order."""
        return list(self._term_names)

    @property
    def total_action_dim(self) -> int:
        """Return the width of the complete flat policy action."""
        return sum(term.action_dim for term in self._terms.values())

    @property
    def action(self) -> torch.Tensor:
        """Return current complete raw policy actions."""
        return self._action

    @property
    def previous_action(self) -> torch.Tensor:
        """Return previous complete raw policy actions."""
        return self._previous_action

    @property
    def descriptors(self) -> tuple[ActionDescriptor, ...]:
        """Return ordered action-term descriptors bound to flat slices."""
        return self._descriptors

    @cached_property
    def single_action_space(self) -> gym.spaces.Box:
        """Return concatenated raw action bounds in term order."""
        if not self._terms:
            return gym.spaces.Box(
                low=np.empty((0,), dtype=np.float32),
                high=np.empty((0,), dtype=np.float32),
                dtype=np.float32,
            )
        lows = [
            np.asarray(term.action_space.low, dtype=np.float32)
            for term in self._terms.values()
        ]
        highs = [
            np.asarray(term.action_space.high, dtype=np.float32)
            for term in self._terms.values()
        ]
        return gym.spaces.Box(
            low=np.concatenate(lows),
            high=np.concatenate(highs),
            dtype=np.float32,
        )

    def process_action(self, action: torch.Tensor) -> None:
        """Validate, store, and distribute one complete flat policy action.

        Args:
            action: Floating tensor with shape `(num_envs, total_action_dim)`.

        Raises:
            TypeError: If the action is not a floating tensor.
            ValueError: If shape or device does not match the environment.
        """
        if not isinstance(action, torch.Tensor):
            raise TypeError(
                "ActionManager expects one flat torch.Tensor policy action."
            )
        expected = (self._env.num_envs, self.total_action_dim)
        if tuple(action.shape) != expected:
            raise ValueError(
                f"Expected action shape {expected}, got {tuple(action.shape)}."
            )
        if not action.is_floating_point():
            raise TypeError("Policy action must use a floating dtype.")
        expected_device = torch.device(self._env.device)
        if action.device != expected_device:
            raise ValueError(
                f"Policy action must be on {expected_device}, got {action.device}."
            )

        self._previous_action.copy_(self._action)
        self._action.copy_(action)
        for name, term in self._terms.items():
            term.process_actions(self._action[:, self._slices[name]])

    def apply_action(self) -> None:
        """Apply every processed term command in configuration order."""
        for term in self._terms.values():
            term.apply_actions()

    def mask_inactive(self, active_mask: torch.Tensor) -> None:
        """Replace inactive vector rows with resource-safe commands.

        Position terms hold measured selected-joint positions. Velocity and
        effort terms use zero commands. This is used by sticky vectorized demo
        execution after term processing and before application.

        Args:
            active_mask: Boolean tensor of shape ``(num_envs,)`` on the
                environment device. ``True`` rows retain processed commands.

        Raises:
            TypeError: If the mask is not boolean.
            ValueError: If mask or processed-command layout is invalid.
            RuntimeError: If a resource type has no defined safe inactive value.
        """
        if not isinstance(active_mask, torch.Tensor) or active_mask.dtype != torch.bool:
            raise TypeError("active_mask must be a boolean torch.Tensor.")
        expected = (self._env.num_envs,)
        if tuple(active_mask.shape) != expected:
            raise ValueError(
                f"Expected active_mask shape {expected}, got {tuple(active_mask.shape)}."
            )
        expected_device = torch.device(self._env.device)
        if active_mask.device != expected_device:
            raise ValueError(
                f"active_mask must be on {expected_device}, got {active_mask.device}."
            )

        measured_qpos: torch.Tensor | None = None
        for name, term in self._terms.items():
            joint_ids = list(term.controlled_joint_ids)
            processed = term.processed_actions
            expected_processed = (self._env.num_envs, len(joint_ids))
            if tuple(processed.shape) != expected_processed:
                raise ValueError(
                    f"Action term {name!r} processed command must have shape "
                    f"{expected_processed}, got {tuple(processed.shape)}."
                )
            if term.command_type == "qpos":
                if measured_qpos is None:
                    measured_qpos = self._env.robot.get_qpos()
                replacement = measured_qpos[:, joint_ids]
            elif term.command_type in {"qvel", "qf"}:
                replacement = torch.zeros_like(processed)
            else:
                raise RuntimeError(
                    f"Action term {name!r} command type {term.command_type!r} "
                    "does not define an inactive-row safety command."
                )
            processed.copy_(torch.where(active_mask[:, None], processed, replacement))

    def reset(
        self, env_ids: list[int] | torch.Tensor | None = None
    ) -> dict[str, float]:
        """Reset manager and term state for selected environments.

        Args:
            env_ids: Selected rows. `None` resets every row.

        Returns:
            Empty diagnostic mapping.
        """
        ids = slice(None) if env_ids is None else env_ids
        self._action[ids] = 0
        self._previous_action[ids] = 0
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        return {}

    def get_term(self, name: str) -> ActionTerm:
        """Return one configured term by name."""
        return self._terms[name]

    def _prepare_functors(self) -> None:
        """Resolve action term classes and bind ordered slices/descriptors."""
        cfg_items = (
            self.cfg.items()
            if isinstance(self.cfg, dict)
            else self.cfg.__dict__.items()
        )
        descriptors: list[ActionDescriptor] = []
        owners: dict[tuple[str, int], tuple[str, str]] = {}
        offset = 0

        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for action term {term_name!r} must be ActionTermCfg, "
                    f"got {type(term_cfg).__name__}."
                )
            if isinstance(term_cfg.func, str):
                term_cfg.func = string_to_callable(term_cfg.func)
            if not inspect.isclass(term_cfg.func) or not issubclass(
                term_cfg.func, ActionTerm
            ):
                raise TypeError(
                    f"Action term {term_name!r} must resolve to an ActionTerm class."
                )

            self._process_functor_cfg_at_play(term_name, term_cfg)
            term = term_cfg.func
            if not isinstance(term, ActionTerm):
                raise TypeError(
                    f"Action term {term_name!r} did not initialize correctly."
                )
            if term.action_dim <= 0:
                raise ValueError(
                    f"Action term {term_name!r} must own at least one dimension."
                )
            if not isinstance(
                term.action_space, gym.spaces.Box
            ) or term.action_space.shape != (term.action_dim,):
                raise ValueError(
                    f"Action term {term_name!r} must expose a Box with shape "
                    f"({term.action_dim},)."
                )

            start = offset
            stop = start + term.action_dim
            self._term_names.append(term_name)
            self._terms[term_name] = term
            self._slices[term_name] = slice(start, stop)
            descriptors.append(
                ActionDescriptor(term_name, start, stop, term.descriptor)
            )
            offset = stop

            controlled_joint_ids = tuple(term.controlled_joint_ids)
            descriptor_joint_names = tuple(term.descriptor.joint_names)
            if term.command_type in {"qpos", "qvel", "qf"}:
                expected_joint_names = tuple(
                    self._env.robot.joint_names[int(joint_id)]
                    for joint_id in controlled_joint_ids
                )
                if descriptor_joint_names != expected_joint_names:
                    raise ValueError(
                        f"Action term {term_name!r} controlled joint IDs must "
                        "correspond exactly to descriptor joint_names."
                    )

                for joint_id, joint_name in zip(
                    controlled_joint_ids,
                    descriptor_joint_names,
                    strict=True,
                ):
                    key = (term.command_type, int(joint_id))
                    if key in owners:
                        previous_name, previous_joint = owners[key]
                        raise ValueError(
                            f"Action terms {previous_name!r} and {term_name!r} "
                            f"both own {term.command_type} command for joint "
                            f"{joint_name or previous_joint!r}."
                        )
                    owners[key] = (term_name, joint_name)

        self._descriptors = tuple(descriptors)
