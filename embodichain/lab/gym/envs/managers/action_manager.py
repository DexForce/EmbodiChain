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

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
import inspect
from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import numpy as np
from prettytable import PrettyTable
from tensordict import TensorDict
import torch

from embodichain.lab.sim.types import EnvAction
from embodichain.utils import logger
from embodichain.utils.string import string_to_callable

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
    """Known policy input and physical command types.

    ``eef_pose`` is a policy input type that resolves to a ``qpos`` command;
    the physical command types accepted by the manager are ``qpos``, ``qvel``
    and ``qf``.
    """

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        self._joint_ids = self._resolve_joint_ids()
        self._clip_command = bool(cfg.params.get("clip", True))

    @property
    @abstractmethod
    def input_key(self) -> str:
        """Structured policy-action key consumed by this term.

        This property is retained for compatibility with existing action configs.
        The physical output type is exposed separately through
        :attr:`command_key`.
        """
        ...

    @property
    def command_key(self) -> str:
        """Physical robot command produced by this term.

        Returns:
            One of ``"qpos"``, ``"qvel"`` or ``"qf"``.
        """
        return self.input_key

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term (policy output dimension)."""
        ...

    @property
    def joint_ids(self) -> list[int]:
        """Robot joint indices controlled by this term."""
        return self._joint_ids

    @property
    def clip_command(self) -> bool:
        """Whether physical commands are clipped to the robot limits."""
        return self._clip_command

    @property
    def action_space(self) -> gym.spaces.Box:
        """Normalized per-term policy action space.

        The default range is ``[-1, 1]``. It can be overridden with the
        ``action_range`` config parameter, specified as ``[low, high]`` where
        either bound may be a scalar or one value per action dimension.
        """
        action_range = self.cfg.params.get("action_range", (-1.0, 1.0))
        try:
            low_value, high_value = action_range
        except (TypeError, ValueError) as error:
            raise ValueError(
                "ActionTermCfg.params['action_range'] must contain [low, high]."
            ) from error
        low = self._expand_action_bound(low_value, "low")
        high = self._expand_action_bound(high_value, "high")
        if np.any(low >= high):
            raise ValueError(
                f"Invalid policy action range for {type(self).__name__}: low must be smaller than high."
            )
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError(
                f"Policy action range for {type(self).__name__} must be finite."
            )
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @abstractmethod
    def process_action(self, action: torch.Tensor) -> EnvAction | torch.Tensor:
        """Process raw action from policy into robot control format.

        Args:
            action: Raw action tensor from policy, shape (num_envs, action_dim).

        Returns:
            Processed action tensor or typed payload ready for robot control.
        """
        ...

    def __call__(self, *args, **kwargs) -> Any:
        """Not used for ActionTerm; use process_action instead."""
        return self.process_action(*args, **kwargs)

    def _resolve_joint_ids(self) -> list[int]:
        """Resolve the static joint selection from term parameters."""
        params = self.cfg.params
        joint_ids = params.get("joint_ids")
        control_part = params.get("control_part")
        if joint_ids is not None and control_part is not None:
            raise ValueError(
                "Specify either 'joint_ids' or 'control_part' for an action term, not both."
            )
        if control_part is not None:
            joint_ids = self._env.robot.get_joint_ids(
                name=control_part, remove_mimic=True
            )
        elif joint_ids is None:
            joint_ids = self._env.active_joint_ids

        resolved = [int(joint_id) for joint_id in joint_ids]
        if len(resolved) == 0:
            raise ValueError(f"{type(self).__name__} must control at least one joint.")
        if len(set(resolved)) != len(resolved):
            raise ValueError(
                f"Duplicate joint ids are not allowed for {type(self).__name__}: {resolved}."
            )
        active_joint_ids = set(int(joint_id) for joint_id in self._env.active_joint_ids)
        invalid_joint_ids = [
            joint_id for joint_id in resolved if joint_id not in active_joint_ids
        ]
        if invalid_joint_ids:
            raise ValueError(
                f"Action term joint ids {invalid_joint_ids} are not active environment joints. "
                f"Active joint ids: {sorted(active_joint_ids)}."
            )
        return resolved

    def _expand_action_bound(self, value: Any, name: str) -> np.ndarray:
        """Expand a scalar or vector action bound to ``action_dim``."""
        bound = np.asarray(value, dtype=np.float32)
        if bound.ndim == 0:
            return np.full((self.action_dim,), float(bound), dtype=np.float32)
        bound = bound.reshape(-1)
        if bound.shape != (self.action_dim,):
            raise ValueError(
                f"Policy action {name} bound for {type(self).__name__} must have "
                f"shape ({self.action_dim},), got {bound.shape}."
            )
        return bound


class ActionManager(ManagerBase):
    """Manager for processing actions sent to the environment.

    The manager separates the flat action sampled by a policy from typed physical
    robot commands. Each pre-processing term owns a stable slice of the policy
    action, produces a ``qpos``, ``qvel`` or ``qf`` command, and records the
    joints to which that command applies. The environment can therefore expose a
    conventional flat :class:`gymnasium.spaces.Box` while still supporting mixed
    control modes on disjoint joint groups.
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
        self._raw_action: torch.Tensor | None = None
        self._processed_action: TensorDict | None = None
        self._processed_term_actions: dict[str, TensorDict] = {}
        super().__init__(cfg, env)
        self._validate_pre_terms()

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
    def single_action_space(self) -> gym.spaces.Box:
        """Flat policy action space formed by concatenating all pre terms."""
        terms = self.get_terms_by_mode("pre")
        if len(terms) == 0:
            qpos_limits = (
                self._env.robot.body_data.qpos_limits[0, self._env.active_joint_ids]
                .cpu()
                .numpy()
            )
            return gym.spaces.Box(
                low=qpos_limits[:, 0], high=qpos_limits[:, 1], dtype=np.float32
            )

        low = np.concatenate([term.action_space.low.reshape(-1) for _, term in terms])
        high = np.concatenate([term.action_space.high.reshape(-1) for _, term in terms])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def raw_action(self) -> torch.Tensor | None:
        """Most recent flat policy action, before term processing."""
        return self._raw_action

    @property
    def processed_action(self) -> TensorDict | None:
        """Most recent typed physical command payload."""
        return self._processed_action

    def convert_policy_action_to_env_action(self, action: torch.Tensor) -> EnvAction:
        """Validate a flat policy action before passing it to ``env.step``.

        .. attention::
            Action conversion now belongs to :meth:`process_action`, which is
            called by the environment. This method remains as a compatibility
            shim for external collectors and returns the validated flat action.

        Args:
            action: Raw action tensor from policy, shape (num_envs, total_action_dim).

        Returns:
            The validated flat policy action on the environment device.
        """
        return self._coerce_action_tensor(
            action, expected_dim=self.total_action_dim, label="policy action"
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

        A flat tensor is split according to term order. Structured mappings may
        instead provide values by term name or by the term's ``input_key``.

        Args:
            action: Raw action from policy (tensor or dict).
            mode: The processing mode - "pre" for preprocessing (default) or "post"
                for postprocessing. When "post", only terms with mode="post" are applied.

        Returns:
            Typed physical commands. Pre-processing returns a TensorDict keyed by
            command type for one term, or by term name for multiple terms. A
            single metadata-free ``qpos`` term retains its historical tensor
            return type; the typed payload is always available through
            :attr:`processed_action`.
        """
        terms = self.get_terms_by_mode(mode)
        if not terms:
            return action

        if mode == "post":
            return self._process_post_action(action, terms)

        term_actions = self._split_policy_action(action, terms)
        self._raw_action = torch.cat(term_actions, dim=-1)
        self._processed_term_actions = {}
        for (term_name, term), term_action in zip(terms, term_actions):
            processed = term.process_action(term_action)
            self._processed_term_actions[term_name] = self._normalize_term_command(
                term_name, term, processed
            )

        if len(terms) == 1:
            self._processed_action = next(iter(self._processed_term_actions.values()))
        else:
            self._processed_action = TensorDict(
                self._processed_term_actions,
                batch_size=[self.num_envs],
                device=self.device,
            )
        # Preserve the established single-qpos return type for task reward code,
        # while keeping the canonical typed payload in ``processed_action``.
        if len(terms) == 1 and set(self._processed_action.keys()) == {"qpos"}:
            return self._processed_action["qpos"]
        return self._processed_action

    def apply_action(self, command_keys: set[str] | None = None) -> None:
        """Apply the most recently processed commands to the robot.

        Args:
            command_keys: Optional subset of ``{"qpos", "qvel", "qf"}`` to
                apply. If omitted, all physical command types are applied.

        Raises:
            RuntimeError: If no policy action has been processed yet.
        """
        if not self._processed_term_actions:
            raise RuntimeError("No processed action is available to apply.")
        selected_keys = {"qpos", "qvel", "qf"} if command_keys is None else command_keys
        unsupported_keys = selected_keys - {"qpos", "qvel", "qf"}
        if unsupported_keys:
            raise ValueError(
                f"Unsupported physical command keys: {sorted(unsupported_keys)}."
            )
        for term_name, command in self._processed_term_actions.items():
            term = self._terms[term_name]
            for command_key in ("qpos", "qvel", "qf"):
                if command_key not in selected_keys or command_key not in command:
                    continue
                value = command[command_key]
                if command_key == "qpos":
                    self._env.robot.set_qpos(qpos=value, joint_ids=term.joint_ids)
                elif command_key == "qvel":
                    self._env.robot.set_qvel(qvel=value, joint_ids=term.joint_ids)
                else:
                    self._env.robot.set_qf(qf=value, joint_ids=term.joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Clear cached action state.

        Args:
            env_ids: Ignored because action caches are shared batched payloads.

        Returns:
            Empty logging information.
        """
        del env_ids
        self._raw_action = None
        self._processed_action = None
        self._processed_term_actions = {}
        return {}

    def _split_policy_action(
        self,
        action: EnvAction,
        terms: list[tuple[str, ActionTerm]],
    ) -> list[torch.Tensor]:
        """Split flat or structured policy actions into per-term tensors."""
        if not isinstance(action, (TensorDict, Mapping)):
            flat_action = self._coerce_action_tensor(
                action,
                expected_dim=sum(term.action_dim for _, term in terms),
                label="policy action",
            )
            return list(
                torch.split(
                    flat_action,
                    [term.action_dim for _, term in terms],
                    dim=-1,
                )
            )

        input_key_counts: dict[str, int] = {}
        for _, term in terms:
            input_key_counts[term.input_key] = (
                input_key_counts.get(term.input_key, 0) + 1
            )

        term_actions: list[torch.Tensor] = []
        for term_name, term in terms:
            if term_name in action:
                value = action[term_name]
            elif term.input_key in action and input_key_counts[term.input_key] == 1:
                value = action[term.input_key]
            else:
                raise KeyError(
                    f"Missing policy action for term '{term_name}'. Provide key "
                    f"'{term_name}'"
                    + (
                        f" or the unambiguous input key '{term.input_key}'."
                        if input_key_counts[term.input_key] == 1
                        else "."
                    )
                )
            term_actions.append(
                self._coerce_action_tensor(
                    value,
                    expected_dim=term.action_dim,
                    label=f"policy action term '{term_name}'",
                )
            )
        return term_actions

    def _coerce_action_tensor(
        self,
        action: Any,
        *,
        expected_dim: int,
        label: str,
    ) -> torch.Tensor:
        """Convert an action-like value and validate its batched shape."""
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action_tensor.ndim == 1 and self.num_envs == 1:
            action_tensor = action_tensor.unsqueeze(0)
        expected_shape = (self.num_envs, expected_dim)
        if tuple(action_tensor.shape) != expected_shape:
            raise ValueError(
                f"Invalid {label} shape: expected {expected_shape}, got "
                f"{tuple(action_tensor.shape)}."
            )
        if not bool(torch.isfinite(action_tensor).all()):
            raise ValueError(f"{label.capitalize()} contains NaN or infinite values.")
        return action_tensor

    def _normalize_term_command(
        self,
        term_name: str,
        term: ActionTerm,
        processed: EnvAction | Mapping[str, Any],
    ) -> TensorDict:
        """Normalize one term result into a typed physical command TensorDict."""
        if isinstance(processed, torch.Tensor):
            values: Mapping[str, Any] = {term.command_key: processed}
        elif isinstance(processed, (TensorDict, Mapping)):
            values = processed
        else:
            raise TypeError(
                f"Action term '{term_name}' returned unsupported type "
                f"{type(processed)!r}."
            )

        data: dict[str, torch.Tensor] = {}
        physical_keys = set()
        for key, value in values.items():
            if key in {"qpos", "qvel", "qf"}:
                command = self._coerce_action_tensor(
                    value,
                    expected_dim=len(term.joint_ids),
                    label=f"{key} command from term '{term_name}'",
                )
                data[key] = self._clip_to_robot_limits(term, key, command)
                physical_keys.add(key)
            else:
                metadata = torch.as_tensor(value, device=self.device)
                if metadata.ndim == 0:
                    if self.num_envs == 1:
                        metadata = metadata.unsqueeze(0)
                    else:
                        raise ValueError(
                            f"Scalar metadata '{key}' from action term '{term_name}' "
                            f"cannot represent {self.num_envs} environments."
                        )
                if metadata.shape[0] != self.num_envs:
                    raise ValueError(
                        f"Metadata '{key}' from action term '{term_name}' must "
                        f"have leading dimension {self.num_envs}, got "
                        f"{tuple(metadata.shape)}."
                    )
                data[key] = metadata

        if not physical_keys:
            raise ValueError(
                f"Action term '{term_name}' did not produce a qpos, qvel or qf command."
            )
        if term.command_key not in physical_keys:
            raise ValueError(
                f"Action term '{term_name}' declares command_key='{term.command_key}' "
                f"but returned {sorted(physical_keys)}."
            )
        return TensorDict(data, batch_size=[self.num_envs], device=self.device)

    def _clip_to_robot_limits(
        self,
        term: ActionTerm,
        command_key: str,
        command: torch.Tensor,
    ) -> torch.Tensor:
        """Clip a physical command to per-joint robot limits when available."""
        if not term.clip_command:
            return command

        body_data = getattr(self._env.robot, "body_data", None)
        if body_data is None:
            return command
        joint_ids = term.joint_ids
        if command_key == "qpos":
            limits = getattr(body_data, "qpos_limits", None)
            if limits is None:
                return command
            limits = limits[:, joint_ids, :].to(command.device)
            return command.clamp(limits[..., 0], limits[..., 1])
        if command_key == "qvel":
            limits = getattr(body_data, "qvel_limits", None)
        else:
            limits = getattr(body_data, "qf_limits", None)
        if limits is None:
            return command
        limits = limits[:, joint_ids].to(command.device).abs()
        return command.clamp(-limits, limits)

    def _process_post_action(
        self,
        action: EnvAction,
        terms: list[tuple[str, ActionTerm]],
    ) -> EnvAction:
        """Apply post terms without modifying cached physical commands."""
        if isinstance(action, torch.Tensor):
            if len(terms) != 1:
                raise ValueError(
                    "A flat post-process action is only valid with one post term."
                )
            return terms[0][1].process_action(action)

        if not isinstance(action, (TensorDict, Mapping)):
            raise TypeError(f"Unsupported post-process action type: {type(action)!r}.")
        result = (
            action.clone()
            if isinstance(action, TensorDict)
            else TensorDict(action, batch_size=[self.num_envs], device=self.device)
        )
        for term_name, term in terms:
            candidate_keys = (term_name, term.command_key, term.input_key)
            key = next((key for key in candidate_keys if key in result), None)
            if key is None:
                raise KeyError(
                    f"Post action term '{term_name}' could not find any of "
                    f"{candidate_keys} in the action payload."
                )
            result[key] = term.process_action(result[key])
        return result

    def _validate_pre_terms(self) -> None:
        """Validate output types, policy spaces and overlapping joint groups."""
        occupied_joints: dict[int, tuple[str, ActionTerm]] = {}
        for term_name, term in self.get_terms_by_mode("pre"):
            if term.command_key not in {"qpos", "qvel", "qf"}:
                raise ValueError(
                    f"Action term '{term_name}' has unsupported physical command "
                    f"type '{term.command_key}'."
                )
            if term.action_space.shape != (term.action_dim,):
                raise ValueError(
                    f"Action term '{term_name}' space shape {term.action_space.shape} "
                    f"does not match action_dim={term.action_dim}."
                )
            for joint_id in term.joint_ids:
                previous = occupied_joints.get(joint_id)
                if previous is None:
                    occupied_joints[joint_id] = (term_name, term)
                    continue
                previous_name, previous_term = previous
                overlap_allowed = bool(
                    term.cfg.params.get("allow_overlap", False)
                    and previous_term.cfg.params.get("allow_overlap", False)
                )
                if not overlap_allowed:
                    raise ValueError(
                        f"Action terms '{previous_name}' and '{term_name}' both "
                        f"control joint {joint_id}. Use disjoint joint_ids/control_part "
                        "selections, or explicitly set allow_overlap=true on both terms."
                    )

        self._warn_for_drive_mismatch()

    def _warn_for_drive_mismatch(self) -> None:
        """Warn when velocity/effort commands conflict with configured drives."""
        body_data = getattr(self._env.robot, "body_data", None)
        if body_data is None:
            return
        stiffness = getattr(body_data, "joint_stiffness", None)
        damping = getattr(body_data, "joint_damping", None)
        for term_name, term in self.get_terms_by_mode("pre"):
            joint_ids = term.joint_ids
            if term.command_key == "qvel" and stiffness is not None:
                if bool((stiffness[:, joint_ids] != 0).any()):
                    logger.log_warning(
                        f"Velocity action term '{term_name}' controls joints with "
                        "non-zero stiffness; the position drive may oppose the velocity target."
                    )
            elif term.command_key == "qf":
                has_stiffness = stiffness is not None and bool(
                    (stiffness[:, joint_ids] != 0).any()
                )
                has_damping = damping is not None and bool(
                    (damping[:, joint_ids] != 0).any()
                )
                if has_stiffness or has_damping:
                    logger.log_warning(
                        f"Effort action term '{term_name}' controls joints with an "
                        "active position/velocity drive; qf will be additive rather than pure torque control."
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
