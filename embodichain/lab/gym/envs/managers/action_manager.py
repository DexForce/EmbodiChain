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
# WITHOUT WARRANTIES OR CONDITIONS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
# ----------------------------------------------------------------------------

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch
from prettytable import PrettyTable
from tensordict import TensorDict

from embodichain.lab.sim.types import EnvAction
from embodichain.utils.math import matrix_from_euler, matrix_from_quat

from embodichain.utils.string import string_to_callable

from .cfg import ActionTermCfg
from .manager_base import Functor, ManagerBase

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


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
    def action_dim(self) -> int:
        """Dimension of the action term (policy output dimension)."""
        raise NotImplementedError

    def process_action(self, action: torch.Tensor) -> EnvAction:
        """Process raw action from policy into robot control format.

        Args:
            action: Raw action tensor from policy, shape (num_envs, action_dim).

        Returns:
            TensorDict with keys such as "qpos", "qvel", or "qf" ready for robot control.
        """
        raise NotImplementedError

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
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active term(s).\n"
        table = PrettyTable()
        table.title = "Active Action Terms"
        table.field_names = ["Index", "Name", "Dimension"]
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        for index, name in enumerate(self._term_names):
            term = self._terms[name]
            table.add_row([index, name, term.action_dim])
        msg += table.get_string()
        msg += "\n"
        return msg

    @property
    def active_functors(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_type(self) -> str:
        """The active action type (term name) for backward compatibility."""
        return self._term_names[0]

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions (sum of all term dimensions)."""
        return sum(t.action_dim for t in self._terms.values())

    def process_action(self, action: EnvAction) -> EnvAction:
        """Process raw action from policy into robot control format.

        Supports:
        1. Tensor input: Passed to the active (first) term.
        2. Dict/TensorDict input: Uses key matching term name, or first term if single.

        Args:
            action: Raw action from policy (tensor or dict).

        Returns:
            TensorDict action ready for robot control.
        """
        if not isinstance(action, (dict, TensorDict)):
            return self._terms[self._term_names[0]].process_action(action)

        # Dict input: find matching term
        for term_name in self._term_names:
            if term_name in action:
                return self._terms[term_name].process_action(action[term_name])
        raise ValueError(f"No valid action keys. Expected one of: {self._term_names}")

    def get_term(self, name: str) -> ActionTerm:
        """Get action term by name."""
        return self._terms[name]

    def _prepare_functors(self) -> None:
        """Parse config and create action terms.

        ActionTerm uses process_action(env, action) rather than __call__(env, env_ids, ...),
        so we skip the base class params signature check and resolve terms directly.
        """
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg. "
                    f"Received: '{type(term_cfg)}'."
                )
            # Resolve string to callable (skip base class params check for ActionTerm)
            if isinstance(term_cfg.func, str):
                term_cfg.func = string_to_callable(term_cfg.func)
            if not callable(term_cfg.func):
                raise AttributeError(
                    f"The action term '{term_name}' is not callable. "
                    f"Received: '{term_cfg.func}'"
                )
            if inspect.isclass(term_cfg.func) and not issubclass(
                term_cfg.func, ActionTerm
            ):
                raise TypeError(
                    f"Configuration for the term '{term_name}' must be a subclass of "
                    f"ActionTerm. Received: '{type(term_cfg.func)}'."
                )
            self._process_functor_cfg_at_play(term_name, term_cfg)
            self._term_names.append(term_name)
            self._terms[term_name] = term_cfg.func


# ----------------------------------------------------------------------------
# Concrete ActionTerm implementations
# ----------------------------------------------------------------------------


class DeltaQposTerm(ActionTerm):
    """Delta joint position action: current_qpos + scale * action -> qpos."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> EnvAction:
        scaled = action * self._scale
        current_qpos = self._env.robot.get_qpos()
        qpos = current_qpos + scaled
        batch_size = qpos.shape[0]
        return TensorDict({"qpos": qpos}, batch_size=[batch_size], device=self.device)


class QposTerm(ActionTerm):
    """Absolute joint position action: scale * action -> qpos."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> EnvAction:
        qpos = action * self._scale
        batch_size = qpos.shape[0]
        return TensorDict({"qpos": qpos}, batch_size=[batch_size], device=self.device)


class QposNormalizedTerm(ActionTerm):
    """Normalized action in [-1, 1] -> denormalize to joint limits -> qpos."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> EnvAction:
        scaled = action * self._scale
        qpos_limits = self._env.robot.body_data.qpos_limits[
            0, self._env.active_joint_ids
        ]
        low = qpos_limits[:, 0]
        high = qpos_limits[:, 1]
        qpos = low + (scaled + 1.0) * 0.5 * (high - low)
        batch_size = qpos.shape[0]
        return TensorDict({"qpos": qpos}, batch_size=[batch_size], device=self.device)


class EefPoseTerm(ActionTerm):
    """End-effector pose (6D or 7D) -> IK -> qpos."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)
        self._pose_dim = cfg.params.get("pose_dim", 7)  # 6 for euler, 7 for quat

    @property
    def action_dim(self) -> int:
        return self._pose_dim

    def process_action(self, action: torch.Tensor) -> EnvAction:
        scaled = action * self._scale
        current_qpos = self._env.robot.get_qpos()
        batch_size = scaled.shape[0]
        target_pose = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        )
        if scaled.shape[-1] == 6:
            target_pose[:, :3, 3] = scaled[:, :3]
            target_pose[:, :3, :3] = matrix_from_euler(scaled[:, 3:6])
        elif scaled.shape[-1] == 7:
            target_pose[:, :3, 3] = scaled[:, :3]
            target_pose[:, :3, :3] = matrix_from_quat(scaled[:, 3:7])
        else:
            raise ValueError(
                f"EEF pose action must be 6D or 7D, got {scaled.shape[-1]}D"
            )
        # Batch IK: robot.compute_ik supports (n_envs, 4, 4) pose and (n_envs, dof) seed
        ret, qpos_ik = self._env.robot.compute_ik(
            pose=target_pose,
            joint_seed=current_qpos,
        )
        # Fallback to current_qpos where IK failed
        result_qpos = torch.where(
            ret.unsqueeze(-1).expand_as(qpos_ik), qpos_ik, current_qpos
        )
        return TensorDict(
            {"qpos": result_qpos},
            batch_size=[batch_size],
            device=self.device,
        )


class QvelTerm(ActionTerm):
    """Joint velocity action: scale * action -> qvel."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> EnvAction:
        qvel = action * self._scale
        batch_size = qvel.shape[0]
        return TensorDict({"qvel": qvel}, batch_size=[batch_size], device=self.device)


class QfTerm(ActionTerm):
    """Joint force/torque action: scale * action -> qf."""

    def __init__(self, cfg: ActionTermCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        self._scale = cfg.params.get("scale", 1.0)

    @property
    def action_dim(self) -> int:
        return len(self._env.active_joint_ids)

    def process_action(self, action: torch.Tensor) -> EnvAction:
        qf = action * self._scale
        batch_size = qf.shape[0]
        return TensorDict({"qf": qf}, batch_size=[batch_size], device=self.device)


__all__ = [
    "ActionTerm",
    "ActionManager",
    "ActionTermCfg",
    "DeltaQposTerm",
    "QposTerm",
    "QposNormalizedTerm",
    "EefPoseTerm",
    "QvelTerm",
    "QfTerm",
]
