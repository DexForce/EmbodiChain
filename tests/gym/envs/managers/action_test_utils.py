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

"""Shared test doubles for action-manager tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import torch

from embodichain.lab.gym.envs.managers.action_manager import ActionManager, ActionTerm
from embodichain.lab.gym.envs.managers.action_types import ActionTermDescriptor
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg

__all__ = ["FakeAction", "FakeRobot", "make_action_env", "make_cfg", "make_manager"]


class FakeRobot:
    """Robot double with stable parts, limits, and command spies."""

    def __init__(
        self,
        num_envs: int,
        joint_names: tuple[str, ...],
        joint_ids_by_part: dict[str, tuple[int, ...]],
    ) -> None:
        self.joint_names = list(joint_names)
        self._joint_ids_by_part = joint_ids_by_part
        self.body_data = SimpleNamespace(
            qpos_limits=torch.tensor([[[-1.0, 1.0]] * len(joint_names)]),
            qvel_limits=torch.ones(1, len(joint_names)),
            qf_limits=torch.ones(1, len(joint_names)),
        )
        self.set_qpos = MagicMock()
        self.set_qvel = MagicMock()
        self.set_qf = MagicMock()
        self.get_qpos = MagicMock(return_value=torch.zeros(num_envs, len(joint_names)))
        self.compute_ik = MagicMock()
        self.cfg = SimpleNamespace(init_qpos=[0.0] * len(joint_names))

    def get_joint_ids(self, name: str, remove_mimic: bool = False) -> list[int]:
        """Return one configured control-part selection."""
        del remove_mimic
        return list(self._joint_ids_by_part[name])


def make_action_env(
    *,
    num_envs: int = 2,
    joint_names: tuple[str, ...] = ("joint_0", "joint_1", "joint_2"),
    parts: dict[str, tuple[int, ...]] | None = None,
) -> SimpleNamespace:
    """Build the environment surface needed by action tests."""
    parts = {"arm": tuple(range(len(joint_names)))} if parts is None else parts
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        robot=FakeRobot(num_envs, joint_names, parts),
        active_joint_ids=list(range(len(joint_names))),
    )


def make_cfg(func: type[ActionTerm], **params: object) -> ActionTermCfg:
    """Build one action-term config."""
    return ActionTermCfg(func=func, params=dict(params))


class FakeAction(ActionTerm):
    """Minimal new-style term used to exercise manager orchestration."""

    def __init__(self, cfg: ActionTermCfg, env: SimpleNamespace) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        self._action_dim = int(params["action_dim"])
        self._joint_ids = tuple(int(value) for value in params["joint_ids"])
        self._joint_names = tuple(
            params.get(
                "joint_names",
                tuple(env.robot.joint_names[index] for index in self._joint_ids),
            )
        )
        self._command_type = str(params.get("command_type", "qpos"))
        self._space = gym.spaces.Box(
            low=float(params["low"]),
            high=float(params["high"]),
            shape=(self._action_dim,),
            dtype=np.float32,
        )
        self._raw = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed = torch.zeros_like(self._raw)
        self.apply_count = 0

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._space

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    @property
    def command_type(self) -> str:
        return self._command_type

    @property
    def controlled_joint_ids(self) -> tuple[int, ...]:
        return self._joint_ids

    @property
    def descriptor(self) -> ActionTermDescriptor:
        return ActionTermDescriptor(
            representation="fake",
            action_dim=self.action_dim,
            feature_names=tuple(f"action_{index}" for index in range(self.action_dim)),
            units=("unitless",) * self.action_dim,
            normalization=None,
            joint_names=self._joint_names,
            metadata={},
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw.copy_(actions)
        self._processed.copy_(actions)

    def apply_actions(self) -> None:
        self.apply_count += 1

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._processed[ids] = 0


def make_manager(
    *named_params: tuple[str, dict[str, object]],
    num_envs: int = 2,
) -> ActionManager:
    """Construct a real manager from ordered fake-term parameters."""
    max_joint_id = max(
        int(joint_id) for _, params in named_params for joint_id in params["joint_ids"]
    )
    joint_names = tuple(f"joint_{index}" for index in range(max_joint_id + 1))
    env = make_action_env(num_envs=num_envs, joint_names=joint_names)
    cfg = {
        name: ActionTermCfg(func=FakeAction, params=params)
        for name, params in named_params
    }
    return ActionManager(cfg, env)
