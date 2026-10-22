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

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock

import gymnasium as gym
import pytest
import torch

from embodichain.learning.rl.evaluation import evaluate_episodes
from embodichain_tasks.manipulation.dexterous import _runtime


class _Policy(torch.nn.Module):
    obs_dim = 1
    action_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_action(self, data: Any, deterministic: bool = False) -> Any:
        data["action"] = torch.zeros(data.batch_size[0], 1)
        return data


@pytest.fixture
def env_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], _runtime.DexSimManipulationEnv]:
    """Exercise the real step/reset bridge with deterministic task events."""
    monkeypatch.setattr(_runtime.wp, "to_torch", lambda x: x)
    monkeypatch.setattr(_runtime.wp, "from_torch", lambda x: x)

    def make(task_id: str):
        env = _runtime.DexSimManipulationEnv.__new__(_runtime.DexSimManipulationEnv)
        env.task_id = task_id
        env.device = torch.device("cpu")
        env.num_envs = 2
        env.diagnostics = False
        env.observation_dim = env.critic_dim = 1
        env.single_observation_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(-1.0, 1.0, (1,))}
        )
        env.single_action_space = gym.spaces.Box(-1.0, 1.0, (1,))
        env.action_format = "tensor"
        env.elapsed_steps = torch.zeros(2, dtype=torch.long)
        env._episode_return = torch.zeros(2)
        env._episode_success = torch.zeros(2, dtype=torch.bool)
        env._last_observation = None
        env.ctrl = torch.zeros(2, 1)
        env.qmap = env.vmap = torch.tensor([0])
        env.data = SimpleNamespace(qpos=torch.zeros(2, 1), qvel=torch.zeros(2, 1))
        env.dexuni_adapter = Mock()
        env.backend = Mock()
        env.mujoco_solver = Mock()
        env.advance_physics = Mock()
        env._state = lambda: None

        def task_reset(ids):
            values = torch.zeros(len(ids), 1)
            if task_id in ("M1", "M2"):
                return values, values, values
            return dict(qpos=values, qvel=values, ctrl=values)

        def observe(state, env_ids=None):
            return torch.zeros(2 if env_ids is None else len(env_ids), 1)

        def step(state, actions):
            # Environment 0 succeeds before termination; environment 1 never does.
            success = (env.elapsed_steps == 0) & torch.tensor([True, False])
            done = env.elapsed_steps >= torch.tensor([2, 3])
            metrics = {"rotation": env.elapsed_steps.float()}
            if task_id != "M5":
                metrics["success"] = success.float()
            result = (torch.zeros(2, 1), torch.ones(2), done)
            if task_id in ("M1", "M2"):
                return (*result, metrics)
            return (*result, torch.zeros(2, dtype=torch.bool), metrics)

        env.task = SimpleNamespace(
            reset=task_reset,
            observe=observe,
            action_targets=lambda x: x,
            control_targets=lambda x: x,
            step=step,
            after_step=step,
        )
        return env

    return make


@pytest.mark.parametrize("task_id", ["M1", "M2", "M3", "M4"])
def test_success_survives_until_autoreset_and_reaches_evaluation(
    env_factory: Callable, task_id: str
) -> None:
    """An early success counts at episode completion even if its pulse has ended."""
    env = env_factory(task_id)
    result = evaluate_episodes(policy=_Policy(), env=env, num_episodes=2, device="cpu")
    assert result["eval/success_rate"] == 0.5
    assert result["eval/metrics/success"] == 0.0
    assert result["eval/avg_length"] == 3.5


@pytest.mark.parametrize("task_id", ["M1", "M3"])
def test_partial_reset_clears_only_selected_success_and_keeps_terminal_info(
    env_factory: Callable, task_id: str
) -> None:
    env = env_factory(task_id)
    env._episode_success[:] = True
    env.reset(options={"reset_ids": torch.tensor([0])})
    assert env._episode_success.tolist() == [False, True]
    env.elapsed_steps[:] = torch.tensor([2, 0])
    _, _, done, _, info = env.step(torch.zeros(2, 1))
    assert done.tolist() == [True, False]
    assert info["success"].tolist() == [False, True]
    env.reset()
    assert not env._episode_success.any()
    assert info["success"].tolist() == [False, True]


def test_continuous_rotation_has_metrics_without_invented_binary_success(
    env_factory: Callable,
) -> None:
    env = env_factory("M5")
    result = evaluate_episodes(policy=_Policy(), env=env, num_episodes=2, device="cpu")
    assert math.isnan(result["eval/success_rate"])
    assert math.isfinite(result["eval/metrics/rotation"])
