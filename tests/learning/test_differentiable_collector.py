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

"""Tests for graph-preserving rollout collection."""

from __future__ import annotations

from typing import Any, Mapping

import pytest
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from tensordict import TensorDict

from embodichain.learning.rl.collector import DifferentiableCollector
from embodichain.learning.rl.models import ActorOnly


class _LinearDifferentiableEnv:
    """CPU environment whose state follows ``next_state = state + action``."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.single_observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1,),
            dtype=float,
        )
        self.single_action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=float,
        )
        self._state = torch.ones((num_envs, 1), device=self.device)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del seed, options
        self._state = torch.ones((self.num_envs, 1), device=self.device)
        return self._state, {}

    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        self._state = self._state + action
        reward = -self._state.square().sum(dim=-1)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(terminated)
        return self._state, reward, terminated, truncated, {}

    def detach_state(self) -> torch.Tensor:
        self._state = self._state.detach()
        return self._state


def _make_policy(device: torch.device) -> ActorOnly:
    actor = nn.Linear(1, 1, bias=False, device=device)
    nn.init.constant_(actor.weight, 0.25)
    return ActorOnly(
        obs_dim=1,
        action_dim=1,
        device=device,
        actor=actor,
    )


def test_collector_preserves_reward_gradient_to_policy() -> None:
    env = _LinearDifferentiableEnv()
    policy = _make_policy(env.device)
    collector = DifferentiableCollector(env=env, policy=policy, device=env.device)

    rollout = collector.collect(num_steps=2, deterministic=True)
    loss = -rollout.rewards.sum()
    loss.backward()

    assert rollout.num_steps == 2
    assert rollout.rewards.grad_fn is not None
    assert policy.actor.weight.grad is not None
    assert torch.isfinite(policy.actor.weight.grad).all()


def test_stochastic_action_uses_reparameterized_sample() -> None:
    env = _LinearDifferentiableEnv()
    policy = _make_policy(env.device)
    collector = DifferentiableCollector(env=env, policy=policy, device=env.device)

    rollout = collector.collect(num_steps=1)
    rollout.transitions[0].action.sum().backward()

    assert rollout.transitions[0].action.grad_fn is not None
    assert policy.actor.weight.grad is not None


def test_standard_policy_action_remains_detached() -> None:
    device = torch.device("cpu")
    policy = _make_policy(device)
    policy_input = TensorDict(
        {"obs": torch.ones((2, 1), device=device)},
        batch_size=[2],
        device=device,
    )

    policy_output = policy.get_action(policy_input)

    assert not policy_output["action"].requires_grad


def test_collector_rejects_empty_rollout() -> None:
    env = _LinearDifferentiableEnv()
    policy = _make_policy(env.device)
    collector = DifferentiableCollector(env=env, policy=policy, device=env.device)

    with pytest.raises(ValueError, match="num_steps must be positive"):
        collector.collect(num_steps=0)
