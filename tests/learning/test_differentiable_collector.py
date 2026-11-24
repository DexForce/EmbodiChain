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

from differentiable_env_support import LinearDifferentiableEnv


import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from embodichain.learning.rl.collector import DifferentiableCollector
from embodichain.learning.rl.policies import ActorOnly


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
    env = LinearDifferentiableEnv()
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
    env = LinearDifferentiableEnv()
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
    env = LinearDifferentiableEnv()
    policy = _make_policy(env.device)
    collector = DifferentiableCollector(env=env, policy=policy, device=env.device)

    with pytest.raises(ValueError, match="num_steps must be positive"):
        collector.collect(num_steps=0)


@pytest.mark.parametrize("training", [True, False])
def test_collector_preserves_caller_selected_policy_mode(training: bool) -> None:
    env = LinearDifferentiableEnv()
    policy = _make_policy(env.device)
    policy.train(training)
    collector = DifferentiableCollector(env=env, policy=policy, device=env.device)

    collector.collect(num_steps=1, deterministic=True)

    assert policy.training is training


def test_observation_snapshots_survive_reused_environment_buffers() -> None:
    class ReusedObservationEnv(LinearDifferentiableEnv):
        def __init__(self):
            super().__init__()
            self.observation_buffer = torch.empty_like(self._state)

        def observation(self):
            self.observation_buffer.copy_(self._state)
            return TensorDict({"state": self.observation_buffer}, [self.num_envs])

        def reset(self, **kwargs):
            _, info = super().reset(**kwargs)
            return self.observation(), info

        def step(self, action):
            _, reward, terminated, truncated, info = super().step(action)
            return self.observation(), reward, terminated, truncated, info

    env = ReusedObservationEnv()
    policy = _make_policy(env.device)
    collector = DifferentiableCollector(env, policy, env.device)
    rollout = collector.collect(2, deterministic=True)
    collector.collect(1, deterministic=True)

    # For x[t + 1] = x[t] + 0.25 * x[t], each owned snapshot has a
    # distinct value even after the environment overwrites its single leaf.
    torch.testing.assert_close(rollout.initial_observation, torch.ones(2, 1))
    for index, step in enumerate(rollout.transitions):
        torch.testing.assert_close(
            step.next_observation, torch.full((2, 1), 1.25 ** (index + 1))
        )
    rollout.rewards.sum().backward()
    expected_gradient = -env.num_envs * (2 * 1.25 + 4 * 1.25**3)
    torch.testing.assert_close(
        policy.actor.weight.grad, torch.full((1, 1), expected_gradient)
    )
