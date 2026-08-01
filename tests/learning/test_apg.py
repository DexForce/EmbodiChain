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

"""Tests for analytic policy-gradient optimization."""

from __future__ import annotations

from typing import Any, Mapping

import pytest
import torch
import torch.nn as nn
from gymnasium.spaces import Box
from tensordict import TensorDict

from embodichain.learning.rl.algo import (
    APG,
    APGCfg,
    build_algo,
    get_registered_algo_names,
    segmented_discounted_return,
)
from embodichain.learning.rl.collector import (
    DifferentiableCollector,
    DifferentiableRollout,
    DifferentiableTransition,
)
from embodichain.learning.rl.models import ActorOnly


class _LinearDifferentiableEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.single_observation_space = Box(-float("inf"), float("inf"), (1,))
        self.single_action_space = Box(-float("inf"), float("inf"), (1,))
        self._state = torch.ones((1, 1))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del seed, options
        self._state = torch.ones((1, 1))
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
        done = torch.zeros(1, dtype=torch.bool)
        return self._state, reward, done, done.clone(), {}

    def detach_state(self) -> torch.Tensor:
        self._state = self._state.detach()
        return self._state


class _NonFiniteRewardEnv(_LinearDifferentiableEnv):
    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        state, reward, terminated, truncated, info = super().step(action)
        return state, reward * float("nan"), terminated, truncated, info


def _make_policy(weight: float = 0.25) -> ActorOnly:
    actor = nn.Linear(1, 1, bias=False)
    nn.init.constant_(actor.weight, weight)
    return ActorOnly(
        obs_dim=1,
        action_dim=1,
        device=torch.device("cpu"),
        actor=actor,
    )


def _objective_at(weight: float) -> float:
    env = _LinearDifferentiableEnv()
    policy = _make_policy(weight)
    rollout = DifferentiableCollector(env, policy, env.device).collect(
        num_steps=3,
        deterministic=True,
    )
    return segmented_discounted_return(rollout, gamma=0.9).mean().detach().item()


def test_apg_is_not_registered_with_standard_trainer() -> None:
    assert "apg" not in get_registered_algo_names()
    with pytest.raises(ValueError, match="differentiable rollouts"):
        build_algo(
            "apg",
            {},
            _make_policy(),
            torch.device("cpu"),
        )


def test_segmented_discounted_return_restarts_discount_after_done() -> None:
    observation = torch.zeros((1, 1))
    transitions = []
    for reward, done in ((1.0, True), (2.0, False), (4.0, False)):
        transitions.append(
            DifferentiableTransition(
                observation=observation,
                policy_output=TensorDict(
                    {"action": torch.zeros((1, 1))},
                    batch_size=[1],
                ),
                reward=torch.tensor([reward]),
                terminated=torch.tensor([done]),
                truncated=torch.tensor([False]),
                next_observation=observation,
                info={},
            )
        )
    rollout = DifferentiableRollout(observation, tuple(transitions))

    returns = segmented_discounted_return(rollout, gamma=0.5)

    assert torch.equal(returns, torch.tensor([5.0]))


def test_apg_entropy_uses_reward_discount_and_done_reset_semantics() -> None:
    policy = _make_policy()
    observation = torch.zeros((1, 1))
    transitions = []
    for entropy, done in ((1.0, True), (2.0, False), (4.0, False)):
        transitions.append(
            DifferentiableTransition(
                observation=observation,
                policy_output=TensorDict(
                    {
                        "action": torch.zeros((1, 1)),
                        "entropy": torch.tensor([entropy]) + policy.log_std[0],
                    },
                    batch_size=[1],
                ),
                reward=policy.actor.weight.sum().reshape(1) * 0.0,
                terminated=torch.tensor([done]),
                truncated=torch.tensor([False]),
                next_observation=observation,
                info={},
            )
        )
    rollout = DifferentiableRollout(observation, tuple(transitions))
    algorithm = APG(
        APGCfg(device="cpu", gamma=0.5, ent_coef=0.1),
        policy,
    )

    metrics = algorithm.update(rollout)

    assert metrics["entropy"] == pytest.approx(5.0)


def test_apg_gradient_matches_central_finite_difference() -> None:
    env = _LinearDifferentiableEnv()
    policy = _make_policy()
    rollout = DifferentiableCollector(env, policy, env.device).collect(
        num_steps=3,
        deterministic=True,
    )
    objective = segmented_discounted_return(rollout, gamma=0.9).mean()
    analytic_gradient = torch.autograd.grad(objective, policy.actor.weight)[0].item()

    epsilon = 1e-4
    finite_difference = (
        _objective_at(0.25 + epsilon) - _objective_at(0.25 - epsilon)
    ) / (2.0 * epsilon)

    assert analytic_gradient == pytest.approx(finite_difference, rel=2e-3, abs=2e-3)


def test_apg_update_improves_deterministic_objective() -> None:
    env = _LinearDifferentiableEnv()
    policy = _make_policy()
    algorithm = APG(
        APGCfg(device="cpu", learning_rate=0.05, gamma=0.9, max_grad_norm=100.0),
        policy,
    )
    collector = DifferentiableCollector(env, policy, env.device)
    initial_objective = _objective_at(policy.actor.weight.detach().item())

    metrics = algorithm.update(collector.collect(num_steps=3, deterministic=True))
    updated_objective = _objective_at(policy.actor.weight.detach().item())

    assert metrics["skipped_update"] == 0.0
    assert metrics["grad_norm"] > 0.0
    assert updated_objective > initial_objective


def test_apg_skips_nonfinite_update_without_changing_policy() -> None:
    env = _NonFiniteRewardEnv()
    policy = _make_policy()
    algorithm = APG(APGCfg(device="cpu"), policy)
    initial_weight = policy.actor.weight.detach().clone()

    metrics = algorithm.update(
        DifferentiableCollector(env, policy, env.device).collect(
            num_steps=1,
            deterministic=True,
        )
    )

    assert metrics["skipped_update"] == 1.0
    assert torch.equal(policy.actor.weight, initial_weight)
