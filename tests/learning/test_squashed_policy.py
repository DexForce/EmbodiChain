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

"""Tests for bounded Gaussian policy actions."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from embodichain.learning.rl.models import ActorCritic, ActorOnly, build_policy


def _make_policy(
    policy_type: type[ActorCritic] | type[ActorOnly],
    *,
    bias: float = 0.25,
    squash_actions: bool = True,
) -> ActorCritic | ActorOnly:
    """Create a deterministic small policy for distribution tests."""
    device = torch.device("cpu")
    actor = torch.nn.Linear(2, 2)
    torch.nn.init.zeros_(actor.weight)
    torch.nn.init.constant_(actor.bias, bias)
    if policy_type is ActorCritic:
        critic = torch.nn.Linear(2, 1)
        policy: ActorCritic | ActorOnly = ActorCritic(
            obs_dim=2,
            action_dim=2,
            device=device,
            actor=actor,
            critic=critic,
            squash_actions=squash_actions,
        )
    else:
        policy = ActorOnly(
            obs_dim=2,
            action_dim=2,
            device=device,
            actor=actor,
            squash_actions=squash_actions,
        )
    with torch.no_grad():
        policy.log_std.fill_(-1.0)
    return policy


@pytest.mark.parametrize("policy_type", [ActorCritic, ActorOnly])
def test_squashed_policy_actions_stay_in_normalized_bounds(policy_type) -> None:
    """Both built-in Gaussian policies obey the manager's normalized range."""
    policy = _make_policy(policy_type, bias=5.0)
    tensordict = TensorDict(
        {"obs": torch.zeros(8, 2)},
        batch_size=[8],
    )

    result = policy(tensordict, deterministic=True)

    assert bool((result["action"] <= 1.0).all())
    assert bool((result["action"] >= -1.0).all())
    assert bool((result["action"] > 0.99).all())


@pytest.mark.parametrize("policy_type", [ActorCritic, ActorOnly])
def test_squashed_policy_evaluation_reproduces_sample_log_prob(policy_type) -> None:
    """PPO-style reevaluation uses the same tanh Jacobian correction."""
    policy = _make_policy(policy_type)
    sample = policy(
        TensorDict({"obs": torch.zeros(16, 2)}, batch_size=[16]),
        deterministic=False,
    )
    evaluation_input = TensorDict(
        {
            "obs": sample["obs"].clone(),
            "action": sample["action"].clone(),
        },
        batch_size=[16],
    )

    evaluated = policy.evaluate_actions(evaluation_input)

    torch.testing.assert_close(
        evaluated["sample_log_prob"],
        sample["sample_log_prob"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_squashed_actor_only_keeps_pathwise_gradients() -> None:
    """Tanh bounding remains compatible with differentiable RL training."""
    policy = _make_policy(ActorOnly)
    result = policy.get_differentiable_action(
        TensorDict({"obs": torch.ones(4, 2)}, batch_size=[4]),
        deterministic=True,
    )

    result["action"].sum().backward()

    assert policy.actor.weight.grad is not None
    assert bool(torch.isfinite(policy.actor.weight.grad).all())
    assert bool((policy.actor.weight.grad != 0).any())


def test_unsquashed_policy_keeps_legacy_action_behavior() -> None:
    """Existing non-simulator users remain unbounded unless explicitly enabled."""
    policy = _make_policy(ActorOnly, bias=2.0, squash_actions=False)

    result = policy(
        TensorDict({"obs": torch.zeros(1, 2)}, batch_size=[1]),
        deterministic=True,
    )

    torch.testing.assert_close(result["action"], torch.full((1, 2), 2.0))


def test_policy_factory_forwards_squash_actions_option() -> None:
    """Configuration can enable the bounded distribution through the factory."""
    actor = torch.nn.Linear(2, 2)

    policy = build_policy(
        {"name": "actor_only", "squash_actions": True},
        obs_space=2,
        action_space=2,
        device=torch.device("cpu"),
        actor=actor,
    )

    assert isinstance(policy, ActorOnly)
    assert policy.squash_actions is True
