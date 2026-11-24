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
from embodichain.learning.rl.policies import ActorCritic, ActorOnly


def _make_policy(
    device: torch.device,
    policy_type: type[ActorOnly] | type[ActorCritic] = ActorOnly,
) -> ActorOnly | ActorCritic:
    actor = nn.Linear(1, 1, bias=False, device=device)
    nn.init.constant_(actor.weight, 0.25)
    kwargs = (
        {"critic": nn.Linear(1, 1, bias=False, device=device)}
        if issubclass(policy_type, ActorCritic)
        else {}
    )
    return policy_type(
        obs_dim=1,
        action_dim=1,
        device=device,
        actor=actor,
        **kwargs,
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


@pytest.mark.parametrize("policy_type", [ActorOnly, ActorCritic])
@pytest.mark.parametrize("deterministic", [False, True])
def test_collection_matches_full_policy_outputs_and_pathwise_gradients(
    policy_type: type[ActorOnly] | type[ActorCritic], deterministic: bool
) -> None:
    env = LinearDifferentiableEnv()
    policy = _make_policy(env.device, policy_type)
    parameters = (policy.actor.weight, policy.log_std)
    seed, horizon, entropy_coefficient = 17, 3, 0.1
    torch.manual_seed(seed)
    rollout = DifferentiableCollector(env, policy, env.device).collect(
        horizon, deterministic=deterministic
    )
    collection_loss = -rollout.rewards.sum() - entropy_coefficient * sum(
        step.policy_output["entropy"].sum() for step in rollout.transitions
    )
    actual_gradients = torch.autograd.grad(collection_loss, parameters)

    # The public full-output sampler is the compatibility reference. It must
    # continue serving callers that also consume likelihoods and values.
    torch.manual_seed(seed)
    observation, _ = env.reset()
    expected_loss = observation.new_zeros(())
    for step in rollout.transitions:
        output = policy.get_differentiable_action(
            TensorDict({"obs": observation}, [env.num_envs]),
            deterministic=deterministic,
        )
        assert {"sample_log_prob", "value"} <= set(output.keys())
        torch.testing.assert_close(step.action, output["action"], rtol=0, atol=0)
        torch.testing.assert_close(
            step.policy_output["entropy"], output["entropy"], rtol=0, atol=0
        )
        observation, reward, *_ = env.step(output["action"])
        expected_loss = (
            expected_loss - reward.sum() - entropy_coefficient * output["entropy"].sum()
        )
    expected_gradients = torch.autograd.grad(expected_loss, parameters)
    torch.testing.assert_close(actual_gradients, expected_gradients)


@pytest.mark.parametrize("policy_type", [ActorOnly, ActorCritic])
@pytest.mark.parametrize(
    "override_target", ["get_differentiable_action", "_sample_action", "instance"]
)
def test_collection_honors_custom_sampling_overrides(
    policy_type: type[ActorOnly] | type[ActorCritic], override_target: str
) -> None:
    class CustomPolicy(policy_type):
        pass

    original = getattr(
        policy_type,
        (
            "get_differentiable_action"
            if override_target == "instance"
            else override_target
        ),
    )

    def shift_action(output):
        output["action"] = output["action"] + 1.0
        return output

    if override_target == "_sample_action":

        def override(self, fields, *, deterministic, reparameterized):
            return shift_action(
                original(
                    self,
                    fields,
                    deterministic=deterministic,
                    reparameterized=reparameterized,
                )
            )

    else:

        def override(self, fields, deterministic=False):
            return shift_action(original(self, fields, deterministic=deterministic))

    if override_target != "instance":
        setattr(CustomPolicy, override_target, override)
    env = LinearDifferentiableEnv()
    policy = _make_policy(env.device, CustomPolicy)
    if override_target == "instance":
        policy.get_differentiable_action = override.__get__(policy)
    rollout = DifferentiableCollector(env, policy, env.device).collect(
        1, deterministic=True
    )
    # Initial state is 1 and the actor weight is 0.25; the override adds 1.
    torch.testing.assert_close(rollout.transitions[0].action, torch.full((2, 1), 1.25))
    assert {"sample_log_prob", "value"} <= set(
        rollout.transitions[0].policy_output.keys()
    )


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
