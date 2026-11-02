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

"""Pathwise derivatives, bootstrap boundaries and actor/critic isolation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from gymnasium.spaces import Box

from embodichain.learning.rl.algo import (
    APGActorCritic,
    APGActorCriticCfg,
    build_algo,
    differentiable_lambda_returns,
)
from embodichain.learning.rl.collector import DifferentiableCollector
from embodichain.learning.rl.differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from embodichain.learning.rl.policies import ActorCritic
from embodichain.learning.rl.utils import OptimizerCfg


class LinearEnv:
    num_envs = 1
    device = torch.device("cpu")
    single_observation_space = Box(-1000, 1000, (1,))
    single_action_space = Box(-1000, 1000, (1,))

    def __init__(
        self,
        boundary: str = "none",
        grouped: bool = False,
        soft: bool = True,
        terminal_snapshot: bool = True,
    ):
        self.boundary, self.grouped, self.soft = boundary, grouped, soft
        self.terminal_snapshot = terminal_snapshot
        self.reset()

    def obs(self):
        return (
            TensorDict({"actor": self.state, "critic": self.state * 2}, [1])
            if self.grouped
            else self.state
        )

    def reset(self, *, seed=None):
        self.state = torch.ones(1, 1, dtype=torch.float64)
        self.steps = 0
        return self.obs(), {}

    def step(self, action):
        self.state = self.state + action
        self.steps += 1
        reward = -self.state[:, 0].square()
        terminated = torch.tensor([self.boundary == "terminated" and self.steps == 1])
        truncated = torch.tensor([self.boundary == "truncated" and self.steps == 1])
        info = {
            "terminal_observation": self.obs(),
            "continuation": (
                self.state[:, 0].sigmoid()
                if self.soft
                else torch.ones(1, dtype=torch.float64)
            ),
        }
        if not self.terminal_snapshot:
            info.pop("terminal_observation")
        if bool((terminated | truncated).any()):
            self.state = torch.full_like(self.state, 100)
        return self.obs(), reward, terminated, truncated, info

    def detach_state(self):
        self.state = self.state.detach()
        return self.obs()


def policy(grouped=False):
    actor, critic = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    nn.init.constant_(actor.weight, 0.2)
    nn.init.constant_(critic.weight, 2)
    return ActorCritic(
        1,
        1,
        torch.device("cpu"),
        actor,
        critic,
        critic_obs_dim=1 if grouped else None,
        actor_obs_groups=("actor",) if grouped else None,
        critic_obs_groups=("critic",) if grouped else None,
    ).double()


def algorithm(p):
    return APGActorCritic(
        APGActorCriticCfg(
            gamma=0.9,
            gae_lambda=0.7,
            normalize_return=False,
            optimizer=OptimizerCfg(name="sgd", learning_rate=0.001),
            critic_optimizer=OptimizerCfg(name="adam", learning_rate=0.001),
            critic_epochs=1,
            critic_minibatches=1,
            max_grad_norm=1e6,
            critic_max_grad_norm=1e6,
        ),
        p,
    )


def objective(weight):
    # Independent three-step scalar dynamics, reward and terminal-value formula.
    s1, s2, s3 = (1 + weight), (1 + weight) ** 2, (1 + weight) ** 3
    c1, c2, c3 = (
        torch.tensor(s, dtype=torch.float64).sigmoid().item() for s in (s1, s2, s3)
    )
    return (
        -(s1**2)
        - 0.9 * c1 * s2**2
        - 0.9**2 * c1 * c2 * s3**2
        + 0.9**3 * c1 * c2 * c3 * 2 * s3
    )


@pytest.mark.parametrize("terminal_snapshot", [False, True])
def test_pathwise_gradient_includes_soft_continuation_and_terminal_value(
    terminal_snapshot,
):
    p = policy()
    algo = algorithm(p)
    rollout = DifferentiableCollector(
        LinearEnv(terminal_snapshot=terminal_snapshot), p, torch.device("cpu")
    ).collect(3, deterministic=True)
    algo.begin_update()
    algo.accumulate_complete_rollout(rollout)
    eps = 1e-5
    expected = -(objective(0.2 + eps) - objective(0.2 - eps)) / (2 * eps)
    assert p.actor.weight.grad.item() == pytest.approx(expected, rel=1e-7)
    assert p.critic.weight.grad is None
    before = p.actor.weight.detach().clone()
    critic_before = p.critic.weight.detach().clone()
    metrics = algo.finish_update()
    assert metrics["objective"] == pytest.approx(objective(0.2), rel=1e-7)
    torch.testing.assert_close(p.actor.weight, before - 0.001 * expected)
    assert not torch.equal(p.critic.weight, critic_before)
    assert p.actor.weight.grad.item() == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize(
    "boundary,expected", [("terminated", -1.44), ("truncated", 0.72)]
)
def test_complete_rollout_stops_at_done_and_truncation_uses_pre_reset_value(
    boundary, expected
):
    p = policy()
    algo = algorithm(p)
    env = LinearEnv(boundary=boundary, soft=False)
    rollout = DifferentiableCollector(env, p, env.device).collect(3, deterministic=True)
    assert rollout.transitions[0].terminal_critic_observation.item() == pytest.approx(
        1.2
    )
    assert rollout.transitions[0].next_observation.item() == 100
    algo.begin_update()
    algo.accumulate_complete_rollout(rollout)
    metrics = algo.finish_update()
    assert metrics["objective"] == pytest.approx(expected, abs=1e-7)
    assert metrics["critic_weight_mean"] == pytest.approx(1 / 3)


def test_lambda_targets_distinguish_termination_truncation_and_rollout_end():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    done = torch.tensor([[False], [True], [False]])
    continuation = torch.ones_like(values)
    result = differentiable_lambda_returns(
        rewards, values, continuation, done, gamma=0.5, gae_lambda=0.5
    )
    torch.testing.assert_close(result, torch.tensor([[6.5], [12.0], [18.0]]))
    continuation[1] = 0
    result = differentiable_lambda_returns(
        rewards, values, continuation, done, gamma=0.5, gae_lambda=0.5
    )
    torch.testing.assert_close(result, torch.tensor([[4.0], [2.0], [18.0]]))


def test_grouped_observations_and_detach_preserve_state_values():
    env, p = LinearEnv(grouped=True), policy(grouped=True)
    collector = DifferentiableCollector(env, p, env.device)
    rollout = collector.collect(2, deterministic=True)
    assert rollout.transitions[0].observation.item() == 1
    assert rollout.transitions[0].critic_observation.item() == 2
    assert rollout.transitions[-1].next_critic_observation.item() == pytest.approx(2.88)
    before = env.state.clone()
    collector.detach_state()
    torch.testing.assert_close(env.state, before)
    assert env.state.grad_fn is None
    assert env.steps == 2
    algorithm(p).update(rollout)


def test_checkpoint_restores_both_optimizers_and_next_update():
    p = policy()
    algo = algorithm(p)
    collector = DifferentiableCollector(LinearEnv(), p, torch.device("cpu"))
    algo.update(collector.collect(3, deterministic=True))
    saved_policy, saved_algo = deepcopy(p.state_dict()), deepcopy(algo.state_dict())
    p2, algo2 = policy(), None
    p2.load_state_dict(saved_policy)
    algo2 = algorithm(p2)
    algo2.load_state_dict(saved_algo)
    for current_policy, current_algo in ((p, algo), (p2, algo2)):
        torch.manual_seed(4)
        current_algo.update(
            DifferentiableCollector(
                LinearEnv(), current_policy, torch.device("cpu")
            ).collect(3, deterministic=True)
        )
    for x, y in zip(p.parameters(), p2.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert algo2.critic_optimizer.state_dict()["state"]


def test_public_single_update_records_counters_and_registered_algorithm():
    p = policy()
    algo = build_algo(
        "apg_actor_critic",
        {"critic_epochs": 1, "critic_minibatches": 1},
        p,
        torch.device("cpu"),
    )
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(segment_length=3, deterministic_actions=True),
        LinearEnv(soft=False),
        p,
        algo,
    )
    metrics = trainer.update_once()
    assert trainer.global_step == 3
    assert trainer.num_updates == len(trainer.train_history) == 1
    assert metrics["critic_updates"] == 1


def test_validity_masks_stop_reward_bootstrap_and_critic_weights():
    p = policy()
    algo = algorithm(p)
    rollout = DifferentiableCollector(
        LinearEnv(soft=False), p, torch.device("cpu")
    ).collect(3, deterministic=True)
    steps = list(rollout.transitions)
    steps[0] = replace(steps[0], next_valid=torch.zeros(1))
    steps[1] = replace(steps[1], valid=torch.zeros(1), next_valid=torch.zeros(1))
    steps[2] = replace(steps[2], valid=torch.zeros(1), next_valid=torch.zeros(1))
    algo.begin_update()
    algo.accumulate_complete_rollout(replace(rollout, transitions=tuple(steps)))
    metrics = algo.finish_update()
    assert metrics["objective"] == 0
    assert metrics["critic_target_mean"] == 0
    assert metrics["critic_weight_mean"] == pytest.approx(1 / 3)
    assert p.actor.weight.grad.item() == 0


def test_truncation_without_terminal_observation_is_rejected():
    p = policy()
    algo = algorithm(p)
    rollout = DifferentiableCollector(
        LinearEnv(boundary="truncated"), p, torch.device("cpu")
    ).collect(1, deterministic=True)
    rollout = replace(
        rollout,
        transitions=(
            replace(rollout.transitions[0], terminal_critic_observation=None),
        ),
    )
    with pytest.raises(ValueError, match="pre-reset"):
        algo.update(rollout)


def test_trainer_checkpoint_restores_actor_critic_and_update_counters(tmp_path):
    p = policy()
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(segment_length=3, deterministic_actions=True),
        LinearEnv(),
        p,
        algorithm(p),
    )
    trainer.update_once()
    checkpoint = trainer.save_checkpoint(tmp_path / "actor_critic.pt")
    p2 = policy()
    restored = DifferentiableTrainer(trainer.cfg, LinearEnv(), p2, algorithm(p2))
    restored.load_checkpoint(checkpoint)
    assert restored.num_updates == 1 and restored.global_step == 3
    assert restored.algorithm.critic_optimizer.state_dict()["state"]
    for x, y in zip(p.parameters(), p2.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    metrics = restored.update_once()
    assert restored.num_updates == 2
    assert metrics["skipped_update"] == 0


def test_rollout_context_freezes_critic_statistics_and_restores_training_mode():
    class CountingCritic(nn.Linear):
        def __init__(self):
            super().__init__(1, 1, bias=False)
            self.register_buffer("calls_in_training", torch.zeros(()))

        def forward(self, value):
            if self.training:
                self.calls_in_training.add_(1)
            return super().forward(value)

    p = policy()
    p.critic = CountingCritic().double()
    algo = algorithm(p)
    collector = DifferentiableCollector(LinearEnv(), p, torch.device("cpu"))
    with algo.rollout_context():
        rollout = collector.collect(3, deterministic=True)
    assert p.critic.training
    assert p.critic.calls_in_training == 0
    algo.begin_update()
    algo.accumulate_complete_rollout(rollout)
    assert p.critic.calls_in_training == 0
    assert p.critic.weight.grad is None
    algo.finish_update()
    assert p.critic.calls_in_training == 1


def test_critic_normalization_refreshes_targets_after_actor_backward():
    p = policy()
    # A known value transform makes old/new-stat targets distinguishable.
    p.register_buffer("value_scale", torch.ones((), dtype=torch.float64))
    original_value = p.get_value
    observed = []

    def get_value(fields):
        output = original_value(fields)
        output["value"] = output["value"] * p.value_scale
        return output

    def update_normalization(inputs):
        assert not inputs.requires_grad
        assert p.actor.weight.grad is not None
        assert p.critic.weight.grad is None
        observed.append(inputs.clone())
        p.value_scale.fill_(2)

    p.get_value = get_value
    p.update_critic_normalization = update_normalization
    algo = algorithm(p)
    algo.cfg.update_critic_normalization = True
    rollout = DifferentiableCollector(LinearEnv(soft=False), p, p.device).collect(
        3, deterministic=True
    )
    algo.begin_update()
    algo.accumulate_complete_rollout(rollout)
    actor_grad = p.actor.weight.grad.clone()
    states = torch.tensor([1.2, 1.44, 1.728], dtype=torch.float64)
    target2 = -states[2] ** 2 + 0.9 * 4 * states[2]
    target1 = -states[1] ** 2 + 0.9 * (0.3 * 4 * states[1] + 0.7 * target2)
    target0 = -states[0] ** 2 + 0.9 * (0.3 * 4 * states[0] + 0.7 * target1)
    metrics = algo.finish_update()
    assert metrics["critic_target_mean"] == pytest.approx(
        float((target0 + target1 + target2) / 3)
    )
    torch.testing.assert_close(
        observed[0].flatten(), torch.cat((torch.ones(1), states))
    )
    torch.testing.assert_close(p.actor.weight.grad, actor_grad)


@pytest.mark.parametrize(
    "boundary,complete,expected",
    [
        ("terminated", True, [1.0]),
        ("truncated", True, [1.0, 1.2]),
        ("reference_end", True, [1.0]),
        ("truncated", False, [1.0, 100.0, 120.0, 1.2, 144.0]),
    ],
)
def test_critic_statistics_exclude_invalid_states_and_include_bootstrap(
    boundary, complete, expected
):
    p = policy()
    samples = []
    p.update_critic_normalization = lambda inputs: samples.append(inputs.clone())
    algo = algorithm(p)
    algo.cfg.update_critic_normalization = True
    rollout = DifferentiableCollector(
        LinearEnv(boundary=boundary, soft=False), p, p.device
    ).collect(3, deterministic=True)
    if boundary == "reference_end":
        steps = tuple(
            replace(
                step, valid=torch.tensor([i == 0]), next_valid=torch.tensor([False])
            )
            for i, step in enumerate(rollout.transitions)
        )
        rollout = replace(rollout, transitions=steps)
    algo.begin_update()
    if complete:
        algo.accumulate_complete_rollout(rollout)
    else:
        algo.accumulate_segment(rollout)
    algo.finish_update()
    torch.testing.assert_close(
        samples[0].flatten(), torch.tensor(expected, dtype=torch.float64)
    )
