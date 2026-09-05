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

"""Tests for the temporary Newton FK reference environment."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("newton")

from embodichain.learning.rl.algo import APG, APGCfg
from embodichain.learning.rl.algo import segmented_discounted_return
from embodichain.learning.rl.collector import DifferentiableCollector
from embodichain.learning.rl.env import DifferentiableVecEnv
from embodichain.learning.rl.experimental.newton import (
    NewtonPlanarReachEnv,
    NewtonPlanarReachEnvCfg,
)
from embodichain.learning.rl.experimental.newton.train_planar_reach import (
    NewtonPlanarReachTrainingCfg,
    _evaluate,
    train_planar_reach,
)
from embodichain.learning.rl.models import ActorOnly
from embodichain.learning.rl.utils import OptimizerCfg


def _make_env() -> NewtonPlanarReachEnv:
    return NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=2,
            device="cpu",
            max_episode_steps=8,
        )
    )


class _ModeRecordingActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 2)
        self.observed_modes: list[bool] = []

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        self.observed_modes.append(self.training)
        return self.linear(observation)


def _reward_at(env: NewtonPlanarReachEnv, action: torch.Tensor) -> float:
    env.reset(seed=7)
    _, reward, _, _, _ = env.step(action)
    return reward.sum().detach().item()


def _make_scalar_policy(env: NewtonPlanarReachEnv, weight: float) -> ActorOnly:
    actor = nn.Linear(8, 2, bias=False)
    nn.init.zeros_(actor.weight)
    with torch.no_grad():
        actor.weight[0, 0] = weight
    return ActorOnly(8, 2, env.device, actor=actor)


def _policy_objective_at(weight: float, *, seed: int = 17) -> float:
    env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=1,
            device="cpu",
            success_threshold=0.0,
            max_episode_steps=16,
        )
    )
    policy = _make_scalar_policy(env, weight)
    collector = DifferentiableCollector(env, policy, env.device)
    collector.reset(seed=seed)
    rollout = collector.collect(num_steps=3, deterministic=True)
    return segmented_discounted_return(rollout, gamma=0.95).mean().detach().item()


def _feature_policy_objective_at(
    weight: float,
    feature_index: int,
    *,
    seed: int = 17,
) -> float:
    env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=1,
            device="cpu",
            success_threshold=0.0,
            max_episode_steps=16,
        )
    )
    actor = nn.Linear(8, 2, bias=False)
    nn.init.zeros_(actor.weight)
    with torch.no_grad():
        actor.weight[0, 0] = 0.05
        actor.weight[0, feature_index] = weight
    policy = ActorOnly(8, 2, env.device, actor=actor)
    collector = DifferentiableCollector(env, policy, env.device)
    collector.reset(seed=seed)
    rollout = collector.collect(num_steps=3, deterministic=True)
    return segmented_discounted_return(rollout, gamma=0.95).mean().detach().item()


def test_newton_env_satisfies_contract_and_fk_matches_analytical_pose() -> None:
    env = _make_env()
    observation, _ = env.reset(seed=3)
    joint_q = observation[:, :2]
    expected_xy = torch.stack(
        [
            env.cfg.first_link_length * torch.sin(joint_q[:, 0])
            + env.cfg.second_link_length * torch.sin(joint_q.sum(dim=-1)),
            -env.cfg.first_link_length * torch.cos(joint_q[:, 0])
            - env.cfg.second_link_length * torch.cos(joint_q.sum(dim=-1)),
        ],
        dim=-1,
    )

    assert isinstance(env, DifferentiableVecEnv)
    assert observation.shape == (2, 8)
    assert torch.allclose(observation[:, 2:4], expected_xy, atol=1e-5)


def test_done_step_returns_next_episode_observation() -> None:
    env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=2,
            device="cpu",
            success_threshold=float("inf"),
        )
    )
    env.reset(seed=5)

    observation, _, terminated, truncated, _ = env.step(torch.ones((2, 2)))

    assert terminated.all()
    assert not truncated.any()
    assert torch.equal(observation, env.detach_state())
    assert torch.equal(observation[:, 6:8], torch.zeros((2, 2)))


def test_newton_reward_gradient_matches_finite_difference() -> None:
    env = _make_env()
    env.reset(seed=7)
    action = torch.tensor([[0.1, -0.2], [0.05, 0.15]], requires_grad=True)
    _, reward, _, _, _ = env.step(action)
    reward.sum().backward()
    analytic_gradient = action.grad[0, 0].item()

    epsilon = 1e-3
    positive = action.detach().clone()
    negative = action.detach().clone()
    positive[0, 0] += epsilon
    negative[0, 0] -= epsilon
    finite_difference = (_reward_at(env, positive) - _reward_at(env, negative)) / (
        2.0 * epsilon
    )

    assert analytic_gradient == pytest.approx(finite_difference, rel=2e-3, abs=2e-3)


def test_auto_reset_keeps_reward_gradient_target_snapshot() -> None:
    env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=2,
            device="cpu",
            max_episode_steps=1,
            success_threshold=0.0,
        )
    )
    env.reset(seed=7)
    action = torch.tensor([[0.1, -0.2], [0.05, 0.15]], requires_grad=True)
    _, reward, _, truncated, _ = env.step(action)
    reward.sum().backward()
    analytic_gradient = action.grad[0, 0].item()

    epsilon = 1e-3
    positive = action.detach().clone()
    negative = action.detach().clone()
    positive[0, 0] += epsilon
    negative[0, 0] -= epsilon
    finite_difference = (_reward_at(env, positive) - _reward_at(env, negative)) / (
        2.0 * epsilon
    )

    assert truncated.all()
    assert analytic_gradient == pytest.approx(finite_difference, rel=2e-3, abs=2e-3)


def test_detach_state_preserves_values_and_removes_graph() -> None:
    env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=2,
            device="cpu",
            success_threshold=0.0,
        )
    )
    env.reset(seed=11)
    action = torch.full((2, 2), 0.1, requires_grad=True)
    observation, _, _, _, _ = env.step(action)

    detached_observation = env.detach_state()
    next_action = torch.full((2, 2), -0.1, requires_grad=True)
    _, next_reward, _, _, _ = env.step(next_action)
    old_gradient, next_gradient = torch.autograd.grad(
        next_reward.sum(),
        (action, next_action),
        allow_unused=True,
    )

    assert torch.equal(detached_observation, observation.detach())
    assert not detached_observation.requires_grad
    assert old_gradient is None
    assert next_gradient is not None
    assert torch.isfinite(next_gradient).all()


def test_multistep_policy_gradient_matches_finite_difference() -> None:
    env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=1,
            device="cpu",
            success_threshold=0.0,
            max_episode_steps=16,
        )
    )
    policy = _make_scalar_policy(env, weight=0.05)
    collector = DifferentiableCollector(env, policy, env.device)
    collector.reset(seed=17)
    rollout = collector.collect(num_steps=3, deterministic=True)
    objective = segmented_discounted_return(rollout, gamma=0.95).mean()
    analytic_gradient = torch.autograd.grad(objective, policy.actor.weight)[0][
        0, 0
    ].item()

    epsilon = 1e-3
    finite_difference = (
        _policy_objective_at(0.05 + epsilon) - _policy_objective_at(0.05 - epsilon)
    ) / (2.0 * epsilon)

    assert analytic_gradient == pytest.approx(
        finite_difference,
        rel=5e-3,
        abs=5e-3,
    )


@pytest.mark.parametrize("feature_index", [2, 6])
def test_multistep_observation_gradient_matches_finite_difference(
    feature_index: int,
) -> None:
    env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=1,
            device="cpu",
            success_threshold=0.0,
            max_episode_steps=16,
        )
    )
    actor = nn.Linear(8, 2, bias=False)
    nn.init.zeros_(actor.weight)
    with torch.no_grad():
        actor.weight[0, 0] = 0.05
        actor.weight[0, feature_index] = 0.03
    policy = ActorOnly(8, 2, env.device, actor=actor)
    collector = DifferentiableCollector(env, policy, env.device)
    collector.reset(seed=17)
    rollout = collector.collect(num_steps=3, deterministic=True)
    objective = segmented_discounted_return(rollout, gamma=0.95).mean()
    analytic_gradient = torch.autograd.grad(objective, actor.weight)[0][
        0, feature_index
    ].item()

    epsilon = 1e-3
    finite_difference = (
        _feature_policy_objective_at(0.03 + epsilon, feature_index)
        - _feature_policy_objective_at(0.03 - epsilon, feature_index)
    ) / (2.0 * epsilon)

    assert analytic_gradient == pytest.approx(
        finite_difference,
        rel=5e-3,
        abs=5e-3,
    )


def test_newton_rollout_drives_apg_policy_update() -> None:
    env = _make_env()
    actor = nn.Linear(8, 2, bias=False)
    nn.init.zeros_(actor.weight)
    policy = ActorOnly(8, 2, env.device, actor=actor)
    algorithm = APG(
        APGCfg(
            device="cpu",
            optimizer=OptimizerCfg(learning_rate=0.01),
            max_grad_norm=100.0,
        ),
        policy,
    )
    rollout = DifferentiableCollector(env, policy, env.device).collect(
        num_steps=2,
        deterministic=True,
    )
    initial_weight = actor.weight.detach().clone()

    metrics = algorithm.update(rollout)

    assert rollout.rewards.grad_fn is not None
    assert metrics["skipped_update"] == 0.0
    assert metrics["grad_norm"] > 0.0
    assert not torch.equal(actor.weight, initial_weight)


@pytest.mark.parametrize("initial_training_mode", [True, False])
def test_evaluation_uses_eval_mode_and_restores_policy_mode(
    initial_training_mode: bool,
) -> None:
    env = _make_env()
    actor = _ModeRecordingActor()
    policy = ActorOnly(8, 2, env.device, actor=actor)
    policy.train(initial_training_mode)
    collector = DifferentiableCollector(env, policy, env.device)

    _evaluate(
        collector,
        NewtonPlanarReachTrainingCfg(
            num_envs=env.num_envs,
            eval_batches=1,
            horizon=2,
        ),
    )

    assert actor.observed_modes
    assert not any(actor.observed_modes)
    assert policy.training is initial_training_mode


@pytest.mark.slow
def test_apg_training_generalizes_to_held_out_reaches() -> None:
    result = train_planar_reach(
        NewtonPlanarReachTrainingCfg(
            device="cpu",
            seed=29,
            eval_batches=1,
            num_envs=32,
            num_updates=300,
            horizon=16,
        )
    )

    assert result["skipped_updates"] == 0
    assert result["num_updates"] == 300
    assert result["global_step"] == 300 * 16 * 32
    assert result["final_return"] > result["initial_return"]
    assert (
        result["final_mean_min_distance"] < 0.25 * result["initial_mean_min_distance"]
    )
    assert result["final_success_rate"] > result["initial_success_rate"]
