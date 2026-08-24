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

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from embodichain.learning.rl.algo import PPO, PPOCfg, compute_gae
from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.models import ActorCritic, EmpiricalNormalizer
from embodichain.learning.rl.utils import OptimizerCfg


class _NormalizationEnv:
    """Small vector environment that exposes distinct actor and critic inputs."""

    num_envs = 2
    device = torch.device("cpu")
    action_manager = None

    def __init__(self) -> None:
        self.step_count = 0

    def _observation(self) -> dict[str, torch.Tensor]:
        return {
            "policy": torch.full((self.num_envs, 2), float(self.step_count)),
            "critic": torch.full((self.num_envs, 3), float(self.step_count + 2)),
        }

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        self.step_count = 0
        return self._observation(), {}

    def step(self, action: torch.Tensor) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        self.step_count += 1
        zeros = torch.zeros(self.num_envs)
        return (
            self._observation(),
            action.sum(dim=-1),
            zeros.bool(),
            zeros.bool(),
            {},
        )


class _TimeLimitEnv:
    """Single-transition environment with a time-limit truncation."""

    num_envs = 1
    device = torch.device("cpu")
    action_manager = None

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        return {"policy": torch.tensor([[2.0]])}, {}

    def step(self, action: torch.Tensor) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        del action
        return (
            {"policy": torch.tensor([[10.0]])},
            torch.tensor([1.0]),
            torch.tensor([False]),
            torch.tensor([True]),
            {},
        )


class _ObservationValuePolicy:
    """Policy whose value equals its scalar observation."""

    obs_dim = 1
    action_dim = 1
    distribution_param_dim = None
    actor_obs_groups = ("policy",)
    critic_obs_groups = ("policy",)
    uses_separate_critic_obs = False

    def train(self) -> None:
        pass

    def get_action(self, tensordict: TensorDict) -> TensorDict:
        batch_size = tensordict.batch_size[0]
        tensordict["action"] = torch.zeros(batch_size, 1)
        tensordict["sample_log_prob"] = torch.zeros(batch_size)
        tensordict["value"] = tensordict["obs"][:, 0]
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        tensordict["value"] = tensordict["obs"][:, 0]
        return tensordict


def _policy(*, normalize: bool = False) -> ActorCritic:
    return ActorCritic(
        obs_dim=2,
        action_dim=1,
        device=torch.device("cpu"),
        actor=nn.Linear(2, 1),
        critic=nn.Linear(3, 1),
        critic_obs_dim=3,
        actor_obs_groups=("policy",),
        critic_obs_groups=("critic",),
        actor_obs_normalization=normalize,
        critic_obs_normalization=normalize,
    )


def _algorithm(policy: ActorCritic, **overrides: object) -> PPO:
    cfg = {
        "device": "cpu",
        "optimizer": OptimizerCfg(learning_rate=1e-3),
        "n_epochs": 1,
        "batch_size": 4,
        "ent_coef": 0.0,
        "vf_coef": 1.0,
        **overrides,
    }
    return PPO(PPOCfg(**cfg), policy)


def test_empirical_normalizer_tracks_and_restores_moments() -> None:
    normalizer = EmpiricalNormalizer(2)
    samples = torch.tensor([[1.0, 2.0], [3.0, 6.0]])

    normalizer.update(samples)

    torch.testing.assert_close(normalizer.mean, torch.tensor([[2.0, 4.0]]))
    torch.testing.assert_close(normalizer.variance, torch.tensor([[1.0, 4.0]]))
    restored = EmpiricalNormalizer(2)
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored(samples), normalizer(samples))
    assert int(restored.count) == 2


def test_collector_updates_normalizers_and_records_old_distribution() -> None:
    env = _NormalizationEnv()
    policy = _policy(normalize=True)
    buffer = RolloutBuffer(
        num_envs=env.num_envs,
        rollout_len=2,
        obs_dim=2,
        action_dim=1,
        device=env.device,
        critic_obs_dim=3,
        distribution_param_dim=1,
    )
    collector = SyncCollector(env, policy, env.device)

    rollout = collector.collect(2, rollout=buffer.start_rollout())

    assert int(policy.actor_obs_normalizer.count) == 4
    assert int(policy.critic_obs_normalizer.count) == 4
    assert torch.isfinite(rollout["action_mean"][:, :-1]).all()
    assert torch.isfinite(rollout["action_std"][:, :-1]).all()


def test_adaptive_kl_reduces_learning_rate() -> None:
    policy = _policy()
    algorithm = _algorithm(policy, schedule="adaptive", desired_kl=0.01)
    batch = TensorDict(
        {
            "action_mean": torch.zeros(4, 1),
            "action_std": torch.ones(4, 1),
        },
        batch_size=[4],
    )
    evaluated = TensorDict(
        {
            "action_mean": torch.ones(4, 1),
            "action_std": torch.ones(4, 1),
        },
        batch_size=[4],
    )

    kl = algorithm._update_adaptive_learning_rate(batch, evaluated)

    assert kl == pytest.approx(0.5)
    assert algorithm.current_learning_rate() == pytest.approx(1e-3 / 1.5)


def test_clipped_value_loss_uses_larger_error() -> None:
    policy = _policy()
    algorithm = _algorithm(policy, use_clipped_value_loss=True, clip_coef=0.2)

    loss = algorithm._value_loss(
        values=torch.tensor([1.0]),
        returns=torch.tensor([1.0]),
        old_values=torch.tensor([0.0]),
    )

    assert float(loss) == pytest.approx(0.64)


def test_non_finite_gradient_is_rejected_before_optimizer_step() -> None:
    policy = _policy()
    algorithm = _algorithm(policy)
    first_parameter = next(policy.actor.parameters())
    first_parameter.grad = torch.full_like(first_parameter, float("inf"))

    with pytest.raises(FloatingPointError, match="optimizer step skipped"):
        algorithm._clip_gradients()


def test_actor_and_critic_gradients_are_clipped_separately() -> None:
    policy = _policy()
    algorithm = _algorithm(policy, max_grad_norm=1.0)
    actor_parameter = next(policy.actor.parameters())
    critic_parameter = next(policy.critic.parameters())
    actor_parameter.grad = torch.full_like(actor_parameter, 2.0)
    critic_parameter.grad = torch.full_like(critic_parameter, 2.0)

    algorithm._clip_gradients()

    actor_norm = torch.linalg.vector_norm(actor_parameter.grad)
    critic_norm = torch.linalg.vector_norm(critic_parameter.grad)
    assert float(actor_norm) == pytest.approx(1.0, rel=1e-5)
    assert float(critic_norm) == pytest.approx(1.0, rel=1e-5)


def test_log_action_standard_deviation_uses_configured_range() -> None:
    policy = ActorCritic(
        obs_dim=2,
        action_dim=1,
        device=torch.device("cpu"),
        actor=nn.Linear(2, 1),
        critic=nn.Linear(2, 1),
        initial_action_std=0.5,
        action_std_range=(0.1, 1.0),
    )

    torch.testing.assert_close(policy.action_std, torch.tensor([0.5]))
    with torch.no_grad():
        policy.log_std.fill_(torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(policy.action_std, torch.tensor([1.0]))


def test_actor_critic_strictly_loads_log_std_checkpoint_state() -> None:
    policy = _policy()
    checkpoint_state = policy.state_dict()
    reloaded_policy = _policy()

    reloaded_policy.load_state_dict(checkpoint_state, strict=True)

    assert "log_std" in reloaded_policy.state_dict()
    assert "std" not in reloaded_policy.state_dict()


def test_gae_bootstraps_time_limit_with_transition_value() -> None:
    gamma = 0.99
    transition_value = 2.0
    rollout = TensorDict(
        {
            "reward": torch.tensor([[1.0, 0.0]]),
            "value": torch.tensor([[transition_value, 10.0]]),
            "done": torch.tensor([[True, False]]),
            "terminated": torch.tensor([[False, False]]),
            "truncated": torch.tensor([[True, False]]),
        },
        batch_size=[1, 2],
    )

    _, returns = compute_gae(rollout, gamma=gamma, gae_lambda=0.95)

    expected = 1.0 + gamma * transition_value
    torch.testing.assert_close(returns, torch.tensor([[expected]]))


def test_gae_does_not_bootstrap_true_termination() -> None:
    rollout = TensorDict(
        {
            "reward": torch.tensor([[1.0, 0.0]]),
            "value": torch.tensor([[2.0, 10.0]]),
            "done": torch.tensor([[True, False]]),
            "terminated": torch.tensor([[True, False]]),
            "truncated": torch.tensor([[False, False]]),
        },
        batch_size=[1, 2],
    )

    _, returns = compute_gae(rollout, gamma=0.99, gae_lambda=0.95)

    torch.testing.assert_close(returns, torch.tensor([[1.0]]))


def test_collector_time_limit_uses_value_before_environment_step() -> None:
    gamma = 0.99
    env = _TimeLimitEnv()
    buffer = RolloutBuffer(
        num_envs=env.num_envs,
        rollout_len=1,
        obs_dim=1,
        action_dim=1,
        device=env.device,
    )
    collector = SyncCollector(env, _ObservationValuePolicy(), env.device)

    rollout = collector.collect(1, rollout=buffer.start_rollout())
    _, returns = compute_gae(rollout, gamma=gamma, gae_lambda=0.95)

    expected = 1.0 + gamma * 2.0
    torch.testing.assert_close(returns, torch.tensor([[expected]]))
