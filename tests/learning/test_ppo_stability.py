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
from gymnasium import spaces
from tensordict import TensorDict

from embodichain.learning.rl.algo import PPO, PPOCfg, compute_gae
from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.models import (
    ActorCritic,
    EmpiricalNormalizer,
    build_policy,
)
from embodichain.learning.rl.train import _build_learning_policy
from embodichain.learning.rl.utils import (
    LRSchedulerCfg,
    OptimizerCfg,
    flatten_observation_groups,
)


class _NormalizationEnv:
    """Small vector environment that exposes distinct actor and critic inputs."""

    num_envs = 2
    device = torch.device("cpu")
    action_manager = None
    single_observation_space = spaces.Dict(
        {
            "policy": spaces.Box(low=-1.0, high=1.0, shape=(2,)),
            "critic": spaces.Box(low=-1.0, high=1.0, shape=(3,)),
        }
    )
    single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))

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
    assert rollout["critic_obs"].shape == (2, 3, 3)
    assert torch.isfinite(rollout["action_mean"][:, :-1]).all()
    assert torch.isfinite(rollout["action_std"][:, :-1]).all()
    losses = _algorithm(policy).update(rollout)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_observation_groups_preserve_order_and_configure_policy() -> None:
    observation = TensorDict(
        {
            "first": torch.ones(2, 2),
            "second": torch.full((2, 1), 2.0),
        },
        batch_size=[2],
    )

    flattened = flatten_observation_groups(observation, ("second", "first"))
    policy = build_policy(
        {
            "name": "actor_critic",
            "obs_groups": {"actor": ["first"], "critic": ["second"]},
        },
        2,
        1,
        torch.device("cpu"),
        actor=nn.Linear(2, 1),
        critic=nn.Linear(1, 1),
        critic_obs_space=1,
    )

    torch.testing.assert_close(flattened[:, 0], torch.full((2,), 2.0))
    torch.testing.assert_close(flattened[:, 1:], torch.ones(2, 2))
    assert policy.actor_obs_groups == ("first",)
    assert policy.critic_obs_groups == ("second",)
    with pytest.raises(KeyError, match="critic_obs"):
        policy.get_value(TensorDict({"obs": torch.zeros(2, 2)}, batch_size=[2]))


def test_lightweight_policy_builds_distinct_actor_and_critic_inputs() -> None:
    policy = _build_learning_policy(
        {
            "name": "actor_critic",
            "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
            "actor": {
                "type": "mlp",
                "network_cfg": {"hidden_sizes": [4], "activation": "tanh"},
            },
            "critic": {
                "type": "mlp",
                "network_cfg": {"hidden_sizes": [4], "activation": "tanh"},
            },
        },
        _NormalizationEnv(),
        torch.device("cpu"),
    )

    assert policy.obs_dim == 2
    assert policy.critic_obs_dim == 3
    assert policy.actor[0].in_features == 2
    assert policy.critic[0].in_features == 3


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


def test_adaptive_kl_rejects_separate_learning_rate_scheduler() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        _algorithm(
            _policy(),
            schedule="adaptive",
            desired_kl=0.01,
            lr_scheduler=LRSchedulerCfg(name="step", kwargs={"step_size": 1}),
        )


def test_clipped_value_loss_uses_larger_error() -> None:
    policy = _policy()
    algorithm = _algorithm(policy, use_clipped_value_loss=True, clip_coef=0.2)

    loss = algorithm._value_loss(
        values=torch.tensor([1.0]),
        returns=torch.tensor([1.0]),
        old_values=torch.tensor([0.0]),
    )

    assert float(loss) == pytest.approx(0.64)


def test_action_bound_loss_penalizes_policy_means_outside_unit_range() -> None:
    action_mean = torch.tensor([[-2.0], [-0.5], [0.5], [2.0]])

    loss = PPO._action_bound_loss(action_mean)

    assert float(loss) == pytest.approx(0.5)


def test_negative_action_bound_coefficient_is_rejected() -> None:
    with pytest.raises(ValueError, match="action_bound_coef must be non-negative"):
        _algorithm(_policy(), action_bound_coef=-1.0)


def test_minibatch_count_preserves_every_transition() -> None:
    algorithm = _algorithm(_policy(), num_mini_batches=3)

    batches = algorithm._minibatch_indices(10)

    assert len(batches) == 3
    torch.testing.assert_close(
        torch.cat(batches).sort().values,
        torch.arange(10),
    )


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


@pytest.mark.parametrize(
    ("terminated", "truncated", "expected_return"),
    [(False, True, 1.0 + 0.99 * 2.0), (True, False, 1.0)],
)
def test_gae_handles_episode_boundaries(
    terminated: bool,
    truncated: bool,
    expected_return: float,
) -> None:
    rollout = TensorDict(
        {
            "reward": torch.tensor([[1.0, 0.0]]),
            "value": torch.tensor([[2.0, 10.0]]),
            "done": torch.tensor([[True, False]]),
            "terminated": torch.tensor([[terminated, False]]),
            "truncated": torch.tensor([[truncated, False]]),
        },
        batch_size=[1, 2],
    )

    _, returns = compute_gae(rollout, gamma=0.99, gae_lambda=0.95)

    torch.testing.assert_close(returns, torch.tensor([[expected_return]]))
