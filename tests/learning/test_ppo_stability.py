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
from embodichain.learning.rl.policies import ActorCritic, build_policy
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


def _policy(*, normalize: bool = False, device: str = "cpu") -> ActorCritic:
    return ActorCritic(
        obs_dim=2,
        action_dim=1,
        device=torch.device(device),
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


def test_ppo_updates_normalizers_after_using_the_rollout_distribution() -> None:
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

    evaluated = policy.evaluate_actions(rollout[:, :-1].reshape(-1))
    probability_ratio = (
        evaluated["sample_log_prob"] - rollout["sample_log_prob"][:, :-1].reshape(-1)
    ).exp()

    torch.testing.assert_close(probability_ratio, torch.ones_like(probability_ratio))
    assert int(policy.actor_obs_normalizer.count) == 0
    assert int(policy.critic_obs_normalizer.count) == 0
    assert rollout["critic_obs"].shape == (2, 3, 3)
    assert torch.isfinite(rollout["action_mean"][:, :-1]).all()
    assert torch.isfinite(rollout["action_std"][:, :-1]).all()
    losses = _algorithm(policy).update(rollout)
    assert int(policy.actor_obs_normalizer.count) == 4
    assert int(policy.critic_obs_normalizer.count) == 4
    torch.testing.assert_close(
        policy.actor_obs_normalizer.mean,
        torch.full((1, 2), 0.5),
    )
    torch.testing.assert_close(
        policy.critic_obs_normalizer.mean,
        torch.full((1, 3), 2.5),
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


@pytest.mark.parametrize(
    ("num_envs", "normalize_per_minibatch", "device"),
    [
        (1, False, "cpu"),
        (3, True, "cpu"),
        pytest.param(3, True, "cuda", marks=pytest.mark.gpu),
    ],
)
def test_ppo_normalization_handles_singleton_rollout_or_tail_batch(
    num_envs: int, normalize_per_minibatch: bool, device: str
) -> None:
    env = _NormalizationEnv()
    env.num_envs = num_envs
    env.device = torch.device(device)
    policy = _policy(device=device)
    buffer = RolloutBuffer(
        num_envs=num_envs,
        rollout_len=1,
        obs_dim=2,
        action_dim=1,
        device=env.device,
        critic_obs_dim=3,
    )
    rollout = SyncCollector(env, policy, env.device).collect(
        1, rollout=buffer.start_rollout()
    )
    algorithm = _algorithm(
        policy,
        device=device,
        batch_size=2,
        normalize_advantage_per_mini_batch=normalize_per_minibatch,
    )

    losses = algorithm.update(rollout)

    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
    # Both singleton rollouts and an uneven 2+1 partition consume every sample.
    expected_steps = (num_envs + 1) // 2
    assert int(algorithm.optimizer.state[policy.log_std]["step"]) == expected_steps


@pytest.mark.parametrize("normalize_per_minibatch", [False, True])
def test_ppo_normalization_preserves_sample_std_for_larger_batches(
    normalize_per_minibatch: bool,
) -> None:
    device = torch.device("cpu")
    policy = ActorCritic(
        4, 1, device, nn.Linear(4, 1, bias=False), nn.Linear(4, 1, bias=False)
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    rollout = RolloutBuffer(4, 1, 4, 1, device).start_rollout()
    for value in rollout.values():
        value.zero_()
    rollout["obs"][:, 0] = torch.eye(4)
    rollout["action"][:, 0] = 1.0
    rollout["reward"][:, 0] = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    rollout["done"][:, 0] = True
    rollout["terminated"][:, 0] = True
    with torch.no_grad():
        rollout["sample_log_prob"][:, 0] = policy.evaluate_actions(rollout[:, 0])[
            "sample_log_prob"
        ]
    algorithm = _algorithm(
        policy,
        optimizer=OptimizerCfg(name="sgd", learning_rate=1.0),
        batch_size=4,
        vf_coef=0.0,
        max_grad_norm=100.0,
        normalize_advantage_per_mini_batch=normalize_per_minibatch,
    )

    algorithm.update(rollout)

    # Rewards are advantages here; their sample variance is 20 / (4 - 1).
    expected = torch.tensor([-3.0, -1.0, 1.0, 3.0]) / (20.0 / 3.0) ** 0.5
    # Unit actions/std, zero means and one-hot inputs give grad_i = -adv_i / 4.
    # One SGD step at lr=1 therefore exposes every normalized advantage.
    torch.testing.assert_close(4 * policy.actor.weight.squeeze(0), expected)


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


@pytest.mark.parametrize("clip", [-0.2, 0.0, 0.2, 1.0, 2.0])
def test_clipped_actor_loss_matches_ratio_objective_and_gradients(clip: float) -> None:
    algorithm = _algorithm(_policy(), clip_coef=clip)
    generator = torch.Generator().manual_seed(20261026)
    log_ratio = (torch.randn(4096, generator=generator) * 2).requires_grad_()
    reference_log_ratio = log_ratio.detach().clone().requires_grad_()
    advantages = torch.randn(4096, generator=generator)
    # Independent reference is the original probability-ratio definition.
    # Include the existing zero/wide clip settings and negative-clip behavior.
    ratio = reference_log_ratio.exp()
    reference = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages
    ).mean()
    actual = algorithm._actor_loss(log_ratio, advantages)
    actual.backward()
    reference.backward()
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        log_ratio.grad, reference_log_ratio.grad, atol=1e-5, rtol=1e-4
    )


@pytest.mark.parametrize("advantage,expected", [(1.0, -1.2), (0.0, 0.0)])
def test_clipped_actor_loss_does_not_backpropagate_inactive_exp_overflow(
    advantage: float, expected: float
) -> None:
    algorithm = _algorithm(_policy(), clip_coef=0.2)
    # exp(100) overflows FP32, although a positive advantage already clips
    # this branch to ratio 1.2. Computing exp first produced a NaN gradient.
    log_ratio = torch.tensor([100.0], requires_grad=True)
    loss = algorithm._actor_loss(log_ratio, torch.tensor([advantage]))
    loss.backward()
    assert float(loss) == pytest.approx(expected)
    torch.testing.assert_close(log_ratio.grad, torch.zeros_like(log_ratio))


def test_unbounded_negative_advantage_still_fails_nonfinite_gradient_guard() -> None:
    policy = _policy()
    algorithm = _algorithm(policy, clip_coef=0.2)
    parameter = next(policy.actor.parameters())
    with torch.no_grad():
        parameter.fill_(100.0 / parameter.numel())
    loss = algorithm._actor_loss(parameter.sum().reshape(1), torch.tensor([-1.0]))
    # Negative advantages retain the genuinely unbounded upper-ratio branch;
    # a global log-ratio clamp would incorrectly make this objective finite.
    assert torch.isinf(loss)
    loss.backward()
    with pytest.raises(FloatingPointError, match="optimizer step skipped"):
        algorithm._clip_gradients()


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


def test_timeout_bootstraps_terminal_state_instead_of_current_or_reset_state() -> None:
    from embodichain.learning.rl.algo import compute_gae
    from embodichain.learning.rl.buffer import RolloutBuffer
    from embodichain.learning.rl.collector import SyncCollector
    from embodichain.learning.rl.policies import ActorCritic

    class TimeoutEnv:
        num_envs = 1
        device = torch.device("cpu")

        def reset(self):
            return {"state": torch.tensor([[2.0]])}, {}

        def step(self, action):
            return (
                {"state": torch.tensor([[99.0]])},
                torch.tensor([1.0]),
                torch.tensor([False]),
                torch.tensor([True]),
                {
                    "terminal_observation": {"state": torch.tensor([[7.0]])},
                    "terminal_env_ids": torch.tensor([0]),
                    "terminal_observation_mask": torch.tensor([True]),
                },
            )

    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1.0)
    policy = ActorCritic(1, 1, torch.device("cpu"), torch.nn.Linear(1, 1), critic)
    env = TimeoutEnv()
    buffer = RolloutBuffer(1, 1, 1, 1, env.device)
    rollout = SyncCollector(env, policy, env.device).collect(
        1, rollout=buffer.start_rollout()
    )
    _, returns = compute_gae(rollout, gamma=0.9, gae_lambda=1.0)
    torch.testing.assert_close(returns, torch.tensor([[7.3]]))


@pytest.mark.parametrize("tensor_dict_info", [False, True])
def test_base_env_preserves_terminal_rows_before_reused_reset_storage(
    tensor_dict_info: bool,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace
    from embodichain.lab.gym.envs import BaseEnv

    state = torch.tensor([[3.0], [7.0], [11.0]])
    observation = {"state": state}
    info = {"success": torch.tensor([True, False, False])}
    if tensor_dict_info:
        info = TensorDict(info, batch_size=[3])

    def reset(**kwargs):
        state[kwargs["options"]["reset_ids"]] = 99.0
        return observation, {}

    env = SimpleNamespace(
        _profiler=SimpleNamespace(section=lambda *args, **kwargs: nullcontext()),
        sim=SimpleNamespace(render_frame=lambda: nullcontext()),
        _preprocess_action=lambda **kwargs: kwargs["action"],
        _step_action=lambda **kwargs: kwargs["action"],
        _postprocess_action=lambda **kwargs: kwargs["action"],
        _advance_physics=lambda: None,
        _update_sim_state=lambda **kwargs: None,
        _update_physical_objective=lambda: None,
        _capture_observation=lambda **kwargs: SimpleNamespace(policy=observation),
        get_obs=lambda **kwargs: observation,
        get_info=lambda **kwargs: info,
        get_reward=lambda **kwargs: torch.ones(3),
        _extend_reward=lambda **kwargs: kwargs["rewards"],
        check_truncated=lambda **kwargs: torch.tensor([False, True, False]),
        _hook_after_sim_step=lambda **kwargs: None,
        _elapsed_steps=torch.zeros(3, dtype=torch.long),
        max_episode_steps=10,
        cfg=SimpleNamespace(ignore_terminations=False),
        num_envs=3,
        device=torch.device("cpu"),
        reset=reset,
    )
    env._get_termination_flags = lambda obs, info: BaseEnv._get_termination_flags(
        env, obs, info
    )
    env._seal_step_outcomes = lambda *args: BaseEnv._seal_step_outcomes(env, *args)
    env._after_observation = lambda *args: BaseEnv._after_observation(env, *args)
    _, _, _, _, info = BaseEnv.step(env, torch.zeros(3, 1))
    torch.testing.assert_close(
        info["terminal_observation"]["state"], torch.tensor([[3.0], [7.0]])
    )
    torch.testing.assert_close(state, torch.tensor([[99.0], [99.0], [11.0]]))
    torch.testing.assert_close(
        info["terminal_observation_mask"], torch.tensor([False, True, False])
    )
