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

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from tensordict import TensorDict

from embodichain.learning.models.normalizer import EmpiricalNormalizer
from embodichain.learning.rl.algo import PPO, PPOCfg, compute_gae
from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.policies import ActorCritic, build_policy
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


class _ExtraFieldPolicy(ActorCritic):
    """Custom action evaluator requiring a transition field beyond built-ins."""

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        evaluated = ActorCritic.evaluate_actions(self, tensordict)
        evaluated["sample_log_prob"] = (
            evaluated["sample_log_prob"] + tensordict["extra"]
        )
        return evaluated


def _policy(
    *,
    normalize: bool | tuple[bool, bool] = False,
    device: str = "cpu",
    policy_type: type[ActorCritic] = ActorCritic,
) -> ActorCritic:
    actor_normalization, critic_normalization = (
        normalize if isinstance(normalize, tuple) else (normalize, normalize)
    )
    return policy_type(
        obs_dim=2,
        action_dim=1,
        device=torch.device(device),
        actor=nn.Linear(2, 1),
        critic=nn.Linear(3, 1),
        critic_obs_dim=3,
        actor_obs_groups=("policy",),
        critic_obs_groups=("critic",),
        actor_obs_normalization=actor_normalization,
        critic_obs_normalization=critic_normalization,
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


@pytest.mark.parametrize("mutated_field", ["sample_log_prob", "return"])
@pytest.mark.parametrize("declare_fields", [False, True])
def test_custom_evaluator_cannot_replace_ppo_reference_targets(
    mutated_field: str,
    declare_fields: bool,
) -> None:
    policy = _policy()
    custom = deepcopy(policy)
    env = _NormalizationEnv()
    buffer = RolloutBuffer(2, 3, 2, 1, env.device, critic_obs_dim=3)
    rollout = SyncCollector(env, policy, env.device).collect(
        3, rollout=buffer.start_rollout()
    )

    def evaluate_actions(batch):
        evaluated = ActorCritic.evaluate_actions(custom, batch)
        if mutated_field == "sample_log_prob":
            batch[mutated_field].copy_(evaluated[mutated_field].detach())
        else:
            batch[mutated_field].zero_()
        return evaluated

    custom.evaluate_actions = evaluate_actions
    if declare_fields:
        from embodichain.learning.rl.policies.policy import _TrainingInputSpec

        custom._training_input_spec = lambda: _TrainingInputSpec(
            fields=("obs", "critic_obs", "action")
        )
    expected = _algorithm(policy, n_epochs=3, batch_size=6)
    actual = _algorithm(custom, n_epochs=3, batch_size=6)
    torch.random.default_generator.manual_seed(31)
    expected_metrics = expected.update(rollout)
    torch.random.default_generator.manual_seed(31)
    actual_metrics = actual.update(rollout)
    assert actual_metrics == pytest.approx(expected_metrics)
    torch.testing.assert_close(custom.state_dict(), policy.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("algorithm_name", ["ppo", "grpo"])
def test_declared_policy_inputs_preserve_updates_and_refresh_between_rollouts(
    algorithm_name: str,
) -> None:
    from embodichain.learning.rl.algo import GRPO, GRPOCfg
    from embodichain.learning.rl.policies.policy import _TrainingInputSpec

    reference = _policy()
    declared = deepcopy(reference)
    env = _NormalizationEnv()
    rollout = SyncCollector(env, reference, env.device).collect(
        3, RolloutBuffer(2, 3, 2, 1, env.device, critic_obs_dim=3).start_rollout()
    )
    rollout["extra"] = torch.full(rollout.batch_size, 0.1)
    # A raw extra cannot replace transition_view's derived successor values.
    rollout["next_value"] = torch.full(rollout.batch_size, -100.0)
    selected_field = "extra"
    declarations = []

    def evaluate(model, batch, *, projected):
        if projected:
            assert "reward" not in batch.keys()
        evaluated = ActorCritic.evaluate_actions(model, batch)
        evaluated["sample_log_prob"] = (
            evaluated["sample_log_prob"] + batch[selected_field]
        )
        return evaluated

    reference.evaluate_actions = lambda batch: evaluate(
        reference, batch, projected=False
    )
    declared.evaluate_actions = lambda batch: evaluate(declared, batch, projected=True)

    def input_spec():
        declarations.append(selected_field)
        return _TrainingInputSpec(
            fields=("obs", "critic_obs", "action", selected_field),
            mutates_inputs=False,
        )

    declared._training_input_spec = input_spec
    if algorithm_name == "ppo":
        expected = _algorithm(reference, n_epochs=2, batch_size=2)
        actual = _algorithm(declared, n_epochs=2, batch_size=2)
    else:
        cfg = GRPOCfg(device="cpu", group_size=2, n_epochs=2, batch_size=2, kl_coef=0.0)
        expected, actual = GRPO(cfg.copy(), reference), GRPO(cfg.copy(), declared)

    for selected_field in ("extra", "next_value"):
        torch.random.default_generator.manual_seed(31)
        expected_metrics = expected.update(rollout)
        torch.random.default_generator.manual_seed(31)
        actual_metrics = actual.update(rollout)
        assert actual_metrics == expected_metrics
        torch.testing.assert_close(
            declared.state_dict(), reference.state_dict(), rtol=0, atol=0
        )
    assert declarations == ["extra", "next_value"]


@pytest.mark.parametrize(
    "policy_type,instance_override,normalization,mutating_normalizer",
    [
        (ActorCritic, False, (False, False), False),
        (ActorCritic, False, (True, False), False),
        (ActorCritic, False, (False, True), False),
        (ActorCritic, False, (True, True), False),
        (_ExtraFieldPolicy, False, (True, True), False),
        (ActorCritic, True, (True, True), False),
        (ActorCritic, False, (True, True), True),
    ],
)
def test_ppo_updates_normalizers_after_using_the_rollout_distribution(
    policy_type: type[ActorCritic],
    instance_override: bool,
    normalization: tuple[bool, bool],
    mutating_normalizer: bool,
) -> None:
    env = _NormalizationEnv()
    policy = _policy(normalize=normalization, policy_type=policy_type)
    if mutating_normalizer:

        class MutatingNormalizer(EmpiricalNormalizer):
            def update(self, value: torch.Tensor) -> None:
                super().update(value)
                value.add_(1000.0)

        policy.actor_obs_normalizer = MutatingNormalizer(policy.actor_obs_dim)
        policy.critic_obs_normalizer = MutatingNormalizer(policy.critic_obs_dim)
    if instance_override:
        policy.evaluate_actions = _ExtraFieldPolicy.evaluate_actions.__get__(policy)
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
    rollout["extra"] = torch.zeros(rollout.batch_size)

    evaluated = policy.evaluate_actions(rollout[:, :-1].reshape(-1))
    probability_ratio = (
        evaluated["sample_log_prob"] - rollout["sample_log_prob"][:, :-1].reshape(-1)
    ).exp()

    torch.testing.assert_close(probability_ratio, torch.ones_like(probability_ratio))
    for normalizer in (policy.actor_obs_normalizer, policy.critic_obs_normalizer):
        if hasattr(normalizer, "count"):
            assert int(normalizer.count) == 0
    assert rollout["critic_obs"].shape == (2, 3, 3)
    assert torch.isfinite(rollout["action_mean"][:, :-1]).all()
    assert torch.isfinite(rollout["action_std"][:, :-1]).all()
    original = rollout.clone()
    losses = _algorithm(policy).update(rollout)
    assert rollout.keys() == original.keys()
    for key in original.keys():
        torch.testing.assert_close(
            rollout[key], original[key], rtol=0, atol=0, equal_nan=True
        )
    for normalizer, enabled, expected_mean in (
        (policy.actor_obs_normalizer, normalization[0], torch.full((1, 2), 0.5)),
        (policy.critic_obs_normalizer, normalization[1], torch.full((1, 3), 2.5)),
    ):
        if enabled:
            assert int(normalizer.count) == 4
            torch.testing.assert_close(normalizer.mean, expected_mean)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


@pytest.mark.parametrize("override_style", ["subclass", "instance"])
def test_custom_normalization_update_keeps_complete_flat_observations(
    override_style: str,
) -> None:
    captured = []

    class CustomNormalization(ActorCritic):
        def update_normalization(self, batch):
            captured.append(batch.clone())

    policy = _policy(
        policy_type=CustomNormalization if override_style == "subclass" else ActorCritic
    )
    if override_style == "instance":
        policy.update_normalization = lambda batch: captured.append(batch.clone())
    env = _NormalizationEnv()
    rollout = SyncCollector(env, policy, env.device).collect(
        2, RolloutBuffer(2, 2, 2, 1, env.device, critic_obs_dim=3).start_rollout()
    )
    _algorithm(policy).update(rollout)

    assert len(captured) == 1
    assert captured[0].batch_size == torch.Size([4])
    assert set(captured[0].keys()) == {"obs", "critic_obs"}
    for name in ("obs", "critic_obs"):
        expected = torch.cat((rollout[name][0, :-1], rollout[name][1, :-1]))
        torch.testing.assert_close(captured[0][name], expected, rtol=0, atol=0)


def test_critic_only_normalization_uses_shared_actor_observations() -> None:
    from embodichain.learning.rl.algo.common import (
        _resolve_training_input_spec,
        _update_observation_normalization,
    )

    policy = ActorCritic(
        2,
        1,
        torch.device("cpu"),
        nn.Linear(2, 1),
        nn.Linear(2, 1),
        critic_obs_normalization=True,
    )
    rollout = TensorDict(
        {"obs": torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])},
        batch_size=[1, 3],
    )
    _update_observation_normalization(
        policy, rollout, _resolve_training_input_spec(policy)
    )

    assert int(policy.critic_obs_normalizer.count) == 2
    torch.testing.assert_close(
        policy.critic_obs_normalizer.mean, torch.tensor([[2.0, 3.0]])
    )
    torch.testing.assert_close(policy.critic_obs_normalizer.variance, torch.ones(1, 2))


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
    from embodichain.learning.rl.runtime import _build_learning_policy

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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "group,bad_value", [("actor", float("inf")), ("critic", float("nan"))]
)
def test_non_finite_gradient_is_rejected_before_optimizer_step(
    device: str, group: str, bad_value: float
) -> None:
    env = _NormalizationEnv()
    env.device = torch.device(device)
    policy = _policy(device=device)
    algorithm = _algorithm(policy, device=device)
    rollout = SyncCollector(env, policy, env.device).collect(
        1, RolloutBuffer(2, 1, 2, 1, env.device, critic_obs_dim=3).start_rollout()
    )
    original = deepcopy(policy.state_dict())
    parameter = next(getattr(policy, group).parameters())
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, bad_value))
    try:
        with pytest.raises(FloatingPointError, match="optimizer step skipped"):
            algorithm.update(rollout)
    finally:
        hook.remove()
    torch.testing.assert_close(policy.state_dict(), original, rtol=0, atol=0)
    assert not algorithm.optimizer.state


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_actor_and_critic_gradients_are_clipped_separately(device: str) -> None:
    policy = _policy(device=device)
    reference = deepcopy(policy)
    algorithm = _algorithm(policy, device=device, max_grad_norm=1.0)
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        actual.grad = torch.full_like(actual, 2.0)
        expected.grad = actual.grad.clone()
    torch.nn.utils.clip_grad_norm_(
        (*reference.actor.parameters(), reference.log_std), 1.0, error_if_nonfinite=True
    )
    torch.nn.utils.clip_grad_norm_(
        reference.critic.parameters(), 1.0, error_if_nonfinite=True
    )

    algorithm._clip_gradients()

    for actual, expected in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_shared_actor_critic_parameters_are_clipped_once(device: str) -> None:
    shared = nn.Linear(2, 3)
    policy = ActorCritic(
        2,
        1,
        torch.device(device),
        nn.Sequential(shared, nn.Linear(3, 1)),
        nn.Sequential(shared, nn.Linear(3, 1)),
    )
    reference = deepcopy(policy)
    algorithm = _algorithm(policy, device=device, max_grad_norm=1.0)
    for actual, expected in zip(policy.parameters(), reference.parameters()):
        actual.grad = torch.ones_like(actual)
        expected.grad = actual.grad.clone()
    actor_parameters = (*reference.actor.parameters(), reference.log_std)
    actor_ids = {id(parameter) for parameter in actor_parameters}
    critic_parameters = tuple(
        parameter
        for parameter in reference.critic.parameters()
        if id(parameter) not in actor_ids
    )
    torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0, error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(critic_parameters, 1.0, error_if_nonfinite=True)

    algorithm._clip_gradients()

    for actual, expected in zip(policy.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def test_custom_gradient_groups_are_refreshed_each_step() -> None:
    policy = _policy()
    reference = deepcopy(policy)
    algorithm = _algorithm(policy, max_grad_norm=1.0)
    for combined in (False, True):
        for actual, expected in zip(policy.parameters(), reference.parameters()):
            actual.grad = torch.full_like(actual, 2.0)
            expected.grad = actual.grad.clone()

        def groups(model):
            actor = tuple(model.actor.parameters())
            critic = tuple(model.critic.parameters())
            return (
                (actor + critic, (model.log_std,))
                if combined
                else (actor, critic + (model.log_std,))
            )

        policy.optimization_parameter_groups = lambda: groups(policy)
        for parameters in groups(reference):
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)

        algorithm._clip_gradients()

        for actual, expected in zip(policy.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


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


@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_gae_mixed_boundaries_preserve_recurrence_and_inputs(
    horizon: int, dtype: torch.dtype, strided: bool
) -> None:
    """Known discounted returns retain timeout and per-row episode boundaries."""
    fields = {
        "reward": torch.tensor([[1, 2, 3, 4, 0], [2, 4, 6, 8, 0]], dtype=dtype),
        "value": torch.tensor([[2, 4, 6, 8, 10], [1, 3, 5, 7, 9]], dtype=dtype),
        "done": torch.tensor(
            [[False, True, False, True, False], [False, True, True, False, False]]
        ),
        "terminated": torch.tensor(
            [[False, True, False, False, False], [False, False, True, False, False]]
        ),
        "truncated": torch.tensor(
            [[False, False, False, True, False], [False, True, True, False, False]]
        ),
        "timeout_value": torch.tensor(
            [[float("nan")] * 3 + [20, 0], [float("nan")] * 2 + [20, 0, 0]],
            dtype=dtype,
        ),
    }
    fields = {name: value[:, : horizon + 1] for name, value in fields.items()}
    if strided:
        fields = {
            name: value.repeat_interleave(2, dim=1)[:, ::2]
            for name, value in fields.items()
        }
    rollout = TensorDict(fields, batch_size=[2, horizon + 1])
    original = rollout.clone()

    advantages, returns = compute_gae(rollout, gamma=0.5, gae_lambda=0.5)

    # At horizon four, row 0's TD residuals are [1, -2, 1, 6]; row 1's
    # are [2.5, 2.5, 1, 5.5]. Only nonterminal steps carry 1/4 of the next
    # advantage. A missing timeout value uses the current value; termination
    # takes precedence when both flags are set.
    expected = torch.tensor(
        [[3.0], [3.5]]
        if horizon == 1
        else [[2.5, 2.0, 8.5, 14.0], [4.125, 5.5, 6.0, 12.5]]
    )
    torch.testing.assert_close(returns, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        advantages, expected - original["value"][:, :-1].float(), rtol=0, atol=0
    )
    for name, value in original.items():
        torch.testing.assert_close(rollout[name], value, equal_nan=True)
    for name in ("advantage", "return"):
        torch.testing.assert_close(rollout[name][:, -1], torch.zeros(2))


@pytest.mark.parametrize("next_ending", ["running", "timeout_without_observation"])
def test_timeout_bootstraps_terminal_state_instead_of_current_or_reset_state(
    next_ending: str,
) -> None:
    from embodichain.learning.rl.algo import compute_gae
    from embodichain.learning.rl.buffer import RolloutBuffer
    from embodichain.learning.rl.collector import SyncCollector
    from embodichain.learning.rl.policies import ActorCritic

    class TimeoutEnv:
        num_envs = 1
        device = torch.device("cpu")
        truncated = True
        has_terminal_observation = True

        def reset(self):
            return {"state": torch.tensor([[2.0]])}, {}

        def step(self, action):
            return (
                {"state": torch.tensor([[99.0]])},
                torch.tensor([1.0]),
                torch.tensor([False]),
                torch.tensor([self.truncated]),
                (
                    {
                        "terminal_observation": {"state": torch.tensor([[7.0]])},
                        "terminal_env_ids": torch.tensor([0]),
                        "terminal_observation_mask": torch.tensor([True]),
                    }
                    if self.has_terminal_observation
                    else {}
                ),
            )

    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1.0)
    policy = ActorCritic(1, 1, torch.device("cpu"), torch.nn.Linear(1, 1), critic)
    env = TimeoutEnv()
    buffer = RolloutBuffer(1, 1, 1, 1, env.device)
    collector = SyncCollector(env, policy, env.device)
    rollout = collector.collect(1, rollout=buffer.start_rollout())
    _, returns = compute_gae(rollout, gamma=0.9, gae_lambda=1.0)
    torch.testing.assert_close(returns, torch.tensor([[7.3]]))

    # Reusing the raw TensorDict must not inherit the prior terminal value,
    # including when a legacy environment cannot provide terminal observations.
    env.truncated = next_ending == "timeout_without_observation"
    env.has_terminal_observation = False
    collector.collect(1, rollout=rollout)
    assert torch.isnan(rollout["timeout_value"]).all()
    _, returns = compute_gae(rollout, gamma=0.9, gae_lambda=1.0)
    torch.testing.assert_close(returns, torch.tensor([[1.0 + 0.9 * 99.0]]))


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize(
    "ending,bootstrap_timeouts,has_timeout_field",
    [
        ("running", True, True),
        ("terminated", True, True),
        ("truncated", True, True),
        ("both", True, True),
        ("truncated", False, True),
        ("truncated", True, False),
    ],
)
def test_terminal_values_only_infer_actual_timeout_rows(
    compact, ending, bootstrap_timeouts, has_timeout_field
) -> None:
    """Full and reordered compact terminal batches use transition flags first."""

    class Env:
        num_envs = 3
        device = torch.device("cpu")

        def reset(self):
            return {"state": torch.tensor([[1.0], [2.0], [3.0]])}, {}

        def step(self, action):
            terminal = torch.tensor([[5.0], [7.0], [11.0]])
            terminated = torch.tensor([True, ending in {"terminated", "both"}, False])
            truncated = torch.tensor([True, ending in {"truncated", "both"}, False])
            info = {"terminal_observation": {"state": terminal}}
            if compact:
                ids = (terminated | truncated).nonzero(as_tuple=False).flatten().flip(0)
                info.update(
                    terminal_env_ids=ids,
                    terminal_observation={"state": terminal[ids]},
                    terminal_observation_mask=truncated & ~terminated,
                )
            next_state = terminal.clone()
            next_state[terminated | truncated] = 99.0
            return (
                {"state": next_state},
                torch.ones(3),
                terminated,
                truncated,
                info,
            )

    critic = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1.0)
    batches = []
    critic.register_forward_pre_hook(
        lambda module, inputs: batches.append(inputs[0].clone())
    )
    policy = ActorCritic(1, 1, torch.device("cpu"), nn.Linear(1, 1), critic)
    buffer = RolloutBuffer(3, 1, 1, 1, torch.device("cpu"))
    rollout = buffer.start_rollout()
    if not has_timeout_field:
        del rollout["timeout_value"]
    elif not bootstrap_timeouts:
        rollout["timeout_value"].fill_(13.0)
    SyncCollector(
        Env(), policy, torch.device("cpu"), bootstrap_timeouts=bootstrap_timeouts
    ).collect(1, rollout=rollout)
    if ending == "truncated" and bootstrap_timeouts and has_timeout_field:
        assert [len(batch) for batch in batches] == [3, 3, 3]
        torch.testing.assert_close(batches[1], torch.tensor([[5.0], [7.0], [11.0]]))
        assert rollout["timeout_value"][1, 0] == 7.0
    else:
        assert [len(batch) for batch in batches] == [3, 3]
        if not has_timeout_field:
            assert "timeout_value" not in rollout.keys()
        elif not bootstrap_timeouts:
            torch.testing.assert_close(
                rollout["timeout_value"], torch.full((3, 2), 13.0)
            )
        else:
            assert torch.isnan(rollout["timeout_value"]).all()
    if bootstrap_timeouts and has_timeout_field:
        assert torch.isnan(rollout["timeout_value"][[0, 2], 0]).all()


@pytest.mark.parametrize("fail_terminal", [False, True])
def test_single_timeout_critic_inference_restores_mode_and_batchnorm(
    fail_terminal,
) -> None:
    """One timeout uses fixed statistics and restores modes on failure."""

    class Critic(nn.Sequential):
        def forward(self, value):
            if not self.training and fail_terminal:
                raise OSError("terminal critic failed")
            return super().forward(value)

    critic = Critic(nn.BatchNorm1d(1), nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        critic[1].weight.fill_(1.0)
    terminal_stats = []

    class Env:
        num_envs = 2
        device = torch.device("cpu")

        def reset(self):
            return {"state": torch.tensor([[-1.0], [1.0]])}, {}

        def step(self, action):
            terminal_stats.append(
                (critic[0].running_mean.clone(), critic[0].running_var.clone())
            )
            return (
                {"state": torch.tensor([[99.0], [3.0]])},
                torch.ones(2),
                torch.tensor([False, False]),
                torch.tensor([True, False]),
                {
                    "terminal_observation": {"state": torch.tensor([[7.0]])},
                    "terminal_env_ids": torch.tensor([0]),
                    "terminal_observation_mask": torch.tensor([True, False]),
                },
            )

    policy = ActorCritic(1, 1, torch.device("cpu"), nn.Linear(1, 1), critic)
    collector = SyncCollector(Env(), policy, torch.device("cpu"))
    rollout = RolloutBuffer(2, 1, 1, 1, torch.device("cpu")).start_rollout()
    if fail_terminal:
        with pytest.raises(OSError, match="terminal critic failed"):
            collector.collect(1, rollout=rollout)
        assert critic[0].num_batches_tracked == 1
    else:
        collector.collect(1, rollout=rollout)
        mean, variance = terminal_stats[0]
        expected = (7.0 - mean) / (variance + critic[0].eps).sqrt()
        torch.testing.assert_close(rollout["timeout_value"][0, 0], expected.squeeze())
        # Only ordinary full-batch action/final-value calls update statistics.
        assert critic[0].num_batches_tracked == 2
    assert all(module.training for module in policy.modules())


@pytest.mark.parametrize("compact", [False, True])
def test_one_timeout_retains_peers_for_batchnorm_without_running_statistics(compact):
    """Batch statistics include hard-terminal and still-active peer states."""
    terminal = torch.tensor([[5.0], [7.0], [11.0]])

    class Env:
        num_envs = 3
        device = torch.device("cpu")

        def reset(self):
            return {"state": torch.tensor([[1.0], [2.0], [3.0]])}, {}

        def step(self, action):
            info = {"terminal_observation": {"state": terminal.clone()}}
            if compact:
                ids = torch.tensor([1, 0])
                info.update(
                    terminal_env_ids=ids, terminal_observation={"state": terminal[ids]}
                )
            return (
                {"state": torch.tensor([[99.0], [99.0], [11.0]])},
                torch.ones(3),
                torch.tensor([True, False, False]),
                torch.tensor([False, True, False]),
                info,
            )

    critic = nn.Sequential(
        nn.BatchNorm1d(1, track_running_stats=False), nn.Linear(1, 1, bias=False)
    )
    with torch.no_grad():
        critic[1].weight.fill_(1.0)
    policy = ActorCritic(1, 1, torch.device("cpu"), nn.Linear(1, 1), critic)
    rollout = SyncCollector(Env(), policy, torch.device("cpu")).collect(
        1, RolloutBuffer(3, 1, 1, 1, torch.device("cpu")).start_rollout()
    )
    expected = (terminal[1, 0] - terminal.mean()) / (
        terminal.var(unbiased=False) + critic[0].eps
    ).sqrt()
    torch.testing.assert_close(rollout["timeout_value"][1, 0], expected)
    assert torch.isnan(rollout["timeout_value"][[0, 2], 0]).all()
    torch.testing.assert_close(rollout["obs"][0, 1], torch.tensor([99.0]))
    assert all(module.training for module in policy.modules())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "source_dtype,amp_dtype,compact",
    [
        (torch.float64, None, True),
        (torch.float16, None, True),
        (torch.uint8, None, True),
        (torch.float64, None, False),
        (torch.float32, torch.float16, True),
        (torch.float32, torch.bfloat16, True),
    ],
)
def test_terminal_bootstrap_preserves_rollout_dtypes(
    device, source_dtype, amp_dtype, compact
):
    """Terminal actor/critic inputs and autocast values use their storage dtypes."""
    if (
        device == "cuda"
        and amp_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 is unavailable on this device")
    target_device = torch.device(device)

    def observation(values):
        state = torch.tensor(values, dtype=source_dtype, device=target_device).reshape(
            -1, 1
        )
        return {
            "policy": state,
            "critic": torch.cat((state, torch.ones_like(state)), dim=-1),
        }

    class Env:
        num_envs = 3
        device = target_device

        def reset(self):
            return observation([1, 2, 3]), {}

        def step(self, action):
            terminal = observation([5, 7, 11])
            info = {"terminal_observation": terminal}
            if compact:
                ids = torch.tensor([1], device=target_device)
                info.update(
                    terminal_env_ids=ids,
                    terminal_observation={
                        key: value[ids] for key, value in terminal.items()
                    },
                )
            return (
                observation([5, 99, 11]),
                torch.ones(3, device=target_device),
                torch.tensor([False, False, False], device=target_device),
                torch.tensor([False, True, False], device=target_device),
                info,
            )

    critic = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1.0)
    policy = ActorCritic(
        1,
        1,
        target_device,
        nn.Linear(1, 1),
        critic,
        critic_obs_dim=2,
        actor_obs_groups=("policy",),
        critic_obs_groups=("critic",),
    )
    buffer = RolloutBuffer(3, 1, 1, 1, target_device, critic_obs_dim=2)
    collector = SyncCollector(Env(), policy, target_device)
    with torch.autocast(
        device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None
    ):
        rollout = collector.collect(1, buffer.start_rollout())
    assert rollout["timeout_value"].dtype == rollout["value"].dtype == torch.float32
    torch.testing.assert_close(
        rollout["timeout_value"][1, 0], torch.tensor(8.0, device=target_device)
    )
    assert torch.isnan(rollout["timeout_value"][[0, 2], 0]).all()
    torch.testing.assert_close(
        rollout["obs"][1, 1], torch.tensor([99.0], device=target_device)
    )


@pytest.mark.parametrize("tensor_dict_info", [False, True])
def test_base_env_preserves_terminal_rows_before_reused_reset_storage(
    tensor_dict_info: bool,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace
    from embodichain.lab.gym.envs import BaseEnv

    state = torch.tensor([[3.0], [7.0], [11.0]])
    rewards = torch.tensor([2.0, 5.0, 8.0])
    terminated = torch.tensor([True, False, False])
    truncated = torch.tensor([False, True, False])
    observation = {"state": state}
    info = {"success": torch.tensor([True, False, False])}
    if tensor_dict_info:
        info = TensorDict(info, batch_size=[3])

    def reset(**kwargs):
        ids = kwargs["options"]["reset_ids"]
        state[ids] = 99.0
        rewards[ids] = 0
        terminated[ids] = False
        truncated[ids] = False
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
        get_reward=lambda **kwargs: rewards,
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
    env._get_termination_flags = lambda obs, info: (terminated, truncated)
    env._seal_step_outcomes = lambda *args: BaseEnv._seal_step_outcomes(env, *args)
    env._after_observation = lambda *args: BaseEnv._after_observation(env, *args)
    _, step_reward, step_terminated, step_truncated, info = BaseEnv.step(
        env, torch.zeros(3, 1)
    )
    torch.testing.assert_close(step_reward, torch.tensor([2.0, 5.0, 8.0]))
    torch.testing.assert_close(step_terminated, torch.tensor([True, False, False]))
    torch.testing.assert_close(step_truncated, torch.tensor([False, True, False]))
    torch.testing.assert_close(
        info["terminal_observation"]["state"], torch.tensor([[3.0], [7.0]])
    )
    torch.testing.assert_close(state, torch.tensor([[99.0], [99.0], [11.0]]))
    torch.testing.assert_close(
        info["terminal_observation_mask"], torch.tensor([False, True, False])
    )


def test_custom_policy_keeps_its_default_config_when_omitted(monkeypatch) -> None:
    from embodichain.learning.rl.policies import Policy
    from embodichain.learning.rl.policies import registry

    class DefaultConfig:
        gain = 3

    class CustomPolicy(Policy):
        def __init__(self, obs_dim, action_dim, device, cfg=DefaultConfig()):
            super().__init__()
            self.gain = cfg.gain

        def forward(self, tensordict, deterministic=False):
            return tensordict

        def get_value(self, tensordict):
            return tensordict

        def evaluate_actions(self, tensordict):
            return tensordict

    monkeypatch.setitem(registry._POLICY_REGISTRY, "custom_default", CustomPolicy)
    policy = build_policy({"name": "custom_default"}, 3, 1, torch.device("cpu"))
    assert policy.gain == 3
