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

"""GRPO valid-transition and distributed optimizer contracts."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.nn.parallel import DistributedDataParallel
from torch.distributions import Distribution, Normal

from embodichain.learning.rl.algo import GRPO, GRPOCfg
from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.policies import ActorOnly
from embodichain.learning.rl.policies.actor_only import _validate_normal
from embodichain.learning.rl.utils import OptimizerCfg


class _ExtraFieldPolicy(ActorOnly):
    """Custom policy retaining access to arbitrary transition-aligned fields."""

    def evaluate_actions(self, tensordict: TensorDict) -> TensorDict:
        evaluated = ActorOnly.evaluate_actions(self, tensordict)
        evaluated["sample_log_prob"] = (
            evaluated["sample_log_prob"] + tensordict["extra"]
        )
        return evaluated


def _policy(device: str = "cpu", policy_type: type[ActorOnly] = ActorOnly) -> ActorOnly:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        return policy_type(2, 1, torch.device(device), torch.nn.Linear(2, 1))


def test_cpu_policy_preserves_rng_without_accessing_accelerators(monkeypatch) -> None:
    def unexpected_accelerator_access(*args, **kwargs):
        pytest.fail("CPU policy construction accessed an accelerator")

    for name in ("device_count", "get_rng_state", "set_rng_state", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, unexpected_accelerator_access)
    monkeypatch.setattr(torch, "manual_seed", unexpected_accelerator_access)
    before = torch.random.get_rng_state()
    first, second = _policy(), _policy()
    assert torch.equal(torch.random.get_rng_state(), before)
    for actual, expected in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mean", "scale", "action", "validate"),
    [
        ([[0.1, 0.2]], [[0.5, 1.0]], [[0.3, 0.4], [0.5, 0.6]], True),
        ([[float("inf"), 0.2]], [[0.5, float("inf")]], [[0.3, 0.4]], True),
        ([[0.1, 0.2]], [[0.5, 1.0]], [[float("inf"), 0.4]], True),
        ([[float("nan"), 0.2]], [[0.5, 1.0]], [[0.3, 0.4]], True),
        ([[0.1, 0.2]], [[float("nan"), 1.0]], [[0.3, 0.4]], True),
        ([[0.1, 0.2]], [[0.0, 1.0]], [[0.3, 0.4]], True),
        ([[0.1, 0.2]], [[0.5, 1.0]], [[float("nan"), 0.4]], True),
        ([[0.1, 0.2]], [[0.5, 1.0]], [[0.3, 0.4, 0.5]], True),
        ([[float("nan"), 0.2]], [[0.5, 1.0]], [[0.3, 0.4, 0.5]], True),
        ([[float("nan"), 0.2]], [[0.0, 1.0]], [[0.3, 0.4]], False),
    ],
)
@pytest.mark.parametrize("parameters_only", [False, True])
def test_actor_only_combined_normal_validation_preserves_torch_contract(
    mean, scale, action, validate, parameters_only, monkeypatch
) -> None:
    monkeypatch.setattr(Distribution, "_validate_args", validate)
    mean, scale, action = (torch.tensor(value) for value in (mean, scale, action))
    try:
        expected_distribution = Normal(mean, scale)
        expected = expected_distribution.log_prob(action)
    except (ValueError, RuntimeError) as error:
        with pytest.raises(type(error)) as actual:
            _validate_normal(
                Normal(mean, scale, validate_args=False),
                None if parameters_only else action,
            ).log_prob(action)
        assert str(actual.value) == str(error)
    else:
        actual_distribution = _validate_normal(
            Normal(mean, scale, validate_args=False),
            None if parameters_only else action,
        )
        torch.testing.assert_close(
            actual_distribution.log_prob(action),
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            actual_distribution.entropy(),
            expected_distribution.entropy(),
            rtol=0,
            atol=0,
            equal_nan=True,
        )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("validate", [False, True])
def test_normal_parameter_validation_retains_default_and_explicit_flags(
    device: str, validate: bool, monkeypatch
) -> None:
    policy = _policy(device)
    obs = torch.zeros(2, 2, device=device)
    for request in ("helper", None, False, True):
        monkeypatch.setattr(Distribution, "_validate_args", validate)
        actual = (
            _validate_normal(
                Normal(obs[:, :1], torch.ones_like(obs[:, :1]), validate_args=False)
            )
            if request == "helper"
            else policy._distribution(obs, validate_args=request)
        )
        expected = Normal(
            actual.loc,
            actual.scale,
            validate_args=None if request == "helper" else request,
        )
        monkeypatch.setattr(Distribution, "_validate_args", not validate)
        action = torch.full_like(actual.loc, float("nan"))
        try:
            expected_value = expected.log_prob(action)
        except ValueError as error:
            with pytest.raises(ValueError) as caught:
                actual.log_prob(action)
            assert str(caught.value) == str(error)
        else:
            torch.testing.assert_close(
                actual.log_prob(action), expected_value, rtol=0, atol=0, equal_nan=True
            )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "case",
    [
        "sample",
        "pathwise",
        "deterministic",
        "collect",
        "nan_mean",
        "nan_scale",
        "nan_sample",
        "nan_collect",
    ],
)
def test_actor_only_sampling_preserves_values_gradients_rng_and_error_timing(
    device: str, case: str, monkeypatch
) -> None:
    monkeypatch.setattr(Distribution, "_validate_args", True)
    policy = _policy(device)
    with torch.no_grad():
        if case == "nan_mean":
            policy.actor.bias.fill_(float("nan"))
        elif case == "nan_scale":
            policy.log_std.fill_(float("nan"))
    oracle = deepcopy(policy)

    def native_distribution(obs: torch.Tensor) -> Normal:
        mean = oracle.actor(obs)
        std = oracle.log_std.clamp(oracle.log_std_min, oracle.log_std_max).exp()
        return Normal(mean, std.expand(mean.shape[0], -1))

    oracle._distribution = native_distribution
    if case in ("nan_sample", "nan_collect"):
        rsample = Normal.rsample

        def invalid_sample(distribution: Normal) -> torch.Tensor:
            return rsample(distribution) * float("nan")

        monkeypatch.setattr(Normal, "rsample", invalid_sample)

    get_rng, set_rng = (
        (torch.cuda.get_rng_state, torch.cuda.set_rng_state)
        if device == "cuda"
        else (torch.get_rng_state, torch.set_rng_state)
    )
    with torch.random.fork_rng(
        devices=[torch.device(device)] if device == "cuda" else []
    ):
        initial_rng = get_rng()
        receipts = []
        for model in (oracle, policy):
            set_rng(initial_rng)
            obs = torch.tensor(
                [[0.1, 0.2], [0.3, 0.4]], device=device, requires_grad=True
            )
            inputs = TensorDict({"obs": obs}, batch_size=[2])
            error = None
            try:
                if case == "sample":
                    model(inputs)
                elif case in ("collect", "nan_collect"):
                    model._collect_differentiable_action(inputs)
                else:
                    model.get_differentiable_action(
                        inputs, deterministic=case == "deterministic"
                    )
            except ValueError as caught:
                error = str(caught)
            if error is None:
                sum(
                    value.sum() for key, value in inputs.items() if key != "obs"
                ).backward()
            receipts.append(
                (
                    inputs,
                    error,
                    get_rng(),
                    [obs.grad] + [p.grad for p in model.parameters()],
                )
            )
        expected, actual = receipts
        assert actual[1] == expected[1]
        assert actual[0].keys() == expected[0].keys()
        for key in expected[0].keys():
            torch.testing.assert_close(
                actual[0][key], expected[0][key], rtol=0, atol=0, equal_nan=True
            )
        assert torch.equal(actual[2], expected[2])
        for actual_grad, expected_grad in zip(actual[3], expected[3], strict=True):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=0, atol=0, equal_nan=True
            )
        if case in ("nan_mean", "nan_scale"):
            assert actual[1] is not None and "action" not in actual[0].keys()
            assert torch.equal(actual[2], initial_rng)
        elif case in ("nan_sample", "nan_collect"):
            assert torch.isnan(actual[0]["action"]).all()
            assert (actual[1] is None) == (case == "nan_collect")
            assert not torch.equal(actual[2], initial_rng)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_actor_only_tensor_subclass_keeps_one_native_constructor(device: str) -> None:
    broadcasts = []

    class ObservedTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is torch.broadcast_tensors:
                broadcasts.append(func)
            return super().__torch_function__(func, types, args, kwargs)

    class Actor(torch.nn.Module):
        def forward(self, obs):
            return obs[:, :1].as_subclass(ObservedTensor)

    policy = ActorOnly(2, 1, torch.device(device), Actor())
    obs = torch.zeros(2, 2, device=device)
    Normal(policy.actor(obs), torch.ones(2, 1, device=device))
    expected_broadcasts = len(broadcasts)
    assert expected_broadcasts == 1
    broadcasts.clear()
    policy._distribution(obs)
    assert len(broadcasts) == expected_broadcasts


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("case", ["finite", "broadcast", "nan_action", "unchecked"])
def test_actor_only_evaluation_preserves_normal_values_and_gradients(
    device: str, case: str, monkeypatch
) -> None:
    monkeypatch.setattr(Distribution, "_validate_args", case != "unchecked")
    policy = _policy(device)
    oracle = deepcopy(policy)
    obs = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device=device)
    action = torch.tensor([[0.4], [0.6]], device=device)
    if case == "broadcast":
        action = action.expand(2, 3)
    elif case in ("nan_action", "unchecked"):
        action[0] = float("nan")
    inputs = TensorDict({"obs": obs, "action": action}, batch_size=[2])
    expected_distribution = oracle._distribution(obs)
    if case == "nan_action":
        with pytest.raises(ValueError) as expected:
            expected_distribution.log_prob(action)
        with pytest.raises(ValueError) as actual:
            policy.evaluate_actions(inputs)
        assert str(actual.value) == str(expected.value)
        return
    expected_log_prob = expected_distribution.log_prob(action).sum(-1)
    expected_entropy = expected_distribution.entropy().sum(-1)
    evaluated = policy.evaluate_actions(inputs)
    for actual, expected in (
        (evaluated["sample_log_prob"], expected_log_prob),
        (evaluated["entropy"], expected_entropy),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if case != "unchecked":
        (evaluated["sample_log_prob"] + evaluated["entropy"]).sum().backward()
        (expected_log_prob + expected_entropy).sum().backward()
        for actual, expected in zip(policy.parameters(), oracle.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("instance_override", [False, True])
def test_actor_only_evaluation_preserves_custom_distribution(
    device: str, instance_override: bool
) -> None:
    class CustomDistributionPolicy(ActorOnly):
        def _distribution(self, obs: torch.Tensor) -> Normal:
            return Normal(self.actor(obs) + 2.0, torch.ones_like(obs[:, :1]) * 3.0)

    policy = _policy(
        device, ActorOnly if instance_override else CustomDistributionPolicy
    )
    if instance_override:
        policy._distribution = CustomDistributionPolicy._distribution.__get__(policy)
    obs = torch.tensor([[0.1, 0.2], [0.3, 0.4]], device=device)
    action = torch.tensor([[0.4], [0.6]], device=device)
    expected = policy._distribution(obs)
    evaluated = policy.evaluate_actions(
        TensorDict({"obs": obs, "action": action}, batch_size=[2])
    )
    torch.testing.assert_close(
        evaluated["sample_log_prob"], expected.log_prob(action).sum(-1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        evaluated["entropy"], expected.entropy().sum(-1), rtol=0, atol=0
    )
    sampled = policy.get_action(
        TensorDict({"obs": obs}, batch_size=[2]), deterministic=True
    )
    torch.testing.assert_close(sampled["action"], expected.mean, rtol=0, atol=0)


def _rollout(
    policy: ActorOnly, lengths: tuple[int, int], padding: int = 0
) -> TensorDict:
    steps = max(lengths) + padding
    rollout = RolloutBuffer(2, steps, 2, 1, policy.device).start_rollout()
    rows = torch.arange(2, dtype=torch.float32, device=policy.device)[:, None].expand(
        2, steps + 1
    )
    times = torch.arange(steps + 1, dtype=torch.float32, device=policy.device)[
        None
    ].expand(2, steps + 1)
    rollout["obs"] = torch.stack((rows / 3, times / 10), -1)
    rollout["action"] = (0.2 + rows / 5 + times / 20)[..., None]
    rollout["extra"] = rows / 7 + times / 100
    for key in ("reward", "value", "done", "terminated", "truncated"):
        rollout[key].zero_()
    with torch.no_grad():
        evaluated = policy.evaluate_actions(rollout.reshape(-1))
        rollout["sample_log_prob"] = evaluated["sample_log_prob"].reshape(2, steps + 1)
    for row, length in enumerate(lengths):
        rollout["reward"][row, :length] = row + 1
        rollout["done"][row, length - 1] = True
        rollout["terminated"][row, length - 1] = True
    return rollout


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_grpo_returns_and_prefix_mask_follow_episode_boundaries(
    truncate: bool, horizon: int, strided: bool, device: str
) -> None:
    """First/last/repeated terminals retain the terminal step in the prefix."""
    rewards = torch.tensor([[1.0, 2.0, 4.0, 8.0]] * 4, device=device)[:, :horizon]
    dones = torch.tensor(
        [
            [True, False, False, False],
            [False, False, False, True],
            [False, True, False, True],
            [False, False, False, False],
        ],
        device=device,
    )[:, :horizon]
    if strided:
        rewards = rewards.repeat_interleave(2, dim=1)[:, ::2]
        dones = dones.repeat_interleave(2, dim=1)[:, ::2]
    original_rewards, original_dones = rewards.clone(), dones.clone()
    algorithm = GRPO(
        GRPOCfg(device=device, gamma=0.5, truncate_at_first_done=truncate),
        _policy(device),
    )

    returns, mask = algorithm._compute_step_returns_and_mask(rewards, dones)

    # A nonterminal return carries half the next return; every terminal cuts
    # that recurrence, even after the prefix mask has excluded later episodes.
    expected_returns = torch.tensor(
        (
            [[1.0]] * 4
            if horizon == 1
            else [
                [1.0, 6.0, 8.0, 8.0],
                [4.0, 6.0, 8.0, 8.0],
                [2.0, 2.0, 8.0, 8.0],
                [4.0, 6.0, 8.0, 8.0],
            ]
        ),
        device=device,
    )
    expected_mask = torch.ones(4, horizon, device=device)
    if truncate:
        expected_mask[0, 1:] = 0
        expected_mask[2, 2:] = 0
    torch.testing.assert_close(returns, expected_returns, rtol=0, atol=0)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)
    torch.testing.assert_close(rewards, original_rewards, rtol=0, atol=0)
    torch.testing.assert_close(dones, original_dones, rtol=0, atol=0)


@pytest.mark.parametrize("kl_coef", [0.0, 0.1])
@pytest.mark.parametrize(
    ("optimizer_name", "device", "policy_type", "instance_override"),
    [
        ("adam", "cpu", ActorOnly, False),
        ("adamw", "cpu", ActorOnly, False),
        ("adamw", "cpu", _ExtraFieldPolicy, False),
        ("adamw", "cpu", ActorOnly, True),
        pytest.param("adamw", "cuda", ActorOnly, False, marks=pytest.mark.gpu),
    ],
)
def test_post_terminal_padding_does_not_advance_grpo_optimizer(
    optimizer_name: str,
    device: str,
    policy_type: type[ActorOnly],
    instance_override: bool,
    kl_coef: float,
) -> None:
    policy = _policy(device, policy_type)
    if instance_override:
        policy.evaluate_actions = _ExtraFieldPolicy.evaluate_actions.__get__(policy)
    padded_policy = deepcopy(policy)
    cfg = GRPOCfg(
        device=device,
        group_size=2,
        n_epochs=2,
        batch_size=2,
        kl_coef=kl_coef,
        ent_coef=0.03,
        optimizer=OptimizerCfg(name=optimizer_name, learning_rate=0.01),
    )
    algorithm = GRPO(cfg.copy(), policy)
    padded_algorithm = GRPO(cfg.copy(), padded_policy)
    rollout = _rollout(policy, (1, 1))
    padded_rollout = _rollout(padded_policy, (1, 1), padding=5)
    original = padded_rollout.clone()

    generator = (
        torch.random.default_generator
        if device == "cpu"
        else torch.cuda.default_generators[torch.cuda.current_device()]
    )
    generator.manual_seed(17)
    expected_metrics = algorithm.update(rollout)
    generator.manual_seed(17)
    actual_metrics = padded_algorithm.update(padded_rollout)

    assert padded_rollout.keys() == original.keys()
    for key in original.keys():
        torch.testing.assert_close(
            padded_rollout[key], original[key], rtol=0, atol=0, equal_nan=True
        )
    assert actual_metrics == pytest.approx(expected_metrics)
    for actual, expected in zip(padded_policy.parameters(), policy.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_state = padded_algorithm.optimizer.state[actual]
        expected_state = algorithm.optimizer.state[expected]
        # One batch of two valid transitions per epoch, independently of padding.
        assert int(actual_state["step"]) == cfg.n_epochs
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


def test_custom_evaluator_cannot_replace_grpo_old_likelihoods() -> None:
    policy = _policy()
    custom = deepcopy(policy)
    rollout = _rollout(policy, (3, 2))

    def evaluate_actions(batch):
        evaluated = ActorOnly.evaluate_actions(custom, batch)
        batch["sample_log_prob"].copy_(evaluated["sample_log_prob"].detach())
        return evaluated

    custom.evaluate_actions = evaluate_actions
    cfg = GRPOCfg(
        device="cpu",
        group_size=2,
        n_epochs=3,
        batch_size=2,
        kl_coef=0.0,
        optimizer=OptimizerCfg(learning_rate=0.01),
    )
    expected, actual = GRPO(cfg.copy(), policy), GRPO(cfg.copy(), custom)
    torch.random.default_generator.manual_seed(31)
    expected_metrics = expected.update(rollout)
    torch.random.default_generator.manual_seed(31)
    actual_metrics = actual.update(rollout)
    assert actual_metrics == pytest.approx(expected_metrics)
    torch.testing.assert_close(custom.state_dict(), policy.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("reference_declared", [False, True])
def test_grpo_projection_preserves_reference_policy_input_requirements(
    reference_declared: bool,
) -> None:
    from embodichain.learning.rl.policies.policy import _TrainingInputSpec

    class ModeDependentInputs(ActorOnly):
        def _training_input_spec(self):
            if not self.declare_inputs or (
                not self.training and not reference_declared
            ):
                return _TrainingInputSpec()
            field = "extra" if self.training else "ref_extra"
            return _TrainingInputSpec(
                fields=("obs", "action", field), mutates_inputs=False
            )

        def evaluate_actions(self, batch):
            evaluated = super().evaluate_actions(batch)
            field = "extra" if self.training else "ref_extra"
            evaluated["sample_log_prob"] = evaluated["sample_log_prob"] + batch[field]
            return evaluated

    declared = _policy(policy_type=ModeDependentInputs)
    declared.declare_inputs = True
    reference = deepcopy(declared)
    reference.declare_inputs = False
    rollout = _rollout(declared, (3, 2))
    rollout["ref_extra"] = torch.full(rollout.batch_size, 0.2)
    cfg = GRPOCfg(device="cpu", group_size=2, n_epochs=2, batch_size=2, kl_coef=0.1)
    expected, actual = GRPO(cfg.copy(), reference), GRPO(cfg.copy(), declared)

    torch.random.default_generator.manual_seed(31)
    expected_metrics = expected.update(rollout)
    torch.random.default_generator.manual_seed(31)
    actual_metrics = actual.update(rollout)

    assert actual_metrics == expected_metrics
    torch.testing.assert_close(
        declared.state_dict(), reference.state_dict(), rtol=0, atol=0
    )


def _distributed_worker(rank: int, group_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{group_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        for batch_size in (128, 2):
            policy = _policy()
            reference = deepcopy(policy)
            short = _rollout(reference, (1, 1), padding=2)
            long = _rollout(reference, (3, 3))
            optimizer = OptimizerCfg(
                name="sgd" if batch_size == 128 else "adamw", learning_rate=0.01
            )
            cfg = GRPOCfg(
                device="cpu",
                group_size=2,
                gamma=1.0,
                n_epochs=1,
                batch_size=batch_size,
                kl_coef=0.0,
                ent_coef=0.03,
                max_grad_norm=100.0,
                optimizer=optimizer,
            )
            if batch_size == 128:
                # Independent single-process oracle: one mean over all 8 valid
                # transitions, rather than a mean of unequal rank-local means.
                global_rollout = torch.cat((short, long), dim=0)
                expected_metrics = GRPO(cfg.copy(), reference).update(global_rollout)
            algorithm = GRPO(cfg.copy(), DistributedDataParallel(policy))
            metrics = algorithm.update(short if rank == 0 else long)
            if batch_size == 128:
                assert metrics == pytest.approx(expected_metrics, abs=1e-6)
                for actual, expected in zip(
                    policy.parameters(), reference.parameters()
                ):
                    torch.testing.assert_close(actual, expected)
            else:
                # Rank 0 has no local samples for the last two batches, but
                # must join rank 1's updates and preserve AdamW momentum.
                for state in algorithm.optimizer.state.values():
                    assert int(state["step"]) == 3
            for parameter in policy.parameters():
                copies = [torch.empty_like(parameter) for _ in range(2)]
                dist.all_gather(copies, parameter.detach())
                torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo"
)
def test_grpo_distributed_unequal_valid_counts_and_empty_tail(tmp_path: Path) -> None:
    processes = mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "group"),),
        nprocs=2,
        join=False,
    )
    try:
        # A bounded join catches regressions in collective participation.
        deadline = time.monotonic() + 45
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("GRPO distributed workers did not finish within 45 seconds")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
