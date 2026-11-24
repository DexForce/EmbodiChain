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
from embodichain.learning.rl.policies.actor_only import _validate_normal_action
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
def test_actor_only_combined_normal_validation_preserves_torch_contract(
    mean, scale, action, validate, monkeypatch
) -> None:
    monkeypatch.setattr(Distribution, "_validate_args", validate)
    mean, scale, action = (torch.tensor(value) for value in (mean, scale, action))
    try:
        expected_distribution = Normal(mean, scale)
        expected = expected_distribution.log_prob(action)
    except (ValueError, RuntimeError) as error:
        with pytest.raises(type(error)) as actual:
            _validate_normal_action(
                Normal(mean, scale, validate_args=False), action
            ).log_prob(action)
        assert str(actual.value) == str(error)
    else:
        actual_distribution = _validate_normal_action(
            Normal(mean, scale, validate_args=False), action
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
        kl_coef=0.0,
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
