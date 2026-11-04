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

from embodichain.learning.rl.algo import GRPO, GRPOCfg
from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.policies import ActorOnly
from embodichain.learning.rl.utils import OptimizerCfg


def _policy(device: str = "cpu") -> ActorOnly:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        return ActorOnly(2, 1, torch.device(device), torch.nn.Linear(2, 1))


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
    ("optimizer_name", "device"),
    [
        ("adam", "cpu"),
        ("adamw", "cpu"),
        pytest.param("adamw", "cuda", marks=pytest.mark.gpu),
    ],
)
def test_post_terminal_padding_does_not_advance_grpo_optimizer(
    optimizer_name: str, device: str
) -> None:
    policy = _policy(device)
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

    generator = (
        torch.random.default_generator
        if device == "cpu"
        else torch.cuda.default_generators[torch.cuda.current_device()]
    )
    generator.manual_seed(17)
    expected_metrics = algorithm.update(rollout)
    generator.manual_seed(17)
    actual_metrics = padded_algorithm.update(padded_rollout)

    assert actual_metrics == pytest.approx(expected_metrics)
    for actual, expected in zip(padded_policy.parameters(), policy.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_state = padded_algorithm.optimizer.state[actual]
        expected_state = algorithm.optimizer.state[expected]
        # One batch of two valid transitions per epoch, independently of padding.
        assert int(actual_state["step"]) == cfg.n_epochs
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


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
