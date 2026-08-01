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

"""Tests for segmented differentiable training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from embodichain.learning.rl import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from embodichain.learning.rl.algo import APG, APGCfg
from embodichain.learning.rl.models import ActorOnly


class _QuadraticActionEnv:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.single_observation_space = Box(-float("inf"), float("inf"), (1,))
        self.single_action_space = Box(-float("inf"), float("inf"), (1,))
        self._state = torch.ones((num_envs, 1))
        self.detach_calls = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del seed, options
        self._state = torch.ones((self.num_envs, 1))
        return self._state, {}

    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        reward = -action.square().sum(dim=-1)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._state, reward, done, done.clone(), {}

    def detach_state(self) -> torch.Tensor:
        self.detach_calls += 1
        self._state = self._state.detach()
        return self._state


def _make_components(
    ent_coef: float = 0.0,
) -> tuple[_QuadraticActionEnv, ActorOnly, APG]:
    env = _QuadraticActionEnv()
    actor = nn.Linear(1, 1, bias=False)
    nn.init.constant_(actor.weight, 0.5)
    policy = ActorOnly(1, 1, env.device, actor=actor)
    algorithm = APG(
        APGCfg(
            device="cpu",
            learning_rate=0.05,
            max_grad_norm=10.0,
            ent_coef=ent_coef,
        ),
        policy,
    )
    return env, policy, algorithm


def test_trainer_updates_policy_and_detaches_each_segment() -> None:
    env, policy, algorithm = _make_components()
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(
            segment_length=2,
            deterministic_actions=True,
        ),
        env,
        policy,
        algorithm,
    )
    initial_weight = policy.actor.weight.detach().abs().item()
    policy.eval()

    summary = trainer.train(total_timesteps=8)

    assert policy.training
    assert policy.actor.weight.detach().abs().item() < initial_weight
    assert summary["global_step"] == 8
    assert summary["num_updates"] == 2
    assert env.detach_calls == 2


def test_update_horizon_keeps_optimizer_budget_fixed_across_segment_lengths() -> None:
    short_env, short_policy, short_algorithm = _make_components(ent_coef=0.2)
    short_trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(
            segment_length=2,
            update_horizon=4,
            deterministic_actions=True,
        ),
        short_env,
        short_policy,
        short_algorithm,
    )
    long_env, long_policy, long_algorithm = _make_components(ent_coef=0.2)
    long_trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(
            segment_length=4,
            update_horizon=4,
            deterministic_actions=True,
        ),
        long_env,
        long_policy,
        long_algorithm,
    )

    short_summary = short_trainer.train(total_timesteps=8)
    long_summary = long_trainer.train(total_timesteps=8)

    assert short_summary["num_updates"] == long_summary["num_updates"] == 1
    assert short_env.detach_calls == 2
    assert long_env.detach_calls == 1
    assert short_summary["last_train_metrics"]["train/objective"] == pytest.approx(
        long_summary["last_train_metrics"]["train/objective"],
        rel=1e-6,
    )
    assert short_summary["last_train_metrics"]["train/entropy"] == pytest.approx(
        long_summary["last_train_metrics"]["train/entropy"],
        rel=1e-6,
    )
    assert torch.allclose(short_policy.actor.weight, long_policy.actor.weight)
    assert torch.allclose(short_policy.log_std, long_policy.log_std)


def test_update_horizon_must_be_divisible_by_segment_length() -> None:
    env, policy, algorithm = _make_components()

    with pytest.raises(ValueError, match="must be divisible"):
        DifferentiableTrainer(
            DifferentiableTrainerCfg(
                segment_length=3,
                update_horizon=4,
            ),
            env,
            policy,
            algorithm,
        )


def test_apg_training_converges_on_quadratic_action_objective() -> None:
    env, policy, algorithm = _make_components()
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(
            segment_length=2,
            deterministic_actions=True,
        ),
        env,
        policy,
        algorithm,
    )
    observation = torch.ones((env.num_envs, 1))
    initial_action = policy.actor(observation).detach()
    initial_objective = -initial_action.square().sum(dim=-1).mean().item()

    summary = trainer.train(total_timesteps=40)

    final_action = policy.actor(observation).detach()
    final_objective = -final_action.square().sum(dim=-1).mean().item()
    assert final_objective > initial_objective
    assert final_action.abs().max().item() < 0.1
    assert summary["num_updates"] == 10
    assert all(
        entry["train/skipped_update"] == 0.0 for entry in summary["train_history"]
    )


def test_checkpoint_restores_policy_optimizer_and_counters(tmp_path: Path) -> None:
    env, policy, algorithm = _make_components()
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(
            segment_length=2,
            deterministic_actions=True,
        ),
        env,
        policy,
        algorithm,
    )
    trainer.train(total_timesteps=4)
    checkpoint_path = trainer.save_checkpoint(tmp_path / "apg.pt")

    restored_env, restored_policy, restored_algorithm = _make_components()
    restored = DifferentiableTrainer(
        trainer.cfg,
        restored_env,
        restored_policy,
        restored_algorithm,
    )
    restored.load_checkpoint(checkpoint_path)

    assert torch.equal(restored_policy.actor.weight, policy.actor.weight)
    assert restored.global_step == trainer.global_step
    assert restored.num_updates == trainer.num_updates
    assert restored_algorithm.optimizer.state_dict()["state"]


def test_checkpoint_load_falls_back_for_older_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, policy, algorithm = _make_components()
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(),
        env,
        policy,
        algorithm,
    )
    checkpoint_path = trainer.save_checkpoint(tmp_path / "apg.pt")
    torch_load = torch.load
    calls: list[bool] = []

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" in kwargs:
            calls.append(True)
            raise TypeError("weights_only is unsupported")
        calls.append(False)
        return torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", compatible_load)

    trainer.load_checkpoint(checkpoint_path)

    assert calls == [True, False]
