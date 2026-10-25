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

from collections import deque
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

from embodichain.learning.rl.evaluation import (
    convert_policy_action_for_env,
    evaluate_episodes,
)
from embodichain.learning.rl.utils.trainer import Trainer


class _DeterministicPolicy(torch.nn.Module):
    obs_dim = 1
    action_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_action(self, tensordict, deterministic: bool = False):
        assert deterministic
        tensordict["action"] = torch.zeros(
            tensordict.batch_size[0], 1, device=tensordict.device
        )
        return tensordict


class _AsyncAutoResetEnv:
    num_envs = 3
    device = torch.device("cpu")
    single_observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, (1,))

    def __init__(self) -> None:
        self.steps = torch.zeros(3, dtype=torch.long)
        self.horizons = torch.tensor([1, 2, 3])
        self.last_seed = None

    def reset(self, *, seed=None, options=None):
        self.last_seed = seed
        self.steps.zero_()
        return self.steps[:, None].float(), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.horizons
        terminal_steps = self.steps.clone()
        info = {
            "success": done.clone(),
            "metrics": {
                "terminal_step": terminal_steps.float(),
                # Non-scalar per-env metric must not be flattened into eval logs.
                "final_position": torch.stack(
                    (terminal_steps.float(), terminal_steps.float() + 10.0),
                    dim=-1,
                ),
            },
        }
        self.steps[done] = 0
        return (
            self.steps[:, None].float(),
            torch.ones(3),
            done,
            torch.zeros(3, dtype=torch.bool),
            info,
        )


class _DeviceCheckingActionManager:
    def __init__(self, device: torch.device) -> None:
        self.device = device


class _MixedDeviceEnv:
    def __init__(self) -> None:
        self.device = torch.device("meta")
        self.action_manager = _DeviceCheckingActionManager(self.device)


def test_trainer_metric_means_transfer_only_one_scalar_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logging preserves mixed metric means without per-value host copies."""
    original_cpu = torch.Tensor.cpu
    transferred_shapes = []

    def counted_cpu(value: torch.Tensor, *args, **kwargs):
        transferred_shapes.append(tuple(value.shape))
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    trainer = Trainer.__new__(Trainer)
    values = trainer._pack_log_dicts(
        {
            "rewards": {
                "progress": torch.tensor([1.0, 3.0], requires_grad=True),
                "other": [1, 5],
            },
            "metrics": {
                "grasped": torch.tensor([True, False]),
                "length": torch.tensor([1, 3], dtype=torch.int64),
                "noncontiguous": torch.arange(8.0).reshape(2, 4)[:, ::2],
                "empty": torch.empty(0),
                "unsupported": "not a numeric metric",
            },
        }
    )
    assert list(values) == [
        "rewards/progress",
        "rewards/other",
        "metrics/grasped",
        "metrics/length",
        "metrics/noncontiguous",
        "metrics/empty",
    ]
    assert list(values.values())[:-1] == [2.0, 3.0, 0.5, 2.0, 3.0]
    assert math.isnan(values["metrics/empty"])
    assert transferred_shapes == [(5,)]


@pytest.mark.parametrize("logging_enabled", [True, False])
def test_trainer_rollout_logging_reuses_means_for_both_sinks(
    monkeypatch: pytest.MonkeyPatch, logging_enabled: bool
) -> None:
    """Each control step retains its tags/global step and shares scalar means."""
    trainer = Trainer.__new__(Trainer)
    trainer.curr_ret = torch.zeros(2)
    trainer.curr_len = torch.zeros(2, dtype=torch.int32)
    trainer.ret_window, trainer.len_window = deque(), deque()
    trainer.global_step = 10
    trainer.distributed, trainer.rank = False, 0
    trainer.writer = MagicMock() if logging_enabled else None
    trainer.use_wandb = logging_enabled
    trainer.buffer_size = 2
    rollout = object()
    trainer.buffer = SimpleNamespace(start_rollout=lambda: rollout, add=MagicMock())
    scalar_means = MagicMock(wraps=trainer._pack_log_dicts)
    monkeypatch.setattr(trainer, "_pack_log_dicts", scalar_means)
    wandb_log = MagicMock()
    monkeypatch.setattr("embodichain.learning.rl.utils.trainer.wandb.log", wandb_log)

    def collect(*, num_steps, rollout, on_step_callback):
        for _ in range(num_steps):
            on_step_callback(
                TensorDict(
                    {"reward": torch.tensor([1.0, 3.0]), "done": torch.zeros(2).bool()},
                    batch_size=[2],
                ),
                {
                    "rewards": {"progress": torch.tensor([1.0, 3.0])},
                    "metrics": {"grasped": torch.tensor([True, False])},
                },
            )
        return rollout

    trainer.collector = SimpleNamespace(collect=collect)
    trainer._collect_rollout()
    assert trainer.global_step == 14
    if logging_enabled:
        assert scalar_means.call_count == 2
        assert trainer.writer.add_scalar.call_args_list == [
            call("rewards/progress", 2.0, 12),
            call("metrics/grasped", 0.5, 12),
            call("rewards/progress", 2.0, 14),
            call("metrics/grasped", 0.5, 14),
        ]
        assert wandb_log.call_args_list == [
            call({"rewards/progress": 2.0, "metrics/grasped": 0.5}, step=12),
            call({"rewards/progress": 2.0, "metrics/grasped": 0.5}, step=14),
        ]
    else:
        scalar_means.assert_not_called()
        wandb_log.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_trainer_without_eval_env_ignores_empty_eval_events(
    tmp_path: Path, rank: int
) -> None:
    """Disabled evaluation and non-evaluating ranks need no eval manager."""
    trainer = Trainer(
        policy=_DeterministicPolicy(),
        env=_AsyncAutoResetEnv(),
        algorithm=SimpleNamespace(device=torch.device("cpu")),
        buffer_size=2,
        batch_size=2,
        writer=None,
        eval_freq=0,
        save_freq=0,
        checkpoint_dir=str(tmp_path),
        exp_name="no_eval",
        use_wandb=False,
        event_cfg={},
        eval_event_cfg={},
        distributed=True,
        rank=rank,
        world_size=2,
    )

    assert trainer.eval_env is None
    assert not hasattr(trainer, "eval_event_manager")
    assert trainer.event_manager._env is trainer.env


def test_trainer_with_eval_env_creates_eval_event_manager(tmp_path: Path) -> None:
    """An enabled evaluation environment retains its own event manager."""
    eval_env = _AsyncAutoResetEnv()
    trainer = Trainer(
        policy=_DeterministicPolicy(),
        env=_AsyncAutoResetEnv(),
        algorithm=SimpleNamespace(device=torch.device("cpu")),
        buffer_size=2,
        batch_size=2,
        writer=None,
        eval_freq=10,
        save_freq=0,
        checkpoint_dir=str(tmp_path),
        exp_name="with_eval",
        use_wandb=False,
        eval_env=eval_env,
        eval_event_cfg={},
    )

    assert trainer.eval_event_manager._env is eval_env


@pytest.mark.parametrize("explicit_legacy", [False, True])
def test_evaluate_episodes_counts_actual_completions_and_restores_mode(
    explicit_legacy: bool,
) -> None:
    policy = _DeterministicPolicy()
    policy.train()
    env = _AsyncAutoResetEnv()

    result = evaluate_episodes(
        policy=policy,
        env=env,
        num_episodes=4,
        device="cpu",
        seed=123,
        **({"episode_allocation": "first_completed"} if explicit_legacy else {}),
    )

    assert policy.training
    assert env.last_seed == 123
    assert result["eval/avg_reward"] == 1.25
    assert result["eval/avg_length"] == 1.25
    assert result["eval/success_rate"] == 1.0
    assert result["eval/completed_episodes"] == 4
    assert result["eval/success_samples"] == 4
    assert result["eval/success_ci95_low"] == pytest.approx(0.5101091635)
    assert result["eval/success_ci95_high"] == pytest.approx(1.0)
    assert result["eval/metrics/terminal_step"] == 1.25
    assert "eval/metrics/final_position" not in result


@pytest.mark.parametrize(
    "episodes,expected_success,expected_length,expected_min,expected_max,controls",
    [
        (6, 2 / 3, 5 / 3, 2, 2, 6),
        (4, 0.5, 2.0, 1, 2, 6),
        (2, 0.5, 2.0, 0, 1, 3),
    ],
)
def test_balanced_evaluation_scores_fixed_quotas_despite_unequal_episode_lengths(
    episodes: int,
    expected_success: float,
    expected_length: float,
    expected_min: int,
    expected_max: int,
    controls: int,
) -> None:
    """Slow failures keep their assigned trials after fast rows exhaust quotas."""

    class TimedOutcomeEnv(_AsyncAutoResetEnv):
        def __init__(self) -> None:
            super().__init__()
            self.horizons = torch.tensor([3, 1, 1])
            self.physical_completions = torch.zeros(3, dtype=torch.long)
            self.control_steps = 0

        def step(self, action):
            observation, reward, done, truncated, info = super().step(action)
            self.physical_completions += done
            self.control_steps += 1
            info["success"] = torch.tensor([False, True, True])
            return observation, reward, done, truncated, info

    env = TimedOutcomeEnv()
    result = evaluate_episodes(
        policy=_DeterministicPolicy(),
        env=env,
        num_episodes=episodes,
        device="cpu",
        episode_allocation="balanced",
    )
    assert result["eval/completed_episodes"] == episodes
    assert result["eval/success_samples"] == episodes
    assert result["eval/success_rate"] == pytest.approx(expected_success)
    assert result["eval/avg_length"] == pytest.approx(expected_length)
    assert result["eval/avg_reward"] == pytest.approx(expected_length)
    assert result["eval/episodes_per_env_min"] == expected_min
    assert result["eval/episodes_per_env_max"] == expected_max
    assert env.control_steps == controls
    # Finished-quota rows still step/reset naturally; their additional fast
    # successes must not enter any scored reward, length, success or metric.
    torch.testing.assert_close(
        env.physical_completions, torch.tensor([controls // 3, controls, controls])
    )
    assert result["eval/metrics/terminal_step"] == pytest.approx(expected_length)


def test_evaluation_rejects_unknown_episode_allocation_before_reset() -> None:
    env = _AsyncAutoResetEnv()
    with pytest.raises(ValueError, match="episode_allocation"):
        evaluate_episodes(
            policy=_DeterministicPolicy(),
            env=env,
            num_episodes=1,
            device="cpu",
            episode_allocation="automatic",
        )
    assert env.last_seed is None


def test_policy_action_moves_to_environment_device_before_preprocessing() -> None:
    """The environment receives an unprocessed action on its own device."""
    env = _MixedDeviceEnv()

    converted = convert_policy_action_for_env(env, torch.ones(1, 2))

    assert converted.device == env.device


def test_trainer_rewinds_external_eval_event_manager() -> None:
    """Repeated model validation starts external event functors from eval_seed."""
    trainer = object.__new__(Trainer)
    trainer.policy = _DeterministicPolicy()
    trainer.eval_env = _AsyncAutoResetEnv()
    trainer.device = torch.device("cpu")
    trainer.eval_seed = 123
    trainer.eval_event_manager = MagicMock()
    trainer.eval_event_manager.available_modes = []
    trainer.global_step = 0
    trainer.eval_history = []
    trainer.last_eval_metrics = {}
    trainer.writer = None
    trainer.rank = 1
    trainer.use_wandb = False
    trainer.best_eval_metric = "missing"
    trainer.best_eval_value = None
    trainer.best_eval_mode = "max"
    trainer._finalize_eval_events = MagicMock()

    trainer._eval_once(num_episodes=1)
    trainer._eval_once(num_episodes=1)

    assert trainer.eval_event_manager.set_seed.call_args_list == [
        call(123),
        call(123),
    ]


def test_evaluation_preserves_separate_actor_and_critic_inputs() -> None:
    """Evaluation uses the observation groups configured on the policy."""

    class GroupedEnv(_AsyncAutoResetEnv):
        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            return {"policy": observation, "critic": observation + 10.0}, info

    class GroupedPolicy(_DeterministicPolicy):
        actor_obs_groups = ("policy",)
        critic_obs_groups = ("critic",)
        uses_separate_critic_obs = True

        def get_action(self, tensordict, deterministic: bool = False):
            torch.testing.assert_close(tensordict["obs"], torch.zeros(3, 1))
            torch.testing.assert_close(
                tensordict["critic_obs"], torch.full((3, 1), 10.0)
            )
            return super().get_action(tensordict, deterministic=deterministic)

    result = evaluate_episodes(
        policy=GroupedPolicy(),
        env=GroupedEnv(),
        num_episodes=1,
        device="cpu",
    )

    assert result["eval/avg_length"] == 1.0
