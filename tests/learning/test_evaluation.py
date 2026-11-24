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

from contextlib import nullcontext
import json
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
    infer_policy_action,
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


class _ManagedEventEnv(_AsyncAutoResetEnv):
    """CPU host that owns event dispatch and reusable recording drains."""

    def __init__(self) -> None:
        from embodichain.lab.gym.envs.managers.event_manager import EventManager

        super().__init__()
        self.event_manager = EventManager({}, env=self)
        self.flush_count = 0

    def add_events(self, event_cfg) -> None:
        self.event_manager.add_events(event_cfg)

    def reset(self, *, seed=None, options=None):
        observation = super().reset(seed=seed, options=options)
        self.event_manager.set_seed(seed)
        self.event_manager.reset()
        return observation

    def step(self, action):
        transition = super().step(action)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval")
        return transition

    def flush_recordings(self) -> None:
        self.flush_count += 1


def _event_trainer(tmp_path: Path, env, **kwargs) -> Trainer:
    from embodichain.learning.rl.policies import ActorCritic

    return Trainer(
        policy=ActorCritic(
            1, 1, torch.device("cpu"), torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        ),
        env=env,
        algorithm=SimpleNamespace(device=torch.device("cpu")),
        buffer_size=2,
        writer=None,
        eval_freq=0,
        save_freq=0,
        checkpoint_dir=str(tmp_path),
        exp_name="events",
        use_wandb=False,
        **kwargs,
    )


@pytest.mark.parametrize("preprocessing", ["transform", "normalizer"])
def test_inference_preprocessing_preserves_environment_observation(preprocessing):
    """Custom in-place preprocessing owns its input, including reused buffers."""
    from embodichain.learning.rl.policies import ActorCritic

    policy = ActorCritic(
        1,
        1,
        torch.device("cpu"),
        torch.nn.Identity(),
        torch.nn.Linear(1, 1),
        actor_normalizer=(
            torch.nn.ReLU(inplace=True) if preprocessing == "normalizer" else None
        ),
    )
    observation = torch.tensor([[-2.0], [2.0]])
    action = infer_policy_action(
        policy,
        observation,
        device="cpu",
        num_envs=2,
        observation_transform=(
            (lambda obs: obs.add_(1.0)) if preprocessing == "transform" else None
        ),
    )
    expected_action = (
        [[-1.0], [3.0]] if preprocessing == "transform" else [[0.0], [2.0]]
    )
    torch.testing.assert_close(action, torch.tensor(expected_action))
    torch.testing.assert_close(observation, torch.tensor([[-2.0], [2.0]]))


@pytest.mark.parametrize("default_device_context", [False, True])
def test_trainer_metric_means_transfer_only_one_scalar_batch(
    monkeypatch: pytest.MonkeyPatch, default_device_context: bool
) -> None:
    """Logging preserves mixed metric means without per-value host copies."""
    original_cpu = torch.Tensor.cpu
    transferred_shapes = []
    cpu_call_depth = 0

    def counted_cpu(value: torch.Tensor, *args, **kwargs):
        nonlocal cpu_call_depth
        # DeviceContext redispatches through this Python wrapper. Count the
        # outer request once; two sequential .cpu() calls must still count twice.
        if cpu_call_depth == 0:
            transferred_shapes.append(tuple(value.shape))
        cpu_call_depth += 1
        try:
            return original_cpu(value, *args, **kwargs)
        finally:
            cpu_call_depth -= 1

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    trainer = Trainer.__new__(Trainer)
    with torch.device("cpu") if default_device_context else nullcontext():
        snapshot = trainer._reduce_log_dicts(
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
        values = trainer._materialize_log_dicts([snapshot])[0]
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
    # A subsequent real batch remains a separate request: redispatch handling
    # must not hide another transfer or retain its reentrancy state.
    with torch.device("cpu") if default_device_context else nullcontext():
        snapshot = trainer._reduce_log_dicts({"metrics": {"next": torch.ones(2)}})
        assert trainer._materialize_log_dicts([snapshot])[0] == {"metrics/next": 1.0}
    assert transferred_shapes == [(5,), (1,)]


@pytest.mark.parametrize("logging_enabled", [True, False])
def test_trainer_rollout_logging_reuses_means_for_both_sinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, logging_enabled: bool
) -> None:
    """Each control step retains its tags/global step and shares scalar means."""
    trainer = Trainer(
        policy=_DeterministicPolicy(),
        env=SimpleNamespace(num_envs=2, reset=lambda: (torch.zeros(2, 1), {})),
        algorithm=SimpleNamespace(device=torch.device("cpu")),
        buffer_size=2,
        writer=MagicMock() if logging_enabled else None,
        eval_freq=0,
        save_freq=0,
        checkpoint_dir=str(tmp_path),
        exp_name="logging",
        use_wandb=logging_enabled,
    )
    trainer.global_step = 10
    wandb_log = MagicMock()
    monkeypatch.setattr("embodichain.learning.rl.utils.trainer.wandb.log", wandb_log)

    def collect(*, num_steps, rollout, on_step_callback):
        reused_metric = torch.tensor([1.0, 3.0])
        rollout["done"].zero_()
        for step in range(num_steps):
            rollout["reward"][:, step] = torch.tensor([1.0, 3.0])
            on_step_callback(
                rollout[:, step],
                {
                    "rewards": {"progress": reused_metric},
                    "metrics": {"grasped": torch.tensor([True, False])},
                },
            )
            reused_metric.add_(2)
        return rollout

    trainer.collector = SimpleNamespace(collect=collect)
    trainer._collect_rollout()
    assert trainer.global_step == 14
    if logging_enabled:
        assert trainer.writer.add_scalar.call_args_list == [
            call("rewards/progress", 2.0, 12),
            call("metrics/grasped", 0.5, 12),
            call("rewards/progress", 4.0, 14),
            call("metrics/grasped", 0.5, 14),
        ]
        assert wandb_log.call_args_list == [
            call({"rewards/progress": 2.0, "metrics/grasped": 0.5}, step=12),
            call({"rewards/progress": 4.0, "metrics/grasped": 0.5}, step=14),
        ]
    else:
        wandb_log.assert_not_called()


def test_training_episode_windows_preserve_completion_order_across_rollouts(
    tmp_path: Path,
) -> None:
    """Bounded histories retain the latest completions, including tied rows."""
    from embodichain.learning.rl.policies import ActorCritic

    class Env(_AsyncAutoResetEnv):
        def step(self, action):
            obs, _, terminated, truncated, info = super().step(action)
            return obs, torch.tensor([1.0, 10.0, 100.0]), terminated, truncated, info

    trainer = Trainer(
        policy=ActorCritic(
            1, 1, torch.device("cpu"), torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        ),
        env=Env(),
        algorithm=SimpleNamespace(device=torch.device("cpu")),
        buffer_size=4,
        writer=None,
        eval_freq=0,
        save_freq=0,
        checkpoint_dir=str(tmp_path),
        exp_name="episode_order",
        use_wandb=False,
    )
    for _ in range(17):
        trainer._collect_rollout()
        trainer.buffer.get(flatten=False)

    # Each row finishes at exact multiples of its horizon. Ties retain row order;
    # the 68-step run exceeds the history capacity and crosses 17 rollout edges.
    episodes = [
        (reward * horizon, horizon)
        for step in range(1, 69)
        for reward, horizon in ((1, 1), (10, 2), (100, 3))
        if step % horizon == 0
    ][-100:]
    assert list(trainer.ret_window) == [reward for reward, _ in episodes]
    assert list(trainer.len_window) == [length for _, length in episodes]
    torch.testing.assert_close(trainer.curr_ret, torch.tensor([0.0, 0.0, 200.0]))
    torch.testing.assert_close(
        trainer.curr_len, torch.tensor([0, 0, 2], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "event_cfg",
    [None, {}, SimpleNamespace(), {"unused": None}, SimpleNamespace(unused=None)],
)
@pytest.mark.parametrize("enable_eval", [False, True])
def test_plain_trainer_accepts_empty_events(
    tmp_path: Path, event_cfg: object, enable_eval: bool
) -> None:
    """Plain hosts train and evaluate without an event lifecycle when unused."""
    eval_env = _AsyncAutoResetEnv() if enable_eval else None
    trainer = _event_trainer(
        tmp_path,
        _AsyncAutoResetEnv(),
        eval_env=eval_env,
        event_cfg=event_cfg,
        eval_event_cfg=event_cfg,
    )
    trainer._collect_rollout()
    assert trainer.global_step == 6
    if enable_eval:
        assert trainer._eval_once(num_episodes=1)["eval/avg_reward"] == 1.0


@pytest.mark.parametrize("explicit_legacy", [False, True])
def test_evaluate_episodes_counts_actual_completions_and_restores_mode(
    explicit_legacy: bool,
) -> None:
    policy = _DeterministicPolicy()
    policy.train()
    policy.normalizer = torch.nn.BatchNorm1d(1).eval()
    policy.dropout = torch.nn.Dropout()
    env = _AsyncAutoResetEnv()
    episodes = []

    def record_episode(record):
        assert not any(module.training for module in policy.modules())
        episodes.append(record)

    result = evaluate_episodes(
        policy=policy,
        env=env,
        num_episodes=4,
        device="cpu",
        seed=123,
        on_scored_episode=record_episode,
        **({"episode_allocation": "first_completed"} if explicit_legacy else {}),
    )

    assert policy.training
    assert not policy.normalizer.training
    assert policy.dropout.training
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
    assert [record["global_episode_index"] for record in episodes] == list(range(4))
    assert [record["env_id"] for record in episodes] == [0, 0, 1, 0]
    assert [record["length"] for record in episodes] == [1, 1, 2, 1]
    assert [record["return"] for record in episodes] == [1.0, 1.0, 2.0, 1.0]
    assert all(record["success"] == 1.0 for record in episodes)
    assert all("final_position" not in record["metrics"] for record in episodes)
    json.dumps(episodes, allow_nan=False)


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
            self.owned_outcomes = []

        def step(self, action):
            observation, reward, done, truncated, info = super().step(action)
            self.physical_completions += done
            self.control_steps += 1
            info["success"] = torch.tensor([False, True, True])
            info["terminal_outcomes"] = {
                int(row): {
                    "env_id": int(row),
                    "physical": {"control": self.control_steps},
                }
                for row in done.nonzero(as_tuple=False).flatten()
            }
            self.owned_outcomes.extend(info["terminal_outcomes"].values())
            return observation, reward, done, truncated, info

    env = TimedOutcomeEnv()
    records = []
    outcomes = []
    result = evaluate_episodes(
        policy=_DeterministicPolicy(),
        env=env,
        num_episodes=episodes,
        device="cpu",
        episode_allocation="balanced",
        on_episode=outcomes.append,
        on_scored_episode=records.append,
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
    assert len(records) == episodes
    assert [outcome["env_id"] for outcome in outcomes] == [
        record["env_id"] for record in records
    ]
    outcomes[0]["physical"]["control"] = -1
    assert all(outcome["physical"]["control"] > 0 for outcome in env.owned_outcomes)
    assert [record["global_episode_index"] for record in records] == list(
        range(episodes)
    )
    quota, remainder = divmod(episodes, env.num_envs)
    assert [sum(record["env_id"] == row for record in records) for row in range(3)] == [
        quota + int(row < remainder) for row in range(3)
    ]
    for record in records:
        expected_steps = 3 if record["env_id"] == 0 else 1
        assert record["length"] == expected_steps
        assert record["return"] == float(expected_steps)
        assert record["success"] == float(record["env_id"] != 0)
        assert record["metrics"] == {"terminal_step": float(expected_steps)}
    json.dumps(records, allow_nan=False)


def test_episode_callback_omits_absent_success_and_restores_mode_on_failure() -> None:
    class UnlabelledEnv(_AsyncAutoResetEnv):
        def step(self, action):
            observation, reward, done, truncated, info = super().step(action)
            del info["success"]
            return observation, reward, done, truncated, info

    policy = _DeterministicPolicy()
    policy.eval()
    policy.normalizer = torch.nn.BatchNorm1d(1)
    records = []

    def reject(record: dict) -> None:
        assert not any(module.training for module in policy.modules())
        records.append(record)
        raise OSError("episode sink unavailable")

    with pytest.raises(OSError, match="episode sink unavailable"):
        evaluate_episodes(
            policy=policy,
            env=UnlabelledEnv(),
            num_episodes=4,
            device="cpu",
            on_scored_episode=reject,
        )
    assert not policy.training
    assert policy.normalizer.training
    assert len(records) == 1
    assert records[0]["global_episode_index"] == 0
    assert "success" not in records[0]


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


@pytest.mark.parametrize("reset_each_rollout", [False, True])
def test_trainer_training_dispatches_interval_events_once(
    tmp_path: Path, reset_each_rollout: bool
) -> None:
    """Host-owned callbacks reset with the environment and run once per step."""
    from embodichain.lab.gym.envs.managers.cfg import EventCfg
    from embodichain.lab.gym.envs.managers.manager_base import Functor

    env = _ManagedEventEnv()
    calls = []

    class Record(Functor):
        def reset(self, env_ids=None) -> None:
            self.calls = 0

        def __call__(self, env, env_ids):
            self.calls += 1
            calls.append((self.calls, env_ids.tolist()))

    trainer = _event_trainer(
        tmp_path,
        env,
        event_cfg={"record": EventCfg(func=Record, mode="interval", interval_step=1)},
    )
    trainer.collector.reset_every_rollout = reset_each_rollout
    for _ in range(2):
        trainer._collect_rollout()
        trainer.buffer.get(flatten=False)

    # Two 2-step rollouts, with all three rows selected on every invocation.
    counts = [1, 2, 1, 2] if reset_each_rollout else [1, 2, 3, 4]
    assert calls == [(count, [0, 1, 2]) for count in counts]
    assert trainer.global_step == 12


def test_trainer_rewinds_eval_events_and_preserves_host_lifecycle(
    tmp_path: Path,
) -> None:
    """Repeated evaluation rewinds interval/term state and deterministic streams."""
    from embodichain.lab.gym.envs.managers.cfg import EventCfg
    from embodichain.lab.gym.envs.managers.manager_base import Functor

    class Record(Functor):
        def reset(self, env_ids=None) -> None:
            self.calls = 0

        def __call__(self, env, env_ids, save_path):
            self.calls += 1
            env.records.append((self.calls, env_ids.tolist(), torch.rand(()).item()))

        def finalize(self, save_path) -> None:
            pytest.fail("The host retains its reusable event functors.")

    eval_env = _ManagedEventEnv()
    eval_env.records = []
    eval_env.flush_count = 0
    events = {
        "record": EventCfg(
            func=Record,
            mode="interval",
            interval_step=2,
            params={"save_path": str(tmp_path)},
        )
    }
    trainer = _event_trainer(
        tmp_path,
        _AsyncAutoResetEnv(),
        eval_env=eval_env,
        eval_seed=123,
        eval_event_cfg=SimpleNamespace(**events),
    )
    first = trainer._eval_once(num_episodes=8)
    first_records = list(eval_env.records)
    eval_env.records.clear()
    second = trainer._eval_once(num_episodes=8)

    # Eight completions take five controls, so the interval fires at 2 and 4.
    assert [(index, rows) for index, rows, _ in first_records] == [
        (1, [0, 1, 2]),
        (2, [0, 1, 2]),
    ]
    assert eval_env.records == first_records
    assert first == second
    assert eval_env.flush_count == 4


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_trainer_eval_failure_drains_host_recordings(
    tmp_path: Path, cleanup_fails: bool
) -> None:
    """Host cleanup runs after an event error without hiding the original error."""
    from embodichain.lab.gym.envs.managers.cfg import EventCfg

    class FailingHost(_ManagedEventEnv):
        def flush_recordings(self) -> None:
            super().flush_recordings()
            if cleanup_fails and self.flush_count == 2:
                raise RuntimeError("drain failed")

    def reject(env, env_ids):
        raise OSError("interval sink unavailable")

    eval_env = FailingHost()
    trainer = _event_trainer(
        tmp_path,
        _AsyncAutoResetEnv(),
        eval_env=eval_env,
        eval_event_cfg={
            "reject": EventCfg(func=reject, mode="interval", interval_step=1)
        },
    )
    with pytest.raises(OSError, match="interval sink unavailable") as failure:
        trainer._eval_once(num_episodes=1)

    assert eval_env.flush_count == 2
    assert trainer.eval_history == []
    if cleanup_fails:
        assert any("drain failed" in note for note in failure.value.__notes__)


@pytest.mark.parametrize("phase", ["train", "eval"])
def test_nonempty_trainer_events_require_host_lifecycle(
    tmp_path: Path, phase: str
) -> None:
    """Even interval-only declarations require a host that owns their lifecycle."""
    from embodichain.lab.gym.envs.managers.cfg import EventCfg

    events = {"interval": EventCfg(func=lambda env, env_ids: None, mode="interval")}
    kwargs = (
        {"event_cfg": events}
        if phase == "train"
        else {
            "eval_env": _AsyncAutoResetEnv(),
            "eval_event_cfg": SimpleNamespace(**events),
        }
    )
    with pytest.raises(ValueError, match=r"environment with add_events\(\)"):
        _event_trainer(tmp_path, _AsyncAutoResetEnv(), **kwargs)


@pytest.mark.parametrize("trainer_kind", ["standard", "differentiable"])
@pytest.mark.parametrize("best_mode", ["min", "max"])
def test_trainer_evaluation_reporting_preserves_best_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    trainer_kind: str,
    best_mode: str,
) -> None:
    """Both trainers retain raw history, finite logs, and their save boundary."""
    from embodichain.learning.rl import DifferentiableTrainer, DifferentiableTrainerCfg

    score = 8.0 if best_mode == "min" else 12.0
    reports = [
        {"eval/avg_reward": score, "eval/absent": float("nan")},
        {"eval/avg_reward": score},
        {"eval/avg_reward": float("nan")},
    ]
    if trainer_kind == "standard":
        trainer = object.__new__(Trainer)
        module = "embodichain.learning.rl.utils.trainer"
        trainer.device = torch.device("cpu")
        trainer.eval_seed = None
        trainer.rank = 0
        trainer.use_wandb = True
        trainer.best_eval_metric = "eval/avg_reward"
        trainer.best_eval_mode = best_mode
        trainer._pending_best = False
        evaluate = trainer._eval_once
    else:
        trainer = object.__new__(DifferentiableTrainer)
        module = "embodichain.learning.rl.differentiable_trainer"
        trainer.cfg = DifferentiableTrainerCfg(
            checkpoint_dir=str(tmp_path), use_wandb=True, best_eval_mode=best_mode
        )
        trainer.algorithm = SimpleNamespace(device=torch.device("cpu"))
        trainer.observation_normalizer = None
        evaluate = trainer._evaluate
    trainer.policy = _DeterministicPolicy()
    trainer.eval_env = _AsyncAutoResetEnv()
    trainer.global_step = 24
    trainer.eval_history = []
    trainer.last_eval_metrics = {}
    trainer.best_eval_value = 10.0
    trainer.best_checkpoint_path = None
    trainer.writer = MagicMock()
    trainer.save_checkpoint = MagicMock()
    monkeypatch.setattr(f"{module}.evaluate_episodes", MagicMock(side_effect=reports))
    wandb_log = MagicMock()
    monkeypatch.setattr(
        "embodichain.learning.rl._trainer_reporting.wandb.log", wandb_log
    )

    for _ in reports:
        evaluate()

    assert trainer.best_eval_value == score
    assert len(trainer.eval_history) == 3
    assert trainer.eval_history[0]["global_step"] == 24
    assert math.isnan(trainer.eval_history[0]["eval/absent"])
    assert math.isnan(trainer.last_eval_metrics["eval/avg_reward"])
    assert trainer.writer.add_scalar.call_args_list == [
        call("eval/avg_reward", score, 24),
        call("eval/avg_reward", score, 24),
    ]
    assert wandb_log.call_args_list == [
        call({"eval/avg_reward": score}, step=24),
        call({"eval/avg_reward": score}, step=24),
        call({}, step=24),
    ]
    if trainer_kind == "standard":
        assert trainer._pending_best
        trainer.save_checkpoint.assert_not_called()
    else:
        trainer.save_checkpoint.assert_called_once()
        assert trainer.best_checkpoint_path.endswith("_best_step_24.pt")


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
