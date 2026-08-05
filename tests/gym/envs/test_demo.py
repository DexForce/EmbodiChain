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

"""Tests for segment-aware demonstration execution and annotations."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.demo import DemoSegment, execute_demo_episode
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv


class _SegmentedEnv:
    """Small environment stub that supports lazy two-segment planning."""

    def __init__(self) -> None:
        self.num_envs = 1
        self.state = 0
        self.actions: list[int] = []
        self.no_auto_reset_during_steps: list[bool] = []
        self.segment_results = []

    def create_demo_segments(self):
        yield DemoSegment(
            actions=(1, 2),
            name="pick_a",
            target_uid="object_a",
            instruction="place object a",
        )
        assert self.state == 2
        yield DemoSegment(
            actions=(3,),
            name="pick_b",
            target_uid="object_b",
            instruction="place object b",
        )

    def step(self, action: int):
        self.actions.append(action)
        self.state = action
        self.no_auto_reset_during_steps.append(self._demo_no_auto_reset)
        success = torch.tensor([action == 3])
        return (
            None,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {"success": success},
        )

    def is_task_success(self) -> torch.Tensor:
        return torch.tensor([self.state == 3])

    def _end_demo_segment_recording(self, result) -> None:
        self.segment_results.append(result)


def test_execute_demo_episode_runs_lazy_segments_as_one_episode() -> None:
    """A task can plan the second object after the first segment executes."""
    env = _SegmentedEnv()

    result = execute_demo_episode(env, episode_index=7)

    assert env.actions == [1, 2, 3]
    assert result.length == 3
    assert result.completed
    assert result.all_success
    assert result.terminal_reason == "success"
    assert [(item.start_step, item.end_step) for item in result.segments] == [
        (0, 2),
        (2, 3),
    ]
    assert [item.target_uid for item in result.segments] == ["object_a", "object_b"]
    assert all(env.no_auto_reset_during_steps)
    assert not env._demo_no_auto_reset


class _TerminatingEnv(_SegmentedEnv):
    def create_demo_segments(self):
        return (DemoSegment(actions=(1, 2, 3), name="pick"),)

    def step(self, action: int):
        self.actions.append(action)
        self.state = action
        self.no_auto_reset_during_steps.append(self._demo_no_auto_reset)
        terminated = torch.tensor([action == 2])
        return (
            None,
            torch.zeros(1),
            terminated,
            torch.zeros(1, dtype=torch.bool),
            {"success": terminated.clone()},
        )


def test_execute_demo_episode_stops_immediately_on_success_termination() -> None:
    """No action after a terminal transition leaks into the next episode."""
    env = _TerminatingEnv()

    result = execute_demo_episode(env)

    assert env.actions == [1, 2]
    assert result.length == 2
    assert result.completed
    assert result.all_success
    assert result.terminal_reason == "success"


class _TruncatingEnv(_TerminatingEnv):
    def step(self, action: int):
        self.actions.append(action)
        return (
            None,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            {"success": torch.ones(1, dtype=torch.bool)},
        )


def test_execute_demo_episode_never_accepts_truncated_rollout() -> None:
    """Truncation wins over a conflicting success flag and discards the episode."""
    env = _TruncatingEnv()

    result = execute_demo_episode(env)

    assert env.actions == [1]
    assert not result.completed
    assert not result.any_success
    assert result.terminal_reason == "truncated"


class _LegacyEnv(_SegmentedEnv):
    create_demo_segments = None

    def create_demo_action_list(self):
        return (3,)


def test_execute_demo_episode_adapts_legacy_action_list() -> None:
    """Existing tasks remain a one-segment episode without code changes."""
    env = _LegacyEnv()

    result = execute_demo_episode(env)

    assert result.all_success
    assert len(result.segments) == 1
    assert result.segments[0].name == "legacy"


def _make_rollout_buffer(num_envs: int, steps: int) -> TensorDict:
    return TensorDict(
        {
            "obs": {"state": torch.zeros(num_envs, steps, 2)},
            "actions": torch.zeros(num_envs, steps, 2),
            "rewards": torch.zeros(num_envs, steps),
            "valid": torch.zeros(num_envs, steps, dtype=torch.bool),
            "episode_step": torch.full((num_envs, steps), -1, dtype=torch.long),
            "segment_id": torch.full((num_envs, steps), -1, dtype=torch.long),
            "segment_step": torch.full((num_envs, steps), -1, dtype=torch.long),
            "segment_start": torch.zeros(num_envs, steps, dtype=torch.bool),
            "segment_end": torch.zeros(num_envs, steps, dtype=torch.bool),
            "terminated": torch.zeros(num_envs, steps, dtype=torch.bool),
            "truncated": torch.zeros(num_envs, steps, dtype=torch.bool),
        },
        batch_size=[num_envs, steps],
    )


class _RolloutWriterStub:
    """Attributes required by EmbodiedEnv's pure rollout writer method."""

    num_envs = 2
    _max_rollout_steps = 5
    _demo_active_segment_id = 4

    def __init__(self) -> None:
        self.rollout_buffer = _make_rollout_buffer(2, 5)
        self.rollout_steps = torch.tensor([0, 2], dtype=torch.long)
        self.current_rollout_step = 2
        self._demo_active_segment_start_steps = torch.tensor([0, 2])


def test_expert_rollout_writer_uses_independent_per_env_lengths() -> None:
    """Partial resets do not overwrite another environment's active episode."""
    env = _RolloutWriterStub()
    obs = TensorDict({"state": torch.tensor([[1.0, 1.0], [2.0, 2.0]])}, batch_size=[2])

    EmbodiedEnv._write_episode_rollout_step(
        env,
        obs=obs,
        action=torch.tensor([[3.0, 3.0], [4.0, 4.0]]),
        rewards=torch.tensor([0.5, 1.0]),
        terminateds=torch.tensor([False, True]),
        truncateds=torch.tensor([False, False]),
    )

    assert env.rollout_steps.tolist() == [1, 3]
    assert env.rollout_buffer["valid"][0, 0]
    assert env.rollout_buffer["valid"][1, 2]
    assert env.rollout_buffer["segment_id"][0, 0].item() == 4
    assert env.rollout_buffer["segment_step"][1, 2].item() == 0
    assert env.rollout_buffer["segment_end"][1, 2]
    assert env.current_rollout_step == 3


def test_clear_expert_rows_preserves_unrelated_environment() -> None:
    """Clearing a completed row is an actual in-place selective mutation."""
    env = _RolloutWriterStub()
    for key in env.rollout_buffer.keys(include_nested=True, leaves_only=True):
        value: Any = env.rollout_buffer[key]
        if value.dtype == torch.bool:
            value[:] = True
        else:
            value[:] = 5

    EmbodiedEnv._clear_expert_rollout_rows(env, torch.tensor([0]))

    assert not env.rollout_buffer["valid"][0].any()
    assert env.rollout_buffer["valid"][1].all()
    assert (env.rollout_buffer["segment_id"][0] == -1).all()
    assert (env.rollout_buffer["segment_id"][1] == 5).all()
