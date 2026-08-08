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

import threading
from typing import Any

import pytest
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


class _StaggeredVectorEnv:
    """Two-row stub whose environments reach success on different steps."""

    def __init__(self) -> None:
        self.num_envs = 2
        self.step_count = 0
        self.actions: list[int] = []
        self.masked_actions: list[tuple[int, tuple[bool, ...]]] = []
        self.requested_second_segment = False

    def create_demo_segments(self):
        yield DemoSegment(actions=(1, 2, 3, 4), name="shared")
        self.requested_second_segment = True
        yield DemoSegment(actions=(5,), name="must_not_run")

    def _mask_demo_action(self, action: int, active_mask: tuple[bool, ...]) -> int:
        self.masked_actions.append((action, active_mask))
        return action

    def step(self, action: int):
        self.actions.append(action)
        self.step_count += 1
        terminated = torch.tensor(
            [self.step_count == 1, self.step_count == 3], dtype=torch.bool
        )
        return (
            None,
            torch.zeros(self.num_envs),
            terminated,
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"success": terminated.clone()},
        )

    def is_task_success(self) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.bool)


def test_execute_demo_episode_supports_staggered_vector_success() -> None:
    """A completed row freezes while unfinished rows continue their shared plan."""
    env = _StaggeredVectorEnv()

    result = execute_demo_episode(env)

    assert env.actions == [1, 2, 3]
    assert env.masked_actions == [(2, (False, True)), (3, (False, True))]
    assert not env.requested_second_segment
    assert result.length == 3
    assert result.lengths == (1, 3)
    assert result.completed_by_env == (True, True)
    assert result.terminal_reasons == ("success", "success")
    assert result.terminated == (True, True)
    assert result.all_success
    assert result.segments[0].end_steps == (1, 3)
    assert result.segments[0].successes == (True, True)


class _VectorFailureEnv(_StaggeredVectorEnv):
    def create_demo_segments(self):
        return (DemoSegment(actions=(1, 2, 3), name="shared"),)

    def step(self, action: int):
        self.actions.append(action)
        self.step_count += 1
        terminated = torch.tensor([self.step_count == 2, False], dtype=torch.bool)
        return (
            None,
            torch.zeros(self.num_envs),
            terminated,
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"success": torch.zeros(self.num_envs, dtype=torch.bool)},
        )


def test_vector_failure_aborts_peer_and_preserves_per_env_reason() -> None:
    """Batch-atomic failure distinguishes the failing row from its peer."""
    env = _VectorFailureEnv()

    result = execute_demo_episode(env)

    assert env.actions == [1, 2]
    assert not result.completed
    assert result.terminal_reason == "failure"
    assert result.terminal_reasons == ("failure", "batch_aborted")
    assert result.terminated == (True, False)
    assert result.success == (False, False)
    assert result.lengths == (2, 2)


class _ValidatedSegmentEnv(_SegmentedEnv):
    def __init__(self, validation: bool) -> None:
        super().__init__()
        self.validation = validation
        self.requested_second_segment = False

    def create_demo_segments(self):
        yield DemoSegment(
            actions=(1,),
            name="pick",
            validator=lambda: torch.tensor([self.validation]),
        )
        self.requested_second_segment = True
        yield DemoSegment(actions=(3,), name="place")


def test_segment_validator_stops_invalid_lazy_plan() -> None:
    """Subtask validation is independent from episode-level Gym termination."""
    env = _ValidatedSegmentEnv(validation=False)

    result = execute_demo_episode(env)

    assert env.actions == [1]
    assert not env.requested_second_segment
    assert not result.completed
    assert result.terminal_reason == "segment_validation_failed"
    assert result.terminal_reasons == ("segment_validation_failed",)
    assert result.segments[0].failure_reason == "segment_validation_failed"


def test_segment_validator_allows_next_lazy_segment() -> None:
    """A validated subtask advances the lazy planner without abusing terminated."""
    env = _ValidatedSegmentEnv(validation=True)

    result = execute_demo_episode(env)

    assert env.actions == [1, 3]
    assert env.requested_second_segment
    assert result.all_success
    assert [segment.name for segment in result.segments] == ["pick", "place"]


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


class _ConflictingTerminalEnv(_TerminatingEnv):
    def step(self, action: int):
        self.actions.append(action)
        return (
            None,
            torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            {
                "success": torch.ones(1, dtype=torch.bool),
                "fail": torch.zeros(1, dtype=torch.bool),
            },
        )


def test_terminated_and_truncated_flags_are_both_sticky() -> None:
    """A conflicting Gym transition keeps both flags while truncation wins."""
    result = execute_demo_episode(_ConflictingTerminalEnv())

    assert result.terminated == (True,)
    assert result.truncated == (True,)
    assert result.success == (False,)
    assert result.terminal_reason == "truncated"


class _SuccessAndFailureEnv(_TerminatingEnv):
    def step(self, action: int):
        self.actions.append(action)
        return (
            None,
            torch.zeros(1),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {
                "success": torch.ones(1, dtype=torch.bool),
                "fail": torch.ones(1, dtype=torch.bool),
            },
        )


def test_explicit_failure_wins_over_conflicting_success() -> None:
    """The task failure signal cannot be committed as a successful episode."""
    result = execute_demo_episode(_SuccessAndFailureEnv())

    assert not result.completed
    assert result.success == (False,)
    assert result.terminal_reason == "failure"


class _InactiveStaleTruncationEnv(_StaggeredVectorEnv):
    def create_demo_segments(self):
        return (DemoSegment(actions=(1, 2), name="shared"),)

    def step(self, action: int):
        self.actions.append(action)
        self.step_count += 1
        if self.step_count == 1:
            terminated = torch.tensor([True, False])
            truncated = torch.tensor([False, False])
            success = torch.tensor([True, False])
        else:
            terminated = torch.tensor([False, True])
            truncated = torch.tensor([True, False])
            success = torch.tensor([True, False])
        return None, torch.zeros(2), terminated, truncated, {"success": success}


def test_inactive_stale_truncation_does_not_change_active_failure_reason() -> None:
    """Signals from a frozen row cannot relabel a peer's current failure."""
    result = execute_demo_episode(_InactiveStaleTruncationEnv())

    assert result.terminal_reason == "failure"
    assert result.terminal_reasons == ("success", "failure")
    assert result.truncated == (False, False)


class _VectorValidatorEnv(_StaggeredVectorEnv):
    def create_demo_segments(self):
        return (
            DemoSegment(
                actions=(1,),
                name="validated",
                validator=lambda: torch.tensor([True, False]),
            ),
        )

    def step(self, action: int):
        self.actions.append(action)
        return (
            None,
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {"success": torch.zeros(2, dtype=torch.bool)},
        )


def test_validator_batch_abort_has_consistent_peer_status() -> None:
    """A batch-aborted peer is not simultaneously marked segment-successful."""
    result = execute_demo_episode(_VectorValidatorEnv())

    assert result.segments[0].successes == (False, False)
    assert result.segments[0].failure_reasons == (
        "batch_aborted",
        "segment_validation_failed",
    )
    assert result.segments[0].failure_reason == "segment_validation_failed"


class _CancellationEnv(_ValidatedSegmentEnv):
    def __init__(self) -> None:
        super().__init__(validation=True)
        self.validator_called = False

    def create_demo_segments(self):
        yield DemoSegment(
            actions=(1,),
            name="first",
            validator=self._validate,
        )
        self.requested_second_segment = True
        yield DemoSegment(actions=(3,), name="second")

    def _validate(self) -> torch.Tensor:
        self.validator_called = True
        return torch.ones(1, dtype=torch.bool)


def test_cancellation_after_last_action_does_not_advance_lazy_plan() -> None:
    """Cancellation is observed before validation or requesting another segment."""
    env = _CancellationEnv()

    result = execute_demo_episode(env, should_stop=lambda: bool(env.actions))

    assert env.actions == [1]
    assert not env.validator_called
    assert not env.requested_second_segment
    assert result.terminal_reason == "interrupted"
    assert result.terminal_reasons == ("interrupted",)


class _LegacyEnv(_SegmentedEnv):
    create_demo_segments = EmbodiedEnv.create_demo_segments

    def create_demo_action_list(self):
        return (3,)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        # Real EmbodiedEnv instances always expose compute_task_state() in
        # info, while legacy expert tasks often only override is_task_success().
        info["success"] = torch.tensor([False])
        return obs, reward, terminated, truncated, info


def test_execute_demo_episode_adapts_legacy_action_list() -> None:
    """Existing tasks remain a one-segment episode without code changes."""
    env = _LegacyEnv()

    result = execute_demo_episode(env)

    assert result.all_success
    assert len(result.segments) == 1
    assert result.segments[0].name == "legacy"


class _NormalizingSegmentedEnv(_SegmentedEnv):
    def create_demo_segments(self):
        return (DemoSegment(actions=(1, 2, 3), name="normalized"),)

    def _normalize_demo_action(self, action: int) -> int:
        return action + 10

    def is_task_success(self) -> torch.Tensor:
        return torch.tensor([self.state == 13])


def test_execute_demo_episode_normalizes_segment_actions() -> None:
    """New segment planners use the same action normalization hook as legacy plans."""
    env = _NormalizingSegmentedEnv()

    result = execute_demo_episode(env)

    assert result.all_success
    assert env.actions == [11, 12, 13]


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


def test_expert_rollout_writer_freezes_inactive_demo_row() -> None:
    """Sticky terminal rows do not receive frames from later shared actions."""
    env = _RolloutWriterStub()
    env._demo_active_mask = torch.tensor([False, True])
    obs = TensorDict({"state": torch.tensor([[1.0, 1.0], [2.0, 2.0]])}, batch_size=[2])

    EmbodiedEnv._write_episode_rollout_step(
        env,
        obs=obs,
        action=torch.tensor([[3.0, 3.0], [4.0, 4.0]]),
        rewards=torch.tensor([0.5, 1.0]),
        terminateds=torch.zeros(2, dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
    )

    assert env.rollout_steps.tolist() == [0, 3]
    assert not env.rollout_buffer["valid"][0].any()
    assert env.rollout_buffer["valid"][1, 2]


class _RobotQposStub:
    def get_qpos(self) -> torch.Tensor:
        return torch.tensor([[0.1, 0.2], [0.3, 0.4]])


class _ActionMaskStub:
    num_envs = 2
    active_joint_ids = [0, 1]
    robot = _RobotQposStub()
    _demo_active_mask = torch.tensor([False, True])


def test_processed_qpos_action_holds_inactive_row() -> None:
    """Inactive qpos-controlled rows hold their measured joint position."""
    env = _ActionMaskStub()

    masked = EmbodiedEnv._mask_processed_demo_action(
        env, torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    )

    assert torch.allclose(masked[0], torch.tensor([0.1, 0.2]))
    assert torch.allclose(masked[1], torch.tensor([8.0, 8.0]))


def test_processed_qpos_action_supports_full_robot_layout() -> None:
    """A full-DOF IK command holds measured full qpos for inactive rows."""
    env = _ActionMaskStub()
    env.active_joint_ids = [0]

    masked = EmbodiedEnv._mask_processed_demo_action(
        env, torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    )

    assert torch.allclose(masked[0], torch.tensor([0.1, 0.2]))
    assert torch.allclose(masked[1], torch.tensor([8.0, 8.0]))


class _FullDofRobotStub:
    def __init__(self) -> None:
        self.command = None
        self.joint_ids = None

    def get_qpos(self) -> torch.Tensor:
        return torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    def set_qpos(self, qpos: torch.Tensor, joint_ids: list[int]) -> None:
        self.command = qpos
        self.joint_ids = joint_ids


def test_full_dof_hold_is_sliced_before_active_joint_step() -> None:
    """Masking and application agree when IK returns full robot qpos."""
    env = type(
        "FullDofActionStub",
        (),
        {
            "num_envs": 2,
            "device": torch.device("cpu"),
            "active_joint_ids": [0, 2],
            "_demo_active_mask": torch.tensor([False, True]),
            "robot": _FullDofRobotStub(),
        },
    )()
    action = TensorDict(
        {"qpos": torch.tensor([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0]])},
        batch_size=[2],
    )

    masked = EmbodiedEnv._mask_processed_demo_action(env, action)
    applied = EmbodiedEnv._step_action(env, masked)

    assert torch.allclose(
        env.robot.command,
        torch.tensor([[0.1, 0.3], [8.0, 8.0]]),
    )
    assert env.robot.joint_ids == [0, 2]
    assert applied["qpos"].shape == (2, 2)


def test_processed_velocity_action_zeros_inactive_row() -> None:
    """Inactive velocity-controlled rows receive a no-op command."""
    env = _ActionMaskStub()
    action = TensorDict(
        {"qvel": torch.tensor([[9.0, 9.0], [8.0, 8.0]])}, batch_size=[2]
    )

    masked = EmbodiedEnv._mask_processed_demo_action(env, action)

    assert torch.equal(masked["qvel"][0], torch.zeros(2))
    assert torch.equal(masked["qvel"][1], torch.tensor([8.0, 8.0]))


class _TrajectoryRobotStub:
    def get_local_pose(self) -> torch.Tensor:
        return torch.ones(2, 7)

    def get_qpos(self) -> torch.Tensor:
        return torch.tensor([[0.1, 0.2], [0.3, 0.4]])


class _TrajectoryWriterStub:
    num_envs = 2
    device = torch.device("cpu")

    def __init__(self) -> None:
        self._traj_buffer = TensorDict(
            {
                "states": {
                    "robot": {
                        "root_pose": torch.zeros(2, 4, 7),
                        "qpos": torch.zeros(2, 4, 2),
                    }
                },
                "actions": torch.zeros(2, 4, 2),
            },
            batch_size=[2, 4],
        )
        self._traj_steps = torch.tensor([1, 1])
        self._traj_raw_action = torch.tensor([[9.0, 9.0], [8.0, 8.0]])
        self._demo_active_mask = torch.tensor([False, True])
        self.robot = _TrajectoryRobotStub()
        self.sim = type(
            "SimulationStub",
            (),
            {"_articulations": {}, "_rigid_objects": {}},
        )()


def test_trajectory_writer_freezes_inactive_demo_row() -> None:
    """Trajectory cursors obey the same sticky activity mask as rollout rows."""
    env = _TrajectoryWriterStub()

    EmbodiedEnv._write_trajectory_step(env)

    assert env._traj_steps.tolist() == [1, 2]
    assert torch.equal(
        env._traj_buffer["states"]["robot"]["qpos"][0, 1], torch.zeros(2)
    )
    assert torch.equal(
        env._traj_buffer["states"]["robot"]["qpos"][1, 1],
        torch.tensor([0.3, 0.4]),
    )


def test_success_status_freezes_after_staggered_demo_completion() -> None:
    """Later stale done signals cannot clear an already successful frozen row."""
    env = type(
        "SuccessStatusStub",
        (),
        {
            "episode_success_status": torch.tensor([True, False]),
            "_demo_no_auto_reset": True,
            "_demo_active_mask": torch.tensor([False, True]),
        },
    )()

    EmbodiedEnv._update_episode_success_status(
        env,
        {"success": torch.tensor([False, True])},
        torch.tensor([True, True]),
    )

    assert env.episode_success_status.tolist() == [True, True]


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


def test_legacy_metadata_reports_consistent_episode_success() -> None:
    """Fallback metadata agrees at the episode and segment levels."""
    env = _RolloutWriterStub()
    env._demo_episode_metadata = [
        {
            "schema_version": 2,
            "episode_index": 0,
            "env_id": env_id,
            "length": 0,
            "completed": False,
            "success": False,
            "terminated": False,
            "truncated": False,
            "terminal_reason": "unknown",
            "segments": [],
        }
        for env_id in range(env.num_envs)
    ]
    env.episode_success_status = torch.tensor([False, True])
    env._task_success = torch.tensor([False, False])

    metadata = EmbodiedEnv.get_demo_episode_metadata(env, 1)

    assert metadata["completed"]
    assert metadata["success"]
    assert metadata["terminal_reason"] == "success"
    assert metadata["segments"][0]["success"]


class _CloseDependencyStub:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.error = error

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error


class _CloseEnvStub:
    def __init__(self, *, finalize_error: Exception | None = None) -> None:
        self._closed = False
        self._close_error = None
        self._close_lock = threading.RLock()
        self.discard = _CloseDependencyStub()
        self.finalize = _CloseDependencyStub(error=finalize_error)
        self.report = _CloseDependencyStub()
        self.destroy = _CloseDependencyStub()
        self.dataset_manager = type(
            "DatasetManagerStub", (), {"finalize": self.finalize}
        )()
        self._profiler = type("ProfilerStub", (), {"report": self.report})()
        self.sim = type("SimulationStub", (), {"destroy": self.destroy})()

    def _discard_pending_recordings(self) -> None:
        self.discard()


def test_close_is_idempotent_and_never_commits_pending_data() -> None:
    """Only the first close aborts pending data and finalizes committed writes."""
    env = _CloseEnvStub()

    EmbodiedEnv.close(env)
    EmbodiedEnv.close(env)

    assert len(env.discard.calls) == 1
    assert len(env.finalize.calls) == 1
    assert len(env.report.calls) == 1
    assert env.destroy.calls == [((), {})]


def test_close_propagates_durability_failure_before_process_exit() -> None:
    """Recorder failures are observable and simulator cleanup cannot hide them."""
    env = _CloseEnvStub(finalize_error=OSError("disk full"))

    with pytest.raises(RuntimeError, match="disk full"):
        EmbodiedEnv.close(env)

    with pytest.raises(RuntimeError, match="disk full"):
        EmbodiedEnv.close(env)

    assert env.destroy.calls == [((), {"exit_process": False})]
    assert len(env.discard.calls) == 1
    assert len(env.finalize.calls) == 1
