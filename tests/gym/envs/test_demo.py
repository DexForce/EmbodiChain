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
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.demo import (
    DemoExecutionCfg,
    DemoSegment,
    DemoSegmentResult,
    execute_demo_episode,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.types import ControllerAction


def test_demo_segment_result_owns_json_safe_lifecycle_metadata() -> None:
    metadata = {
        "runtime": {"status": "completed"},
        "validation": {"accepted_mask": [True, False]},
    }
    result = DemoSegmentResult(
        segment_id=0,
        name="place",
        start_step=0,
        end_step=2,
        success=False,
        metadata=metadata,
    )

    metadata["runtime"]["status"] = "mutated"
    exported = result.to_metadata()
    exported["metadata"]["validation"]["accepted_mask"][0] = False

    assert result.metadata["runtime"]["status"] == "completed"
    assert result.metadata["validation"]["accepted_mask"] == [True, False]


def test_demo_segment_result_rejects_non_json_metadata() -> None:
    with pytest.raises(TypeError, match="non-JSON value Tensor"):
        DemoSegmentResult(
            segment_id=0,
            name="place",
            start_step=0,
            end_step=1,
            success=True,
            metadata={"mask": torch.tensor([True])},
        )


@pytest.mark.parametrize(
    ("validation", "expected"),
    [
        (
            {
                "runtime_success_mask": [False],
                "post_policy_success_mask": None,
                "validators": [],
                "accepted_mask": [False],
            },
            "runtime_failed",
        ),
        (
            {
                "runtime_success_mask": [True],
                "post_policy_success_mask": [False],
                "validators": [],
                "accepted_mask": [False],
            },
            "post_policy_failed",
        ),
        (
            {
                "runtime_success_mask": [True],
                "post_policy_success_mask": [True],
                "validators": [{"result_mask": [False]}],
                "accepted_mask": [False],
            },
            "validation_failed",
        ),
    ],
)
def test_demo_segment_result_preserves_first_authoritative_failure_phase(
    validation: dict[str, object], expected: str
) -> None:
    result = DemoSegmentResult(
        segment_id=0,
        name="place",
        start_step=0,
        end_step=2,
        success=False,
        failure_reason="segment_validation_failed",
        metadata={"validation": validation},
        active=(True,),
        start_steps=(0,),
        end_steps=(2,),
        successes=(False,),
        failure_reasons=("segment_validation_failed",),
    )

    assert result.outcome_kind == expected
    assert result.outcome_kinds == (expected,)
    assert result.to_metadata(0)["outcome_kind"] == expected


def test_demo_execution_cfg_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        DemoExecutionCfg(mode="resume")  # type: ignore[arg-type]


def test_demo_execution_cfg_rejects_failed_fragment_policy_in_continuous_mode() -> None:
    with pytest.raises(ValueError, match="only valid in segment_fragments"):
        DemoExecutionCfg(save_failed_fragments=True)


def _controller_action_env() -> EmbodiedEnv:
    env = object.__new__(EmbodiedEnv)
    env._num_envs = 2
    env._traj_buffer = None
    env.action_manager = Mock(
        process_action=Mock(side_effect=lambda action, mode: action)
    )
    env._demo_no_auto_reset = False
    env.active_joint_ids = [0, 1, 2]
    env.robot = Mock()
    env.robot.get_qpos.return_value = torch.zeros(2, 3)
    env.sim = Mock(device=torch.device("cpu"))
    return env


def test_embodied_env_skips_preprocessing_for_controller_action() -> None:
    env = _controller_action_env()
    action = ControllerAction(value=torch.ones(2, 3))

    normalized = env._normalize_demo_action(action)
    controller = env._preprocess_action(normalized)

    assert isinstance(normalized, ControllerAction)
    assert normalized is not action
    assert torch.equal(controller, action.value)
    env.action_manager.process_action.assert_not_called()


def test_embodied_env_preprocesses_raw_action_before_controller_validation() -> None:
    env = _controller_action_env()
    raw_action = torch.ones(2, 3)

    controller = env._preprocess_action(raw_action)

    assert torch.equal(controller, raw_action)
    env.action_manager.process_action.assert_called_once_with(raw_action, mode="pre")


def test_embodied_env_applies_post_terms_to_controller_action() -> None:
    env = _controller_action_env()
    action = ControllerAction(value=torch.ones(2, 3))

    controller = env._preprocess_action(action)
    env._postprocess_action(controller)

    env.action_manager.process_action.assert_called_once_with(controller, mode="post")


def test_embodied_env_validates_controller_action_batch_size() -> None:
    env = _controller_action_env()
    action = ControllerAction(value=torch.ones(1, 3))

    with pytest.raises(ValueError, match=r"shape \(num_envs, D\)"):
        env._preprocess_action(action)


def test_embodied_env_rejects_unsupported_controller_action_key() -> None:
    env = _controller_action_env()
    action = ControllerAction(
        value=TensorDict({"eef_pose": torch.ones(2, 7)}, batch_size=[2])
    )

    with pytest.raises(ValueError, match="must contain at least one"):
        env._preprocess_action(action)


def test_embodied_env_preserves_controller_action_auxiliary_fields() -> None:
    env = _controller_action_env()
    action = ControllerAction(
        value=TensorDict(
            {
                "qpos": torch.ones(2, 3),
                "ik_success": torch.tensor([True, False]),
            },
            batch_size=[2],
        )
    )

    controller = env._preprocess_action(action)

    assert torch.equal(controller["ik_success"], torch.tensor([True, False]))


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


def test_execute_demo_episode_exposes_declared_progress_total_for_lazy_actions() -> (
    None
):
    """A progress wrapper can render a total without consuming a generator."""

    class _ProgressTotalEnv(_SegmentedEnv):
        def create_demo_segments(self):
            def actions():
                yield from (1, 2, 3)

            return (
                DemoSegment(
                    actions=actions(),
                    name="move_cube",
                    progress_total_steps=3,
                ),
            )

    totals: list[int] = []

    def progress(actions, description: str):
        del description
        totals.append(len(actions))
        return actions

    result = execute_demo_episode(_ProgressTotalEnv(), progress=progress)

    assert result.all_success
    assert totals == [3]


class _LifecycleMetadataEnv:
    """Populate one shared metadata mapping at lazy lifecycle boundaries."""

    def __init__(self) -> None:
        self.num_envs = 1
        self.lifecycle = {"runtime": None, "validation": None}

    def create_demo_segments(self):
        def actions():
            yield 1
            self.lifecycle["runtime"] = {"status": "completed"}

        def validate() -> bool:
            self.lifecycle["validation"] = {"accepted_mask": [True]}
            return True

        return (
            DemoSegment(
                actions=actions(),
                name="lifecycle",
                metadata=self.lifecycle,
                validator=validate,
            ),
        )

    def step(self, action: int):
        del action
        return (
            None,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {"success": torch.tensor([True])},
        )

    def is_task_success(self) -> torch.Tensor:
        return torch.tensor([True])


class _EmptySuccessfulSegmentEnv:
    """Expose an empty ordinary segment whose callbacks otherwise succeed."""

    num_envs = 1

    def __init__(self) -> None:
        self.validator_calls = 0
        self.step_calls = 0

    def create_demo_segments(self):
        return (
            DemoSegment(
                actions=(),
                name="empty",
                validator=self._validate,
            ),
        )

    def _validate(self) -> bool:
        self.validator_calls += 1
        return True

    def step(self, action: object):
        del action
        self.step_calls += 1
        raise AssertionError("An empty segment must not call env.step().")

    @staticmethod
    def is_task_success() -> torch.Tensor:
        return torch.tensor([True])


def test_execute_demo_episode_snapshots_finalized_lifecycle_metadata() -> None:
    env = _LifecycleMetadataEnv()

    result = execute_demo_episode(env)
    env.lifecycle["runtime"]["status"] = "mutated"

    assert result.segments[0].metadata == {
        "runtime": {"status": "completed"},
        "validation": {"accepted_mask": [True]},
    }


def test_empty_ordinary_segment_keeps_existing_empty_segment_guard() -> None:
    env = _EmptySuccessfulSegmentEnv()

    result = execute_demo_episode(env)

    assert env.step_calls == 0
    assert env.validator_calls == 0
    assert not result.completed
    assert result.terminal_reason == "empty_segment"
    assert result.segments[0].failure_reason == "empty_segment"


class _GeneratorFailureEnv:
    """Raise between lazy actions and expose an emergency hold callback."""

    def __init__(self) -> None:
        self.num_envs = 1
        self.actions: list[int] = []
        self.abort_calls: list[tuple[str, bool]] = []

    def create_demo_segments(self):
        def actions():
            yield 1
            raise ValueError("planner stream failed")

        def abort(reason: str, *, last_action_consumed: bool):
            self.abort_calls.append((reason, last_action_consumed))
            yield 0

        return (DemoSegment(actions=actions(), abort_actions=abort),)

    def step(self, action: int):
        self.actions.append(action)
        return (
            None,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )


def test_action_generator_failure_safe_stops_before_propagating() -> None:
    env = _GeneratorFailureEnv()

    with pytest.raises(RuntimeError, match="action generation") as error:
        execute_demo_episode(env)

    assert isinstance(error.value.__cause__, ValueError)
    assert env.actions == [1, 0]
    assert env.abort_calls == [("action_generation_failed", True)]


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


class _RowIndependentFailureEnv(_VectorFailureEnv):
    def create_demo_segments(self):
        return (
            DemoSegment(
                actions=(1, 2, 3),
                name="shared",
                failure_policy="row_independent",
            ),
        )


def test_row_independent_failure_freezes_only_failed_environment() -> None:
    env = _RowIndependentFailureEnv()

    result = execute_demo_episode(env)

    assert env.actions == [1, 2, 3]
    assert env.masked_actions == [(3, (False, True))]
    assert result.completed_by_env == (False, True)
    assert result.terminal_reasons == ("failure", "success")
    assert result.success == (False, True)
    assert result.lengths == (2, 3)


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


class _RowIndependentValidatorEnv(_VectorValidatorEnv):
    def create_demo_segments(self):
        return (
            DemoSegment(
                actions=(1,),
                name="validated",
                validator=lambda: torch.tensor([True, False]),
                failure_policy="row_independent",
            ),
        )


def test_row_independent_validator_keeps_accepted_peer_active() -> None:
    result = execute_demo_episode(_RowIndependentValidatorEnv())

    assert result.segments[0].successes == (True, False)
    assert result.segments[0].failure_reasons == (
        None,
        "segment_validation_failed",
    )
    assert result.completed_by_env == (True, False)
    assert result.terminal_reasons == ("success", "segment_validation_failed")


def test_demo_segment_rejects_unknown_failure_policy() -> None:
    with pytest.raises(ValueError, match="failure_policy"):
        DemoSegment(actions=(1,), failure_policy="continue")


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
    metadata = {"dataset": {"instruction": {"lang": "Complete the legacy task"}}}

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
    assert result.segments[0].instruction == "Complete the legacy task"


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
            "segment_accepted": torch.zeros(num_envs, steps, dtype=torch.bool),
            "segment_attempt_id": torch.full((num_envs, steps), -1, dtype=torch.long),
            "continuity_id": torch.full((num_envs, steps), -1, dtype=torch.long),
            "terminated": torch.zeros(num_envs, steps, dtype=torch.bool),
            "truncated": torch.zeros(num_envs, steps, dtype=torch.bool),
        },
        batch_size=[num_envs, steps],
    )


class _RolloutWriterStub:
    """Attributes required by EmbodiedEnv's pure rollout writer method."""

    num_envs = 2
    device = torch.device("cpu")
    _max_rollout_steps = 5
    _demo_active_segment_id = 4
    _seed_expert_observations = EmbodiedEnv._seed_expert_observations

    def __init__(self) -> None:
        self.rollout_buffer = _make_rollout_buffer(2, 5)
        self.rollout_steps = torch.tensor([0, 2], dtype=torch.long)
        self.current_rollout_step = 2
        self._demo_attempt_id = 3
        self._demo_continuity_id = 0
        self._demo_active_segment_start_steps = torch.tensor([0, 2])
        self.rollout_buffer["obs"]["state"][0, 0] = 10.0
        self.rollout_buffer["obs"]["state"][1, 2] = 20.0


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
    assert env.rollout_buffer["segment_attempt_id"][0, 0].item() == 3
    assert env.rollout_buffer["continuity_id"][1, 2].item() == 0
    assert torch.equal(
        env.rollout_buffer["obs"]["state"][0, 0], torch.tensor([10.0, 10.0])
    )
    assert torch.equal(
        env.rollout_buffer["obs"]["state"][1, 2], torch.tensor([20.0, 20.0])
    )
    assert torch.equal(
        env.rollout_buffer["obs"]["state"][0, 1], torch.tensor([1.0, 1.0])
    )
    assert torch.equal(
        env.rollout_buffer["obs"]["state"][1, 3], torch.tensor([2.0, 2.0])
    )
    assert env.current_rollout_step == 3


def test_end_segment_retroactively_annotates_accepted_frame_spans() -> None:
    """The bridge result, not a new evaluator, qualifies every segment frame."""
    env = _RolloutWriterStub()
    env._demo_segment_participants = torch.tensor([True, True])
    env._demo_active_segment_start_steps = torch.tensor([0, 0])
    env._demo_active_rollout_start_steps = torch.tensor([0, 0])
    env._demo_steps = torch.tensor([2, 2])
    env.rollout_steps = torch.tensor([2, 2])
    env._demo_episode_metadata = [{"segments": []}, {"segments": []}]
    env.rollout_buffer["valid"][:, :2] = True

    result = DemoSegmentResult(
        segment_id=0,
        name="pick",
        start_step=0,
        end_step=2,
        success=False,
        active=(True, True),
        start_steps=(0, 0),
        end_steps=(2, 2),
        successes=(True, False),
        failure_reasons=(None, "segment_validation_failed"),
        attempt_id=3,
        continuity_id=0,
    )

    EmbodiedEnv._end_demo_segment_recording(env, result)

    assert env.rollout_buffer["segment_accepted"][0, :2].all()
    assert not env.rollout_buffer["segment_accepted"][1, :2].any()
    assert env.rollout_buffer["segment_end"][:, 1].all()
    assert env._demo_episode_metadata[0]["segments"][0]["outcome_kind"] == ("succeeded")
    assert env._demo_episode_metadata[1]["segments"][0]["outcome_kind"] == (
        "validation_failed"
    )


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


def test_controller_qpos_action_holds_inactive_row() -> None:
    """Inactive qpos-controlled rows hold their measured joint position."""
    env = _ActionMaskStub()

    masked = EmbodiedEnv._mask_controller_demo_action(
        env, torch.tensor([[9.0, 9.0], [8.0, 8.0]])
    )

    assert torch.allclose(masked[0], torch.tensor([0.1, 0.2]))
    assert torch.allclose(masked[1], torch.tensor([8.0, 8.0]))


def test_controller_qpos_action_supports_full_robot_layout() -> None:
    """A full-DOF IK command holds measured full qpos for inactive rows."""
    env = _ActionMaskStub()
    env.active_joint_ids = [0]

    masked = EmbodiedEnv._mask_controller_demo_action(
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

    masked = EmbodiedEnv._mask_controller_demo_action(env, action)
    applied = EmbodiedEnv._step_action(env, masked)

    assert torch.allclose(
        env.robot.command,
        torch.tensor([[0.1, 0.3], [8.0, 8.0]]),
    )
    assert env.robot.joint_ids == [0, 2]
    assert applied["qpos"].shape == (2, 2)


def test_controller_velocity_action_zeros_inactive_row() -> None:
    """Inactive velocity-controlled rows receive a no-op command."""
    env = _ActionMaskStub()
    action = TensorDict(
        {"qvel": torch.tensor([[9.0, 9.0], [8.0, 8.0]])}, batch_size=[2]
    )

    masked = EmbodiedEnv._mask_controller_demo_action(env, action)

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
    _seed_trajectory_states = EmbodiedEnv._seed_trajectory_states

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
        self._traj_buffer["states"]["robot"]["qpos"][1, 1] = torch.tensor([0.3, 0.4])


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
    assert torch.equal(env._traj_buffer["actions"][1, 1], torch.tensor([8.0, 8.0]))


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
    task_instruction = "Hold still while recording"
    env.metadata = {"dataset": {"instruction": {"lang": task_instruction}}}
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
    assert metadata["segments"][0]["instruction"] == task_instruction


def test_explicit_segment_instructions_are_preserved() -> None:
    """Dataset-level fallback never overwrites explicit segment instructions."""
    env = _RolloutWriterStub()
    env.metadata = {"dataset": {"instruction": {"lang": "Complete the overall task"}}}
    expected_segments = [
        {
            "segment_id": 0,
            "name": "pick",
            "start_step": 0,
            "end_step": 1,
            "instruction": "Pick up the cube",
        },
        {
            "segment_id": 1,
            "name": "place",
            "start_step": 1,
            "end_step": 2,
            "instruction": "Place the cube",
        },
    ]
    env._demo_episode_metadata = [
        {
            "schema_version": 2,
            "episode_index": 0,
            "env_id": env_id,
            "length": 2,
            "completed": True,
            "success": True,
            "terminated": True,
            "truncated": False,
            "terminal_reason": "success",
            "segments": expected_segments,
        }
        for env_id in range(env.num_envs)
    ]

    metadata = EmbodiedEnv.get_demo_episode_metadata(env, 1)

    assert metadata["segments"] == expected_segments


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
