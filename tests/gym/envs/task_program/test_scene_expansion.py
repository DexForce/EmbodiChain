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

"""Scene candidate integration through real demo and recording boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from embodichain.lab.gym.envs.demo import (
    DemoExecutionCfg,
    DemoSegment,
    execute_demo_episode,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.managers.record import record_camera_data
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene_expansion import ScenePoseChange, SceneVariant
from embodichain.lab.task_program import (
    CompiledTaskProgram,
    InvokeCfg,
    PickCfg,
    TaskProgramCfg,
    TaskProgramIntegrationCfg,
)
from embodichain.lab.task_program.integrations.scene_expansion import (
    ScenePreparationResult,
    execute_scene_variant,
)

_GOAL_X = 0.5
_START_X = 0.2


def _program() -> TaskProgramCfg:
    return TaskProgramCfg(
        program_id="scene-expansion-fixture",
        integration=TaskProgramIntegrationCfg(
            robot_profile="fixture-robot",
            scene_registry="fixture-scene",
            runtime_preset="fixture-runtime",
        ),
        program=InvokeCfg(call=PickCfg(object="cube")),
    )


def _checks(status: str = "passed") -> ValidationResult:
    return ValidationResult((ValidationCheck("measured-object-position", status),))


def _variant() -> SceneVariant:
    pose = torch.eye(4).tolist()
    pose[0][3] = _START_X
    return SceneVariant(
        parent_scene_id="fixture-scene-v1",
        seed=7,
        ordinal=0,
        pose_changes=(ScenePoseChange("cube", pose),),
        metadata=(("source", "fixed-fixture"),),
    )


class _CameraRecorder(record_camera_data):
    """Exercise the production camera save/discard routing without IO."""

    def __init__(self) -> None:
        self.saved = 0
        self.discarded = 0

    def save_and_clear(self, env_ids: torch.Tensor | None = None) -> None:
        self.saved += 1

    def discard_and_clear(self, env_ids: torch.Tensor | None = None) -> None:
        self.discarded += 1


class _DatasetRecorder:
    """Capture the real buffered metadata when reset requests a save."""

    available_modes = ("save",)
    save_failed_episodes = True

    def __init__(self, env: _CandidateEnv) -> None:
        self.env = env
        self.saved: list[dict[str, Any]] = []

    def apply(self, *, mode: str, env_ids: torch.Tensor) -> None:
        assert mode == "save"
        for env_id in env_ids.tolist():
            self.saved.append(self.env.get_demo_episode_metadata(env_id))

    def reset(self, *, env_ids: Any) -> None:
        pass


class _Bridge:
    """A deterministic trusted runtime producing one controller command."""

    expansion_records = ()

    def __init__(self, *, runtime_passes: bool) -> None:
        self.program_completed = False
        self.completion_mask = torch.tensor([runtime_passes])

    def iter_segments(self) -> Iterator[DemoSegment]:
        yield DemoSegment(
            actions=(ControllerAction(value=torch.tensor([[_GOAL_X]])),),
            name="move-cube",
            target_uid="cube",
            validator=lambda: self.completion_mask,
        )
        self.program_completed = True


class _CandidateEnv(EmbodiedEnv):
    """Use production bridge selection, demo execution, metadata, and reset IO.

    Only physics and runtime planning are replaced by deterministic CPU ports.
    This lets the tests distinguish program completion from measured state and
    exercise the real delayed persistence boundary.
    """

    def __init__(self) -> None:
        self._num_envs = 1
        self.sim = SimpleNamespace(device=torch.device("cpu"))
        self.cfg = SimpleNamespace(
            task_program=_program(),
            seed=7,
            events=True,
            observations=None,
            rewards=None,
            dataset=True,
            trajectory_auto_save=True,
        )
        self._profiler = SimpleNamespace(section=lambda *args, **kwargs: nullcontext())
        self._demo_steps = torch.zeros(1, dtype=torch.long)
        self.rollout_steps = torch.zeros(1, dtype=torch.long)
        self.rollout_buffer = None
        self._demo_active_segment_ids = torch.zeros(1, dtype=torch.long)
        self._demo_active_mask = torch.ones(1, dtype=torch.bool)
        self._demo_segment_participants = torch.zeros(1, dtype=torch.bool)
        self._demo_active_segment_start_steps = self._demo_steps.clone()
        self._demo_active_rollout_start_steps = self.rollout_steps.clone()
        self._demo_episode_metadata = [self._new_demo_episode_metadata(0)]
        self._active_task_program_bridge = None
        self._demo_no_auto_reset = False
        self.episode_success_status = torch.zeros(1, dtype=torch.bool)
        self._task_success = torch.zeros(1, dtype=torch.bool)
        self.action_manager = None
        self.camera_recorder = _CameraRecorder()
        self.event_manager = SimpleNamespace(
            _mode_functor_cfgs={"record": [SimpleNamespace(func=self.camera_recorder)]},
            available_modes=(),
            reset=lambda **kwargs: None,
        )
        self.dataset_manager = _DatasetRecorder(self)
        self._traj_buffer = object()
        self._traj_steps = torch.zeros(1, dtype=torch.long)
        self.trajectory_saves = 0
        self.object_x = 0.0
        self.first_observations: list[float] = []
        self.bridge_initial_positions: list[float] = []
        self.runtime_passes = True
        self.physics_moves = True
        self.fail_step = False

    def compile_task_program(self, program: TaskProgramCfg) -> CompiledTaskProgram:
        return object.__new__(CompiledTaskProgram)

    def create_task_program_bridge(self, program: CompiledTaskProgram) -> _Bridge:
        self.bridge_initial_positions.append(self.object_x)
        return _Bridge(runtime_passes=self.runtime_passes)

    def step(self, action: ControllerAction) -> tuple[Any, ...]:
        assert self._demo_no_auto_reset
        self.first_observations.append(self.object_x)
        if self.fail_step:
            raise RuntimeError("physics port failed")
        if self.physics_moves:
            self.object_x = float(action.value[0, 0])
        self._demo_steps += 1
        self.rollout_steps += 1
        self._traj_steps += 1
        return (
            None,
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    def _save_trajectory_for_env(self, env_id: int) -> str:
        self.trajectory_saves += 1
        return "fixture.pt"

    def prepare(self, variant: SceneVariant) -> ScenePreparationResult:
        self._initialize_episode(save_data=False)
        self._active_task_program_bridge = None
        self.object_x = variant.pose_changes[0].arena_pose[0][3]
        return ScenePreparationResult(
            initial_state_id=f"settled-cube-{self.object_x}",
            validation=_checks(),
            metadata={"scene_revision": "revision-1"},
        )

    def measure(self) -> ValidationResult:
        status = "passed" if abs(self.object_x - _GOAL_X) < 1e-6 else "failed"
        return _checks(status)

    def persist_via_reset(self, **kwargs: Any) -> None:
        # BaseEnv.reset computes task success before _initialize_episode.
        self._task_success = self.is_task_success()
        self._initialize_episode(**kwargs)
        self._active_task_program_bridge = None


def test_fixed_candidate_uses_fresh_bridge_and_actual_recording_metadata() -> None:
    env = _CandidateEnv()
    variant = _variant()

    result = execute_scene_variant(
        env,
        variant,
        prepare_scene=env.prepare,
        measured_validator=env.measure,
        episode_index=3,
        attempt_id=2,
    )

    assert result.accepted
    assert result.execution.length == 1
    assert result.execution.episode_index == 3
    assert result.execution.attempt_id == 2
    assert env.bridge_initial_positions == [_START_X]
    assert env.first_observations == [_START_X]
    assert env.dataset_manager.saved == []
    provenance = env.get_demo_episode_metadata(0)["scene_expansion"]
    assert provenance["variant"]["variant_id"] == variant.variant_id
    assert provenance["variant"]["parent_scene_id"] == variant.parent_scene_id
    assert provenance["initial_state_id"] == result.preparation.initial_state_id
    assert provenance["initial_state_id"] != variant.variant_id
    assert provenance["measured_validation"][0]["status"] == "passed"

    env.persist_via_reset()

    assert len(env.dataset_manager.saved) == 1
    assert env.dataset_manager.saved[0]["scene_expansion"] == provenance
    assert env.camera_recorder.saved == 1
    assert env.trajectory_saves == 1


@pytest.mark.parametrize("mode", ("continuous", "segment_fragments"))
def test_physical_failure_blocks_all_later_reset_persistence(mode: str) -> None:
    env = _CandidateEnv()
    env.physics_moves = False

    result = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
        execution_cfg=DemoExecutionCfg(
            mode=mode, save_failed_fragments=(mode == "segment_fragments")
        ),
    )

    assert result.execution.all_success
    assert result.execution.completed
    assert not result.measured_validation.accepted
    assert not result.accepted
    assert env.is_task_success().tolist() == [True]

    env.persist_via_reset(save_data=True, commit_env_ids=[0])

    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.camera_recorder.discarded == 2  # Preparation and rejected attempt.
    assert env.trajectory_saves == 0


@pytest.mark.parametrize("status", ("failed", "unavailable", "not_run"))
def test_invalid_initial_scene_prevents_program_execution(status: str) -> None:
    env = _CandidateEnv()

    def prepare(variant: SceneVariant) -> ScenePreparationResult:
        prepared = env.prepare(variant)
        return ScenePreparationResult(prepared.initial_state_id, _checks(status))

    def must_not_measure() -> ValidationResult:
        raise AssertionError("an invalid initial scene must not execute")

    result = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=prepare,
        measured_validator=must_not_measure,
    )

    assert not result.accepted
    assert result.execution is None
    assert env.bridge_initial_positions == []
    assert env.first_observations == []
    env.persist_via_reset()
    assert env.dataset_manager.saved == []


@pytest.mark.parametrize(
    "measured",
    (ValidationResult(()), _checks("unavailable"), _checks("not_run")),
)
def test_missing_measured_evidence_cannot_accept_program_success(
    measured: ValidationResult,
) -> None:
    env = _CandidateEnv()
    result = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=lambda: measured,
    )

    assert result.execution.all_success
    assert not result.accepted
    env.persist_via_reset()
    assert env.dataset_manager.saved == []


@pytest.mark.parametrize("value", (True, "passed", {"accepted": True}))
def test_validator_rejects_arbitrary_truthy_values_and_blocks_saving(
    value: object,
) -> None:
    env = _CandidateEnv()

    with pytest.raises(TypeError, match="measured_validator must return"):
        execute_scene_variant(
            env,
            _variant(),
            prepare_scene=env.prepare,
            measured_validator=lambda: value,
        )

    assert env.get_demo_episode_metadata(0)["scene_expansion"]["accepted"] is False
    env.persist_via_reset()
    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.trajectory_saves == 0


def test_measurement_exception_marks_completed_program_ineligible_for_saving() -> None:
    env = _CandidateEnv()

    def fail_measurement() -> ValidationResult:
        raise RuntimeError("physical evidence unavailable")

    with pytest.raises(RuntimeError, match="physical evidence unavailable"):
        execute_scene_variant(
            env,
            _variant(),
            prepare_scene=env.prepare,
            measured_validator=fail_measurement,
        )

    assert env.is_task_success().tolist() == [True]
    assert (
        env.get_demo_episode_metadata(0)["scene_expansion"]["error_type"]
        == "RuntimeError"
    )
    env.persist_via_reset(commit_env_ids=[0])
    assert env.dataset_manager.saved == []
    assert env.trajectory_saves == 0


def test_preparation_failure_does_not_leave_the_previous_episode_saveable() -> None:
    env = _CandidateEnv()
    env.episode_success_status.fill_(True)

    def fail_preparation(variant: SceneVariant) -> ScenePreparationResult:
        raise RuntimeError("scene restore failed")

    with pytest.raises(RuntimeError, match="scene restore failed"):
        execute_scene_variant(
            env,
            _variant(),
            prepare_scene=fail_preparation,
            measured_validator=env.measure,
        )

    env.persist_via_reset()
    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0


def test_successful_measurements_cannot_override_program_failure() -> None:
    env = _CandidateEnv()
    env.runtime_passes = False

    result = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
    )

    assert result.measured_validation.accepted
    assert not result.execution.all_success
    assert not result.accepted
    env.persist_via_reset()
    assert env.dataset_manager.saved == []


def test_execution_error_fails_closed_after_begin_recording_replaced_metadata() -> None:
    env = _CandidateEnv()
    env.fail_step = True

    with pytest.raises(RuntimeError, match="emergency safe-stop attempt") as error:
        execute_scene_variant(
            env,
            _variant(),
            prepare_scene=env.prepare,
            measured_validator=env.measure,
        )

    assert str(error.value.__cause__) == "physics port failed"
    assert env.get_demo_episode_metadata(0)["scene_expansion"]["accepted"] is False
    env.persist_via_reset(commit_env_ids=[0])
    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0


def test_batch_and_legacy_expert_inputs_fail_before_scene_preparation() -> None:
    env = _CandidateEnv()
    env._num_envs = 2
    with pytest.raises(ValueError, match="B=1"):
        execute_scene_variant(
            env, _variant(), prepare_scene=env.prepare, measured_validator=env.measure
        )

    env._num_envs = 1
    env.cfg.task_program = None
    with pytest.raises(TypeError, match="requires a Task Program"):
        execute_scene_variant(
            env, _variant(), prepare_scene=env.prepare, measured_validator=env.measure
        )
    assert env.bridge_initial_positions == []


def test_episode_program_override_uses_the_same_task_program_path() -> None:
    env = _CandidateEnv()
    env.cfg.task_program = None

    result = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
        task_program=_program(),
    )

    assert result.accepted
    assert len(env.bridge_initial_positions) == 1


def test_normal_demo_recording_is_unchanged_without_scene_expansion() -> None:
    env = _CandidateEnv()
    result = execute_demo_episode(env)

    assert result.all_success
    assert "scene_expansion" not in env.get_demo_episode_metadata(0)
    env.persist_via_reset()
    assert len(env.dataset_manager.saved) == 1
    assert env.camera_recorder.saved == 1
    assert env.trajectory_saves == 1


def test_rejected_scene_does_not_poison_the_next_fresh_attempt() -> None:
    env = _CandidateEnv()
    env.physics_moves = False
    rejected = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
    )
    env.persist_via_reset()
    env.physics_moves = True
    accepted = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
        attempt_id=1,
    )
    env.persist_via_reset()

    assert not rejected.accepted
    assert accepted.accepted
    assert env.bridge_initial_positions == [_START_X, _START_X]
    assert len(env.dataset_manager.saved) == 1
    assert env.dataset_manager.saved[0]["attempt_id"] == 1
    assert env.dataset_manager.saved[0]["scene_expansion"]["accepted"] is True


def test_preparation_callback_requires_an_explicit_validation_contract() -> None:
    env = _CandidateEnv()

    with pytest.raises(TypeError, match="prepare_scene must return"):
        execute_scene_variant(
            env,
            _variant(),
            prepare_scene=lambda variant: True,
            measured_validator=env.measure,
        )

    assert env.bridge_initial_positions == []
    env.persist_via_reset(commit_env_ids=[0])
    assert env.dataset_manager.saved == []


class _LegacySegmentOverrideEnv(_CandidateEnv):
    """A legacy task override must not replace the selected Task Program."""

    def __init__(self) -> None:
        super().__init__()
        self.legacy_calls = 0

    def create_demo_segments(self, **kwargs: Any) -> tuple[DemoSegment, ...]:
        self.legacy_calls += 1
        return (
            DemoSegment(
                actions=(ControllerAction(value=torch.tensor([[_GOAL_X]])),),
                name="legacy-expert",
            ),
        )

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        return torch.ones(1, dtype=torch.bool)


def test_scene_execution_cannot_be_replaced_by_a_legacy_segment_override() -> None:
    env = _LegacySegmentOverrideEnv()

    result = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
    )

    assert result.accepted
    assert env.legacy_calls == 0
    assert env.bridge_initial_positions == [_START_X]
    assert result.execution.segments[0].name == "move-cube"


def test_explicit_demo_segments_keep_instruction_fallback_without_replanning() -> None:
    env = _LegacySegmentOverrideEnv()
    env.metadata = {"dataset": {"instruction": {"lang": "Move the cube"}}}

    result = execute_demo_episode(
        env,
        segments=(
            DemoSegment(
                actions=(ControllerAction(value=torch.tensor([[_GOAL_X]])),),
                name="explicit",
            ),
        ),
    )

    assert result.all_success
    assert env.legacy_calls == 0
    assert result.segments[0].instruction == "Move the cube"
    assert (
        env.get_demo_episode_metadata(0)["segments"][0]["instruction"]
        == "Move the cube"
    )


def test_explicit_demo_segments_require_the_normal_segment_contract() -> None:
    env = _LegacySegmentOverrideEnv()

    with pytest.raises(TypeError, match="must yield DemoSegment"):
        execute_demo_episode(env, segments=(object(),))

    assert env.legacy_calls == 0
    assert env.first_observations == []


def test_empty_explicit_demo_segments_do_not_fall_back_to_a_legacy_planner() -> None:
    env = _LegacySegmentOverrideEnv()

    result = execute_demo_episode(env, segments=())

    assert not result.all_success
    assert result.terminal_reason == "empty_plan"
    assert env.legacy_calls == 0


def test_preparation_metadata_owns_and_freezes_nested_provenance() -> None:
    source = {"nested": {"ids": ["snapshot-v1"]}}
    prepared = ScenePreparationResult("initial-state-v1", _checks(), source)
    source["nested"]["ids"][0] = "mutated-source"

    assert prepared.metadata["nested"]["ids"] == ("snapshot-v1",)
    with pytest.raises(TypeError):
        prepared.metadata["nested"]["ids"][0] = "mutated-preparation"
    with pytest.raises(TypeError):
        prepared.metadata["nested"]["ids"] = ()

    env = _CandidateEnv()
    result = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=lambda variant: prepared,
        measured_validator=env.measure,
    )
    exported = result.to_metadata()
    exported["preparation"]["nested"]["ids"][0] = "mutated-export"
    assert prepared.metadata["nested"]["ids"] == ("snapshot-v1",)
    assert result.to_metadata()["preparation"]["nested"]["ids"] == ["snapshot-v1"]
