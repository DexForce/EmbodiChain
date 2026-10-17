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
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.gym.envs.demo import (
    DemoEpisodeResult,
    DemoExecutionCfg,
    DemoSegment,
    execute_demo_episode,
)
from embodichain.lab.gym.envs.base_env import BaseEnv
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.managers.record import record_camera_data
from embodichain.lab.gym.envs.task_program.bridge import (
    BufferedGymCommandSink,
    EnvironmentStepClock,
    RuntimeCommandFrameEncoder,
    TaskProgramDemoBridge,
)
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.envs.wrapper.no_fail import NoFailWrapper
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
)
from embodichain.lab.sim.atomic_actions.state import (
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
from embodichain.lab.task_program import (
    CompiledTaskProgram,
    InvokeCfg,
    PickCfg,
    TaskProgramCfg,
    TaskProgramIntegrationCfg,
)
from embodichain.lab.task_program.integrations.scene_expansion import (
    ScenePreparationResult,
    SceneVariantEpisodeResult,
    execute_scene_variant,
)
from embodichain.lab.task_program.runtime.results import (
    SemanticExecutionResult,
    SemanticExecutionStatus,
)

_GOAL_X = 0.5
_START_X = 0.2
_STEP_DT = 0.02


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
        self._scene_expansion_attempt_active = False
        self._scene_expansion_recording_rejected = False
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


class _RuntimePort:
    """Single-row runtime port for the production bridge and safe-stop sink."""

    def __init__(self, sink: BufferedGymCommandSink) -> None:
        self.sink = sink
        self.cancel_count = 0
        self.frame = RuntimeCommandFrame(
            commands=(
                EndpointCommand(
                    JointPositionTarget(control_part="arm", joint_ids=(0,)),
                    JointPositionPayload(positions=torch.tensor([[_GOAL_X]])),
                ),
            ),
            active_mask=torch.ones(1, dtype=torch.bool),
            env_ids=torch.tensor([0]),
            hold_duration=torch.tensor([_STEP_DT]),
        )
        self._set_status(SemanticExecutionStatus.IDLE)

    def _set_status(self, status: SemanticExecutionStatus) -> SemanticExecutionResult:
        self.status = status
        self.result = SemanticExecutionResult(
            status=status,
            workflow_id="scene-expansion-fixture/segment-0",
            current_call_index=0 if status is SemanticExecutionStatus.RUNNING else None,
            env_ids=torch.tensor([0]),
            success_mask=torch.tensor([status is SemanticExecutionStatus.COMPLETED]),
            failure_mask=torch.zeros(1, dtype=torch.bool),
            cancelled_mask=torch.tensor([status is SemanticExecutionStatus.CANCELLED]),
            eligible_mask=torch.tensor(
                [status is not SemanticExecutionStatus.CANCELLED]
            ),
            task_state=TaskState.empty(1, "cpu"),
        )
        return self.result

    def start(self, *calls: Any, **kwargs: Any) -> SemanticExecutionResult:
        self.sink.send(self.frame, timeout=1.0)
        return self._set_status(SemanticExecutionStatus.RUNNING)

    def step(self) -> SemanticExecutionResult:
        self._hold()
        return self._set_status(SemanticExecutionStatus.COMPLETED)

    def cancel(self, reason: str) -> SemanticExecutionResult:
        self.cancel_count += 1
        self.sink.cancel(self.frame.targets, timeout=1.0)
        self._hold()
        return self._set_status(SemanticExecutionStatus.CANCELLED)

    def _hold(self) -> None:
        qpos = torch.tensor([[_GOAL_X]])  # Measured after the command was consumed.
        self.sink.hold(
            self.frame.targets,
            PlanningContext(
                robot=RobotObservation(timestamp=0.0, qpos=qpos, qvel=qpos.clone()),
                task=TaskState.empty(1, "cpu"),
                scene=SceneSnapshot.empty(),
                env_ids=torch.tensor([0]),
            ),
            timeout=1.0,
        )

    def adopt_verified_task_state(
        self, task_state: TaskState
    ) -> SemanticExecutionResult:
        return self.result


class _RealBridgeEnv(_CandidateEnv):
    """Use the production bridge while replacing physical and semantic ports."""

    def __init__(
        self, *, terminal_success: bool, terminal_action_kind: str = "runtime_command"
    ) -> None:
        super().__init__()
        self.terminal_success = terminal_success
        self.terminal_action_kind = terminal_action_kind
        self.action_kinds: list[str | None] = []
        self.post_policy_port = SimpleNamespace(
            validate_policy=Mock(),
            actions=Mock(return_value=(torch.tensor([[_GOAL_X]]),)),
            post_policy_result=Mock(return_value=True),
            post_policy_metadata=Mock(return_value={}),
        )
        self.validator_port = SimpleNamespace(
            validate_validator=Mock(),
            validate=Mock(return_value=True),
            validator_metadata=Mock(return_value={}),
        )

    def create_task_program_bridge(
        self, program: CompiledTaskProgram
    ) -> TaskProgramDemoBridge:
        clock = EnvironmentStepClock(_STEP_DT)
        encoder = RuntimeCommandFrameEncoder(
            SimpleNamespace(current_qpos=lambda env_ids: torch.zeros(1, 1))
        )
        sink = BufferedGymCommandSink(encoder, clock)
        self.runtime = _RuntimePort(sink)
        call = SimpleNamespace(call_index=0, call="pick")
        segment = SimpleNamespace(
            segment_id="segment-0",
            instruction=None,
            segment_index=0,
            name="move-cube",
            source_path=("program",),
            implicit=False,
            calls=(call,),
            post_policies=(SimpleNamespace(),),
            validators=(SimpleNamespace(),),
        )
        compiled = SimpleNamespace(
            program_id="scene-expansion-fixture",
            iter_segments=lambda: iter((segment,)),
            sequential_execution_analysis=lambda index: SimpleNamespace(
                calls=(call.call,), execution_prefix_length=1
            ),
        )
        return TaskProgramDemoBridge(
            compiled,
            self.runtime,
            sink,
            clock,
            post_policy_port=self.post_policy_port,
            validator_port=self.validator_port,
        )

    def step(self, action: ControllerAction) -> tuple[Any, ...]:
        self.action_kinds.append(action.metadata.get("bridge_action_kind"))
        obs, reward, terminated, truncated, info = super().step(action)
        if (
            self.terminal_success
            and action.metadata.get("bridge_action_kind") == self.terminal_action_kind
        ):
            terminated = torch.ones(1, dtype=torch.bool)
            info = {"success": torch.ones(1, dtype=torch.bool)}
        return obs, reward, terminated, truncated, info


@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize(
    "terminal_action_kind", ("runtime_command", "program_post_policy")
)
def test_early_gym_success_cannot_accept_an_unfinished_canonical_program(
    wrapped: bool, terminal_action_kind: str
) -> None:
    env = _RealBridgeEnv(
        terminal_success=True, terminal_action_kind=terminal_action_kind
    )
    result = execute_scene_variant(
        NoFailWrapper(env) if wrapped else env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
    )

    assert result.execution.completed and result.execution.all_success
    assert not env._active_task_program_bridge.program_completed
    assert not result.accepted
    assert result.to_metadata()["program_success"] is False
    assert env.runtime.cancel_count == int(terminal_action_kind == "runtime_command")
    assert "runtime_abort_safe_hold" in env.action_kinds
    if terminal_action_kind == "runtime_command":
        env.post_policy_port.actions.assert_not_called()
    else:
        env.post_policy_port.actions.assert_called_once()
    env.validator_port.validate.assert_not_called()
    env.persist_via_reset(commit_env_ids=[0])
    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.trajectory_saves == 0


def test_normal_canonical_completion_includes_post_policy_and_validation() -> None:
    env = _RealBridgeEnv(terminal_success=False)
    result = execute_scene_variant(
        env, _variant(), prepare_scene=env.prepare, measured_validator=env.measure
    )

    assert result.accepted
    assert result.program_completed
    assert result.program_completion_mask == (True,)
    env.post_policy_port.actions.assert_called_once()
    env.validator_port.validate.assert_called_once()
    assert "program_post_policy" in env.action_kinds
    env.persist_via_reset()
    assert len(env.dataset_manager.saved) == 1


def test_no_fail_wrapper_cannot_override_a_completed_bridge_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _CandidateEnv()
    bridge = _Bridge(runtime_passes=True)

    def segments() -> Iterator[DemoSegment]:
        yield DemoSegment(
            actions=(ControllerAction(value=torch.tensor([[_GOAL_X]])),),
            validator=lambda: True,
        )
        bridge.program_completed = True
        bridge.completion_mask.fill_(False)

    monkeypatch.setattr(bridge, "iter_segments", segments)
    monkeypatch.setattr(env, "create_task_program_bridge", lambda program: bridge)
    result = execute_scene_variant(
        NoFailWrapper(env),
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
    )

    assert result.execution.completed and result.execution.all_success
    assert not result.accepted
    assert result.program_completed
    assert result.program_completion_mask == (False,)
    env.persist_via_reset(commit_env_ids=[0])
    assert env.dataset_manager.saved == []


class _SettlingEnv(_CandidateEnv):
    """Exercise production auto-reset dispatch with CPU simulation ports."""

    def __init__(self) -> None:
        super().__init__()
        self._elapsed_steps = torch.zeros(1, dtype=torch.long)
        self.max_episode_steps = 1  # One settling step reaches the ordinary limit.
        self.cfg.ignore_terminations = False
        self.sim.render_frame = lambda: nullcontext()
        self.automatic_resets = 0

    def _preprocess_action(self, action: Any) -> Any:
        return action

    def _step_action(self, action: Any) -> Any:
        return action

    def _advance_physics(self) -> None:
        pass

    def _update_sim_state(self, **kwargs: Any) -> None:
        pass

    def _update_physical_objective(self) -> None:
        pass

    def get_obs(self, **kwargs: Any) -> None:
        return None

    def get_info(self, **kwargs: Any) -> dict[str, Any]:
        return {}

    def get_reward(self, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(1)

    def _extend_reward(self, rewards: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return rewards

    def _postprocess_action(self, action: Any) -> Any:
        return action

    def check_truncated(self, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(1, dtype=torch.bool)

    def _hook_after_sim_step(self, **kwargs: Any) -> None:
        self._demo_steps += 1
        self.rollout_steps += 1
        self._traj_steps += 1

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[None, dict[str, Any]]:
        self.automatic_resets += 1
        self.persist_via_reset(**(options or {}))
        self._elapsed_steps.zero_()
        return None, {}

    step = BaseEnv.step


def test_preparation_settling_cannot_auto_reset_and_commit_unverified_data() -> None:
    env = _SettlingEnv()

    def prepare(variant: SceneVariant) -> ScenePreparationResult:
        prepared = env.prepare(variant)
        env.step(ControllerAction(value=torch.zeros(1, 1)))
        return ScenePreparationResult(prepared.initial_state_id, _checks("failed"))

    result = execute_scene_variant(
        env, _variant(), prepare_scene=prepare, measured_validator=env.measure
    )

    assert not result.accepted
    assert result.execution is None
    assert env.automatic_resets == 0
    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.trajectory_saves == 0
    assert not env._demo_no_auto_reset
    assert not env._scene_expansion_attempt_active


@pytest.mark.parametrize("phase", ("prepare", "measure"))
def test_explicit_reset_inside_an_attempt_cannot_commit_pending_data(
    phase: str,
) -> None:
    env = _SettlingEnv()
    env.max_episode_steps = 2  # The explicit reset owns this boundary.

    def prepare(variant: SceneVariant) -> ScenePreparationResult:
        prepared = env.prepare(variant)
        if phase == "prepare":
            env.step(ControllerAction(value=torch.tensor([[_GOAL_X]])))
            env.reset(options={"commit_env_ids": [0]})
        return prepared

    def measure() -> ValidationResult:
        if phase == "measure":
            env.reset(options={"commit_env_ids": [0]})
        return _checks()

    result = execute_scene_variant(
        env, _variant(), prepare_scene=prepare, measured_validator=measure
    )

    assert result.accepted
    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.trajectory_saves == 0


@pytest.mark.parametrize("explicit_commit", (False, True))
def test_rejected_buffers_remain_blocked_after_ordinary_demo_replaces_metadata(
    explicit_commit: bool,
) -> None:
    env = _CandidateEnv()
    env.physics_moves = False
    rejected = execute_scene_variant(
        env, _variant(), prepare_scene=env.prepare, measured_validator=env.measure
    )
    assert not rejected.accepted

    env.physics_moves = True
    retry = execute_demo_episode(env)
    assert retry.all_success
    assert "scene_expansion" not in env.get_demo_episode_metadata(0)
    env.persist_via_reset(**({"commit_env_ids": [0]} if explicit_commit else {}))

    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.trajectory_saves == 0


@pytest.mark.parametrize("reset_buffers", (False, True))
def test_a_later_scene_can_save_only_after_rejected_buffers_are_reset(
    reset_buffers: bool,
) -> None:
    env = _CandidateEnv()
    env.physics_moves = False
    execute_scene_variant(
        env, _variant(), prepare_scene=env.prepare, measured_validator=env.measure
    )
    env.physics_moves = True
    prepare = (
        env.prepare
        if reset_buffers
        else lambda variant: ScenePreparationResult("same-recording", _checks())
    )

    accepted = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=prepare,
        measured_validator=env.measure,
        attempt_id=1,
    )
    assert accepted.accepted
    env.persist_via_reset(commit_env_ids=[0])

    assert len(env.dataset_manager.saved) == int(reset_buffers)
    assert env.camera_recorder.saved == int(reset_buffers)
    assert env.trajectory_saves == int(reset_buffers)


def test_empty_reset_cannot_clear_rejected_buffer_eligibility() -> None:
    env = _CandidateEnv()
    env.physics_moves = False
    execute_scene_variant(
        env, _variant(), prepare_scene=env.prepare, measured_validator=env.measure
    )
    env._initialize_episode(env_ids=[], save_data=False)
    env.physics_moves = True
    execute_demo_episode(env)
    env.persist_via_reset(commit_env_ids=[0])

    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.trajectory_saves == 0


def test_empty_reset_during_preparation_does_not_reject_a_fresh_recording() -> None:
    env = _CandidateEnv()

    def prepare(variant: SceneVariant) -> ScenePreparationResult:
        env._initialize_episode(env_ids=[], save_data=False)
        env.object_x = variant.pose_changes[0].arena_pose[0][3]
        return ScenePreparationResult("fresh-recording", _checks())

    result = execute_scene_variant(
        env, _variant(), prepare_scene=prepare, measured_validator=env.measure
    )
    assert result.accepted
    env.persist_via_reset()

    assert len(env.dataset_manager.saved) == 1
    assert env.camera_recorder.saved == 1
    assert env.trajectory_saves == 1


def test_failed_discard_preserves_rejection_after_metadata_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _CandidateEnv()
    env.physics_moves = False
    execute_scene_variant(
        env, _variant(), prepare_scene=env.prepare, measured_validator=env.measure
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            env.camera_recorder,
            "discard_and_clear",
            Mock(side_effect=RuntimeError("camera discard failed")),
        )
        with pytest.raises(RuntimeError, match="camera discard failed"):
            env.persist_via_reset()

    env.physics_moves = True
    execute_demo_episode(env)
    env.persist_via_reset(commit_env_ids=[0])

    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.trajectory_saves == 0


@pytest.mark.parametrize("previous_no_auto_reset", (False, True))
@pytest.mark.parametrize("failure_phase", (None, "prepare", "execute", "measure"))
def test_attempt_flags_restore_after_success_and_each_error_boundary(
    previous_no_auto_reset: bool, failure_phase: str | None
) -> None:
    env = _CandidateEnv()
    env._demo_no_auto_reset = previous_no_auto_reset
    env.fail_step = failure_phase == "execute"

    def prepare(variant: SceneVariant) -> ScenePreparationResult:
        assert env._scene_expansion_attempt_active and env._demo_no_auto_reset
        if failure_phase == "prepare":
            raise RuntimeError("preparation failed")
        return env.prepare(variant)

    def measure() -> ValidationResult:
        assert env._scene_expansion_attempt_active and env._demo_no_auto_reset
        if failure_phase == "measure":
            raise RuntimeError("measurement failed")
        return env.measure()

    if failure_phase is None:
        assert execute_scene_variant(
            env, _variant(), prepare_scene=prepare, measured_validator=measure
        ).accepted
    else:
        with pytest.raises(RuntimeError):
            execute_scene_variant(
                env, _variant(), prepare_scene=prepare, measured_validator=measure
            )
        assert env._scene_expansion_recording_rejected

    assert env._demo_no_auto_reset is previous_no_auto_reset
    assert not env._scene_expansion_attempt_active


def test_measurement_interrupt_restores_flags_and_keeps_buffers_ineligible() -> None:
    env = _CandidateEnv()

    def interrupt() -> ValidationResult:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_scene_variant(
            env, _variant(), prepare_scene=env.prepare, measured_validator=interrupt
        )

    assert not env._demo_no_auto_reset
    assert not env._scene_expansion_attempt_active
    execute_demo_episode(env)
    env.persist_via_reset(commit_env_ids=[0])
    assert env.dataset_manager.saved == []
    assert env.camera_recorder.saved == 0
    assert env.trajectory_saves == 0


def _successful_demo() -> DemoEpisodeResult:
    return DemoEpisodeResult(
        episode_index=0,
        length=1,
        completed=True,
        success=(True,),
        terminated=(False,),
        truncated=(False,),
        terminal_reason="success",
    )


def test_result_requires_owned_canonical_program_completion_proof() -> None:
    result = SceneVariantEpisodeResult(
        _variant(),
        ScenePreparationResult("settled-v1", _checks()),
        _successful_demo(),
        _checks(),
    )
    assert not result.accepted
    assert not result.program_success

    mask = [True]
    accepted = replace(result, program_completed=True, program_completion_mask=mask)
    mask[0] = False
    assert accepted.accepted
    assert accepted.program_completion_mask == (True,)
    assert not replace(accepted, program_completion_mask=(False,)).accepted


@pytest.mark.parametrize(
    "proof,error",
    (
        ({"program_completed": 1}, TypeError),
        ({"program_completion_mask": (1,)}, TypeError),
        ({"program_completion_mask": None}, TypeError),
        ({"program_completed": True, "program_completion_mask": ()}, ValueError),
        (
            {"program_completed": True, "program_completion_mask": (True, False)},
            ValueError,
        ),
    ),
)
def test_result_rejects_malformed_program_completion_proof(
    proof: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error, match="program_completed|program_completion_mask"):
        SceneVariantEpisodeResult(
            _variant(),
            ScenePreparationResult("settled-v1", _checks()),
            _successful_demo(),
            _checks(),
            **proof,
        )


def test_scene_program_override_preserves_selected_language_snapshot() -> None:
    env = _CandidateEnv()
    env.cfg.task_program.instruction = "Static task"
    selected = _program()
    selected.instruction = "Selected scene task"
    result = execute_scene_variant(
        env,
        _variant(),
        prepare_scene=env.prepare,
        measured_validator=env.measure,
        task_program=selected,
    )
    assert result.accepted
    assert result.execution.instruction == "Selected scene task"
    assert result.execution.instruction_source == "task_program"
    assert result.execution.segments[0].instruction == "Selected scene task"
    assert result.execution.segments[0].instruction_source == "task_fallback"
