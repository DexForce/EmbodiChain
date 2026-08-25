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

"""Configuration and non-physical bridge vertical slices for Expert Programs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramCompiler,
    decode_expert_program,
)
from embodichain.lab.gym.envs.expert_program.bridge import (
    AtomicDemoBridge,
    BufferedGymCommandSink,
    EnvironmentStepClock,
    RuntimeCommandFrameEncoder,
)
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    EntityState,
    ObjectSemantics,
    SlideAffordance,
    SlideOptions,
    TaskState,
)
from embodichain.lab.sim.skills.calls import Pick, Place, RegisteredSemanticCall
from embodichain.lab.sim.skills.runtime import SkillResult, SkillStatus
from embodichain.lab.sim.skills.scene import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
)
from embodichain_tasks.expert_program import open_drawer as drawer_task
from embodichain_tasks.expert_program import repeated_pick_place as cube_task

_REPOSITORY_ROOT = Path(__file__).parents[4]
_REPEATED_CUBE_PROGRAM = Path("expert_program/repeated_pick_place.yaml")
_OPEN_DRAWER_PROGRAM = Path("expert_program/open_drawer.yaml")
_LIFECYCLE_BATCH_SIZE = 2
_LIFECYCLE_ROBOT_DOF = 3
_LIFECYCLE_STEP_DT = 0.02


class _NeverObserveProvider:
    """Reject dynamic observations during configuration decoding/compilation."""

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        raise AssertionError("Task configuration compilation must not observe state.")


class _FixedQposProvider:
    """Return a finite full-qpos hold for the bridge's unused command sink."""

    def current_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (env_ids.numel(), _LIFECYCLE_ROBOT_DOF),
            dtype=torch.float32,
            device=env_ids.device,
        )


class _FreshObservationPort:
    """Issue one distinct observation generation for every segment runtime."""

    def __init__(self) -> None:
        self.generations: list[int] = []

    def capture(self) -> int:
        generation = len(self.generations) + 1
        self.generations.append(generation)
        return generation


class _CompletedSegmentRuntime:
    """Complete each semantic prefix from one freshly captured observation."""

    def __init__(
        self,
        observation: _FreshObservationPort,
        lifecycle_events: list[tuple[str, int]],
    ) -> None:
        self._observation = observation
        self._lifecycle_events = lifecycle_events
        self._status = SkillStatus.IDLE
        self._result = self._make_result(
            status=SkillStatus.IDLE,
            workflow_id=None,
            eligible_mask=torch.ones(_LIFECYCLE_BATCH_SIZE, dtype=torch.bool),
            generation=0,
        )
        self.analysis_window_lengths: list[int] = []
        self.executed_semantic_ids: list[str] = []
        self.eligible_masks: list[torch.Tensor | None] = []

    @staticmethod
    def _make_result(
        *,
        status: SkillStatus,
        workflow_id: str | None,
        eligible_mask: torch.Tensor,
        generation: int,
    ) -> SkillResult:
        terminal = status is SkillStatus.COMPLETED
        return SkillResult(
            status=status,
            workflow_id=workflow_id,
            current_call_index=None,
            env_ids=torch.arange(_LIFECYCLE_BATCH_SIZE, dtype=torch.long),
            success_mask=(
                eligible_mask.clone() if terminal else torch.zeros_like(eligible_mask)
            ),
            failure_mask=torch.zeros_like(eligible_mask),
            cancelled_mask=torch.zeros_like(eligible_mask),
            eligible_mask=eligible_mask,
            task_state=TaskState.empty(_LIFECYCLE_BATCH_SIZE, "cpu"),
            message=f"observation_generation={generation}",
        )

    @property
    def result(self) -> SkillResult:
        return self._result

    @property
    def status(self) -> SkillStatus:
        return self._status

    def start(
        self,
        *calls: object,
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SkillResult:
        call_values = tuple(calls[0]) if len(calls) == 1 else tuple(calls)
        if execution_prefix_length is None:
            raise AssertionError("A packaged sequential segment requires a prefix.")
        selected = (
            torch.ones(_LIFECYCLE_BATCH_SIZE, dtype=torch.bool)
            if eligible_mask is None
            else eligible_mask.clone()
        )
        execution_calls = call_values[:execution_prefix_length]
        generation = self._observation.capture()
        self._lifecycle_events.append(("observe", generation))
        self.analysis_window_lengths.append(len(call_values))
        self.executed_semantic_ids.extend(
            str(getattr(call, "semantic_id")) for call in execution_calls
        )
        self.eligible_masks.append(
            None if eligible_mask is None else eligible_mask.clone()
        )
        self._status = SkillStatus.COMPLETED
        self._result = self._make_result(
            status=SkillStatus.COMPLETED,
            workflow_id=workflow_id,
            eligible_mask=selected,
            generation=generation,
        )
        return self._result

    def step(self) -> SkillResult:
        raise AssertionError("A terminal fake runtime must not be stepped.")

    def cancel(self, reason: str) -> SkillResult:
        raise AssertionError(f"A completed fake runtime cannot be cancelled: {reason}")

    def adopt_verified_task_state(self, task_state: TaskState) -> SkillResult:
        del task_state
        return self._result


class _LifecyclePostPolicyPort:
    """Run every packaged settle policy and expose deterministic metadata."""

    def __init__(
        self,
        observation: _FreshObservationPort,
        lifecycle_events: list[tuple[str, int]],
    ) -> None:
        self._observation = observation
        self._lifecycle_events = lifecycle_events
        self.active_masks: list[torch.Tensor] = []
        self._metadata: dict[int, dict[str, object]] = {}

    def validate_policy(self, policy: object, *, segment: object) -> None:
        del policy, segment

    def actions(
        self,
        policy: object,
        *,
        segment: object,
        active_mask: torch.Tensor,
    ):
        segment_index = int(getattr(segment, "segment_index"))
        generation = self._observation.generations[-1]
        self._lifecycle_events.append(("settle", segment_index))
        self.active_masks.append(active_mask.clone())
        self._metadata[id(policy)] = {
            "status": "settled",
            "segment_index": segment_index,
            "observation_generation": generation,
        }
        yield torch.zeros(
            (_LIFECYCLE_BATCH_SIZE, _LIFECYCLE_ROBOT_DOF),
            dtype=torch.float32,
        )

    def post_policy_result(
        self,
        policy: object,
        *,
        segment: object,
    ) -> torch.Tensor:
        del policy, segment
        return self.active_masks[-1].clone()

    def post_policy_metadata(
        self,
        policy: object,
        *,
        segment: object,
    ) -> dict[str, object]:
        del segment
        return dict(self._metadata[id(policy)])


class _LifecycleValidatorPort:
    """Validate every segment and filter one row after the first cycle."""

    def __init__(
        self,
        observation: _FreshObservationPort,
        lifecycle_events: list[tuple[str, int]],
    ) -> None:
        self._observation = observation
        self._lifecycle_events = lifecycle_events
        self._metadata: dict[int, dict[str, object]] = {}

    def validate_validator(self, validator: object, *, segment: object) -> None:
        del validator, segment

    def validate(self, validator: object, *, segment: object) -> torch.Tensor:
        segment_index = int(getattr(segment, "segment_index"))
        generation = self._observation.generations[-1]
        self._lifecycle_events.append(("validate", segment_index))
        result = (
            torch.tensor([True, False])
            if segment_index == 0
            else torch.ones(_LIFECYCLE_BATCH_SIZE, dtype=torch.bool)
        )
        self._metadata[id(validator)] = {
            "segment_index": segment_index,
            "observation_generation": generation,
            "accepted_mask": result.tolist(),
        }
        return result

    def validator_metadata(
        self,
        validator: object,
        *,
        segment: object,
    ) -> dict[str, object]:
        del segment
        return dict(self._metadata[id(validator)])


def _read_payload(relative_path: Path) -> dict[str, object]:
    """Load one packaged JSON/YAML example as inert data."""
    path = _REPOSITORY_ROOT / "embodichain_tasks/configs" / relative_path
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert type(payload) is dict
    return payload


def _cube_compiler() -> ExpertProgramCompiler:
    """Build the smallest typed identity registry needed by the cube program."""
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_NeverObserveProvider(),
            ),
        )
    )
    return ExpertProgramCompiler.from_scene_registry(registry)


def _drawer_compiler() -> ExpertProgramCompiler:
    """Build typed drawer and handle identities without any motion code."""
    provider = _NeverObserveProvider()
    drawer = SceneArticulationRef("drawer")
    registry = SceneRegistry(
        (
            SceneEntityRegistration(ref=drawer, state_provider=provider),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("drawer_handle"),
                parent=drawer,
                native_name="large_handle_bar",
                affordance=Affordance(),
                relative_pose=torch.eye(4),
            ),
        )
    )
    return ExpertProgramCompiler.from_scene_registry(registry)


def test_repeated_cube_program_is_three_lazy_semantic_segments() -> None:
    """The packaged cube task expands to three independently scoped cycles."""
    config = decode_expert_program(_read_payload(_REPEATED_CUBE_PROGRAM))

    assert config.integration.scene_registry == cube_task.SCENE_REGISTRY_ID
    assert config.integration.robot_profile == cube_task.ROBOT_PROFILE_ID

    segments = tuple(_cube_compiler().compile(config))

    assert [segment.name for segment in segments] == ["move_cube"] * 3
    assert [segment.segment_index for segment in segments] == [0, 1, 2]
    assert [len(segment.calls) for segment in segments] == [2, 2, 2]
    assert all(type(segment.calls[0].call) is Pick for segment in segments)
    assert all(type(segment.calls[1].call) is Place for segment in segments)
    assert [
        segment.calls[1].target_selections[0].value_index for segment in segments
    ] == [0, 1, 0]
    assert [
        segment.validators[0].target_selection.value_index for segment in segments
    ] == [
        0,
        1,
        0,
    ]
    assert all(
        segment.post_policies[0].cfg.kind == "wait_stable" for segment in segments
    )
    assert all(
        segment.validators[0].cfg.position_tolerance == 0.12 for segment in segments
    )


def test_packaged_repeated_cube_runs_three_lazy_bridge_lifecycles() -> None:
    """The real packaged program owns three ordered observable lifecycles."""
    config = decode_expert_program(_read_payload(_REPEATED_CUBE_PROGRAM))
    compiled = _cube_compiler().compile(config)
    lifecycle_events: list[tuple[str, int]] = []
    observation = _FreshObservationPort()
    clock = EnvironmentStepClock(_LIFECYCLE_STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_FixedQposProvider()),
        clock,
    )
    runtime = _CompletedSegmentRuntime(observation, lifecycle_events)
    post_port = _LifecyclePostPolicyPort(observation, lifecycle_events)
    validator_port = _LifecycleValidatorPort(observation, lifecycle_events)
    bridge = AtomicDemoBridge(
        compiled,
        runtime,
        sink,
        clock,
        post_policy_port=post_port,
        validator_port=validator_port,
    )

    iterator = iter(bridge.iter_segments())
    segment_names: list[str | None] = []
    segment_metadata: list[dict[str, object]] = []
    action_metadata: list[dict[str, object]] = []
    accepted_masks: list[list[bool]] = []
    for segment_index in range(3):
        observation_count = len(observation.generations)
        demo_segment = next(iterator)
        segment_names.append(demo_segment.name)

        # Merely requesting the next lazy segment must not capture live state.
        assert len(observation.generations) == observation_count
        actions = tuple(demo_segment.actions)

        assert observation.generations == list(range(1, segment_index + 2))
        assert len(actions) == 1
        assert demo_segment.metadata["validation"] is None
        action_metadata.append(dict(actions[0].metadata))
        accepted_masks.append(demo_segment.validator().tolist())
        segment_metadata.append(dict(demo_segment.metadata))

    with pytest.raises(StopIteration):
        next(iterator)

    assert segment_names == ["move_cube"] * 3
    assert runtime.analysis_window_lengths == [6, 4, 2]
    assert runtime.executed_semantic_ids == ["pick", "place"] * 3
    assert observation.generations == [1, 2, 3]
    assert lifecycle_events == [
        ("observe", 1),
        ("settle", 0),
        ("validate", 0),
        ("observe", 2),
        ("settle", 1),
        ("validate", 1),
        ("observe", 3),
        ("settle", 2),
        ("validate", 2),
    ]
    assert runtime.eligible_masks[0] is None
    assert [mask.tolist() for mask in runtime.eligible_masks[1:]] == [
        [True, False],
        [True, False],
    ]
    assert [mask.tolist() for mask in post_port.active_masks] == [
        [True, True],
        [True, False],
        [True, False],
    ]
    assert accepted_masks == [[True, False]] * 3

    for segment_index, metadata in enumerate(segment_metadata):
        eligible_before = [True, True] if segment_index == 0 else [True, False]
        validator_result = [True, False] if segment_index == 0 else [True, True]
        assert metadata["expert_program_id"] == compiled.program_id
        assert metadata["program_segment_index"] == segment_index
        assert metadata["semantic_call_indices"] == [
            2 * segment_index,
            2 * segment_index + 1,
        ]
        assert metadata["post_policy_count"] == 1
        assert metadata["validator_count"] == 1
        runtime_metadata = metadata["runtime"]
        assert isinstance(runtime_metadata, dict)
        assert runtime_metadata["message"] == (
            f"observation_generation={segment_index + 1}"
        )
        post_policies = metadata["post_policies"]
        assert isinstance(post_policies, list)
        assert post_policies[0]["kind"] == "wait_stable"
        assert post_policies[0]["result_mask"] == eligible_before
        assert post_policies[0]["result"] == {
            "status": "settled",
            "segment_index": segment_index,
            "observation_generation": segment_index + 1,
        }
        validation = metadata["validation"]
        assert isinstance(validation, dict)
        assert validation["eligible_mask_before_validation"] == eligible_before
        assert validation["accepted_mask"] == [True, False]
        validators = validation["validators"]
        assert validators[0]["kind"] == "object_near_target"
        assert validators[0]["result_mask"] == validator_result
        assert validators[0]["result"] == {
            "segment_index": segment_index,
            "observation_generation": segment_index + 1,
            "accepted_mask": validator_result,
        }
        json.dumps(metadata, allow_nan=False, sort_keys=True)

        assert action_metadata[segment_index]["bridge_action_kind"] == (
            "program_post_policy"
        )
        assert action_metadata[segment_index]["program_segment_index"] == (
            segment_index
        )


def test_cube_variant_extends_by_data_without_motion_generation_code() -> None:
    """A fourth destination and cycle require only serialized-data changes."""
    payload = deepcopy(_read_payload(_REPEATED_CUBE_PROGRAM))
    target = payload["targets"]["drop_pose"]
    target["values"].extend(
        (
            {
                "position": [-0.25, -0.20, 0.10],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "position": [-0.25, 0.20, 0.10],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        )
    )
    payload["program"]["count"] = 4

    segments = tuple(_cube_compiler().compile(decode_expert_program(payload)))

    assert len(segments) == 4
    last_place = segments[-1].calls[-1].call
    assert type(last_place) is Place
    assert last_place.at is not None
    assert last_place.at.position.tolist() == pytest.approx([-0.25, 0.20, 0.10])


def test_open_drawer_program_compiles_to_registered_slide_call() -> None:
    """The drawer task supplies a validated call, never a trajectory."""
    payload = _read_payload(_OPEN_DRAWER_PROGRAM)
    config = decode_expert_program(payload)

    assert config.integration.scene_registry == drawer_task.SCENE_REGISTRY_ID
    assert config.integration.robot_profile == drawer_task.ROBOT_PROFILE_ID

    segments = tuple(_drawer_compiler().compile(config))

    assert len(segments) == 1
    assert segments[0].name == "open_drawer"
    assert len(segments[0].calls) == 1
    call = segments[0].calls[0].call
    assert type(call) is RegisteredSemanticCall
    assert call.call_id == "embodichain_tasks.open_drawer"
    assert dict(call.arguments) == {
        "handle": "drawer_handle",
    }
    assert dict(call.resources) == {"primary": "manipulator"}
    validator = segments[0].validators[0].cfg
    assert validator.articulation == "drawer"
    assert validator.joint == "cabinet_to_drawer"
    assert validator.minimum_position == 0.10


def test_open_drawer_lowerer_preserves_legacy_payload_compatibility() -> None:
    """Schema-v1 option fields remain accepted only as preset-matching input."""
    options = SlideOptions(
        direction="pull",
        hand_interp_steps=12,
        approach_distance=0.10,
        translation_distance=0.18,
    )
    lowerer = drawer_task._OpenDrawerSlideLowerer(
        ObjectSemantics(
            label="drawer_handle",
            entity_id=drawer_task.HANDLE_ENTITY_ID,
            geometry={},
            affordance=SlideAffordance(
                mesh_vertices=torch.tensor(
                    [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
                translation_axis=torch.tensor([0.0, 1.0, 0.0]),
            ),
        )
    )
    minimal = {"handle": drawer_task.HANDLE_ENTITY_ID}
    legacy = {
        **minimal,
        "direction": options.direction,
        "hand_interp_steps": options.hand_interp_steps,
        "approach_distance": options.approach_distance,
        "translation_distance": options.translation_distance,
    }

    for arguments in (minimal, legacy):
        lowering = lowerer.lower(
            RegisteredSemanticCall(
                call_id=drawer_task.OPEN_DRAWER_CALL_ID,
                arguments=arguments,
            ),
            context=None,  # type: ignore[arg-type]
            bound=None,  # type: ignore[arg-type]
            option_template=options,
        )
        assert lowering.goal.target_pose.entity_id == drawer_task.HANDLE_ENTITY_ID

    with pytest.raises(ValueError, match="legacy option fields"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id=drawer_task.OPEN_DRAWER_CALL_ID,
                arguments={**legacy, "direction": "push"},
            ),
            context=None,  # type: ignore[arg-type]
            bound=None,  # type: ignore[arg-type]
            option_template=options,
        )


def test_task_classes_do_not_override_motion_or_demo_generation() -> None:
    """Both environments delegate planning and execution to the shared runtime."""
    forbidden_overrides = {
        "create_demo_action_list",
        "create_demo_segments",
        "_generate_eef_motion",
        "_initialize_atomic_actions",
        "_observe_grasp_constraint",
        "_plan_pick_place_cycle",
    }

    for env_type in (
        cube_task.ExpertProgramRepeatedPickPlaceEnv,
        drawer_task.ExpertProgramOpenDrawerEnv,
    ):
        assert forbidden_overrides.isdisjoint(env_type.__dict__)


def test_cube_task_declares_the_canonical_scene_and_profile_ids() -> None:
    """The repeated task owns one canonical scene/profile integration."""
    assert cube_task.SCENE_REGISTRY_ID == "expert_program_repeated_pick_place"
    assert cube_task.ROBOT_PROFILE_ID == "expert_program_ur5_pick_place"


def test_vertical_slice_payloads_expose_no_motion_layer_fields() -> None:
    """Official examples remain semantic data without controller/planner knobs."""
    forbidden_fields = {
        "action",
        "control_part",
        "eef",
        "joint_ids",
        "motion_generator",
        "planner",
        "qpos",
        "sample_count",
        "tcp",
        "trajectory",
    }

    def keys(value: object) -> set[str]:
        if type(value) is dict:
            return set(value).union(*(keys(item) for item in value.values()))
        if type(value) is list:
            return set().union(*(keys(item) for item in value))
        return set()

    for path in (_REPEATED_CUBE_PROGRAM, _OPEN_DRAWER_PROGRAM):
        assert forbidden_fields.isdisjoint(keys(_read_payload(path)))


__all__: list[str] = []
