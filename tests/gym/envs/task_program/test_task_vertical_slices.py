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

"""Configuration and non-physical bridge vertical slices for Task Programs."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from embodichain.lab.task_program import TaskProgramCompiler, decode_task_program
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.task_program.integrations import (
    SimulationRobotSkillProfileBinding,
)
from embodichain.lab.task_program.integrations._configured_composition import (
    _load_configured_task_program_deployment,
)
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.lab.gym.envs.task_program.bridge import (
    TaskProgramDemoBridge,
    BufferedGymCommandSink,
    EnvironmentStepClock,
    RuntimeCommandFrameEncoder,
)
from embodichain.lab.sim.atomic_actions import (
    Affordance,
    AtomicActionEngine,
    DynamicCollisionMode,
    EntityState,
    ObjectSemantics,
    Slide,
    SlideAffordance,
    SlideOptions,
    TaskState,
    TimedTerminalAcceptance,
)
from embodichain.lab.task_program.semantics.calls import (
    Pick,
    Place,
    RegisteredSemanticCall,
)
from embodichain.lab.task_program.runtime.results import (
    SemanticExecutionResult,
    SemanticExecutionStatus,
)
from embodichain.lab.task_program.semantics.scene import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRegistration,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
)

_REPOSITORY_ROOT = Path(__file__).parents[4]
_REPEATED_CUBE_PROGRAM = Path(
    "tasks/manipulation/repeated_pick_place/task_program/program.yaml"
)
_REPEATED_CUBE_INTEGRATION = Path(
    "tasks/manipulation/repeated_pick_place/task_program/integration.yaml"
)
_REPEATED_CUBE_GYM_CONFIG = Path("tasks/manipulation/repeated_pick_place/env.ur5.yaml")
_OPEN_DRAWER_PROGRAM = Path("tasks/manipulation/open_drawer/task_program/program.yaml")
_OPEN_DRAWER_GYM_CONFIG = Path("tasks/manipulation/open_drawer/env.ur5.yaml")
_TRAJECTORY_POLICY = Path("components/execution_policies/trajectory_open_loop.yaml")
_UR5_COMPONENT = Path("components/embodiments/ur5_dh_pgi_140_80.yaml")
_OPEN_DRAWER_CALL_ID = "simulation.articulation_link_slide"
_OPEN_DRAWER_ENTITY_ID = "drawer"
_OPEN_DRAWER_HANDLE_ID = "drawer_handle"
_OPEN_DRAWER_HANDLE_LINK_NAME = "large_handle_bar"
_OPEN_DRAWER_SCENE_ID = "task_program_open_drawer"
_OPEN_DRAWER_PROFILE_ID = "ur5_dh_pgi_140_80"
_LIFECYCLE_BATCH_SIZE = 2
_LIFECYCLE_ROBOT_DOF = 3
_LIFECYCLE_STEP_DT = 0.02
_EXPECTED_TRAJECTORY_SAMPLE_COUNT = 40
_EXPECTED_GRASP_SAMPLES = 1_000


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
        self._status = SemanticExecutionStatus.IDLE
        self._result = self._make_result(
            status=SemanticExecutionStatus.IDLE,
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
        status: SemanticExecutionStatus,
        workflow_id: str | None,
        eligible_mask: torch.Tensor,
        generation: int,
    ) -> SemanticExecutionResult:
        terminal = status is SemanticExecutionStatus.COMPLETED
        return SemanticExecutionResult(
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
    def result(self) -> SemanticExecutionResult:
        return self._result

    @property
    def status(self) -> SemanticExecutionStatus:
        return self._status

    def start(
        self,
        *calls: object,
        workflow_id: str = "semantic_workflow",
        eligible_mask: torch.Tensor | None = None,
        execution_prefix_length: int | None = None,
    ) -> SemanticExecutionResult:
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
        self._status = SemanticExecutionStatus.COMPLETED
        self._result = self._make_result(
            status=SemanticExecutionStatus.COMPLETED,
            workflow_id=workflow_id,
            eligible_mask=selected,
            generation=generation,
        )
        return self._result

    def step(self) -> SemanticExecutionResult:
        raise AssertionError("A terminal fake runtime must not be stepped.")

    def cancel(self, reason: str) -> SemanticExecutionResult:
        raise AssertionError(f"A completed fake runtime cannot be cancelled: {reason}")

    def adopt_verified_task_state(
        self, task_state: TaskState
    ) -> SemanticExecutionResult:
        del task_state
        return self._result


def _read_payload(relative_path: Path) -> dict[str, object]:
    """Load one packaged JSON/YAML example as inert data."""
    path = _REPOSITORY_ROOT / "embodichain_tasks/configs" / relative_path
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert type(payload) is dict
    return payload


def _deployment(relative_path: Path):
    """Compose one packaged environment deployment."""
    path = _REPOSITORY_ROOT / "embodichain_tasks/configs" / relative_path
    payload = _read_payload(relative_path)
    physical = _resolve_gym_components(payload, base_dir=path.parent)
    assert physical.embodiment_skill_profile is not None
    assert physical.scene_task_program is not None
    return _load_configured_task_program_deployment(
        task_program=payload["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        scene=physical.scene_task_program,
        base_dir=path.parent,
    )


def _cube_integration():
    """Compose a fresh cube integration from the packaged deployment."""
    return _deployment(_REPEATED_CUBE_GYM_CONFIG).integration


def _drawer_integration():
    """Compose a fresh drawer integration from the packaged deployment."""
    return _deployment(_OPEN_DRAWER_GYM_CONFIG).integration


def _decode_deployed_program(
    payload: dict[str, object],
    *,
    deployment_path: Path,
):
    """Bind a trusted deployment selection before strict decoding."""
    deployment = _deployment(deployment_path)
    selected = deepcopy(payload)
    selected["integration"] = {
        "robot_profile": deployment.selection.robot_profile,
        "scene_registry": deployment.selection.scene_registry,
        "runtime_preset": deployment.selection.runtime_preset,
    }
    return decode_task_program(
        selected,
        validation_context=deployment.integration.registration.catalog,
    )


def _cube_robot_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Return the config-declared repeated Pick/Place robot profile."""
    return _cube_integration().registration.robot_profile_binding


def _drawer_robot_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Return the config-declared Open Drawer robot profile."""
    return _drawer_integration().registration.robot_profile_binding


def _configure_cube_environment():
    """Load the packaged config and register its declared integration."""
    path = _REPOSITORY_ROOT / "embodichain_tasks/configs" / _REPEATED_CUBE_GYM_CONFIG
    payload = _read_payload(_REPEATED_CUBE_GYM_CONFIG)
    cfg = config_to_cfg(payload, source_path=path)
    return payload, cfg, REGISTERED_ENVS[str(payload["id"])]


def _cube_compiler() -> TaskProgramCompiler:
    """Build the smallest typed identity registry needed by the cube program."""
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_NeverObserveProvider(),
            ),
        )
    )
    return TaskProgramCompiler.from_scene_registry(registry)


def _drawer_compiler() -> TaskProgramCompiler:
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
    return TaskProgramCompiler.from_scene_registry(registry)


def test_repeated_cube_program_is_three_lazy_semantic_segments() -> None:
    """The packaged cube task expands to three independently scoped cycles."""
    config = _decode_deployed_program(
        _read_payload(_REPEATED_CUBE_PROGRAM),
        deployment_path=_REPEATED_CUBE_GYM_CONFIG,
    )

    integration = _cube_integration()
    assert (
        config.integration.scene_registry
        == integration.registration.scene_binding.registry_id
    )
    assert (
        config.integration.robot_profile
        == integration.registration.robot_profile_binding.profile_id
    )

    segments = tuple(_cube_compiler().compile(config))

    assert [segment.name for segment in segments] == ["move_cube"] * 3
    assert [segment.segment_index for segment in segments] == [0, 1, 2]
    assert [len(segment.calls) for segment in segments] == [2, 2, 2]
    assert all(type(segment.calls[0].call) is Pick for segment in segments)
    assert all(type(segment.calls[1].call) is Place for segment in segments)
    assert [
        segment.calls[1].target_selections[0].value_index for segment in segments
    ] == [0, 1, 0]
    assert all(segment.post_policies == () for segment in segments)
    assert all(segment.validators == () for segment in segments)


def test_packaged_repeated_cube_runs_three_lazy_bridge_lifecycles() -> None:
    """The real packaged program owns three ordered observable lifecycles."""
    config = _decode_deployed_program(
        _read_payload(_REPEATED_CUBE_PROGRAM),
        deployment_path=_REPEATED_CUBE_GYM_CONFIG,
    )
    compiled = _cube_compiler().compile(config)
    lifecycle_events: list[tuple[str, int]] = []
    observation = _FreshObservationPort()
    clock = EnvironmentStepClock(_LIFECYCLE_STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_FixedQposProvider()),
        clock,
    )
    runtime = _CompletedSegmentRuntime(observation, lifecycle_events)
    bridge = TaskProgramDemoBridge(
        compiled,
        runtime,
        sink,
        clock,
    )

    iterator = iter(bridge.iter_segments())
    segment_names: list[str | None] = []
    segment_metadata: list[dict[str, object]] = []
    accepted_masks: list[list[bool]] = []
    for segment_index in range(3):
        observation_count = len(observation.generations)
        demo_segment = next(iterator)
        segment_names.append(demo_segment.name)

        # Merely requesting the next lazy segment must not capture live state.
        assert len(observation.generations) == observation_count
        actions = tuple(demo_segment.actions)

        assert observation.generations == list(range(1, segment_index + 2))
        assert actions == ()
        assert demo_segment.metadata["validation"] is None
        accepted_masks.append(demo_segment.validator().tolist())
        segment_metadata.append(dict(demo_segment.metadata))

    with pytest.raises(StopIteration):
        next(iterator)

    assert segment_names == ["move_cube"] * 3
    assert runtime.analysis_window_lengths == [6, 4, 2]
    assert runtime.executed_semantic_ids == ["pick", "place"] * 3
    assert observation.generations == [1, 2, 3]
    assert lifecycle_events == [("observe", 1), ("observe", 2), ("observe", 3)]
    assert runtime.eligible_masks[0] is None
    assert [mask.tolist() for mask in runtime.eligible_masks[1:]] == [
        [True, True],
        [True, True],
    ]
    assert accepted_masks == [[True, True]] * 3

    for segment_index, metadata in enumerate(segment_metadata):
        assert metadata["task_program_id"] == compiled.program_id
        assert metadata["program_segment_index"] == segment_index
        assert metadata["semantic_call_indices"] == [
            2 * segment_index,
            2 * segment_index + 1,
        ]
        assert metadata["post_policy_count"] == 0
        assert metadata["validator_count"] == 0
        runtime_metadata = metadata["runtime"]
        assert isinstance(runtime_metadata, dict)
        assert runtime_metadata["message"] == (
            f"observation_generation={segment_index + 1}"
        )
        assert metadata["post_policies"] == []
        validation = metadata["validation"]
        assert isinstance(validation, dict)
        assert validation["eligible_mask_before_validation"] == [True, True]
        assert validation["accepted_mask"] == [True, True]
        assert validation["validators"] == []
        json.dumps(metadata, allow_nan=False, sort_keys=True)


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

    config = _decode_deployed_program(
        payload,
        deployment_path=_REPEATED_CUBE_GYM_CONFIG,
    )
    segments = tuple(_cube_compiler().compile(config))

    assert len(segments) == 4
    last_place = segments[-1].calls[-1].call
    assert type(last_place) is Place
    assert last_place.at is not None
    assert last_place.at.position.tolist() == pytest.approx([-0.25, 0.20, 0.10])


def test_open_drawer_program_compiles_to_registered_slide_call() -> None:
    """The drawer config supplies one registered call with no acceptance hooks."""
    payload = _read_payload(_OPEN_DRAWER_PROGRAM)
    config = _decode_deployed_program(
        payload,
        deployment_path=_OPEN_DRAWER_GYM_CONFIG,
    )

    assert config.integration.scene_registry == _OPEN_DRAWER_SCENE_ID
    assert config.integration.robot_profile == _OPEN_DRAWER_PROFILE_ID

    segments = tuple(_drawer_compiler().compile(config))

    assert len(segments) == 1
    assert segments[0].name == "open_drawer"
    assert len(segments[0].calls) == 1
    call = segments[0].calls[0].call
    assert type(call) is RegisteredSemanticCall
    assert call.call_id == _OPEN_DRAWER_CALL_ID
    assert dict(call.arguments) == {
        "handle": "drawer_handle",
    }
    assert dict(call.resources) == {}
    assert segments[0].post_policies == ()
    assert segments[0].validators == ()


def test_open_drawer_config_owns_registered_lowerer_factory() -> None:
    """The decoded registration owns call discovery and fresh live lowerers."""
    registration = _drawer_integration().registration

    assert tuple(registration.catalog.registered_semantic_lowerer_declarations) == (
        _OPEN_DRAWER_CALL_ID,
    )
    assert (
        registration.catalog.call_catalog.discover(
            _OPEN_DRAWER_CALL_ID
        ).target_descriptor
        == Slide.descriptor()
    )

    class Drawer:
        link_names = (_OPEN_DRAWER_HANDLE_LINK_NAME,)

        @staticmethod
        def get_link_vert_face(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            assert name == _OPEN_DRAWER_HANDLE_LINK_NAME
            return (
                torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                torch.tensor([[0, 1, 2]]),
            )

    drawer_ref = SceneArticulationRef(_OPEN_DRAWER_ENTITY_ID)
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=drawer_ref,
                state_provider=_NeverObserveProvider(),
            ),
            SceneEntityRegistration(
                ref=SceneLinkRef(_OPEN_DRAWER_HANDLE_ID),
                state_provider=_NeverObserveProvider(),
                parent=drawer_ref,
                native_name=_OPEN_DRAWER_HANDLE_LINK_NAME,
            ),
        )
    )
    robot = object()
    engine = AtomicActionEngine.__new__(AtomicActionEngine)
    engine._planning_services = SimpleNamespace(  # type: ignore[attr-defined]
        robot=robot,
        device=torch.device("cpu"),
    )
    simulation = SimpleNamespace(
        get_articulation=lambda identifier: (
            Drawer() if identifier == "drawer" else None
        )
    )

    first = registration.create_registered_semantic_lowerers(
        simulation=simulation,
        robot=robot,
        scene_registry=registry,
        engine=engine,
    )
    second = registration.create_registered_semantic_lowerers(
        simulation=simulation,
        robot=robot,
        scene_registry=registry,
        engine=engine,
    )

    assert type(first[0]) is type(second[0])
    assert type(first[0]).call_id == _OPEN_DRAWER_CALL_ID
    assert first[0] is not second[0]


def test_open_drawer_lowerer_accepts_only_canonical_payload() -> None:
    """Serialized calls name the handle while presets own motion options."""
    options = SlideOptions(
        direction="pull",
        hand_interp_steps=12,
        approach_distance=0.10,
        translation_distance=0.18,
    )
    from embodichain.lab.task_program.integrations._configured_services import (
        _ArticulationLinkSlideLowerer,
    )

    lowerer = _ArticulationLinkSlideLowerer(
        ObjectSemantics(
            label="drawer_handle",
            entity_id=_OPEN_DRAWER_HANDLE_ID,
            geometry={},
            affordance=SlideAffordance(
                mesh_vertices=torch.tensor(
                    [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
                translation_axis=torch.tensor([0.0, 1.0, 0.0]),
            ),
        ),
        _OPEN_DRAWER_HANDLE_ID,
    )
    minimal = {"handle": _OPEN_DRAWER_HANDLE_ID}
    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id=_OPEN_DRAWER_CALL_ID,
            arguments=minimal,
        ),
        context=None,  # type: ignore[arg-type]
        bound=None,  # type: ignore[arg-type]
        option_template=options,
    )
    assert lowering.goal.target_pose.entity_id == _OPEN_DRAWER_HANDLE_ID

    with pytest.raises(ValueError, match="motion options belong"):
        lowerer.lower(
            RegisteredSemanticCall(
                call_id=_OPEN_DRAWER_CALL_ID,
                arguments={**minimal, "direction": "push"},
            ),
            context=None,  # type: ignore[arg-type]
            bound=None,  # type: ignore[arg-type]
            option_template=options,
        )


def test_open_drawer_lowerer_owns_a_snapshot_of_the_current_target_pose() -> None:
    """Snapshot mode lowers one current pose without a dynamic scene reference."""
    from embodichain.lab.task_program.integrations._configured_services import (
        _ArticulationLinkSlideLowerer,
    )

    observed_pose = torch.eye(4, dtype=torch.float32)
    lowerer = _ArticulationLinkSlideLowerer(
        ObjectSemantics(
            label="drawer_handle",
            entity_id=_OPEN_DRAWER_HANDLE_ID,
            geometry={},
            affordance=SlideAffordance(
                mesh_vertices=torch.tensor(
                    [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                mesh_triangles=torch.tensor([[0, 1, 2]]),
                translation_axis=torch.tensor([0.0, 1.0, 0.0]),
            ),
        ),
        _OPEN_DRAWER_HANDLE_ID,
        target_pose_mode="snapshot",
    )
    context = SimpleNamespace(
        scene=SimpleNamespace(
            entities={
                _OPEN_DRAWER_HANDLE_ID: SimpleNamespace(pose=observed_pose),
            }
        )
    )

    lowering = lowerer.lower(
        RegisteredSemanticCall(
            call_id=_OPEN_DRAWER_CALL_ID,
            arguments={"handle": _OPEN_DRAWER_HANDLE_ID},
        ),
        context=context,  # type: ignore[arg-type]
        bound=None,  # type: ignore[arg-type]
        option_template=SlideOptions(),
    )
    observed_pose[0, 0] = 2.0

    assert isinstance(lowering.goal.target_pose, torch.Tensor)
    assert torch.equal(lowering.goal.target_pose, torch.eye(4))


def test_examples_have_no_task_specific_environment_modules() -> None:
    """Both examples are assembled from config against plain EmbodiedEnv."""
    for module_name in (
        "embodichain_tasks.manipulation.repeated_pick_place",
        "embodichain_tasks.manipulation.open_drawer",
    ):
        assert importlib.util.find_spec(module_name) is None
    _, _, cube_spec = _configure_cube_environment()
    assert cube_spec.cls is EmbodiedEnv


def test_cube_config_registers_embodied_env_with_its_integration_factory() -> None:
    """Repeated Pick/Place needs no environment subclass or task module."""
    payload, cfg, spec = _configure_cube_environment()
    expected = _cube_integration()

    assert payload["id"] == "TaskProgramRepeatedPickPlace-v1"
    assert cfg.task_program is not None
    assert spec.cls is EmbodiedEnv
    assert spec.max_episode_steps == payload["max_episode_steps"]
    assert spec.task_program_registration is not None
    assert (
        spec.task_program_registration.fingerprint == expected.registration.fingerprint
    )
    assert spec.task_program_adapter_factory is not None
    assert (
        spec.task_program_adapter_factory.registration is spec.task_program_registration
    )
    assert (
        spec.task_program_adapter_factory.integration_fingerprint
        == expected.integration_fingerprint
    )


def test_cube_config_declares_the_canonical_scene_and_profile_ids() -> None:
    """The repeated config owns one canonical scene/profile integration."""
    registration = _cube_integration().registration

    assert registration.scene_binding.registry_id == (
        "task_program_repeated_pick_place"
    )
    assert registration.robot_profile_binding.profile_id == ("ur5_dh_pgi_140_80")


@pytest.mark.parametrize(
    ("create_binding", "expected_grasp_samples"),
    (
        (_cube_robot_profile_binding, _EXPECTED_GRASP_SAMPLES),
        (_drawer_robot_profile_binding, _EXPECTED_GRASP_SAMPLES),
    ),
)
def test_example_profiles_execute_only_open_loop_trajectories(
    create_binding: Callable[[], SimulationRobotSkillProfileBinding],
    expected_grasp_samples: int,
) -> None:
    """Both tutorials use timed execution without effects or retry layers."""
    binding = create_binding()
    preset = binding.presets[0]

    assert preset.preset_id == "trajectory"
    assert preset.motion_policy.sample_count == _EXPECTED_TRAJECTORY_SAMPLE_COUNT
    assert preset.motion_policy.dynamic_collision_mode is DynamicCollisionMode.OFF
    assert preset.tracking_policy.in_flight is None
    assert isinstance(preset.tracking_policy.terminal, TimedTerminalAcceptance)
    assert preset.tracking_policy.terminal.settle_duration == 0.0
    assert preset.recovery_policy.max_replans == 0
    assert preset.recovery_policy.max_action_retries == 0
    assert preset.workflow_recovery_policy.max_recovery_attempts == 0
    assert dict(preset.effect_monitors) == {}
    assert preset.runner_cfg.hold_on_completion is False
    assert preset.runner_cfg.hold_during_effect_verification is False
    assert expected_grasp_samples == _EXPECTED_GRASP_SAMPLES


def test_cube_policy_and_skill_parameters_have_single_component_owners() -> None:
    """Execution and grasp tuning live in policy and embodiment components."""
    integration = _read_payload(_REPEATED_CUBE_INTEGRATION)
    policy = _read_payload(_TRAJECTORY_POLICY)
    embodiment = _read_payload(_UR5_COMPONENT)

    assert "motion" not in integration["profile"]
    assert "grasp_pose_generators" not in integration.get("runtime_services", {})
    assert policy["motion"]["sample_count"] == _EXPECTED_TRAJECTORY_SAMPLE_COUNT
    services = embodiment["skill_profile"]["runtime_services"]
    assert services["grasp_pose_generators"]["hand"]["sample_count"] == (
        _EXPECTED_GRASP_SAMPLES
    )


def test_cube_registration_has_no_contact_evidence_route() -> None:
    """The trajectory tutorial does not construct its former contact observer."""
    declaration = (
        _cube_integration().registration.catalog.control_part_evidence_declaration
    )
    assert declaration is None


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("tasks/manipulation/repeated_pick_place/env.ur5.yaml"),
        Path("tasks/manipulation/open_drawer/env.ur5.yaml"),
    ),
)
def test_example_gym_configs_omit_auxiliary_environment_mechanisms(
    relative_path: Path,
) -> None:
    """Runnable examples keep only deterministic simulation and motion inputs."""
    payload = _read_payload(relative_path)

    assert set(payload["task_program"]) == {
        "program",
        "integration",
        "execution_policy",
    }
    assert set(payload["embodiment"]) <= {"component", "overrides"}
    assert payload["scene"] == {"component": "task_program/scene.yaml"}
    assert "version" not in payload
    assert "task_program_path" not in payload
    assert "task_program_integration_path" not in payload
    assert "task_program_runtime" not in payload
    assert payload["env"]["events"] == {}
    assert payload["env"]["dataset"] == {}
    assert "physics_config" not in payload


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
