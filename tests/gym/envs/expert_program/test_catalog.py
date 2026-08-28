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

"""Tests for task-registration-owned Expert Program integration catalogs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import ClassVar

import pytest

from embodichain.lab.expert_program import (
    ExpertProgramValidationError,
    decode_expert_program,
)
from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramIntegrationCatalog,
    IntegrationFingerprintMismatch,
    SimulationArticulationLinkBinding,
    SimulationExpertProgramAdapterFactory,
    SimulationExpertProgramRegistration,
    SimulationRigidObjectBinding,
    SimulationSceneBinding,
    SupportSurfaceAffordanceBinding,
)
from embodichain.lab.gym.envs.expert_program._configured_runtime_decoder import (
    _decode_configured_expert_program_runtime,
)
from embodichain.lab.gym.utils.registration import EnvSpec
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    Affordance,
    AtomicActionEngine,
    PlanningContext,
    SceneProvider,
    SceneSnapshot,
    SkillDescriptor,
)
from embodichain.lab.semantic_skills import (
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    PLACEMENT_TARGET_AFFORDANCE_REVISION,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    BoundSemanticCall,
    ControlPartEndpoint,
    EffectAssurance,
    EffectEvidenceProvider,
    HandOver,
    SemanticCallCatalog,
    SceneAffordanceRef,
    SceneDynamics,
    SceneEntityManifest,
    SceneManifest,
    SceneObjectRef,
    SceneRegistry,
    RegisteredSemanticCall,
    SemanticCallDescriptor,
    SkillPolicyPreset,
    SupportSurfaceAffordance,
    WorkflowRecoveryPolicy,
    builtin_semantic_call_catalog,
)
from embodichain.lab.expert_program._semantic_compiler import (
    HandOverPoseProvider,
    HandOverPoseTargets,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
    SemanticLowering,
    SemanticRelationTarget,
    SupportSurfaceRelationTargetGrounder,
)
from embodichain.lab.sim.atomic_actions.tracking import (
    InFlightTrackingPolicy,
    TimedTerminalAcceptance,
    TrackingMetricCfg,
    TrackingPolicy,
)
from embodichain.lab.semantic_skills.effects import EffectMonitorRef
from embodichain.lab.expert_program._parallel_executor import (
    ParallelCommandSafetyValidator,
)

_CATALOG_REGISTERED_CALL_ID = "test.catalog_call"
_CUBE_RUNTIME_PRESET_ID = "trajectory"
_REPOSITORY_ROOT = Path(__file__).parents[4]
_CUBE_GYM_CONFIG_PATH = (
    _REPOSITORY_ROOT
    / "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json"
)


def _cube_runtime():
    """Decode a fresh cube integration from the production Gym config."""
    payload = json.loads(_CUBE_GYM_CONFIG_PATH.read_text(encoding="utf-8"))
    return _decode_configured_expert_program_runtime(payload["expert_program_runtime"])


def create_cube_scene_binding() -> SimulationSceneBinding:
    """Return a fresh provider-free cube scene binding from config."""
    return _cube_runtime().registration.scene_binding


def create_cube_robot_profile_binding():
    """Return a fresh provider-free cube robot profile binding from config."""
    return _cube_runtime().registration.robot_profile_binding


_CUBE_REFERENCE_RUNTIME = _cube_runtime()
CUBE_ROBOT_PROFILE_ID = (
    _CUBE_REFERENCE_RUNTIME.registration.robot_profile_binding.profile_id
)
CUBE_SCENE_REGISTRY_ID = _CUBE_REFERENCE_RUNTIME.registration.scene_binding.registry_id
_CATALOG_REGISTERED_TARGET = (
    builtin_semantic_call_catalog().descriptors["pick"].target_descriptor
)
assert _CATALOG_REGISTERED_TARGET is not None


class _CatalogRegisteredLowerer(RegisteredSemanticLowerer):
    """Live lowerer sentinel for registration factory lifecycle tests."""

    call_id: ClassVar[str] = _CATALOG_REGISTERED_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = _CATALOG_REGISTERED_TARGET

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Remain unreachable in registration lifecycle tests."""
        del call, context, bound, option_template
        raise AssertionError("Catalog tests must not lower semantic calls.")


@dataclass(frozen=True, slots=True)
class _CatalogRegisteredLowererFactory:
    """Frozen declaration creating a fresh lowerer per runtime assembly."""

    call_id: ClassVar[str] = _CATALOG_REGISTERED_CALL_ID
    revision: ClassVar[str] = "1"

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Return one fresh stateless lowerer."""
        del simulation, robot, scene_registry, engine
        return _CatalogRegisteredLowerer()


class _CatalogRelationGrounder(RelationTargetGrounder):
    """Typed relation-grounder sentinel for registration validation."""

    capability: ClassVar[str] = "test.catalog_relation"
    affordance_type: ClassVar[type[Affordance]] = Affordance
    affordance_revision: ClassVar[str] = "test-v1"

    def ground(
        self,
        relation: SemanticRelationTarget,
        *,
        affordance: Affordance,
        context: PlanningContext,
    ) -> object:
        """Remain unreachable in provider-free catalog tests."""
        del relation, affordance, context
        raise AssertionError("Catalog tests must not execute live providers.")


class _CatalogPlaceAffordance(Affordance):
    """Typed provider-free payload marker for relation-linking tests."""


@dataclass(frozen=True, slots=True)
class _CatalogHandOverPoseProvider(HandOverPoseProvider):
    """Frozen declaration used to prove malicious drift detection."""

    provider_id: ClassVar[str] = "test.catalog_handover"
    transfer_height: float

    def resolve(
        self,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        """Remain unreachable in provider-free catalog tests."""
        del call, context, bound
        raise AssertionError("Catalog tests must not execute live providers.")


class _SecondCatalogRelationGrounder(_CatalogRelationGrounder):
    """Second stateless grounder used for ordering regressions."""

    capability: ClassVar[str] = "test.catalog_relation.second"
    affordance_revision: ClassVar[str] = "test-v2"


class _SecondCatalogHandOverPoseProvider(HandOverPoseProvider):
    """Second stateless hand-over provider used for ordering regressions."""

    provider_id: ClassVar[str] = "test.catalog_handover.second"

    def resolve(
        self,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        """Remain unreachable in provider-free catalog tests."""
        del call, context, bound
        raise AssertionError("Catalog tests must not execute live providers.")


class _StatefulCatalogRelationGrounder(_CatalogRelationGrounder):
    """Invalid non-dataclass provider with public instance state."""

    capability: ClassVar[str] = "test.catalog_relation.stateful"

    def __init__(self) -> None:
        self.height = 0.5


@dataclass(frozen=True, slots=True)
class _NestedMutableCatalogRelationGrounder(_CatalogRelationGrounder):
    """Invalid frozen provider retaining one mutable nested configuration."""

    capability: ClassVar[str] = "test.catalog_relation.mutable_nested"
    offsets: list[float]


class _PrivateSlotHandOverPoseProvider(HandOverPoseProvider):
    """Invalid provider whose state is hidden behind a mangled slot name."""

    __slots__ = ("__height",)

    provider_id: ClassVar[str] = "test.catalog_handover.private_slot"

    def __init__(self) -> None:
        self.__height = 0.5

    def resolve(
        self,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        """Remain unreachable because registration rejects this provider."""
        del call, context, bound
        raise AssertionError("Rejected providers must never execute.")


class _InheritedCachedHandOverPoseProvider(_CatalogHandOverPoseProvider):
    """Invalid non-dataclass subclass adding state to a frozen declaration."""

    __slots__ = ("cache",)

    provider_id: ClassVar[str] = "test.catalog_handover.inherited_cache"

    def __init__(self) -> None:
        super().__init__(transfer_height=0.5)
        object.__setattr__(self, "cache", {})


@dataclass(frozen=True, slots=True)
class _OpaqueHandOverPoseProvider(HandOverPoseProvider):
    """Provider declaration containing an unsupported opaque nested value."""

    provider_id: ClassVar[str] = "test.catalog_handover.opaque"
    opaque: object

    def resolve(
        self,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        """Remain unreachable because fingerprinting rejects this provider."""
        del call, context, bound
        raise AssertionError("Opaque providers must never reach runtime.")


class _AcceptParallelSafety:
    """Stateless safety sentinel returned by the registration-owned factory."""

    def validate(self, *, branch_frames: object, merged_frame: object) -> None:
        """Accept the provider-free test command without observing simulation."""
        del branch_frames, merged_frame


class _CatalogControlPartEvidenceProvider(EffectEvidenceProvider):
    """Live evidence sentinel returned by the registration-owned factory."""

    provider_id = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision = CONTROL_PART_EVIDENCE_PROVIDER_REVISION

    def collect(self, queries: object, context: object) -> object:
        """Remain unreachable in factory lifecycle tests."""
        del queries, context
        raise AssertionError("Catalog lifecycle tests must not collect evidence.")


@dataclass(frozen=True, slots=True)
class _CatalogControlPartEvidenceFactory:
    """Frozen declaration for the built-in control-part evidence route."""

    provider_id: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_REVISION
    source: str = "test.contact"

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        scene_provider: SceneProvider,
    ) -> EffectEvidenceProvider:
        """Return one fresh provider for every runtime assembly."""
        del simulation, robot, scene_registry, engine, scene_provider
        return _CatalogControlPartEvidenceProvider()


class _CatalogSceneProvider:
    """Structurally satisfy the scene-provider runtime boundary."""

    def snapshot(
        self,
        *,
        timestamp: float,
        env_ids: object,
    ) -> SceneSnapshot:
        del timestamp, env_ids
        raise AssertionError("Catalog lifecycle tests must not snapshot scenes.")


@dataclass(frozen=True, slots=True)
class _CatalogParallelSafetyFactory:
    """Frozen declaration covering the built-in transport exactly."""

    validator_id: ClassVar[str] = "test.catalog_parallel_safety"
    revision: ClassVar[str] = "1"
    supported_transport_ids: ClassVar[frozenset[str]] = frozenset(
        {"robot.joint_position"}
    )
    margin: float = 0.02

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> ParallelCommandSafetyValidator:
        """Return one independent protocol-compatible safety gate."""
        del simulation, robot, scene_registry, engine
        return _AcceptParallelSafety()


class _SerializedParallelSafetyFactory:
    """Instrument concurrent create calls without carrying instance state."""

    validator_id: ClassVar[str] = "test.serialized_parallel_safety"
    revision: ClassVar[str] = "1"
    supported_transport_ids: ClassVar[frozenset[str]] = frozenset(
        {"robot.joint_position"}
    )
    _state_lock: ClassVar[Lock] = Lock()
    _first_entered: ClassVar[Event] = Event()
    _second_entered: ClassVar[Event] = Event()
    _release_first: ClassVar[Event] = Event()
    _calls: ClassVar[int] = 0
    _active: ClassVar[int] = 0
    _max_active: ClassVar[int] = 0

    @classmethod
    def reset(cls) -> None:
        """Reset class-owned concurrency instrumentation for one test."""
        cls._first_entered = Event()
        cls._second_entered = Event()
        cls._release_first = Event()
        cls._calls = 0
        cls._active = 0
        cls._max_active = 0

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> ParallelCommandSafetyValidator:
        """Block the first call so a second call can attempt registration entry."""
        del simulation, robot, scene_registry, engine
        with self._state_lock:
            call_index = self._calls
            type(self)._calls += 1
            type(self)._active += 1
            type(self)._max_active = max(self._max_active, self._active)
        if call_index == 0:
            self._first_entered.set()
            if not self._release_first.wait(timeout=2.0):
                raise TimeoutError("Timed out waiting to release first safety create.")
        else:
            self._second_entered.set()
        with self._state_lock:
            type(self)._active -= 1
        return _AcceptParallelSafety()


@dataclass(frozen=True, slots=True)
class _CatalogCustomTrackingMetric(TrackingMetricCfg):
    """Metric with no built-in exact evaluator registration."""

    metric_id: ClassVar[str] = "test.catalog_metric"
    revision: ClassVar[str] = "1"
    channel_id: ClassVar[str] = "joint.position"


def _program_payload(
    *,
    scene_registry: str = CUBE_SCENE_REGISTRY_ID,
    runtime_preset: str = _CUBE_RUNTIME_PRESET_ID,
    object_id: str = "cube",
) -> dict[str, object]:
    """Return one minimal catalog-linked program payload."""
    return {
        "program_id": "catalog_pick",
        "integration": {
            "robot_profile": CUBE_ROBOT_PROFILE_ID,
            "scene_registry": scene_registry,
            "runtime_preset": runtime_preset,
        },
        "targets": {},
        "program": {
            "kind": "invoke",
            "call": {"kind": "pick", "object": object_id},
        },
    }


def _registration() -> SimulationExpertProgramRegistration:
    """Build one isolated provider-free task registration."""
    return SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=create_cube_robot_profile_binding(),
    )


def _parallel_live_inputs() -> tuple[object, SceneRegistry, AtomicActionEngine]:
    """Build minimal identity-consistent inputs for factory lifecycle tests."""
    robot = object()
    engine = AtomicActionEngine.__new__(AtomicActionEngine)
    engine._planning_services = SimpleNamespace(robot=robot)  # type: ignore[attr-defined]
    return robot, SceneRegistry(), engine


def _place_relation_catalog(
    *,
    install_grounder_key: bool,
) -> ExpertProgramIntegrationCatalog:
    """Build one provider-free placement catalog with an optional grounder key."""
    base = _registration().catalog
    support_ref = SceneObjectRef("support")
    affordance_ref = SceneAffordanceRef("support_top")
    scene = SceneManifest(
        (
            SceneEntityManifest(ref=SceneObjectRef("cube")),
            SceneEntityManifest(
                ref=support_ref,
                default_affordances={
                    PLACE_ON_AFFORDANCE_CAPABILITY: affordance_ref,
                },
            ),
            SceneEntityManifest(
                ref=affordance_ref,
                parent=support_ref,
                native_name="support_top_surface",
                affordance_capabilities=frozenset({PLACE_ON_AFFORDANCE_CAPABILITY}),
                affordance_payload_type=_CatalogPlaceAffordance,
                affordance_revision="test-v1",
            ),
        )
    )
    grounder_keys = (
        frozenset(
            {
                (
                    PLACE_ON_AFFORDANCE_CAPABILITY,
                    _CatalogPlaceAffordance,
                    "test-v1",
                )
            }
        )
        if install_grounder_key
        else frozenset()
    )
    return ExpertProgramIntegrationCatalog(
        scene_registry_id="relation_scene",
        robot_profile_id=base.robot_profile_id,
        scene=scene,
        robot_profile=base.robot_profile,
        call_catalog=base.call_catalog,
        relation_grounder_keys=grounder_keys,
        settle_preset_ids=base.settle_preset_ids,
        endpoint_adapter_declarations=base.endpoint_adapter_declarations,
        runtime_transport_declarations=base.runtime_transport_declarations,
        parallel_safety_declaration=base.parallel_safety_declaration,
        control_part_evidence_declaration=(base.control_part_evidence_declaration),
        registered_semantic_lowerer_declarations=(
            base.registered_semantic_lowerer_declarations
        ),
        fingerprint="0" * 64,
        _required_skills={},
    )


def _place_relation_payload() -> dict[str, object]:
    """Return one Place(on=object) program requiring relation grounding."""
    return {
        "program_id": "catalog_place_relation",
        "integration": {
            "robot_profile": CUBE_ROBOT_PROFILE_ID,
            "scene_registry": "relation_scene",
            "runtime_preset": _CUBE_RUNTIME_PRESET_ID,
        },
        "targets": {},
        "program": {
            "kind": "invoke",
            "call": {
                "kind": "place",
                "object": "cube",
                "on": "support",
            },
        },
    }


def _parallel_pick_payload() -> dict[str, object]:
    """Return one parallel program rooted at an exact config path."""
    return {
        "program_id": "catalog_parallel_pick",
        "integration": {
            "robot_profile": CUBE_ROBOT_PROFILE_ID,
            "scene_registry": CUBE_SCENE_REGISTRY_ID,
            "runtime_preset": _CUBE_RUNTIME_PRESET_ID,
        },
        "targets": {},
        "program": {
            "kind": "parallel",
            "branches": [
                {
                    "kind": "invoke",
                    "call": {"kind": "pick", "object": "cube"},
                },
                {
                    "kind": "invoke",
                    "call": {"kind": "pick", "object": "cube"},
                },
            ],
            "barrier": {
                "kind": "barrier",
                "name": "catalog_join",
                "timeout_steps": 40,
                "failure_policy": "fail_fast",
            },
        },
    }


def _registration_with_preset(
    preset: SkillPolicyPreset,
) -> SimulationExpertProgramRegistration:
    """Replace the Cube task's sole preset for registration validation tests."""
    binding = create_cube_robot_profile_binding()
    return SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=replace(
            binding,
            presets=(preset,),
            default_preset=preset.preset_id,
        ),
    )


def test_catalog_decodes_compiles_and_links_without_simulation() -> None:
    """All external references are linked before a simulation is available."""
    registration = _registration()

    program = decode_expert_program(
        _program_payload(),
        validation_context=registration.catalog,
    )
    compiled = registration.catalog.preflight(program)

    assert tuple(compiled.iter_segments())[0].calls[0].call.semantic_id == "pick"


def test_catalog_declares_builtin_endpoint_and_ordered_transport_contracts() -> None:
    """The standard provider-free catalog contains its exact built-in wiring."""
    catalog = _registration().catalog

    adapter = catalog.endpoint_adapter_declarations[ControlPartEndpoint]

    assert adapter.adapter_id == "control_part"
    assert adapter.runtime_transport_ids == frozenset({"robot.joint_position"})
    assert tuple(
        value.transport_id for value in catalog.runtime_transport_declarations
    ) == ("robot.joint_position",)


def test_control_part_evidence_factory_is_fingerprinted_and_fresh() -> None:
    """The catalog owns the physical route and each assembly gets a provider."""
    factory = _CatalogControlPartEvidenceFactory()
    registration = SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=create_cube_robot_profile_binding(),
        control_part_evidence_factory=factory,
    )
    without_factory = _registration()
    declaration = registration.catalog.control_part_evidence_declaration

    assert declaration is not None
    assert declaration.provider_id == CONTROL_PART_EVIDENCE_PROVIDER_ID
    assert declaration.revision == CONTROL_PART_EVIDENCE_PROVIDER_REVISION
    assert registration.fingerprint != without_factory.fingerprint

    robot, scene_registry, engine = _parallel_live_inputs()
    first = registration.create_control_part_evidence_provider(
        simulation=object(),
        robot=robot,
        scene_registry=scene_registry,
        engine=engine,
        scene_provider=_CatalogSceneProvider(),
    )
    second = registration.create_control_part_evidence_provider(
        simulation=object(),
        robot=robot,
        scene_registry=scene_registry,
        engine=engine,
        scene_provider=_CatalogSceneProvider(),
    )

    assert isinstance(first, EffectEvidenceProvider)
    assert isinstance(second, EffectEvidenceProvider)
    assert first is not second


def test_parallel_preflight_requires_registered_safety_factory_at_exact_path() -> None:
    """Parallel programs cannot defer physical-safety wiring to live startup."""
    registration = _registration()
    program = decode_expert_program(
        _parallel_pick_payload(),
        validation_context=registration.catalog,
    )

    with pytest.raises(ExpertProgramValidationError) as error:
        registration.catalog.preflight(program)

    assert error.value.code == "parallel_safety_factory_not_registered"
    assert error.value.path == ("program",)


def test_parallel_preflight_accepts_exact_registration_owned_safety_factory() -> None:
    """A factory declaration covers preflight and creates a fresh live gate."""
    registration = SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=create_cube_robot_profile_binding(),
        parallel_safety_factory=_CatalogParallelSafetyFactory(),
    )
    program = decode_expert_program(
        _parallel_pick_payload(),
        validation_context=registration.catalog,
    )

    compiled = registration.catalog.preflight(program)
    robot, scene_registry, engine = _parallel_live_inputs()
    validator = registration.create_parallel_safety_validator(
        simulation=object(),
        robot=robot,
        scene_registry=scene_registry,
        engine=engine,
    )

    assert tuple(compiled.iter_segments())[0].parallel_block is not None
    assert isinstance(validator, ParallelCommandSafetyValidator)


def test_parallel_safety_factory_must_return_a_validator() -> None:
    """A malformed registration-owned factory fails before runtime dispatch."""

    class InvalidParallelSafetyFactory:
        validator_id: ClassVar[str] = "test.invalid_parallel_safety"
        revision: ClassVar[str] = "1"
        supported_transport_ids: ClassVar[frozenset[str]] = frozenset(
            {"robot.joint_position"}
        )

        def create(
            self,
            *,
            simulation: object,
            robot: object,
            scene_registry: SceneRegistry,
            engine: AtomicActionEngine,
        ) -> object:
            del simulation, robot, scene_registry, engine
            return object()

    registration = SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=create_cube_robot_profile_binding(),
        parallel_safety_factory=InvalidParallelSafetyFactory(),
    )

    robot, scene_registry, engine = _parallel_live_inputs()
    with pytest.raises(TypeError, match="must return a ParallelCommandSafetyValidator"):
        registration.create_parallel_safety_validator(
            simulation=object(),
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )


def test_parallel_safety_creation_and_history_are_one_registration_lock_scope() -> None:
    """Concurrent assemblies cannot enter one registration factory together."""
    factory_type = _SerializedParallelSafetyFactory
    factory_type.reset()
    registration = SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=create_cube_robot_profile_binding(),
        parallel_safety_factory=factory_type(),
    )
    robot, scene_registry, engine = _parallel_live_inputs()

    def create_validator() -> ParallelCommandSafetyValidator | None:
        return registration.create_parallel_safety_validator(
            simulation=object(),
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(create_validator)
        assert factory_type._first_entered.wait(timeout=1.0)
        second = executor.submit(create_validator)
        assert not factory_type._second_entered.wait(timeout=0.05)
        factory_type._release_first.set()
        assert isinstance(first.result(timeout=1.0), ParallelCommandSafetyValidator)
        assert isinstance(second.result(timeout=1.0), ParallelCommandSafetyValidator)

    assert factory_type._calls == 2
    assert factory_type._max_active == 1


def test_standard_registration_requires_exact_registered_lowerer_factory_coverage() -> (
    None
):
    """Every registered descriptor has one fingerprinted lowerer factory."""
    catalog = builtin_semantic_call_catalog()
    target = _CATALOG_REGISTERED_TARGET
    assert target is not None and target.binding_contract is not None
    custom = SemanticCallDescriptor(
        call_id=_CATALOG_REGISTERED_CALL_ID,
        spec_type=RegisteredSemanticCall,
        target_descriptor=target,
    )

    with pytest.raises(ValueError, match="lowerer factories"):
        SimulationExpertProgramRegistration(
            scene_binding=create_cube_scene_binding(),
            robot_profile_binding=create_cube_robot_profile_binding(),
            call_catalog=catalog.with_descriptor(custom),
        )

    registration = SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=create_cube_robot_profile_binding(),
        call_catalog=catalog.with_descriptor(custom),
        registered_semantic_lowerer_factories=(_CatalogRegisteredLowererFactory(),),
    )

    declaration = registration.catalog.registered_semantic_lowerer_declarations[
        _CATALOG_REGISTERED_CALL_ID
    ]
    assert declaration.factory_type is _CatalogRegisteredLowererFactory
    assert declaration.revision == "1"

    robot, scene_registry, engine = _parallel_live_inputs()
    first = registration.create_registered_semantic_lowerers(
        simulation=object(),
        robot=robot,
        scene_registry=scene_registry,
        engine=engine,
    )
    second = registration.create_registered_semantic_lowerers(
        simulation=object(),
        robot=robot,
        scene_registry=scene_registry,
        engine=engine,
    )
    assert len(first) == len(second) == 1
    assert first[0] is not second[0]


def test_standard_registration_rejects_nonbuiltin_effect_monitor() -> None:
    """Custom effect-monitor factories cannot be injected after registration."""
    base = create_cube_robot_profile_binding().presets[0]
    preset = SkillPolicyPreset(
        base.preset_id,
        action_option_templates=base.action_option_templates,
        effect_assurance=EffectAssurance.VERIFIED,
        motion_policy=base.motion_policy,
        tracking_policy=base.tracking_policy,
        recovery_policy=base.recovery_policy,
        runner_cfg=base.runner_cfg,
        effect_monitors={"pick": EffectMonitorRef("test.monitor", "1")},
    )

    with pytest.raises(ValueError, match="non-built-in effect monitor"):
        _registration_with_preset(preset)


def test_standard_registration_rejects_tracking_metric_without_builtin_evaluator() -> (
    None
):
    """Metric evaluator availability is proven before simulation startup."""
    base = create_cube_robot_profile_binding().presets[0]
    preset = SkillPolicyPreset(
        base.preset_id,
        action_option_templates=base.action_option_templates,
        effect_assurance=base.effect_assurance,
        motion_policy=base.motion_policy,
        tracking_policy=TrackingPolicy(
            in_flight=InFlightTrackingPolicy(
                metrics=(_CatalogCustomTrackingMetric(),),
            ),
            terminal=TimedTerminalAcceptance(),
        ),
        recovery_policy=base.recovery_policy,
        runner_cfg=base.runner_cfg,
        effect_monitors=base.effect_monitors,
    )

    with pytest.raises(ValueError, match="no exact built-in evaluator"):
        _registration_with_preset(preset)


def test_catalog_rejects_linked_place_relation_without_exact_grounder() -> None:
    """A linked affordance cannot defer a missing typed grounder to runtime."""
    catalog = _place_relation_catalog(install_grounder_key=False)
    program = decode_expert_program(
        _place_relation_payload(),
        validation_context=catalog,
    )

    with pytest.raises(ExpertProgramValidationError) as error:
        catalog.preflight(program)

    assert error.value.code == "relation_grounder_not_registered"
    assert error.value.path == ("program", "call", "on")


def test_catalog_accepts_linked_place_relation_with_exact_grounder_key() -> None:
    """The capability, payload type, and revision must all match exactly."""
    catalog = _place_relation_catalog(install_grounder_key=True)
    program = decode_expert_program(
        _place_relation_payload(),
        validation_context=catalog,
    )

    compiled = catalog.preflight(program)

    assert tuple(compiled.iter_segments())[0].calls[0].call.semantic_id == "place"


def test_standard_support_binding_installs_grounder_without_task_code() -> None:
    """A task relation declaration supplies its production grounder implicitly."""
    base = create_cube_scene_binding()
    scene = replace(
        base,
        registry_id="relation_scene",
        rigid_objects=(
            *base.rigid_objects,
            SimulationRigidObjectBinding(
                entity_id="support",
                simulation_uid="support",
                dynamics=SceneDynamics.STATIC,
                semantic_type="support_surface",
            ),
        ),
        support_surfaces=(
            SupportSurfaceAffordanceBinding(
                entity_id="support_top",
                parent_id="support",
                native_name="top_object_target",
                is_default=True,
            ),
        ),
    )
    registration = SimulationExpertProgramRegistration(
        scene_binding=scene,
        robot_profile_binding=create_cube_robot_profile_binding(),
    )

    assert len(registration.relation_grounders) == 1
    assert type(registration.relation_grounders[0]) is (
        SupportSurfaceRelationTargetGrounder
    )
    assert registration.catalog.relation_grounder_keys == frozenset(
        {
            (
                PLACE_ON_AFFORDANCE_CAPABILITY,
                SupportSurfaceAffordance,
                PLACEMENT_TARGET_AFFORDANCE_REVISION,
            )
        }
    )

    program = decode_expert_program(
        _place_relation_payload(),
        validation_context=registration.catalog,
    )
    compiled = registration.catalog.preflight(program)

    assert tuple(compiled.iter_segments())[0].calls[0].call.semantic_id == "place"


@pytest.mark.parametrize(
    ("overrides", "path"),
    (
        ({"scene_registry": "other_scene"}, ("integration",)),
        ({"runtime_preset": "unknown"}, ("integration",)),
        ({"object_id": "unknown_object"}, ("program", "call", "object")),
    ),
)
def test_catalog_rejects_unknown_references_at_decode_time(
    overrides: dict[str, str],
    path: tuple[str, ...],
) -> None:
    """Invalid task integration references retain exact config paths."""
    registration = _registration()

    with pytest.raises(ExpertProgramValidationError) as error:
        decode_expert_program(
            _program_payload(**overrides),
            validation_context=registration.catalog,
        )

    assert error.value.path == path


def test_scene_declare_rejects_orphan_link_without_simulation() -> None:
    """Canonical topology failures do not reach native entity lookup."""
    binding = SimulationSceneBinding(
        registry_id="orphan_scene",
        links=(
            SimulationArticulationLinkBinding(
                entity_id="handle",
                articulation_id="missing_drawer",
                native_link_name="handle_link",
            ),
        ),
    )

    with pytest.raises(KeyError, match="missing_drawer"):
        binding.declare()


def test_fingerprint_is_stable_for_equivalent_declarations() -> None:
    """Fresh equivalent registrations produce the same canonical digest."""
    left = _registration()
    right = _registration()

    assert left.fingerprint == right.fingerprint
    assert len(left.fingerprint) == 64


def test_fingerprint_covers_workflow_recovery_policy() -> None:
    """A recovery budget is immutable registration-owned runtime behavior."""
    base = create_cube_robot_profile_binding().presets[0]
    changed = SkillPolicyPreset(
        base.preset_id,
        action_option_templates=base.action_option_templates,
        effect_assurance=base.effect_assurance,
        motion_policy=base.motion_policy,
        tracking_policy=base.tracking_policy,
        recovery_policy=base.recovery_policy,
        workflow_recovery_policy=WorkflowRecoveryPolicy(
            max_recovery_attempts=1,
        ),
        runner_cfg=base.runner_cfg,
        effect_monitors=base.effect_monitors,
    )

    assert _registration().fingerprint != _registration_with_preset(changed).fingerprint


def test_fingerprint_is_independent_of_catalog_and_provider_insertion_order() -> None:
    """Semantically equivalent unordered registration inputs hash identically."""
    descriptors = tuple(builtin_semantic_call_catalog().descriptors.values())
    first_relation = _CatalogRelationGrounder()
    second_relation = _SecondCatalogRelationGrounder()
    first_handover = _CatalogHandOverPoseProvider(transfer_height=0.6)
    second_handover = _SecondCatalogHandOverPoseProvider()
    common = {
        "scene_binding": create_cube_scene_binding(),
        "robot_profile_binding": create_cube_robot_profile_binding(),
    }
    forward = SimulationExpertProgramRegistration(
        **common,
        call_catalog=SemanticCallCatalog(descriptors),
        relation_grounders=(first_relation, second_relation),
        handover_pose_providers=(first_handover, second_handover),
    )
    reversed_registration = SimulationExpertProgramRegistration(
        **common,
        call_catalog=SemanticCallCatalog(tuple(reversed(descriptors))),
        relation_grounders=(second_relation, first_relation),
        handover_pose_providers=(second_handover, first_handover),
    )

    assert forward.fingerprint == reversed_registration.fingerprint


def test_fingerprint_owns_provider_ids_and_declarative_fields() -> None:
    """Provider identity and dataclass configuration are registration data."""
    provider = _CatalogHandOverPoseProvider(transfer_height=0.6)
    registration = SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=create_cube_robot_profile_binding(),
        relation_grounders=(_CatalogRelationGrounder(),),
        handover_pose_providers=(provider,),
    )
    changed_value = SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(),
        robot_profile_binding=create_cube_robot_profile_binding(),
        relation_grounders=(_CatalogRelationGrounder(),),
        handover_pose_providers=(_CatalogHandOverPoseProvider(transfer_height=0.7),),
    )

    assert registration.handover_pose_providers == (provider,)
    assert registration.fingerprint != changed_value.fingerprint
    object.__setattr__(provider, "transfer_height", 0.8)
    with pytest.raises(IntegrationFingerprintMismatch, match="changed"):
        registration.assert_unchanged()


def test_fingerprint_rejects_opaque_nested_declaration_values() -> None:
    """Unknown nested values cannot silently collapse to their Python type."""
    with pytest.raises(TypeError, match="unsupported value type"):
        SimulationExpertProgramRegistration(
            scene_binding=create_cube_scene_binding(),
            robot_profile_binding=create_cube_robot_profile_binding(),
            handover_pose_providers=(_OpaqueHandOverPoseProvider(opaque=object()),),
        )


def test_registration_rejects_duplicate_provider_keys_and_ids() -> None:
    """Provider lookup tables remain unambiguous before simulation startup."""
    common = {
        "scene_binding": create_cube_scene_binding(),
        "robot_profile_binding": create_cube_robot_profile_binding(),
    }

    with pytest.raises(ValueError, match="Duplicate relation grounder key"):
        SimulationExpertProgramRegistration(
            **common,
            relation_grounders=(
                _CatalogRelationGrounder(),
                _CatalogRelationGrounder(),
            ),
        )
    with pytest.raises(ValueError, match="Duplicate handover pose provider"):
        SimulationExpertProgramRegistration(
            **common,
            handover_pose_providers=(
                _CatalogHandOverPoseProvider(transfer_height=0.6),
                _CatalogHandOverPoseProvider(transfer_height=0.7),
            ),
        )


def test_registration_requires_immutable_provider_tuples() -> None:
    """Mutable provider containers cannot enter task registration metadata."""
    with pytest.raises(TypeError, match="relation_grounders must be an exact tuple"):
        SimulationExpertProgramRegistration(
            scene_binding=create_cube_scene_binding(),
            robot_profile_binding=create_cube_robot_profile_binding(),
            relation_grounders=[_CatalogRelationGrounder()],  # type: ignore[arg-type]
        )
    with pytest.raises(
        TypeError,
        match="handover_pose_providers must be an exact tuple",
    ):
        SimulationExpertProgramRegistration(
            scene_binding=create_cube_scene_binding(),
            robot_profile_binding=create_cube_robot_profile_binding(),
            handover_pose_providers=[  # type: ignore[arg-type]
                _CatalogHandOverPoseProvider(transfer_height=0.6)
            ],
        )


@pytest.mark.parametrize(
    ("field_name", "provider"),
    (
        ("relation_grounders", _StatefulCatalogRelationGrounder()),
        ("handover_pose_providers", _PrivateSlotHandOverPoseProvider()),
        (
            "handover_pose_providers",
            _InheritedCachedHandOverPoseProvider(),
        ),
    ),
)
def test_registration_rejects_stateful_non_dataclass_providers(
    field_name: str,
    provider: object,
) -> None:
    """Public and name-mangled provider state cannot evade fingerprinting."""
    kwargs = {field_name: (provider,)}

    with pytest.raises(TypeError, match="Use a frozen dataclass"):
        SimulationExpertProgramRegistration(
            scene_binding=create_cube_scene_binding(),
            robot_profile_binding=create_cube_robot_profile_binding(),
            **kwargs,
        )


def test_registration_rejects_nested_mutable_relation_grounder_state() -> None:
    """Catalog providers reuse the standard recursive immutability boundary."""
    with pytest.raises(TypeError, match="deeply immutable"):
        SimulationExpertProgramRegistration(
            scene_binding=create_cube_scene_binding(),
            robot_profile_binding=create_cube_robot_profile_binding(),
            relation_grounders=(_NestedMutableCatalogRelationGrounder(offsets=[0.1]),),
        )


def test_nested_declaration_drift_is_detected_before_live_build() -> None:
    """Mutable nested config cannot silently change a registered binding."""
    registration = _registration()
    object.__setattr__(
        registration.scene_binding.rigid_objects[0],
        "semantic_type",
        "changed_cube",
    )

    with pytest.raises(IntegrationFingerprintMismatch, match="changed"):
        registration.assert_unchanged()


def test_env_spec_keeps_typed_registration_out_of_gym_kwargs() -> None:
    """The integration catalog is metadata, not a duplicated Gym config source."""

    class _Environment:
        pass

    registration = _registration()
    spec = EnvSpec(
        "CatalogTest-v1",
        _Environment,
        default_kwargs={"physical_option": 3},
        expert_program_registration=registration,
    )

    assert spec.expert_program_registration is registration
    assert spec.gym_spec.kwargs == {"physical_option": 3}


def test_env_spec_derives_registration_and_injects_runtime_factory() -> None:
    """One runtime declaration owns static preflight and live environment binding."""

    class _Environment:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    registration = _registration()
    factory = SimulationExpertProgramAdapterFactory(registration)
    spec = EnvSpec(
        "RuntimeFactoryTest-v1",
        _Environment,
        default_kwargs={"physical_option": 3},
        expert_program_adapter_factory=factory,
    )

    environment = spec.make(cfg=object())

    assert spec.expert_program_registration is registration
    assert spec.expert_program_adapter_factory is factory
    assert environment.kwargs["expert_program_adapter_factory"] is factory
    assert spec.gym_spec.kwargs == {"physical_option": 3}

    with pytest.raises(ValueError, match="cannot be overridden"):
        spec.make(
            cfg=object(),
            expert_program_adapter_factory=SimulationExpertProgramAdapterFactory(
                registration
            ),
        )
