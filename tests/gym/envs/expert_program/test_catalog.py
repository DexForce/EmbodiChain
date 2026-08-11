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

from dataclasses import dataclass
from typing import ClassVar

import pytest

from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramIntegrationCatalog,
    ExpertProgramValidationError,
    IntegrationFingerprintMismatch,
    SimulationArticulationLinkBinding,
    SimulationExpertProgramRegistration,
    SimulationSceneBinding,
    decode_expert_program,
)
from embodichain.lab.gym.utils.registration import EnvSpec
from embodichain.lab.sim.atomic_actions import Affordance, PlanningContext
from embodichain.lab.sim.skills import (
    PLACE_ON_AFFORDANCE_CAPABILITY,
    BoundSemanticCall,
    HandOver,
    HandOverPoseProvider,
    HandOverPoseTargets,
    OperateArticulation,
    RelationTargetGrounder,
    SemanticCallCatalog,
    SceneAffordanceRef,
    SceneEntityManifest,
    SceneManifest,
    SceneObjectRef,
    SemanticRelationTarget,
    builtin_semantic_call_catalog,
)
from embodichain_tasks.multi_segments.cube_pick_place import (
    CUBE_ROBOT_PROFILE_ID,
    CUBE_SCENE_REGISTRY_ID,
    create_cube_robot_profile_binding,
    create_cube_scene_binding,
)
from embodichain_tasks.tableware.open_drawer import (
    DRAWER_HANDLE_AFFORDANCE_ID,
    DRAWER_ROBOT_PROFILE_ID,
    DRAWER_SCENE_REGISTRY_ID,
    DRAWER_UID,
    OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION,
)


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


def _program_payload(
    *,
    scene_registry: str = CUBE_SCENE_REGISTRY_ID,
    runtime_preset: str = "safe",
    object_id: str = "cube",
) -> dict[str, object]:
    """Return one minimal catalog-linked program payload."""
    return {
        "schema_version": 1,
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
        scene_binding=create_cube_scene_binding(grasp_samples=32),
        robot_profile_binding=create_cube_robot_profile_binding(),
    )


def _operate_articulation_payload(
    *,
    target: str,
    handle: str | None = None,
) -> dict[str, object]:
    """Return one named drawer-operation program with an optional handle."""
    call: dict[str, object] = {
        "kind": "operate_articulation",
        "articulation": DRAWER_UID,
        "target": target,
    }
    if handle is not None:
        call["handle"] = handle
    return {
        "schema_version": 1,
        "program_id": "catalog_open_drawer",
        "integration": {
            "robot_profile": DRAWER_ROBOT_PROFILE_ID,
            "scene_registry": DRAWER_SCENE_REGISTRY_ID,
            "runtime_preset": "safe",
        },
        "targets": {},
        "program": {"kind": "invoke", "call": call},
    }


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
        articulation_operation_targets={},
        settle_preset_ids=base.settle_preset_ids,
        fingerprint="0" * 64,
        _required_skills={},
    )


def _place_relation_payload() -> dict[str, object]:
    """Return one Place(on=object) program requiring relation grounding."""
    return {
        "schema_version": 1,
        "program_id": "catalog_place_relation",
        "integration": {
            "robot_profile": CUBE_ROBOT_PROFILE_ID,
            "scene_registry": "relation_scene",
            "runtime_preset": "safe",
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


def test_catalog_decodes_compiles_and_links_without_simulation() -> None:
    """All external references are linked before a simulation is available."""
    registration = _registration()

    program = decode_expert_program(
        _program_payload(),
        validation_context=registration.catalog,
    )
    compiled = registration.catalog.preflight(program)

    assert tuple(compiled.iter_segments())[0].calls[0].call.semantic_id == "pick"


@pytest.mark.parametrize("validation_stage", ("decode", "preflight"))
def test_catalog_rejects_unknown_named_articulation_target_at_exact_path(
    validation_stage: str,
) -> None:
    """Unknown provider-owned target IDs fail before simulation startup."""
    catalog = OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION.catalog
    payload = _operate_articulation_payload(target="does_not_exist")

    with pytest.raises(ExpertProgramValidationError) as error:
        if validation_stage == "decode":
            decode_expert_program(payload, validation_context=catalog)
        else:
            catalog.preflight(decode_expert_program(payload))

    assert error.value.code == "unknown_articulation_operation_target"
    assert error.value.path == ("program", "call", "target")


@pytest.mark.parametrize("handle", (None, DRAWER_HANDLE_AFFORDANCE_ID))
def test_catalog_accepts_named_target_through_default_or_explicit_affordance(
    handle: str | None,
) -> None:
    """Both handle-selection forms resolve the same registered target table."""
    catalog = OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION.catalog
    program = decode_expert_program(
        _operate_articulation_payload(target="open", handle=handle),
        validation_context=catalog,
    )

    compiled = catalog.preflight(program)

    call = tuple(compiled.iter_segments())[0].calls[0].call
    assert type(call) is OperateArticulation
    assert call.target == "open"


def test_catalog_owns_immutable_articulation_operation_target_metadata() -> None:
    """Named target IDs are a read-only task-registration catalog surface."""
    targets = (
        OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION.catalog.articulation_operation_targets
    )

    assert targets == {DRAWER_HANDLE_AFFORDANCE_ID: frozenset({"open"})}
    with pytest.raises(TypeError):
        targets[DRAWER_HANDLE_AFFORDANCE_ID] = frozenset()  # type: ignore[index]


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


def test_fingerprint_is_independent_of_catalog_and_provider_insertion_order() -> None:
    """Semantically equivalent unordered registration inputs hash identically."""
    descriptors = tuple(builtin_semantic_call_catalog().descriptors.values())
    first_relation = _CatalogRelationGrounder()
    second_relation = _SecondCatalogRelationGrounder()
    first_handover = _CatalogHandOverPoseProvider(transfer_height=0.6)
    second_handover = _SecondCatalogHandOverPoseProvider()
    common = {
        "scene_binding": create_cube_scene_binding(grasp_samples=32),
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
        scene_binding=create_cube_scene_binding(grasp_samples=32),
        robot_profile_binding=create_cube_robot_profile_binding(),
        relation_grounders=(_CatalogRelationGrounder(),),
        handover_pose_providers=(provider,),
    )
    changed_value = SimulationExpertProgramRegistration(
        scene_binding=create_cube_scene_binding(grasp_samples=32),
        robot_profile_binding=create_cube_robot_profile_binding(),
        relation_grounders=(_CatalogRelationGrounder(),),
        handover_pose_providers=(_CatalogHandOverPoseProvider(transfer_height=0.7),),
    )

    assert registration.handover_pose_providers == (provider,)
    assert registration.fingerprint != changed_value.fingerprint
    object.__setattr__(provider, "transfer_height", 0.8)
    with pytest.raises(IntegrationFingerprintMismatch, match="changed"):
        registration.assert_unchanged()


def test_registration_rejects_duplicate_provider_keys_and_ids() -> None:
    """Provider lookup tables remain unambiguous before simulation startup."""
    common = {
        "scene_binding": create_cube_scene_binding(grasp_samples=32),
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
            scene_binding=create_cube_scene_binding(grasp_samples=32),
            robot_profile_binding=create_cube_robot_profile_binding(),
            relation_grounders=[_CatalogRelationGrounder()],  # type: ignore[arg-type]
        )
    with pytest.raises(
        TypeError,
        match="handover_pose_providers must be an exact tuple",
    ):
        SimulationExpertProgramRegistration(
            scene_binding=create_cube_scene_binding(grasp_samples=32),
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
            scene_binding=create_cube_scene_binding(grasp_samples=32),
            robot_profile_binding=create_cube_robot_profile_binding(),
            **kwargs,
        )


def test_nested_declaration_drift_is_detected_before_live_build() -> None:
    """Mutable nested config cannot silently change a registered binding."""
    registration = _registration()
    generator_cfg = registration.scene_binding.antipodal_grasps[0].generator_cfg
    assert generator_cfg is not None
    generator_cfg.antipodal_sampler_cfg.n_sample = 64

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
