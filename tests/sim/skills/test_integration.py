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

"""Pure-Python tests for static semantic integration."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    Affordance,
    AntipodalAffordance,
    AtomicActionEngine,
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    ControlPartCommandProfile,
    EntityState,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
)
from embodichain.lab.sim.skills.calls import (
    Pick,
    RegisteredSemanticCall,
    SemanticCallCatalog,
    SemanticCallDescriptor,
    builtin_semantic_call_catalog,
)
from embodichain.lab.sim.skills.integration import (
    BoundSemanticCall,
    SceneEntityManifest,
    SceneManifest,
    SemanticIntegrationManifest,
    SemanticValidationError,
)
from embodichain.lab.sim.skills.profiles import (
    ControlPartEndpoint,
    RobotResource,
    RobotSkillProfile,
    SkillPolicyPreset,
)
from embodichain.lab.sim.skills.scene import (
    AmbiguousSceneAffordanceError,
    GRASP_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    SceneAffordanceRef,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
    UnsupportedSceneAffordanceError,
)

_MOTION_CAPABILITIES = frozenset(
    {
        BATCH_INVERSE_KINEMATICS_CAPABILITY,
        CARTESIAN_POSE_CAPABILITY,
        FORWARD_KINEMATICS_CAPABILITY,
    }
)


class _NeverObservedStateProvider:
    """Fail if provider-backed state leaks into static validation."""

    def __init__(self) -> None:
        self.calls = 0

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        self.calls += 1
        raise AssertionError("static semantic validation must not observe providers")


class _CopyTrackedAffordance(AntipodalAffordance):
    """Count payload copies so metadata projection can prove it performs none."""

    copies = 0

    def __deepcopy__(self, memo: dict[int, object]) -> _CopyTrackedAffordance:
        del memo
        type(self).copies += 1
        return _CopyTrackedAffordance()


def _scene_registry(
    *,
    with_default: bool,
) -> tuple[SceneRegistry, _NeverObservedStateProvider]:
    provider = _NeverObservedStateProvider()
    object_ref = SceneObjectRef("cube")
    side_grasp = SceneAffordanceRef("cube.grasp.side")
    top_grasp = SceneAffordanceRef("cube.grasp.top")
    defaults = {GRASP_AFFORDANCE_CAPABILITY: top_grasp} if with_default else {}
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=object_ref,
                state_provider=provider,
                aliases=("sim_cube",),
                default_affordances=defaults,
            ),
            SceneEntityRegistration(
                ref=side_grasp,
                parent=object_ref,
                native_name="side_grasp",
                affordance=AntipodalAffordance(),
                affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                affordance_revision="grasp-v1",
                relative_pose=torch.eye(4),
            ),
            SceneEntityRegistration(
                ref=top_grasp,
                parent=object_ref,
                native_name="top_grasp",
                affordance=AntipodalAffordance(),
                affordance_capabilities=frozenset(
                    {
                        GRASP_AFFORDANCE_CAPABILITY,
                        PLACE_ON_AFFORDANCE_CAPABILITY,
                    }
                ),
                affordance_revision="grasp-v1",
                relative_pose=torch.eye(4),
            ),
        )
    )
    return registry, provider


def _semantic_integration(
    registry: SceneRegistry,
) -> SemanticIntegrationManifest:
    robot_profile = RobotSkillProfile(
        profile_id="test_robot",
        resources={
            "manipulator": RobotResource(
                resource_id="manipulator",
                endpoints={
                    "motion": ControlPartEndpoint(
                        control_part="arm",
                        capabilities=_MOTION_CAPABILITIES,
                    ),
                    "grasp": ControlPartEndpoint(
                        control_part="hand",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                    ),
                },
            )
        },
        command_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.0]),
                grasp=torch.tensor([1.0]),
            )
        },
        presets={"safe": SkillPolicyPreset("safe")},
        default_preset="safe",
    )
    return SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=robot_profile,
        call_catalog=builtin_semantic_call_catalog(),
    )


def _engine_for_integration(
    integration: SemanticIntegrationManifest,
) -> AtomicActionEngine:
    """Build a minimal live engine whose resource graph matches the manifest."""
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 2
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_qpos.return_value = torch.zeros(1, 2)
    robot.get_qvel.return_value = torch.zeros(1, 2)
    robot.get_joint_ids.side_effect = lambda name: {
        "arm": [0],
        "hand": [1],
    }[name]
    robot.get_solver.side_effect = lambda name=None: (
        object() if name == "arm" else None
    )
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    return AtomicActionEngine(
        generator,
        skill_profile=integration.robot_profile,
    )


def test_scene_registry_filters_capabilities_and_uses_scoped_default() -> None:
    registry, _ = _scene_registry(with_default=True)

    assert registry.affordances(
        "sim_cube",
        capability=GRASP_AFFORDANCE_CAPABILITY,
    ) == (
        SceneAffordanceRef("cube.grasp.side"),
        SceneAffordanceRef("cube.grasp.top"),
    )
    assert registry.affordances(
        "cube",
        capability=PLACE_ON_AFFORDANCE_CAPABILITY,
    ) == (SceneAffordanceRef("cube.grasp.top"),)
    assert registry.resolve_affordance(
        "cube",
        capability=GRASP_AFFORDANCE_CAPABILITY,
    ) == SceneAffordanceRef("cube.grasp.top")
    assert registry.resolve_affordance(
        "cube",
        capability=GRASP_AFFORDANCE_CAPABILITY,
        explicit="cube.grasp.side",
    ) == SceneAffordanceRef("cube.grasp.side")


def test_scene_registry_rejects_ambiguous_or_unsupported_affordance() -> None:
    registry, _ = _scene_registry(with_default=False)

    with pytest.raises(AmbiguousSceneAffordanceError, match="multiple affordances"):
        registry.resolve_affordance(
            "cube",
            capability=GRASP_AFFORDANCE_CAPABILITY,
        )
    with pytest.raises(UnsupportedSceneAffordanceError, match="no affordance"):
        registry.resolve_affordance(
            "cube",
            capability="affordance.place.inside",
        )


def test_scene_registry_rejects_untyped_or_unversioned_grasp_capability() -> None:
    object_ref = SceneObjectRef("cube")
    base = dict(
        ref=SceneAffordanceRef("cube.grasp"),
        parent=object_ref,
        native_name="grasp",
        relative_pose=torch.eye(4),
        affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
    )

    with pytest.raises(TypeError, match="AntipodalAffordance"):
        SceneEntityRegistration(
            **base,
            affordance=Affordance(),
            affordance_revision="v1",
        )
    with pytest.raises(ValueError, match="affordance_revision"):
        SceneEntityRegistration(
            **base,
            affordance=AntipodalAffordance(),
        )


def test_scene_registry_rejects_default_reference_subclass() -> None:
    class SpecialAffordanceRef(SceneAffordanceRef):
        pass

    with pytest.raises(TypeError, match="SceneAffordanceRef"):
        SceneEntityRegistration(
            ref=SceneObjectRef("cube"),
            state_provider=_NeverObservedStateProvider(),
            default_affordances={
                GRASP_AFFORDANCE_CAPABILITY: SpecialAffordanceRef("cube.grasp")
            },
        )


def test_scene_manifest_projection_does_not_copy_affordance_payload() -> None:
    provider = _NeverObservedStateProvider()
    object_ref = SceneObjectRef("cube")
    _CopyTrackedAffordance.copies = 0
    registry = SceneRegistry(
        (
            SceneEntityRegistration(ref=object_ref, state_provider=provider),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("cube.grasp"),
                parent=object_ref,
                native_name="grasp",
                relative_pose=torch.eye(4),
                affordance=_CopyTrackedAffordance(),
                affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                affordance_revision="v1",
            ),
        )
    )
    _CopyTrackedAffordance.copies = 0

    SceneManifest.from_registry(registry)

    assert _CopyTrackedAffordance.copies == 0
    assert provider.calls == 0


def test_scene_manifest_detects_grounding_metadata_drift() -> None:
    provider = _NeverObservedStateProvider()
    object_ref = SceneObjectRef("cube")

    def registry(native_name: str, revision: str) -> SceneRegistry:
        return SceneRegistry(
            (
                SceneEntityRegistration(ref=object_ref, state_provider=provider),
                SceneEntityRegistration(
                    ref=SceneAffordanceRef("cube.grasp"),
                    parent=object_ref,
                    native_name=native_name,
                    relative_pose=torch.eye(4),
                    affordance=AntipodalAffordance(),
                    affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                    affordance_revision=revision,
                ),
            )
        )

    manifest = SceneManifest.from_registry(registry("grasp", "v1"))

    with pytest.raises(SemanticValidationError) as error:
        manifest.validate_registry(registry("changed", "v2"))

    assert error.value.diagnostic.code == "scene_manifest_mismatch"


def test_scene_manifest_rejects_impossible_typed_topology() -> None:
    affordance = SceneAffordanceRef("self")

    with pytest.raises(ValueError, match="object, articulation, or link"):
        SceneEntityManifest(
            ref=affordance,
            parent=affordance,
            native_name="self",
            affordance_payload_type=AntipodalAffordance,
            affordance_revision="v1",
        )


def test_scene_manifest_rejects_entry_subclass_with_live_state() -> None:
    class LiveManifest(SceneEntityManifest):
        live_handle = object()

    with pytest.raises(TypeError, match="exact SceneEntityManifest"):
        SceneManifest((LiveManifest(ref=SceneObjectRef("cube")),))


def test_semantic_integration_rejects_catalog_subclass_with_behavior() -> None:
    class LiveCatalog(SemanticCallCatalog):
        live_handle = object()

    registry, _ = _scene_registry(with_default=True)
    integration = _semantic_integration(registry)
    live_catalog = LiveCatalog(integration.call_catalog.descriptors.values())

    with pytest.raises(TypeError, match="exactly SemanticCallCatalog"):
        SemanticIntegrationManifest(
            scene=integration.scene,
            robot_profile=integration.robot_profile,
            call_catalog=live_catalog,
        )


def test_scene_manifest_reports_structured_pathful_diagnostic() -> None:
    manifest = SceneManifest((SceneEntityManifest(ref=SceneObjectRef("cube")),))

    with pytest.raises(SemanticValidationError) as error:
        manifest.resolve(
            "missing",
            expected_type=SceneObjectRef,
            path=("program", 2, "object"),
        )

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "unknown_entity"
    assert diagnostic.path == ("program", 2, "object")
    assert diagnostic.rendered_path == "program[2].object"
    assert diagnostic.candidates == ("cube",)
    assert str(error.value).startswith("program[2].object:")


def test_static_integration_links_resources_and_affordances_without_observation() -> (
    None
):
    registry, provider = _scene_registry(with_default=True)
    integration = _semantic_integration(registry)

    linked = integration.link_call(
        Pick(
            object=SceneObjectRef("cube"),
            resources={"primary": "manipulator"},
        ),
        path=("program", 0),
    )
    integration.scene.validate_registry(registry)

    assert linked.descriptor.skill_id == "pick_up"
    assert linked.preset_id == "safe"
    assert linked.call.resources == {"primary": "manipulator"}
    assert isinstance(linked.call, Pick)
    assert linked.call.grasp == SceneAffordanceRef("cube.grasp.top")
    assert linked.affordances == {"grasp": SceneAffordanceRef("cube.grasp.top")}
    assert provider.calls == 0


def test_static_integration_rejects_unknown_resource_with_complete_path() -> None:
    registry, provider = _scene_registry(with_default=True)
    integration = _semantic_integration(registry)

    with pytest.raises(SemanticValidationError) as error:
        integration.link_call(
            Pick(
                object=SceneObjectRef("cube"),
                resources={"primary": "missing"},
            ),
            path=("program", 3),
        )

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "unknown_resource"
    assert diagnostic.path == ("program", 3, "resources", "primary")
    assert diagnostic.rendered_path == "program[3].resources.primary"
    assert diagnostic.candidates == ("manipulator",)
    assert provider.calls == 0


def test_static_integration_preserves_scene_path_without_observing_provider() -> None:
    registry, provider = _scene_registry(with_default=True)
    integration = _semantic_integration(registry)

    with pytest.raises(SemanticValidationError) as error:
        integration.link_call(
            Pick(
                object=SceneObjectRef("missing"),
                resources={"primary": "manipulator"},
            ),
            path=("program", 4),
        )

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "unknown_entity"
    assert diagnostic.rendered_path == "program[4].object"
    assert provider.calls == 0


def test_registered_payload_scene_refs_are_statically_resolved() -> None:
    registry, provider = _scene_registry(with_default=True)
    integration = _semantic_integration(registry)
    pick = integration.call_catalog.discover("pick")
    extension = SemanticCallDescriptor(
        call_id="vendor.inspect",
        spec_type=RegisteredSemanticCall,
        skill_id=pick.skill_id,
        binding_contract=pick.binding_contract,
        target_descriptor=pick.target_descriptor,
    )
    integration = SemanticIntegrationManifest(
        scene=integration.scene,
        robot_profile=integration.robot_profile,
        call_catalog=integration.call_catalog.with_descriptor(extension),
    )

    with pytest.raises(SemanticValidationError) as error:
        integration.link_call(
            RegisteredSemanticCall(
                call_id="vendor.inspect",
                arguments={"object": SceneObjectRef("missing")},
                resources={"primary": "manipulator"},
            ),
            path=("program", 5, "call"),
        )

    assert error.value.diagnostic.code == "unknown_entity"
    assert error.value.diagnostic.rendered_path == ("program[5].call.arguments.object")
    assert provider.calls == 0


def test_bound_semantic_call_is_factory_owned_by_installed_profile() -> None:
    registry, _ = _scene_registry(with_default=True)
    integration = _semantic_integration(registry)
    engine = _engine_for_integration(integration)
    bound_integration = integration.bind(registry, engine)

    result = bound_integration.link_call(Pick(object=SceneObjectRef("cube")))

    assert result.robot_profile is bound_integration.robot_profile
    assert result.binding.action_binding.owner_id == engine.binding_owner_id
    with pytest.raises(TypeError, match="created by"):
        BoundSemanticCall()


def test_bound_semantic_integration_rejects_engine_profile_rebind() -> None:
    registry, _ = _scene_registry(with_default=True)
    integration = _semantic_integration(registry)
    engine = _engine_for_integration(integration)
    stale = integration.bind(registry, engine)

    engine.bind_skill_profile(integration.robot_profile)

    with pytest.raises(SemanticValidationError) as error:
        stale.link_call(Pick(object=SceneObjectRef("cube")))

    assert error.value.diagnostic.code == "semantic_profile_stale"


def test_bound_semantic_integration_rejects_manifest_subclass_behavior() -> None:
    class LiveManifest(SemanticIntegrationManifest):
        live_handle = object()

    registry, _ = _scene_registry(with_default=True)
    integration = _semantic_integration(registry)
    engine = _engine_for_integration(integration)
    bound_profile = engine.skill_profile
    assert bound_profile is not None
    live_manifest = LiveManifest(
        scene=integration.scene,
        robot_profile=integration.robot_profile,
        call_catalog=integration.call_catalog,
    )

    with pytest.raises(TypeError, match="exactly SemanticIntegrationManifest"):
        type(integration.bind(registry, engine))(
            manifest=live_manifest,
            scene_registry=registry,
            robot_profile=bound_profile,
            engine=engine,
        )
