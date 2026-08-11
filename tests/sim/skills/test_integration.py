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
    DynamicCollisionMode,
    EntityState,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    HandOverOptions,
    MotionPolicy,
    OperateArticulationOptions,
    PickUpOptions,
    PlaceOptions,
)
from embodichain.lab.sim.skills.calls import (
    Pick,
    RegisteredSemanticCall,
    SemanticCallCatalog,
    SemanticCallDescriptor,
    builtin_semantic_call_catalog,
)
from embodichain.lab.sim.skills.effects import EffectMonitorRef
from embodichain.lab.sim.skills.integration import (
    BoundSemanticCall,
    BoundSemanticIntegration,
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
    WorkflowRecoveryPolicy,
)
from embodichain.lab.sim.skills.scene import (
    AmbiguousSceneAffordanceError,
    GRASP_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    SceneAffordanceRef,
    SceneCollisionRole,
    SceneCollisionWorldMode,
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


def _action_option_templates() -> dict[str, object]:
    """Return exact built-in semantic-call option declarations."""
    return {
        "pick": PickUpOptions(),
        "place": PlaceOptions(),
        "hand_over": HandOverOptions(),
        "operate_articulation": OperateArticulationOptions(),
    }


def _preset(preset_id: str, **kwargs: object) -> SkillPolicyPreset:
    """Build one complete schema-v3 test preset."""
    kwargs.setdefault("action_option_templates", _action_option_templates())
    return SkillPolicyPreset(preset_id, **kwargs)


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


class _GeometryProvider:
    """Return one opaque planner-facing geometry descriptor."""

    def get_geometry(self) -> object:
        return object()


def _scene_registry(
    *,
    with_default: bool,
    dynamic_collision: bool = False,
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
                geometry_provider=(_GeometryProvider() if dynamic_collision else None),
                collision_role=(
                    SceneCollisionRole.DYNAMIC
                    if dynamic_collision
                    else SceneCollisionRole.NONE
                ),
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
        ),
        collision_world_mode=(
            SceneCollisionWorldMode.PER_ENV if dynamic_collision else None
        ),
    )
    return registry, provider


def _semantic_integration(
    registry: SceneRegistry,
    *,
    preset: SkillPolicyPreset | None = None,
    additional_presets: tuple[SkillPolicyPreset, ...] = (),
    default_preset: str | None = None,
    skill_presets: dict[str, str] | None = None,
    runtime_preset: str | None = None,
) -> SemanticIntegrationManifest:
    selected_preset = _preset("safe") if preset is None else preset
    presets = {selected_preset.preset_id: selected_preset}
    presets.update(
        {
            additional_preset.preset_id: additional_preset
            for additional_preset in additional_presets
        }
    )
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
        presets=presets,
        default_preset=(
            selected_preset.preset_id if default_preset is None else default_preset
        ),
        skill_presets={} if skill_presets is None else skill_presets,
    )
    return SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=robot_profile,
        call_catalog=builtin_semantic_call_catalog(),
        runtime_preset=runtime_preset,
    )


def _engine_for_integration(
    integration: SemanticIntegrationManifest,
    *,
    supports_dynamic_collision_world: bool = False,
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
    generator.supports_dynamic_collision_world = supports_dynamic_collision_world
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


def test_semantic_integration_rejects_monitor_for_unknown_call_with_path() -> None:
    registry, _ = _scene_registry(with_default=True)
    unknown_semantic_id = "not_catalogued"

    with pytest.raises(SemanticValidationError) as error:
        _semantic_integration(
            registry,
            preset=_preset(
                "safe",
                effect_monitors={
                    unknown_semantic_id: EffectMonitorRef("test.monitor", "1")
                },
            ),
        )

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "unknown_effect_monitor_call"
    assert diagnostic.path == (
        "integration",
        "robot_profile",
        "presets",
        "safe",
        "effect_monitors",
        unknown_semantic_id,
    )
    assert diagnostic.rendered_path == (
        "integration.robot_profile.presets.safe.effect_monitors.not_catalogued"
    )


def test_semantic_integration_rejects_unknown_action_option_call() -> None:
    registry, _ = _scene_registry(with_default=True)

    with pytest.raises(SemanticValidationError) as error:
        _semantic_integration(
            registry,
            preset=SkillPolicyPreset(
                "safe",
                action_option_templates={"vendor.unknown": PickUpOptions()},
            ),
        )

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "unknown_action_option_call"
    assert diagnostic.path[-2:] == (
        "action_option_templates",
        "vendor.unknown",
    )


def test_semantic_integration_validates_exact_action_option_type() -> None:
    registry, _ = _scene_registry(with_default=True)

    with pytest.raises(SemanticValidationError) as error:
        _semantic_integration(
            registry,
            preset=SkillPolicyPreset(
                "safe",
                action_option_templates={"pick": PlaceOptions()},
            ),
        )

    assert error.value.diagnostic.code == "incompatible_action_option_template"
    assert error.value.diagnostic.path[-1] == "pick"


def test_semantic_integration_rejects_compiler_owned_option_fields() -> None:
    registry, _ = _scene_registry(with_default=True)

    with pytest.raises(SemanticValidationError) as pick_error:
        _semantic_integration(
            registry,
            preset=SkillPolicyPreset(
                "safe",
                action_option_templates={
                    "pick": PickUpOptions(
                        downstream_object_target_poses=(torch.eye(4),)
                    )
                },
            ),
        )
    assert pick_error.value.diagnostic.code == "reserved_action_option_field"
    assert pick_error.value.diagnostic.path[-1] == ("downstream_object_target_poses")

    with pytest.raises(SemanticValidationError) as handover_error:
        _semantic_integration(
            registry,
            preset=SkillPolicyPreset(
                "safe",
                action_option_templates={
                    "hand_over": HandOverOptions(
                        middle_object_pose=torch.eye(4),
                    )
                },
            ),
        )
    assert handover_error.value.diagnostic.code == "reserved_action_option_field"
    assert handover_error.value.diagnostic.path[-1] == "middle_object_pose"


def test_static_link_requires_selected_preset_action_option_template() -> None:
    registry, _ = _scene_registry(with_default=True)
    integration = _semantic_integration(
        registry,
        preset=SkillPolicyPreset("safe", action_option_templates={}),
    )

    with pytest.raises(SemanticValidationError) as error:
        integration.link_call(Pick(object=SceneObjectRef("cube")))

    assert error.value.diagnostic.code == "missing_action_option_template"
    assert error.value.diagnostic.path == (
        "integration",
        "robot_profile",
        "presets",
        "safe",
        "action_option_templates",
        "pick",
    )
    assert "selected at call" in error.value.diagnostic.message


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


@pytest.mark.parametrize(
    "source_mode",
    [
        DynamicCollisionMode.AUTO,
        DynamicCollisionMode.OFF,
        DynamicCollisionMode.REQUIRED,
    ],
)
def test_safe_preset_requires_dynamic_collision_for_dynamic_scene(
    source_mode: DynamicCollisionMode,
) -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(
                strategy="motion_gen",
                dynamic_collision_mode=source_mode,
            ),
            workflow_recovery_policy=WorkflowRecoveryPolicy(
                max_recovery_attempts=2,
            ),
        ),
    )
    engine = _engine_for_integration(
        integration,
        supports_dynamic_collision_world=True,
    )

    bound = integration.bind(registry, engine).link_call(
        Pick(object=SceneObjectRef("cube"))
    )

    assert (
        bound.preset.motion_policy.dynamic_collision_mode
        is DynamicCollisionMode.REQUIRED
    )
    assert (
        integration.robot_profile.presets["safe"].motion_policy.dynamic_collision_mode
        is source_mode
    )
    assert bound.preset.workflow_recovery_policy.max_recovery_attempts == 2
    assert provider.calls == 0


def test_safe_preset_rejects_unsupported_dynamic_planner_before_observation() -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(strategy="motion_gen"),
        ),
    )
    engine = _engine_for_integration(integration)
    bind_skill_profile = Mock(wraps=engine.bind_skill_profile)
    engine.bind_skill_profile = bind_skill_profile  # type: ignore[method-assign]

    with pytest.raises(SemanticValidationError) as error:
        integration.bind(registry, engine)

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "safe_dynamic_collision_unsupported"
    assert diagnostic.path == (
        "integration",
        "robot_profile",
        "presets",
        "safe",
        "motion_policy",
        "dynamic_collision_mode",
    )
    assert diagnostic.candidates == ()
    assert "('cube',)" in diagnostic.message
    bind_skill_profile.assert_not_called()
    assert provider.calls == 0


def test_per_skill_safe_preset_is_conservatively_preflighted() -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    pick_skill_id = builtin_semantic_call_catalog().descriptors["pick"].skill_id
    integration = _semantic_integration(
        registry,
        preset=_preset("fast"),
        additional_presets=(
            _preset(
                "safe",
                motion_policy=MotionPolicy(strategy="motion_gen"),
            ),
        ),
        skill_presets={pick_skill_id: "safe"},
    )
    engine = _engine_for_integration(integration)

    with pytest.raises(SemanticValidationError) as error:
        integration.bind(registry, engine)

    assert error.value.diagnostic.code == "safe_dynamic_collision_unsupported"
    assert provider.calls == 0


def test_fully_overridden_safe_default_is_not_reachable() -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    catalog = builtin_semantic_call_catalog()
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(strategy="motion_gen"),
        ),
        additional_presets=(_preset("fast"),),
        skill_presets={
            descriptor.skill_id: "fast" for descriptor in catalog.descriptors.values()
        },
    )
    engine = _engine_for_integration(integration)

    bound = integration.bind(registry, engine)

    assert (
        bound.link_call(Pick(object=SceneObjectRef("cube"))).preset.preset_id == "fast"
    )
    assert provider.calls == 0


def test_runtime_non_safe_override_makes_safe_default_unreachable() -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(strategy="motion_gen"),
        ),
        additional_presets=(_preset("fast"),),
        runtime_preset="fast",
    )
    engine = _engine_for_integration(integration)

    bound = integration.bind(registry, engine)

    assert (
        bound.link_call(Pick(object=SceneObjectRef("cube"))).preset.preset_id == "fast"
    )
    assert provider.calls == 0


def test_bound_integration_cannot_bypass_safe_dynamic_planner_preflight() -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(strategy="motion_gen"),
        ),
    )
    engine = _engine_for_integration(integration)
    bound_profile = engine.bind_skill_profile(integration.robot_profile)

    with pytest.raises(SemanticValidationError) as error:
        BoundSemanticIntegration(
            manifest=integration,
            scene_registry=registry,
            robot_profile=bound_profile,
            engine=engine,
        )

    assert error.value.diagnostic.code == "safe_dynamic_collision_unsupported"
    assert error.value.diagnostic.path[-1] == "dynamic_collision_mode"
    assert provider.calls == 0


def test_bind_rejects_invalid_engine_before_safe_capability_lookup() -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(strategy="motion_gen"),
        ),
    )

    with pytest.raises(TypeError, match="engine must be an AtomicActionEngine"):
        integration.bind(registry, object())  # type: ignore[arg-type]

    assert provider.calls == 0


def test_safe_preset_rejects_non_motion_generator_strategy_for_dynamic_scene() -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(strategy="ik_interp"),
        ),
    )
    engine = _engine_for_integration(
        integration,
        supports_dynamic_collision_world=True,
    )

    with pytest.raises(SemanticValidationError) as error:
        integration.bind(registry, engine)

    assert error.value.diagnostic.code == "safe_dynamic_collision_unsupported"
    assert error.value.diagnostic.path[-1] == "strategy"
    assert provider.calls == 0


@pytest.mark.parametrize(
    "source_mode",
    [DynamicCollisionMode.AUTO, DynamicCollisionMode.OFF],
)
def test_non_safe_preset_preserves_dynamic_collision_policy(
    source_mode: DynamicCollisionMode,
) -> None:
    registry, provider = _scene_registry(
        with_default=True,
        dynamic_collision=True,
    )
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "fast",
            motion_policy=MotionPolicy(dynamic_collision_mode=source_mode),
        ),
    )
    engine = _engine_for_integration(integration)

    bound = integration.bind(registry, engine).link_call(
        Pick(object=SceneObjectRef("cube"))
    )

    assert bound.preset.preset_id == "fast"
    assert bound.preset.motion_policy.dynamic_collision_mode is source_mode
    assert provider.calls == 0


@pytest.mark.parametrize(
    "source_mode",
    [DynamicCollisionMode.AUTO, DynamicCollisionMode.OFF],
)
def test_safe_preset_preserves_policy_without_dynamic_collision(
    source_mode: DynamicCollisionMode,
) -> None:
    registry, provider = _scene_registry(with_default=True)
    integration = _semantic_integration(
        registry,
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(dynamic_collision_mode=source_mode),
        ),
    )
    engine = _engine_for_integration(integration)

    bound = integration.bind(registry, engine).link_call(
        Pick(object=SceneObjectRef("cube"))
    )

    assert bound.preset.motion_policy.dynamic_collision_mode is source_mode
    assert provider.calls == 0


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
