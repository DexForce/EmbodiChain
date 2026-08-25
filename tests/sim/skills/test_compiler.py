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

"""Tests for static semantic analysis and JIT invocation lowering."""

from __future__ import annotations

from types import MethodType
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
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
    GraspGoal,
    HandOverGoal,
    HandOverOptions,
    HeldObjectState,
    MotionPolicy,
    ObjectSemantics,
    PickUp,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    PlanningContext,
    RobotObservation,
    SceneEntityPose,
    SkillDescriptor,
    TaskState,
)
from embodichain.lab.sim.atomic_actions.tracking import (
    JointPositionTrackingMetric,
    TrackingPolicy,
)
from embodichain.lab.sim.skills.calls import (
    HandOver,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallDescriptor,
    SemanticPose,
    builtin_semantic_call_catalog,
)
from embodichain.lab.sim.skills.compiler import (
    ContainerRelationTargetGrounder,
    GroundedSemanticCall,
    HandOverPoseProvider,
    HandOverPoseTargets,
    HeldObjectGuardBaseline,
    RegisteredSemanticLowerer,
    RelationTargetGrounder,
    SemanticLowering,
    SemanticObjectTarget,
    SemanticRelationTarget,
    SemanticSkillCompiler,
    SemanticWorkflow,
    SupportSurfaceRelationTargetGrounder,
)
from embodichain.lab.sim.skills.effects import (
    BinaryEffectClause,
    BinaryEvidenceKind,
    COMPOSITE_EFFECT_MONITOR_ID,
    COMPOSITE_EFFECT_MONITOR_REVISION,
    CompositeEffectMonitorFactory,
    ControlPartEvidenceAddress,
    EffectMonitor,
    EffectMonitorRef,
    EffectMonitorRegistry,
    HeldObjectRelation,
    HeldObjectStateExpectation,
    PoseRelationClause,
    PoseRelationExpectation,
    SemanticEffectKind,
    SemanticEffectSpec,
    SymbolicStateKey,
)
from embodichain.lab.sim.skills.integration import (
    BoundSemanticCall,
    SceneManifest,
    SemanticIntegrationManifest,
    SemanticValidationError,
)
from embodichain.lab.sim.skills.profiles import (
    ControlPartEndpoint,
    ResourceBinding,
    RobotResource,
    RobotSkillProfile,
    SkillPolicyPreset,
)
from embodichain.lab.sim.skills.scene import (
    ContainerAffordance,
    GRASP_AFFORDANCE_CAPABILITY,
    PLACEMENT_TARGET_AFFORDANCE_REVISION,
    PLACE_IN_AFFORDANCE_CAPABILITY,
    PLACE_ON_AFFORDANCE_CAPABILITY,
    SceneAffordanceRef,
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
    SupportSurfaceAffordance,
)

_MOTION_CAPABILITIES = frozenset(
    {
        BATCH_INVERSE_KINEMATICS_CAPABILITY,
        CARTESIAN_POSE_CAPABILITY,
        FORWARD_KINEMATICS_CAPABILITY,
    }
)
_PICK_TARGET = PickUp.descriptor()


def _action_option_templates(*, registered: bool = False) -> dict[str, ActionOptions]:
    """Return complete exact options for the test semantic catalog."""
    templates: dict[str, ActionOptions] = {
        "pick": PickUpOptions(),
        "place": PlaceOptions(),
        "hand_over": HandOverOptions(),
    }
    if registered:
        templates["vendor.inspect"] = PickUpOptions()
    return templates


def _preset(
    preset_id: str,
    *,
    registered: bool = False,
    **kwargs: object,
) -> SkillPolicyPreset:
    """Build one complete schema-v3 test preset."""
    kwargs.setdefault(
        "action_option_templates",
        _action_option_templates(registered=registered),
    )
    return SkillPolicyPreset(preset_id, **kwargs)


class _PoseProvider:
    """Return a fixed pose while exposing observation call count."""

    def __init__(self, pose: torch.Tensor) -> None:
        self.pose = pose
        self.calls = 0

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        self.calls += 1
        return EntityState(self.pose)


class _GeometryProvider:
    """Return one opaque planner-facing geometry descriptor."""

    def get_geometry(self) -> object:
        return object()


class _FrameRelationGrounder(RelationTargetGrounder):
    """Explicit test contract: relation frame equals target object frame."""

    capability: ClassVar[str] = PLACE_ON_AFFORDANCE_CAPABILITY
    affordance_type: ClassVar[type[Affordance]] = Affordance
    affordance_revision: ClassVar[str] = "relation-v1"

    def ground(
        self,
        relation: SemanticRelationTarget,
        *,
        affordance: Affordance,
        context: PlanningContext,
    ) -> SceneEntityPose:
        del affordance, context
        return SceneEntityPose(relation.affordance.entity_id)


class _InspectLowerer(RegisteredSemanticLowerer):
    """Test extension proving a lowerer cannot replace compiler ownership."""

    call_id: ClassVar[str] = "vendor.inspect"
    schema_version: ClassVar[int] = 1
    target_descriptor: ClassVar[SkillDescriptor] = _PICK_TARGET

    def __init__(self) -> None:
        self.option_templates: list[ActionOptions] = []

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: object,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        del call, context, bound
        self.option_templates.append(option_template)
        return SemanticLowering(
            goal=GraspGoal(
                semantics=ObjectSemantics(
                    affordance=AntipodalAffordance(),
                    geometry={},
                    entity_id="cube",
                )
            )
        )


class _DerivedGraspGoal(GraspGoal):
    """Executable subclass that an extension must not smuggle into the core."""


class _SubclassOutputLowerer(RegisteredSemanticLowerer):
    """Try to bypass exact target contracts with executable subclasses."""

    call_id: ClassVar[str] = "vendor.inspect"
    schema_version: ClassVar[int] = 1
    target_descriptor: ClassVar[SkillDescriptor] = _PICK_TARGET

    def __init__(self, output: str) -> None:
        self.output = output

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        del call, context, bound, option_template
        semantics = ObjectSemantics(
            affordance=AntipodalAffordance(),
            geometry={},
            entity_id="cube",
        )
        if self.output == "goal":
            return SemanticLowering(goal=_DerivedGraspGoal(semantics=semantics))
        return SemanticLowering(
            goal=GraspGoal(semantics=semantics),
            skill_options=PickUpOptions(pre_grasp_distance=0.99),
        )


class _DualCenterHandOverProvider(HandOverPoseProvider):
    """Resolve named dual-arm handover poses without observing during analysis."""

    provider_id: ClassVar[str] = "dual_center"

    def __init__(self) -> None:
        self.calls = 0

    def resolve(
        self,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        del call, context, bound
        self.calls += 1
        return HandOverPoseTargets(
            middle=SemanticObjectTarget(
                pose=SemanticPose(
                    (0.5, 0.0, 0.4),
                    (1.0, 0.0, 0.0, 0.0),
                )
            ),
            final=SemanticObjectTarget(pose=SceneEntityPose("table_top")),
        )


class _CountingRelationMonitorFactory(CompositeEffectMonitorFactory):
    """Count monitor construction without changing built-in behavior."""

    def __init__(self) -> None:
        self.calls = 0

    def create(
        self,
        spec: SemanticEffectSpec,
        ref: EffectMonitorRef,
    ) -> EffectMonitor:
        self.calls += 1
        return super().create(spec, ref)


class _BadCreatingRelationMonitorFactory(CompositeEffectMonitorFactory):
    """Return an invalid monitor value after successful static validation."""

    def create(
        self,
        spec: SemanticEffectSpec,
        ref: EffectMonitorRef,
    ) -> EffectMonitor:
        del spec, ref
        return object()  # type: ignore[return-value]


def _scene_registry(
    *,
    dynamic_collision: bool = False,
) -> tuple[SceneRegistry, tuple[_PoseProvider, _PoseProvider]]:
    cube_provider = _PoseProvider(torch.eye(4).repeat(2, 1, 1))
    table_pose = torch.eye(4).repeat(2, 1, 1)
    table_pose[:, 0, 3] = 0.6
    table_provider = _PoseProvider(table_pose)
    cube = SceneObjectRef("cube")
    table = SceneObjectRef("table")
    grasp = SceneAffordanceRef("cube_grasp")
    table_top = SceneAffordanceRef("table_top")
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=cube,
                state_provider=cube_provider,
                semantic_type="cube",
                default_affordances={GRASP_AFFORDANCE_CAPABILITY: grasp},
                geometry_provider=(_GeometryProvider() if dynamic_collision else None),
                collision_role=(
                    SceneCollisionRole.DYNAMIC
                    if dynamic_collision
                    else SceneCollisionRole.NONE
                ),
            ),
            SceneEntityRegistration(
                ref=grasp,
                parent=cube,
                native_name="grasp",
                affordance=AntipodalAffordance(),
                affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                affordance_revision="grasp-v1",
                relative_pose=torch.eye(4),
            ),
            SceneEntityRegistration(
                ref=table,
                state_provider=table_provider,
                semantic_type="table",
                default_affordances={PLACE_ON_AFFORDANCE_CAPABILITY: table_top},
            ),
            SceneEntityRegistration(
                ref=table_top,
                parent=table,
                native_name="top",
                affordance=Affordance(),
                affordance_capabilities=frozenset({PLACE_ON_AFFORDANCE_CAPABILITY}),
                affordance_revision="relation-v1",
                relative_pose=torch.eye(4),
            ),
        ),
        collision_world_mode=(
            SceneCollisionWorldMode.PER_ENV if dynamic_collision else None
        ),
    )
    return registry, (cube_provider, table_provider)


def _profile(
    *,
    preset: SkillPolicyPreset | None = None,
    registered: bool = False,
) -> RobotSkillProfile:
    return RobotSkillProfile(
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
        presets={
            "safe": (
                _preset("safe", registered=registered) if preset is None else preset
            )
        },
        default_preset="safe",
    )


def _dual_profile(
    *,
    provider_id: str | None = "dual_center",
    preset: SkillPolicyPreset | None = None,
) -> RobotSkillProfile:
    resources = {
        side: RobotResource(
            resource_id=side,
            endpoints={
                "motion": ControlPartEndpoint(
                    control_part=f"{side}_arm",
                    capabilities=_MOTION_CAPABILITIES,
                ),
                "grasp": ControlPartEndpoint(
                    control_part=f"{side}_hand",
                    capabilities=frozenset({GRASP_CAPABILITY}),
                ),
            },
        )
        for side in ("left", "right")
    }
    return RobotSkillProfile(
        profile_id="dual_robot",
        resources=resources,
        command_profiles={
            f"{side}_hand": ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.0]),
                grasp=torch.tensor([1.0]),
            )
            for side in ("left", "right")
        },
        defaults={
            "pick_up": ResourceBinding({"primary": "left"}),
            "hand_over": ResourceBinding({"source": "left", "destination": "right"}),
        },
        presets={"safe": _preset("safe") if preset is None else preset},
        default_preset="safe",
        grounding_providers=({} if provider_id is None else {"hand_over": provider_id}),
    )


def _engine(
    profile: RobotSkillProfile,
    *,
    supports_dynamic_collision_world: bool = False,
) -> AtomicActionEngine:
    robot = Mock()
    robot.device = torch.device("cpu")
    control_parts = tuple(
        sorted(
            {
                endpoint.control_part
                for resource in profile.resources.values()
                for endpoint in resource.endpoints.values()
                if type(endpoint) is ControlPartEndpoint
            }
        )
    )
    joint_ids = {name: [index] for index, name in enumerate(control_parts)}
    robot.dof = len(control_parts)
    robot.control_parts = {name: object() for name in control_parts}
    robot.get_qpos.return_value = torch.zeros(2, robot.dof)
    robot.get_qvel.return_value = torch.zeros(2, robot.dof)
    robot.get_joint_ids.side_effect = lambda name: joint_ids[name]
    robot.get_solver.return_value = object()
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    generator.supports_dynamic_collision_world = supports_dynamic_collision_world
    return AtomicActionEngine(generator, skill_profile=profile)


def _integration(
    registry: SceneRegistry,
    *,
    registered: bool = False,
    profile: RobotSkillProfile | None = None,
    supports_dynamic_collision_world: bool = False,
) -> tuple[SemanticIntegrationManifest, AtomicActionEngine]:
    selected_profile = _profile(registered=registered) if profile is None else profile
    catalog = builtin_semantic_call_catalog()
    if registered:
        assert _PICK_TARGET.binding_contract is not None
        catalog = catalog.with_descriptor(
            SemanticCallDescriptor(
                call_id="vendor.inspect",
                spec_type=RegisteredSemanticCall,
                target_descriptor=_PICK_TARGET,
            )
        )
    manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=selected_profile,
        call_catalog=catalog,
    )
    return manifest, _engine(
        selected_profile,
        supports_dynamic_collision_world=supports_dynamic_collision_world,
    )


def _compiler(
    registry: SceneRegistry,
    *,
    registered: bool = False,
    relation_grounders: tuple[RelationTargetGrounder, ...] = (
        _FrameRelationGrounder(),
    ),
    registered_lowerers: tuple[RegisteredSemanticLowerer, ...] = (),
    handover_pose_providers: tuple[HandOverPoseProvider, ...] = (),
    profile: RobotSkillProfile | None = None,
    effect_monitor_registry: EffectMonitorRegistry | None = None,
    supports_dynamic_collision_world: bool = False,
) -> tuple[SemanticSkillCompiler, AtomicActionEngine]:
    manifest, engine = _integration(
        registry,
        registered=registered,
        profile=profile,
        supports_dynamic_collision_world=supports_dynamic_collision_world,
    )
    bound = manifest.bind(registry, engine)
    return (
        SemanticSkillCompiler(
            bound,
            relation_grounders=relation_grounders,
            registered_lowerers=registered_lowerers,
            handover_pose_providers=handover_pose_providers,
            effect_monitor_registry=effect_monitor_registry,
        ),
        engine,
    )


def _context(
    registry: SceneRegistry,
    *,
    task: TaskState | None = None,
    timestamp: float = 0.0,
    robot_dof: int = 2,
) -> PlanningContext:
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    scene = registry.make_scene_provider(
        translation_threshold=0.0,
        rotation_threshold=0.0,
    ).snapshot(timestamp=timestamp, env_ids=env_ids)
    return PlanningContext(
        robot=RobotObservation(
            timestamp=timestamp,
            qpos=torch.zeros(2, robot_dof),
            qvel=torch.zeros(2, robot_dof),
        ),
        task=TaskState.empty(2, "cpu") if task is None else task,
        scene=scene,
        env_ids=env_ids,
    )


def _held_context(
    registry: SceneRegistry,
    semantics: ObjectSemantics,
    object_to_eef: torch.Tensor,
    *,
    env_mask: torch.Tensor | None = None,
    task_state_key: str = "manipulator",
    robot_dof: int = 2,
) -> PlanningContext:
    held = HeldObjectState(
        semantics=semantics,
        object_to_eef=object_to_eef,
        grasp_xpos=torch.eye(4).repeat(2, 1, 1),
        env_mask=env_mask,
    )
    return _context(
        registry,
        task=TaskState(
            batch_size=2,
            device="cpu",
            held_objects={task_state_key: held},
        ),
        robot_dof=robot_dof,
    )


@pytest.mark.parametrize(
    ("grounder", "capability", "affordance_type"),
    (
        (
            SupportSurfaceRelationTargetGrounder(),
            PLACE_ON_AFFORDANCE_CAPABILITY,
            SupportSurfaceAffordance,
        ),
        (
            ContainerRelationTargetGrounder(),
            PLACE_IN_AFFORDANCE_CAPABILITY,
            ContainerAffordance,
        ),
    ),
)
def test_builtin_relation_grounders_preserve_late_pose_and_confidence(
    grounder: RelationTargetGrounder,
    capability: str,
    affordance_type: type[Affordance],
) -> None:
    """Production relation grounders keep target frames live and typed."""
    registry, _ = _scene_registry()
    relation = SemanticRelationTarget(
        capability=capability,
        affordance=SceneAffordanceRef("declared_target"),
        payload_type=affordance_type,
        payload_revision=PLACEMENT_TARGET_AFFORDANCE_REVISION,
    )

    target = grounder.ground(
        relation,
        affordance=affordance_type(minimum_confidence=0.65),
        context=_context(registry),
    )

    assert type(target) is SceneEntityPose
    assert target.entity_id == "declared_target"
    assert target.relative_pose is None
    assert target.minimum_confidence == pytest.approx(0.65)


def test_curated_analysis_selects_exact_preset_monitor_without_creating_it() -> None:
    registry, providers = _scene_registry()
    factory = _CountingRelationMonitorFactory()
    compiler, _ = _compiler(
        registry,
        effect_monitor_registry=EffectMonitorRegistry((factory,)),
    )

    workflow = compiler.analyze((Pick(object=SceneObjectRef("cube")),))

    monitor_ref = workflow.calls[0].effect_monitor_ref
    assert monitor_ref is not None
    assert monitor_ref.monitor_id == COMPOSITE_EFFECT_MONITOR_ID
    assert monitor_ref.revision == COMPOSITE_EFFECT_MONITOR_REVISION
    assert workflow.calls[0].symbolic_writes == frozenset(
        {SymbolicStateKey.held_object("manipulator")}
    )
    assert not workflow.calls[0].opaque_symbolic_effect
    assert factory.calls == 0
    assert [provider.calls for provider in providers] == [0, 0]


def test_curated_analysis_rejects_explicitly_missing_monitor() -> None:
    registry, _ = _scene_registry()
    profile = _profile(
        preset=_preset("safe", effect_monitors={}),
    )
    compiler, _ = _compiler(registry, profile=profile)

    with pytest.raises(SemanticValidationError) as error:
        compiler.analyze((Pick(object=SceneObjectRef("cube")),))

    assert error.value.diagnostic.code == "missing_effect_monitor"


def test_uninstalled_effect_monitor_fails_analysis_without_factory_creation() -> None:
    registry, providers = _scene_registry()
    factory = _CountingRelationMonitorFactory()
    profile = _profile(
        preset=_preset(
            "safe",
            effect_monitors={
                "pick": EffectMonitorRef("test.not_installed", "1"),
            },
        ),
    )
    compiler, _ = _compiler(
        registry,
        profile=profile,
        effect_monitor_registry=EffectMonitorRegistry((factory,)),
    )

    with pytest.raises(SemanticValidationError) as error:
        compiler.analyze((Pick(object=SceneObjectRef("cube")),))

    assert error.value.diagnostic.code == "effect_monitor_not_installed"
    assert factory.calls == 0
    assert [provider.calls for provider in providers] == [0, 0]


def test_invalid_effect_monitor_config_fails_analysis_without_side_effects() -> None:
    registry, providers = _scene_registry()
    factory = _CountingRelationMonitorFactory()
    profile = _profile(
        preset=_preset(
            "safe",
            effect_monitors={
                "pick": EffectMonitorRef(
                    COMPOSITE_EFFECT_MONITOR_ID,
                    COMPOSITE_EFFECT_MONITOR_REVISION,
                    {
                        "attached_translation_threshold": 0.10,
                        "detached_translation_threshold": 0.05,
                    },
                ),
            },
        ),
    )
    compiler, _ = _compiler(
        registry,
        profile=profile,
        effect_monitor_registry=EffectMonitorRegistry((factory,)),
    )

    with pytest.raises(SemanticValidationError) as error:
        compiler.analyze((Pick(object=SceneObjectRef("cube")),))

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "invalid_effect_monitor_config"
    assert diagnostic.path == ("workflow", 0, "effect_monitor")
    assert factory.calls == 0
    assert [provider.calls for provider in providers] == [0, 0]


def test_pick_effect_spec_binds_destination_and_fresh_monitor_per_grounding() -> None:
    registry, _ = _scene_registry()
    compiler, _ = _compiler(registry)
    workflow = compiler.analyze((Pick(object=SceneObjectRef("cube")),))
    context = _context(registry)

    first = compiler.ground(workflow, 0, context)
    repeated = compiler.ground(workflow, 0, context)
    revised = compiler.ground(workflow, 0, context, revision=1)

    spec = first.effect_spec
    assert spec is not None
    assert spec.semantic_id == "pick"
    assert spec.effect_kind is SemanticEffectKind.ATTACH
    assert spec.skill_id == first.invocation.skill_id
    assert spec.invocation_id == first.invocation.invocation_id
    assert spec.invocation_revision == 0
    torch.testing.assert_close(spec.env_ids, context.env_ids)
    assert len(spec.state_expectations) == 1
    relation = spec.state_expectations[0]
    assert isinstance(relation, HeldObjectStateExpectation)
    assert relation.expectation_id == "destination"
    assert relation.relation is HeldObjectRelation.ATTACHED
    assert relation.object_id == "cube"
    assert relation.slot_id == "primary"
    assert relation.resource_id == "manipulator"
    assert relation.task_state_key == "manipulator"
    pose, constraint = spec.clauses
    assert isinstance(pose, PoseRelationClause)
    assert pose.expectation is PoseRelationExpectation.MATCHED
    assert pose.baseline_object_to_endpoint is None
    assert pose.source.address == ControlPartEvidenceAddress("arm", "pose_relation")
    assert isinstance(constraint, BinaryEffectClause)
    assert constraint.evidence_kind is BinaryEvidenceKind.CONSTRAINT
    assert constraint.expected is True
    assert constraint.source.address == ControlPartEvidenceAddress("hand", "constraint")
    assert (
        first.analyzed.bound.binding.action_binding.endpoint(
            "primary", "motion"
        ).task_state_key
        == "manipulator"
    )
    assert first.effect_monitor is not None
    assert repeated.effect_monitor is not None
    assert revised.effect_monitor is not None
    assert repeated.effect_monitor is not first.effect_monitor
    assert revised.effect_monitor is not first.effect_monitor
    assert repeated.effect_spec is not None
    assert repeated.effect_spec.invocation_revision == 0
    assert revised.effect_spec is not None
    assert revised.effect_spec.invocation_revision == 1
    assert revised.effect_monitor.spec.invocation_revision == 1
    assert len(first.effect_guards) == 1
    guard = first.effect_guards[0]
    assert guard.guard_id == "destination_attached"
    assert guard.active_segments == ("lift",)
    assert guard.baseline is HeldObjectGuardBaseline.PLANNED_EFFECT
    assert guard.task_state_key == "manipulator"
    assert guard.invalidation_task_state_keys == ("manipulator",)
    assert guard.retry_action is True
    assert guard.effect_monitor is not first.effect_monitor
    assert guard.effect_spec.effect_kind is SemanticEffectKind.ATTACH
    assert repeated.effect_guards[0].effect_monitor is not guard.effect_monitor
    assert len(first.effect_gates) == 1
    gate = first.effect_gates[0]
    assert gate.gate_id == "destination_acquired"
    assert gate.segment_name == "lift"
    assert gate.retry_action is True
    assert gate.effect_monitor is not first.effect_monitor
    assert gate.effect_monitor is not guard.effect_monitor
    assert gate.effect_spec.state_expectations[0].expectation_id == "destination"
    assert gate.effect_spec.effect_kind is SemanticEffectKind.ATTACH
    assert first.invocation.phase_effect_gates == (gate.requirement,)
    assert repeated.effect_gates[0].effect_monitor is not gate.effect_monitor


def test_place_effect_spec_binds_source_and_verified_detach_baseline() -> None:
    registry, _ = _scene_registry()
    compiler, _ = _compiler(registry)
    pick_workflow = compiler.analyze((Pick(object=SceneObjectRef("cube")),))
    pick = compiler.ground(pick_workflow, 0, _context(registry))
    semantics = pick.invocation.goal.semantics
    object_to_eef = torch.eye(4).repeat(2, 1, 1)
    object_to_eef[:, 2, 3] = 0.12
    context = _held_context(registry, semantics, object_to_eef)
    workflow = compiler.analyze(
        (
            Place(
                object=SceneObjectRef("cube"),
                at=SemanticPose(
                    (0.5, -0.2, 0.4),
                    (1.0, 0.0, 0.0, 0.0),
                ),
            ),
        )
    )

    assert workflow.calls[0].symbolic_writes == frozenset(
        {SymbolicStateKey.held_object("manipulator")}
    )
    grounded = compiler.ground(workflow, 0, context)

    spec = grounded.effect_spec
    assert spec is not None
    assert spec.semantic_id == "place"
    assert spec.effect_kind is SemanticEffectKind.RELEASE
    assert len(spec.state_expectations) == 1
    relation = spec.state_expectations[0]
    assert isinstance(relation, HeldObjectStateExpectation)
    assert relation.expectation_id == "source"
    assert relation.relation is HeldObjectRelation.DETACHED
    assert relation.object_id == "cube"
    assert relation.slot_id == "primary"
    assert relation.resource_id == "manipulator"
    assert relation.task_state_key == "manipulator"
    pose, constraint = spec.clauses
    assert isinstance(pose, PoseRelationClause)
    assert pose.expectation is PoseRelationExpectation.SEPARATED
    assert pose.baseline_object_to_endpoint is not None
    torch.testing.assert_close(
        pose.baseline_object_to_endpoint,
        object_to_eef,
    )
    assert isinstance(constraint, BinaryEffectClause)
    assert constraint.expected is False
    assert len(grounded.effect_guards) == 1
    guard = grounded.effect_guards[0]
    assert guard.guard_id == "source_attached"
    assert guard.active_segments == ("approach",)
    assert guard.baseline is HeldObjectGuardBaseline.VERIFIED_TASK_STATE
    assert guard.task_state_key == "manipulator"
    assert guard.invalidation_task_state_keys == ("manipulator",)
    assert guard.retry_action is False
    guard_pose, guard_constraint = guard.effect_spec.clauses
    assert isinstance(guard_pose, PoseRelationClause)
    assert guard_pose.expectation is PoseRelationExpectation.MATCHED
    assert guard_pose.baseline_object_to_endpoint is None
    assert isinstance(guard_constraint, BinaryEffectClause)
    assert guard_constraint.expected is True
    assert len(grounded.effect_gates) == 1
    gate = grounded.effect_gates[0]
    assert gate.gate_id == "source_released"
    assert gate.segment_name == "retract"
    assert gate.retry_action is True
    assert gate.effect_spec.effect_kind is SemanticEffectKind.RELEASE
    gate_relation = gate.effect_spec.state_expectations[0]
    assert isinstance(gate_relation, HeldObjectStateExpectation)
    assert gate_relation.expectation_id == "source"
    assert gate_relation.relation is HeldObjectRelation.DETACHED
    assert grounded.invocation.phase_effect_gates == (gate.requirement,)


def test_handover_effect_spec_binds_source_and_destination_relations() -> None:
    registry, _ = _scene_registry()
    provider = _DualCenterHandOverProvider()
    compiler, _ = _compiler(
        registry,
        profile=_dual_profile(),
        handover_pose_providers=(provider,),
    )
    workflow = compiler.analyze((HandOver(object=SceneObjectRef("cube")),))

    assert workflow.calls[0].symbolic_writes == frozenset(
        {
            SymbolicStateKey.held_object("left"),
            SymbolicStateKey.held_object("right"),
        }
    )
    grounded = compiler.ground(workflow, 0, _context(registry, robot_dof=4))

    spec = grounded.effect_spec
    assert spec is not None
    assert spec.semantic_id == "hand_over"
    assert spec.effect_kind is SemanticEffectKind.RELEASE
    assert tuple(relation.expectation_id for relation in spec.state_expectations) == (
        "source",
        "destination",
    )
    source, destination = spec.state_expectations
    assert isinstance(source, HeldObjectStateExpectation)
    assert source.relation is HeldObjectRelation.DETACHED
    assert source.object_id == "cube"
    assert source.slot_id == "source"
    assert source.resource_id == "left"
    assert source.task_state_key == "left"
    source_constraint, destination_constraint = spec.clauses
    assert isinstance(source_constraint, BinaryEffectClause)
    assert source_constraint.expected is False
    assert isinstance(destination, HeldObjectStateExpectation)
    assert destination.relation is HeldObjectRelation.DETACHED
    assert destination.object_id == "cube"
    assert destination.slot_id == "destination"
    assert destination.resource_id == "right"
    assert destination.task_state_key == "right"
    assert isinstance(destination_constraint, BinaryEffectClause)
    assert destination_constraint.expected is False
    assert type(grounded.invocation.goal) is HandOverGoal
    assert type(grounded.invocation.skill_options) is HandOverOptions
    assert tuple(guard.guard_id for guard in grounded.effect_guards) == (
        "source_attached",
        "destination_attached",
    )
    source_guard, destination_guard = grounded.effect_guards
    assert source_guard.active_segments == (
        "pickup_transport",
        "receive_approach",
        "receive_close",
    )
    assert source_guard.baseline is HeldObjectGuardBaseline.PLANNED_EFFECT
    assert source_guard.task_state_key == "left"
    assert source_guard.invalidation_task_state_keys == ("left",)
    assert source_guard.retry_action is True
    assert destination_guard.active_segments == ("handover_release", "place")
    assert destination_guard.baseline is HeldObjectGuardBaseline.PLANNED_EFFECT
    assert destination_guard.task_state_key == "right"
    assert destination_guard.invalidation_task_state_keys == ("left", "right")
    assert destination_guard.retry_action is True
    assert tuple(gate.gate_id for gate in grounded.effect_gates) == (
        "source_acquired",
        "destination_acquired",
        "source_released",
    )
    source_acquired, destination_acquired, source_released = grounded.effect_gates
    assert source_acquired.segment_name == "pickup_transport"
    assert destination_acquired.segment_name == "handover_release"
    assert source_released.segment_name == "place"
    assert all(gate.retry_action for gate in grounded.effect_gates)
    assert source_acquired.effect_monitor is not source_guard.effect_monitor
    assert destination_acquired.effect_monitor is not destination_guard.effect_monitor
    source_acquired_relation = source_acquired.effect_spec.state_expectations[0]
    destination_acquired_relation = destination_acquired.effect_spec.state_expectations[
        0
    ]
    source_released_relation = source_released.effect_spec.state_expectations[0]
    assert isinstance(source_acquired_relation, HeldObjectStateExpectation)
    assert isinstance(destination_acquired_relation, HeldObjectStateExpectation)
    assert isinstance(source_released_relation, HeldObjectStateExpectation)
    assert source_acquired_relation.relation is HeldObjectRelation.ATTACHED
    assert destination_acquired_relation.relation is HeldObjectRelation.ATTACHED
    assert source_released_relation.relation is HeldObjectRelation.DETACHED
    assert grounded.invocation.phase_effect_gates == tuple(
        gate.requirement for gate in grounded.effect_gates
    )


def test_registered_call_without_monitor_has_no_effect_contract() -> None:
    registry, _ = _scene_registry()
    factory = _CountingRelationMonitorFactory()
    templates = _action_option_templates(registered=True)
    templates["vendor.inspect"] = PickUpOptions(pre_grasp_distance=0.07)
    profile = _profile(
        preset=_preset(
            "safe",
            registered=True,
            action_option_templates=templates,
        )
    )
    lowerer = _InspectLowerer()
    compiler, _ = _compiler(
        registry,
        registered=True,
        registered_lowerers=(lowerer,),
        profile=profile,
        effect_monitor_registry=EffectMonitorRegistry((factory,)),
    )
    workflow = compiler.analyze((RegisteredSemanticCall(call_id="vendor.inspect"),))

    grounded = compiler.ground(workflow, 0, _context(registry))

    assert workflow.calls[0].symbolic_writes == frozenset()
    assert workflow.calls[0].opaque_symbolic_effect
    assert workflow.calls[0].effect_monitor_ref is None
    assert grounded.effect_spec is None
    assert grounded.effect_monitor is None
    assert grounded.effect_gates == ()
    assert grounded.invocation.phase_effect_gates == ()
    options = grounded.invocation.skill_options
    assert type(options) is PickUpOptions
    assert options.pre_grasp_distance == 0.07
    assert len(lowerer.option_templates) == 1
    assert lowerer.option_templates[0] is not options
    assert factory.calls == 0


def test_registered_monitor_without_effect_grounder_fails_during_analysis() -> None:
    registry, _ = _scene_registry()
    profile = _profile(
        preset=_preset(
            "safe",
            registered=True,
            effect_monitors={
                "vendor.inspect": EffectMonitorRef(
                    COMPOSITE_EFFECT_MONITOR_ID,
                    COMPOSITE_EFFECT_MONITOR_REVISION,
                )
            },
        )
    )
    compiler, _ = _compiler(
        registry,
        registered=True,
        registered_lowerers=(_InspectLowerer(),),
        profile=profile,
    )

    with pytest.raises(SemanticValidationError) as error:
        compiler.analyze((RegisteredSemanticCall(call_id="vendor.inspect"),))

    assert error.value.diagnostic.code == "registered_effect_contract_not_installed"
    assert error.value.diagnostic.path == ("workflow", 0, "effect_monitor")


def test_ground_wraps_effect_monitor_factory_contract_failure_with_path() -> None:
    registry, _ = _scene_registry()
    compiler, _ = _compiler(
        registry,
        effect_monitor_registry=EffectMonitorRegistry(
            (_BadCreatingRelationMonitorFactory(),)
        ),
    )
    workflow = compiler.analyze((Pick(object=SceneObjectRef("cube")),))

    with pytest.raises(SemanticValidationError) as error:
        compiler.ground(workflow, 0, _context(registry))

    assert error.value.diagnostic.code == "effect_monitor_creation_failed"
    assert error.value.diagnostic.path == ("workflow", 0, "effect_monitor")


def test_analysis_is_provider_free_and_propagates_object_target() -> None:
    registry, providers = _scene_registry()
    templates = _action_option_templates()
    templates["pick"] = PickUpOptions(
        pick_object_part="top",
        pre_grasp_distance=0.08,
    )
    compiler, engine = _compiler(
        registry,
        profile=_profile(
            preset=_preset("safe", action_option_templates=templates),
        ),
    )
    drop = SemanticPose((0.4, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0))

    workflow = compiler.analyze(
        (
            Pick(object=SceneObjectRef("cube")),
            Place(object=SceneObjectRef("cube"), at=drop),
        ),
        workflow_id="pick_place",
    )

    assert [provider.calls for provider in providers] == [0, 0]
    assert workflow.calls[0].downstream_object_targets[0].pose is not drop
    assert workflow.effect_dependencies[0].producer_index == 0
    context = _context(registry)
    grounded = compiler.ground(workflow, 0, context)
    assert type(grounded.invocation.goal) is GraspGoal
    assert grounded.invocation.goal.semantics.entity_id == "cube"
    options = grounded.invocation.skill_options
    assert type(options) is PickUpOptions
    assert options.pick_object_part == "top"
    assert options.pre_grasp_distance == 0.08
    torch.testing.assert_close(
        options.downstream_object_target_poses[0],
        drop.to_matrix(),
    )
    engine.resolve(grounded.invocation)


def test_grounded_safe_invocation_requires_registered_dynamic_collision() -> None:
    registry, _ = _scene_registry(dynamic_collision=True)
    profile = _profile(
        preset=_preset(
            "safe",
            motion_policy=MotionPolicy(strategy="motion_gen"),
            tracking_policy=TrackingPolicy.joint_position(
                in_flight_max_abs_error=0.125,
                terminal_max_abs_error=0.125,
            ),
        )
    )
    compiler, engine = _compiler(
        registry,
        profile=profile,
        supports_dynamic_collision_world=True,
    )
    workflow = compiler.analyze((Pick(object=SceneObjectRef("cube")),))

    grounded = compiler.ground(workflow, 0, _context(registry))

    assert (
        grounded.invocation.motion_policy.dynamic_collision_mode
        is DynamicCollisionMode.REQUIRED
    )
    assert (
        engine.resolve(grounded.invocation).motion_policy.dynamic_collision_mode
        is DynamicCollisionMode.REQUIRED
    )
    invocation_tracking = grounded.invocation.tracking_policy.in_flight
    resolved_tracking = engine.resolve(grounded.invocation).tracking_policy.in_flight
    assert invocation_tracking is not None
    assert resolved_tracking is not None
    assert isinstance(invocation_tracking.metrics[0], JointPositionTrackingMetric)
    assert isinstance(resolved_tracking.metrics[0], JointPositionTrackingMetric)
    assert invocation_tracking.metrics[0].tolerance == 0.125
    assert resolved_tracking.metrics[0].tolerance == 0.125


def test_pick_relation_lookahead_stays_late_bound_scene_dependency() -> None:
    registry, _ = _scene_registry()
    compiler, engine = _compiler(registry)
    workflow = compiler.analyze(
        (
            Pick(object=SceneObjectRef("cube")),
            Place(
                object=SceneObjectRef("cube"),
                on=SceneObjectRef("table"),
            ),
        )
    )

    grounded = compiler.ground(workflow, 0, _context(registry))

    options = grounded.invocation.skill_options
    assert type(options) is PickUpOptions
    assert len(options.downstream_object_target_poses) == 1
    downstream = options.downstream_object_target_poses[0]
    assert type(downstream) is SceneEntityPose
    assert downstream.entity_id == "table_top"
    request = engine.resolve(grounded.invocation)
    action = engine.actions["pick_up"]
    assert "table_top" in action._scene_dependencies(request)


def test_pick_replan_resolves_downstream_target_from_latest_snapshot() -> None:
    registry, providers = _scene_registry()
    compiler, engine = _compiler(registry)
    workflow = compiler.analyze(
        (
            Pick(object=SceneObjectRef("cube")),
            Place(
                object=SceneObjectRef("cube"),
                on=SceneObjectRef("table"),
            ),
        )
    )
    first_context = _context(registry, timestamp=0.0)
    invocation = compiler.ground(workflow, 0, first_context).invocation
    action = engine.actions["pick_up"]
    captured: list[torch.Tensor] = []

    def fail_after_capture(
        self: object,
        semantics: object,
        object_pose: torch.Tensor,
        start_qpos: torch.Tensor,
        manipulator: object,
        grasp_target_id: str,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self, semantics, start_qpos, manipulator, approach_direction
        target = options.downstream_object_target_poses[0]
        assert isinstance(target, torch.Tensor)
        captured.append(target.clone())
        return (
            torch.zeros(2, dtype=torch.bool),
            object_pose.clone(),
        )

    action._resolve_grasp_pose = MethodType(  # type: ignore[method-assign]
        fail_after_capture,
        action,
    )
    request = engine.resolve(invocation)
    engine.plan_request(request, first_context)
    moved_table_pose = torch.eye(4).repeat(2, 1, 1)
    moved_table_pose[:, 0, 3] = 0.9
    providers[1].pose = moved_table_pose
    second_context = _context(registry, timestamp=1.0)
    engine.plan_request(request, second_context)

    assert captured[0][0, 0, 3].item() == pytest.approx(0.6)
    assert captured[1][0, 0, 3].item() == pytest.approx(0.9)


def test_handover_uses_profile_selected_provider_for_unified_goal() -> None:
    registry, providers = _scene_registry()
    profile = _dual_profile()
    manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=profile,
        call_catalog=builtin_semantic_call_catalog(),
    )
    engine = _engine(profile)
    provider = _DualCenterHandOverProvider()
    compiler = SemanticSkillCompiler(
        manifest.bind(registry, engine),
        relation_grounders=(_FrameRelationGrounder(),),
        handover_pose_providers=(provider,),
    )
    workflow = compiler.analyze((HandOver(object=SceneObjectRef("cube")),))

    assert provider.calls == 0
    assert [scene_provider.calls for scene_provider in providers] == [0, 0]
    assert workflow.calls[0].downstream_object_targets == ()
    handover = compiler.ground(workflow, 0, _context(registry, robot_dof=4))
    assert provider.calls == 1
    goal = handover.invocation.goal
    assert type(goal) is HandOverGoal
    assert type(goal.target_pose) is SceneEntityPose
    assert goal.target_pose.entity_id == "table_top"
    options = handover.invocation.skill_options
    assert type(options) is HandOverOptions
    request = engine.resolve(handover.invocation)
    action = engine.actions["hand_over"]
    assert action._scene_dependencies(request) == ("cube", "table_top")


def test_handover_requires_profile_selection_and_installed_provider() -> None:
    registry, _ = _scene_registry()
    call = HandOver(object=SceneObjectRef("cube"))

    unconfigured_profile = _dual_profile(provider_id=None)
    unconfigured_manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=unconfigured_profile,
        call_catalog=builtin_semantic_call_catalog(),
    )
    unconfigured_engine = _engine(unconfigured_profile)
    unconfigured = SemanticSkillCompiler(
        unconfigured_manifest.bind(registry, unconfigured_engine)
    )
    with pytest.raises(SemanticValidationError) as unconfigured_error:
        unconfigured.analyze((call,))
    assert unconfigured_error.value.diagnostic.code == "handover_grounding_unconfigured"

    missing_profile = _dual_profile(provider_id="not_installed")
    missing_manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=missing_profile,
        call_catalog=builtin_semantic_call_catalog(),
    )
    missing_engine = _engine(missing_profile)
    missing = SemanticSkillCompiler(missing_manifest.bind(registry, missing_engine))
    with pytest.raises(SemanticValidationError) as missing_error:
        missing.analyze((call,))
    assert (
        missing_error.value.diagnostic.code
        == "handover_grounding_provider_not_installed"
    )


def test_relation_call_requires_exact_typed_versioned_grounder() -> None:
    registry, _ = _scene_registry()
    compiler, _ = _compiler(registry, relation_grounders=())

    with pytest.raises(SemanticValidationError) as error:
        compiler.analyze(
            (
                Place(
                    object=SceneObjectRef("cube"),
                    on=SceneObjectRef("table"),
                ),
            )
        )

    assert error.value.diagnostic.code == "relation_grounder_not_installed"


def test_place_uses_verified_object_to_eef_transform() -> None:
    registry, _ = _scene_registry()
    templates = _action_option_templates()
    templates["place"] = PlaceOptions(
        lift_height=0.22,
        cartesian_waypoint_count=3,
    )
    compiler, engine = _compiler(
        registry,
        profile=_profile(
            preset=_preset("safe", action_option_templates=templates),
        ),
    )
    drop = SemanticPose((0.5, -0.2, 0.4), (1.0, 0.0, 0.0, 0.0))
    workflow = compiler.analyze((Place(object=SceneObjectRef("cube"), at=drop),))
    pick_workflow = compiler.analyze((Pick(object=SceneObjectRef("cube")),))
    semantics = compiler.ground(
        pick_workflow,
        0,
        _context(registry),
    ).invocation.goal.semantics
    object_to_eef = torch.eye(4).repeat(2, 1, 1)
    object_to_eef[:, 2, 3] = 0.12
    context = _held_context(registry, semantics, object_to_eef)

    grounded = compiler.ground(workflow, 0, context)

    assert type(grounded.invocation.goal) is PlaceGoal
    options = grounded.invocation.skill_options
    assert type(options) is PlaceOptions
    assert options.lift_height == 0.22
    assert options.cartesian_waypoint_count == 3
    expected = torch.bmm(drop.to_matrix().repeat(2, 1, 1), object_to_eef)
    torch.testing.assert_close(grounded.invocation.goal.xpos, expected)
    engine.resolve(grounded.invocation)


def test_relation_place_composes_late_target_with_verified_transform() -> None:
    registry, _ = _scene_registry()
    compiler, _ = _compiler(registry)
    pick_workflow = compiler.analyze((Pick(object=SceneObjectRef("cube")),))
    semantics = compiler.ground(
        pick_workflow,
        0,
        _context(registry),
    ).invocation.goal.semantics
    object_to_eef = torch.eye(4).repeat(2, 1, 1)
    object_to_eef[:, 0, 3] = 0.08
    context = _held_context(registry, semantics, object_to_eef)
    workflow = compiler.analyze(
        (
            Place(
                object=SceneObjectRef("cube"),
                on=SceneObjectRef("table"),
            ),
        )
    )

    grounded = compiler.ground(workflow, 0, context)

    goal = grounded.invocation.goal
    assert type(goal) is PlaceGoal
    assert type(goal.xpos) is SceneEntityPose
    assert goal.xpos.entity_id == "table_top"
    torch.testing.assert_close(goal.xpos.relative_pose, object_to_eef)


def test_place_rejects_wrong_or_inactive_verified_holder() -> None:
    registry, _ = _scene_registry()
    compiler, _ = _compiler(registry)
    workflow = compiler.analyze(
        (
            Place(
                object=SceneObjectRef("cube"),
                at=SemanticPose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            ),
        )
    )
    wrong = ObjectSemantics(
        affordance=AntipodalAffordance(),
        geometry={},
        entity_id="other",
    )
    wrong_context = _held_context(
        registry,
        wrong,
        torch.eye(4).repeat(2, 1, 1),
    )

    with pytest.raises(SemanticValidationError) as wrong_error:
        compiler.ground(workflow, 0, wrong_context)
    assert wrong_error.value.diagnostic.code == "verified_held_object_required"

    pick_workflow = compiler.analyze((Pick(object=SceneObjectRef("cube")),))
    semantics = compiler.ground(
        pick_workflow,
        0,
        _context(registry),
    ).invocation.goal.semantics
    partial_context = _held_context(
        registry,
        semantics,
        torch.eye(4).repeat(2, 1, 1),
        env_mask=torch.tensor([True, False]),
    )
    with pytest.raises(SemanticValidationError) as inactive_error:
        compiler.ground(workflow, 0, partial_context)
    assert inactive_error.value.diagnostic.code == "verified_held_object_required"

    grounded = compiler.ground(
        workflow,
        0,
        partial_context,
        eligible_mask=torch.tensor([True, False]),
    )
    assert grounded.eligible_mask.tolist() == [True, False]
    with pytest.raises(TypeError, match="created by"):
        GroundedSemanticCall(
            analyzed=grounded.analyzed,
            invocation=grounded.invocation,
            eligible_mask=torch.tensor([True, False]),
        )


def test_registered_lowerer_is_explicit_and_opaque_to_lookahead() -> None:
    registry, _ = _scene_registry()
    without_lowerer, _ = _compiler(registry, registered=True)
    registered = RegisteredSemanticCall(call_id="vendor.inspect")

    with pytest.raises(SemanticValidationError) as error:
        without_lowerer.analyze((registered,))
    assert error.value.diagnostic.code == "semantic_lowerer_not_installed"

    compiler, engine = _compiler(
        registry,
        registered=True,
        registered_lowerers=(_InspectLowerer(),),
    )
    workflow = compiler.analyze(
        (
            Pick(object=SceneObjectRef("cube")),
            registered,
            Place(
                object=SceneObjectRef("cube"),
                at=SemanticPose((0.3, 0.0, 0.2), (1.0, 0.0, 0.0, 0.0)),
            ),
        )
    )

    assert workflow.calls[0].downstream_object_targets == ()
    assert workflow.effect_dependencies[0].producer_index is None
    grounded = compiler.ground(workflow, 1, _context(registry))
    assert grounded.invocation.skill_id == "pick_up"
    engine.resolve(grounded.invocation)


@pytest.mark.parametrize(
    ("output", "message"),
    (("goal", "produced"), ("options", "must not return skill_options")),
)
def test_registered_lowerer_cannot_replace_owned_contracts(
    output: str,
    message: str,
) -> None:
    registry, _ = _scene_registry()
    compiler, _ = _compiler(
        registry,
        registered=True,
        registered_lowerers=(_SubclassOutputLowerer(output),),
    )
    workflow = compiler.analyze((RegisteredSemanticCall(call_id="vendor.inspect"),))

    with pytest.raises(TypeError, match=message):
        compiler.ground(workflow, 0, _context(registry))


def test_workflow_is_factory_owned_and_cannot_cross_compilers() -> None:
    registry, _ = _scene_registry()
    first, _ = _compiler(registry)
    second, _ = _compiler(registry)
    workflow = first.analyze((Pick(object=SceneObjectRef("cube")),))

    with pytest.raises(TypeError, match="created by"):
        SemanticWorkflow()
    with pytest.raises(SemanticValidationError) as error:
        second.ground(workflow, 0, _context(registry))
    assert error.value.diagnostic.code in {
        "semantic_program_stale",
        "semantic_workflow_owner_mismatch",
    }
