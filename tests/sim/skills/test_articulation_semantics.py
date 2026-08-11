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

"""Tests for first-class semantic articulation operations."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    Affordance,
    ArticulationOperationAffordance,
    ArticulationOperationTarget,
    AtomicActionEngine,
    CARTESIAN_POSE_CAPABILITY,
    ControlPartCommandProfile,
    EntityState,
    GRASP_CAPABILITY,
    JOINT_POSITION_CAPABILITY,
    ObservedArticulationJointState,
    OperateArticulationGoal,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.skills.calls import (
    OperateArticulation,
    builtin_semantic_call_catalog,
)
from embodichain.lab.sim.skills.compiler import SemanticSkillCompiler
from embodichain.lab.sim.skills.effects import (
    ArticulationJointStateExpectation,
    JointStateEffectClause,
    SemanticEffectKind,
    SymbolicStateKey,
)
from embodichain.lab.sim.skills.integration import (
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
    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY,
    SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID,
    SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION,
    ArticulationJointEvidenceAddress,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRegistration,
    SceneRegistry,
)

_BATCH_SIZE = 2
_TARGET_POSITION = 0.42
_TARGET_DISPLACEMENT = 0.4
_POSITION_SCALE = 0.5


class _MutablePoseProvider:
    """Expose a mutable pose and count semantic observation calls."""

    def __init__(
        self,
        pose: torch.Tensor,
        *,
        joint_position: torch.Tensor | None = None,
    ) -> None:
        self.pose = pose.clone()
        self.joint_position = None if joint_position is None else joint_position.clone()
        self.calls = 0
        self.joint_calls = 0

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        self.calls += 1
        return EntityState(self.pose)

    def observe_joints(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> dict[str, ObservedArticulationJointState]:
        del timestamp, env_ids
        self.joint_calls += 1
        if self.joint_position is None:
            raise RuntimeError("This provider has no articulation joint fixture.")
        return {"drawer_slide": ObservedArticulationJointState(self.joint_position)}


def _translated_offset(x: float, y: float, z: float) -> torch.Tensor:
    """Build one test-only proper local offset."""
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 3] = torch.tensor((x, y, z), dtype=torch.float32)
    return pose


def _operation_affordance() -> ArticulationOperationAffordance:
    """Build the canonical drawer-handle fixture."""
    return ArticulationOperationAffordance(
        joint_id="drawer_slide",
        approach_offset=_translated_offset(0.0, 0.0, -0.1),
        contact_offset=torch.eye(4),
        operation_offset=_translated_offset(0.0, 0.02, 0.0),
        retract_offset=_translated_offset(0.0, 0.0, -0.1),
        operation_axis=torch.tensor((1.0, 0.0, 0.0)),
        position_scale=_POSITION_SCALE,
        semantic_targets={
            "open": ArticulationOperationTarget(
                target_position=_TARGET_POSITION,
                displacement=_TARGET_DISPLACEMENT,
            )
        },
    )


def _registry() -> tuple[SceneRegistry, _MutablePoseProvider, _MutablePoseProvider]:
    """Build an articulation plus one directly registered handle affordance."""
    articulation = SceneArticulationRef("drawer")
    handle = SceneAffordanceRef("drawer_handle")
    articulation_provider = _MutablePoseProvider(
        torch.eye(4).repeat(_BATCH_SIZE, 1, 1),
        joint_position=torch.zeros(_BATCH_SIZE, 1),
    )
    handle_pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
    handle_pose[:, 0, 3] = 0.3
    handle_provider = _MutablePoseProvider(handle_pose)
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=articulation,
                state_provider=articulation_provider,
                joint_state_provider=articulation_provider,
                semantic_type="drawer",
                default_affordances={
                    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY: handle
                },
            ),
            SceneEntityRegistration(
                ref=handle,
                state_provider=handle_provider,
                parent=articulation,
                native_name="handle",
                affordance=_operation_affordance(),
                affordance_capabilities=frozenset(
                    {ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY}
                ),
                affordance_revision="drawer-handle-v1",
            ),
        )
    )
    return registry, articulation_provider, handle_provider


def _profile() -> RobotSkillProfile:
    """Build one resource satisfying motion and interaction endpoints."""
    return RobotSkillProfile(
        profile_id="articulation_test_robot",
        resources={
            "manipulator": RobotResource(
                resource_id="manipulator",
                endpoints={
                    "motion": ControlPartEndpoint(
                        control_part="arm",
                        capabilities=frozenset(
                            {
                                CARTESIAN_POSE_CAPABILITY,
                                JOINT_POSITION_CAPABILITY,
                            }
                        ),
                    ),
                    "interaction": ControlPartEndpoint(
                        control_part="hand",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                    ),
                },
            )
        },
        command_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.tensor((0.0,)),
                grasp=torch.tensor((1.0,)),
            )
        },
        presets={"safe": SkillPolicyPreset("safe")},
        default_preset="safe",
    )


def _engine(profile: RobotSkillProfile) -> AtomicActionEngine:
    """Construct a CPU-only engine with the minimum typed robot surface."""
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 2
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_qpos.return_value = torch.zeros(_BATCH_SIZE, robot.dof)
    robot.get_qvel.return_value = torch.zeros(_BATCH_SIZE, robot.dof)
    robot.get_joint_ids.side_effect = lambda name: {"arm": [0], "hand": [1]}[name]
    robot.get_solver.return_value = object()
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    return AtomicActionEngine(generator, skill_profile=profile)


def _compiler(registry: SceneRegistry) -> SemanticSkillCompiler:
    """Bind the curated semantic catalog to the test scene and profile."""
    profile = _profile()
    engine = _engine(profile)
    manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=profile,
        call_catalog=builtin_semantic_call_catalog(),
    )
    return SemanticSkillCompiler(manifest.bind(registry, engine))


def _context(scene: SceneSnapshot, *, timestamp: float) -> PlanningContext:
    """Build one immutable planning observation around a supplied scene."""
    env_ids = torch.arange(_BATCH_SIZE, dtype=torch.long)
    return PlanningContext(
        robot=RobotObservation(
            timestamp=timestamp,
            qpos=torch.zeros(_BATCH_SIZE, 2),
            qvel=torch.zeros(_BATCH_SIZE, 2),
        ),
        task=TaskState.empty(_BATCH_SIZE, "cpu"),
        scene=scene,
        env_ids=env_ids,
    )


def test_articulation_affordance_owns_configuration_and_grounds_geometry() -> None:
    axis = torch.tensor((2.0, 0.0, 0.0))
    operation_offset = _translated_offset(0.0, 0.02, 0.0)
    targets = {
        "open": ArticulationOperationTarget(
            _TARGET_POSITION,
            _TARGET_DISPLACEMENT,
        )
    }
    affordance = ArticulationOperationAffordance(
        joint_id="drawer_slide",
        operation_axis=axis,
        operation_offset=operation_offset,
        position_scale=_POSITION_SCALE,
        semantic_targets=targets,
    )
    axis.zero_()
    operation_offset.zero_()
    targets.clear()

    handle = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
    handle[:, 0, 3] = 0.3
    _, _, operation, _ = affordance.ground_poses(
        handle,
        displacement=_TARGET_DISPLACEMENT,
    )

    assert tuple(affordance.semantic_targets) == ("open",)
    assert torch.allclose(affordance.operation_axis, torch.tensor((1.0, 0.0, 0.0)))
    assert torch.allclose(
        operation[:, :3, 3],
        torch.tensor((0.5, 0.02, 0.0)).repeat(_BATCH_SIZE, 1),
    )


def test_registry_returns_owned_articulation_affordance_snapshots() -> None:
    registry, _, _ = _registry()

    first = registry.lookup(
        SceneAffordanceRef("drawer_handle"),
        expected_type=SceneAffordanceRef,
    ).affordance
    second = registry.lookup(
        SceneAffordanceRef("drawer_handle"),
        expected_type=SceneAffordanceRef,
    ).affordance

    assert type(first) is ArticulationOperationAffordance
    assert type(second) is ArticulationOperationAffordance
    assert first is not second
    first.operation_offset[0, 3] = 99.0
    assert second.operation_offset[0, 3].item() == 0.0


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"target_position": _TARGET_POSITION},
        {"target_displacement": _TARGET_DISPLACEMENT},
        {
            "target": "open",
            "target_position": _TARGET_POSITION,
            "target_displacement": _TARGET_DISPLACEMENT,
        },
    ),
)
def test_articulation_call_requires_exactly_one_complete_target(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        OperateArticulation(
            articulation=SceneArticulationRef("drawer"),
            **kwargs,
        )


def test_static_link_selects_default_without_observing_scene() -> None:
    registry, articulation_provider, handle_provider = _registry()
    compiler = _compiler(registry)

    workflow = compiler.analyze(
        (
            OperateArticulation(
                articulation=SceneArticulationRef("drawer"),
                target="open",
            ),
        )
    )

    analyzed = workflow.calls[0]
    assert analyzed.effect_kind is SemanticEffectKind.ARTICULATION
    assert analyzed.bound.linked.affordances["handle"] == SceneAffordanceRef(
        "drawer_handle"
    )
    assert analyzed.bound.linked.descriptor.skill_id == "operate_articulation"
    assert analyzed.symbolic_writes == frozenset(
        {SymbolicStateKey.articulation_joint("drawer", "drawer_slide")}
    )
    assert not analyzed.opaque_symbolic_effect
    assert articulation_provider.calls == 0
    assert handle_provider.calls == 0


def test_static_link_rejects_unknown_explicit_handle_with_path() -> None:
    registry, _, _ = _registry()
    compiler = _compiler(registry)

    with pytest.raises(SemanticValidationError) as error:
        compiler.analyze(
            (
                OperateArticulation(
                    articulation=SceneArticulationRef("drawer"),
                    handle=SceneAffordanceRef("missing_handle"),
                    target="open",
                ),
            )
        )

    assert error.value.diagnostic.path == ("workflow", 0, "call", "handle")


def test_grounding_uses_fresh_handle_pose_and_lowers_typed_effect() -> None:
    registry, articulation_provider, handle_provider = _registry()
    compiler = _compiler(registry)
    workflow = compiler.analyze(
        (
            OperateArticulation(
                articulation=SceneArticulationRef("drawer"),
                target="open",
            ),
        )
    )
    env_ids = torch.arange(_BATCH_SIZE, dtype=torch.long)
    scene_provider = registry.make_scene_provider(
        translation_threshold=0.0,
        rotation_threshold=0.0,
    )
    first_context = _context(
        scene_provider.snapshot(timestamp=0.0, env_ids=env_ids),
        timestamp=0.0,
    )
    first = compiler.ground(workflow, 0, first_context)
    handle_provider.pose[:, 0, 3] = 0.7
    assert articulation_provider.joint_position is not None
    articulation_provider.joint_position[:, 0] = 0.1
    second_context = _context(
        scene_provider.snapshot(timestamp=1.0, env_ids=env_ids),
        timestamp=1.0,
    )
    second = compiler.ground(workflow, 0, second_context, revision=1)

    first_goal = first.invocation.goal
    second_goal = second.invocation.goal
    assert type(first_goal) is OperateArticulationGoal
    assert type(second_goal) is OperateArticulationGoal
    first_poses = first_goal.geometry.resolve(
        first_context,
        displacement=torch.full((_BATCH_SIZE,), _TARGET_DISPLACEMENT),
    )
    second_poses = second_goal.geometry.resolve(
        second_context,
        displacement=torch.full((_BATCH_SIZE,), _TARGET_DISPLACEMENT),
    )
    assert torch.allclose(first_poses[0][:, 0, 3], torch.full((2,), 0.3))
    assert torch.allclose(second_poses[0][:, 0, 3], torch.full((2,), 0.7))
    assert torch.allclose(second_poses[2][:, 0, 3], torch.full((2,), 0.9))
    assert torch.equal(
        first_goal.source_position,
        torch.zeros(_BATCH_SIZE, 1),
    )
    assert torch.equal(
        second_goal.source_position,
        torch.full((_BATCH_SIZE, 1), 0.1),
    )
    assert second_goal.target_displacement == _TARGET_DISPLACEMENT
    assert torch.allclose(
        second_goal.target_position,
        torch.full((_BATCH_SIZE, 1), _TARGET_POSITION),
    )

    effect = second.effect_spec
    assert effect is not None
    assert effect.effect_kind is SemanticEffectKind.ARTICULATION
    expectation = effect.state_expectations[0]
    clause = effect.clauses[0]
    assert type(expectation) is ArticulationJointStateExpectation
    assert expectation.articulation_id == "drawer"
    assert expectation.joint_id == "drawer_slide"
    assert type(clause) is JointStateEffectClause
    assert torch.equal(clause.target_position, second_goal.target_position)
    assert clause.source.provider_id == SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID
    assert clause.source.revision == SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION
    assert type(clause.source.address) is ArticulationJointEvidenceAddress
    assert clause.source.address.articulation_id == "drawer"
    assert clause.source.address.joint_id == "drawer_slide"


def test_explicit_target_pair_records_live_source_joint_state() -> None:
    registry, _, _ = _registry()
    compiler = _compiler(registry)
    workflow = compiler.analyze(
        (
            OperateArticulation(
                articulation=SceneArticulationRef("drawer"),
                target_position=0.25,
                target_displacement=-0.1,
            ),
        )
    )
    env_ids = torch.arange(_BATCH_SIZE, dtype=torch.long)
    scene = registry.make_scene_provider(
        translation_threshold=0.0,
        rotation_threshold=0.0,
    ).snapshot(timestamp=0.0, env_ids=env_ids)

    grounded = compiler.ground(workflow, 0, _context(scene, timestamp=0.0))

    goal = grounded.invocation.goal
    assert type(goal) is OperateArticulationGoal
    assert torch.allclose(goal.target_position, torch.full((_BATCH_SIZE, 1), 0.25))
    assert torch.equal(goal.source_position, torch.zeros(_BATCH_SIZE, 1))
    assert goal.target_displacement == -0.1
    operation = goal.geometry.resolve(
        _context(scene, timestamp=0.0),
        displacement=torch.full((_BATCH_SIZE,), -0.1),
    )[2]
    assert torch.allclose(operation[:, 0, 3], torch.full((2,), 0.25))


def test_unknown_named_target_has_strict_grounding_diagnostic() -> None:
    registry, _, _ = _registry()
    compiler = _compiler(registry)
    workflow = compiler.analyze(
        (
            OperateArticulation(
                articulation=SceneArticulationRef("drawer"),
                target="closed",
            ),
        )
    )
    env_ids = torch.arange(_BATCH_SIZE, dtype=torch.long)
    scene = registry.make_scene_provider(
        translation_threshold=0.0,
        rotation_threshold=0.0,
    ).snapshot(timestamp=0.0, env_ids=env_ids)

    with pytest.raises(SemanticValidationError) as error:
        compiler.ground(workflow, 0, _context(scene, timestamp=0.0))

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "unknown_articulation_target"
    assert diagnostic.path == ("workflow", 0, "call", "target")
    assert diagnostic.candidates == ("open",)


def test_missing_handle_pose_has_strict_grounding_diagnostic() -> None:
    registry, _, _ = _registry()
    compiler = _compiler(registry)
    workflow = compiler.analyze(
        (
            OperateArticulation(
                articulation=SceneArticulationRef("drawer"),
                target="open",
            ),
        )
    )
    scene = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={"drawer": EntityState(torch.eye(4).repeat(_BATCH_SIZE, 1, 1))},
    )

    with pytest.raises(SemanticValidationError) as error:
        compiler.ground(workflow, 0, _context(scene, timestamp=0.0))

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "missing_handle_observation"
    assert diagnostic.path == ("workflow", 0, "call", "handle")


def test_missing_live_joint_state_has_strict_grounding_diagnostic() -> None:
    registry, _, _ = _registry()
    compiler = _compiler(registry)
    workflow = compiler.analyze(
        (
            OperateArticulation(
                articulation=SceneArticulationRef("drawer"),
                target="open",
            ),
        )
    )
    handle = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
    scene = SceneSnapshot(
        timestamp=0.0,
        version=0,
        entities={"drawer_handle": EntityState(handle)},
    )

    with pytest.raises(SemanticValidationError) as error:
        compiler.ground(workflow, 0, _context(scene, timestamp=0.0))

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "missing_articulation_joint_observation"
    assert diagnostic.path == ("workflow", 0, "call", "articulation")


def test_articulation_capability_rejects_untyped_affordance_payload() -> None:
    articulation = SceneArticulationRef("drawer")
    handle = SceneAffordanceRef("drawer_handle")
    provider = _MutablePoseProvider(torch.eye(4).repeat(_BATCH_SIZE, 1, 1))

    with pytest.raises(TypeError, match="ArticulationOperationAffordance"):
        SceneRegistry(
            (
                SceneEntityRegistration(
                    ref=articulation,
                    state_provider=provider,
                    default_affordances={
                        ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY: handle
                    },
                ),
                SceneEntityRegistration(
                    ref=handle,
                    state_provider=provider,
                    parent=articulation,
                    native_name="handle",
                    affordance=Affordance(),
                    affordance_capabilities=frozenset(
                        {ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY}
                    ),
                    affordance_revision="bad-v1",
                ),
            )
        )
