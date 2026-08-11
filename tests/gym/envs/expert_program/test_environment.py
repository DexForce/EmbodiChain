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

"""Tests for reusable environment-backed Expert Program assembly."""

from __future__ import annotations

from collections import Counter
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.gym.envs.expert_program.bridge import (
    AtomicDemoBridge,
    DemoBridgeError,
    EnvironmentStepClock,
    GymPlanningObservationProvider,
)
from embodichain.lab.gym.envs.expert_program.cfg import (
    EXPERT_PROGRAM_SCHEMA_VERSION,
    EXPERT_PROGRAM_SCHEMA_VERSION_V2,
    BarrierCfg,
    CyclicPoseTargetCfg,
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    InvokeCfg,
    ObjectNearTargetValidatorCfg,
    OperateArticulationCfg,
    ParallelCfg,
    PickCfg,
    PlaceCfg,
    PoseCfg,
    ProgramNodeCfg,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    WaitStablePostCfg,
)
from embodichain.lab.gym.envs.expert_program.environment import (
    ExpertProgramEnvironmentAdapter,
    ExpertProgramEnvironmentMixin,
    PlanningObservationPort,
)
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    ArticulationOperationAffordance,
    ArticulationOperationTarget,
    AtomicActionEngine,
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    ControlPartCommandProfile,
    EntityState,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    JOINT_POSITION_CAPABILITY,
    MotionPolicy,
    PlanningContext,
    RobotObservation,
    TaskState,
)
from embodichain.lab.sim.skills import (
    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY,
    GRASP_AFFORDANCE_CAPABILITY,
    ControlPartEndpoint,
    RobotResource,
    RobotSkillProfile,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
    SkillPolicyPreset,
)
from embodichain.lab.sim.skills.compiler import SemanticSkillCompiler
from embodichain.lab.sim.skills.evidence import EffectEvidenceProvider
from embodichain.lab.sim.skills.integration import SemanticValidationError

_BATCH_SIZE = 2
_ROBOT_DOF = 2
_STEP_DT = 0.02


class _PoseProvider:
    """Return a stable owned pose for the fake environment scene."""

    def __init__(self) -> None:
        self._pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
        self.calls = 0

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        """Return rows aligned to the requested environment IDs."""
        del timestamp
        self.calls += 1
        return EntityState(self._pose.index_select(0, env_ids))


class _GeometryProvider:
    """Return one opaque planner-facing geometry descriptor."""

    def get_geometry(self) -> object:
        return object()


def _scene_registry(
    *,
    dynamic_collision: bool = False,
    pose_provider: _PoseProvider | None = None,
) -> SceneRegistry:
    """Build an explicitly named object and default grasp affordance."""
    cube = SceneObjectRef("cube")
    grasp = SceneAffordanceRef("cube_grasp")
    selected_pose_provider = _PoseProvider() if pose_provider is None else pose_provider
    return SceneRegistry(
        (
            SceneEntityRegistration(
                ref=cube,
                state_provider=selected_pose_provider,
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
        ),
        collision_world_mode=(
            SceneCollisionWorldMode.PER_ENV if dynamic_collision else None
        ),
    )


def _robot_profile(
    profile_id: str = "fake_robot",
    *,
    safe_motion_policy: MotionPolicy | None = None,
) -> RobotSkillProfile:
    """Build the declarative resource graph used by the fake backend."""
    return RobotSkillProfile(
        profile_id=profile_id,
        resources={
            "manipulator": RobotResource(
                resource_id="manipulator",
                endpoints={
                    "motion": ControlPartEndpoint(
                        control_part="arm",
                        capabilities=frozenset(
                            {
                                BATCH_INVERSE_KINEMATICS_CAPABILITY,
                                CARTESIAN_POSE_CAPABILITY,
                                FORWARD_KINEMATICS_CAPABILITY,
                                JOINT_POSITION_CAPABILITY,
                            }
                        ),
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
                open=torch.tensor((0.0,)),
                grasp=torch.tensor((1.0,)),
            )
        },
        presets={
            "safe": SkillPolicyPreset(
                "safe",
                motion_policy=safe_motion_policy,
            )
        },
        default_preset="safe",
    )


def _parallel_articulation_scene_registry() -> SceneRegistry:
    """Build one drawer whose exact joint key is statically discoverable."""
    drawer = SceneArticulationRef("drawer")
    handle = SceneAffordanceRef("drawer_handle")
    return SceneRegistry(
        (
            SceneEntityRegistration(
                ref=drawer,
                state_provider=_PoseProvider(),
                semantic_type="drawer",
                default_affordances={
                    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY: handle
                },
            ),
            SceneEntityRegistration(
                ref=handle,
                state_provider=_PoseProvider(),
                parent=drawer,
                native_name="handle",
                affordance=ArticulationOperationAffordance(
                    joint_id="drawer_slide",
                    operation_axis=torch.tensor((1.0, 0.0, 0.0)),
                    semantic_targets={
                        "open": ArticulationOperationTarget(
                            target_position=0.4,
                            displacement=0.35,
                        )
                    },
                ),
                affordance_capabilities=frozenset(
                    {ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY}
                ),
                affordance_revision="drawer-operation-v1",
            ),
        )
    )


def _parallel_articulation_profile() -> RobotSkillProfile:
    """Build two physically disjoint resources that can address one drawer."""

    def resource(resource_id: str) -> RobotResource:
        return RobotResource(
            resource_id=resource_id,
            endpoints={
                "motion": ControlPartEndpoint(
                    control_part=f"{resource_id}_arm",
                    capabilities=frozenset(
                        {CARTESIAN_POSE_CAPABILITY, JOINT_POSITION_CAPABILITY}
                    ),
                ),
                "interaction": ControlPartEndpoint(
                    control_part=f"{resource_id}_hand",
                    capabilities=frozenset({GRASP_CAPABILITY}),
                ),
            },
        )

    return RobotSkillProfile(
        profile_id="parallel_articulation_robot",
        resources={
            "left": resource("left"),
            "right": resource("right"),
        },
        command_profiles={
            hand: ControlPartCommandProfile.joint_positions(
                open=torch.tensor((0.0,)),
                grasp=torch.tensor((1.0,)),
            )
            for hand in ("left_hand", "right_hand")
        },
        presets={"safe": SkillPolicyPreset("safe")},
        default_preset="safe",
    )


def _parallel_articulation_engine(
    profile: RobotSkillProfile,
) -> AtomicActionEngine:
    """Build the disjoint four-control-part engine used only for preflight."""
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 4
    robot.control_parts = {
        "left_arm": object(),
        "left_hand": object(),
        "right_arm": object(),
        "right_hand": object(),
    }
    robot.get_qpos.return_value = torch.zeros(_BATCH_SIZE, robot.dof)
    robot.get_qvel.return_value = torch.zeros(_BATCH_SIZE, robot.dof)
    joint_ids = {
        "left_arm": [0],
        "left_hand": [1],
        "right_arm": [2],
        "right_hand": [3],
    }
    robot.get_joint_ids.side_effect = lambda name: joint_ids[name]
    robot.get_solver.return_value = object()
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "fake_planner"
    return AtomicActionEngine(generator, skill_profile=profile)


def _engine(profile: RobotSkillProfile) -> AtomicActionEngine:
    """Build a CPU-only engine around a minimal typed robot surface."""
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = _ROBOT_DOF
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_qpos.return_value = torch.zeros(_BATCH_SIZE, _ROBOT_DOF)
    robot.get_qvel.return_value = torch.zeros(_BATCH_SIZE, _ROBOT_DOF)
    robot.get_joint_ids.side_effect = lambda name: {"arm": [0], "hand": [1]}[name]
    robot.get_solver.return_value = object()
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "fake_planner"
    return AtomicActionEngine(generator, skill_profile=profile)


class _FakeEnvironmentFactory:
    """Count every explicit factory boundary used by the production adapter."""

    scene_registry_id = "fake_scene"
    robot_profile_id = "fake_robot"

    def __init__(self, *, returned_profile_id: str = "fake_robot") -> None:
        self.returned_profile_id = returned_profile_id
        self.calls: Counter[str] = Counter()
        self.observation_samples = 0

    def create_scene_registry(self) -> SceneRegistry:
        """Create a fresh live registry."""
        self.calls["scene"] += 1
        return _scene_registry()

    def create_robot_skill_profile(self) -> RobotSkillProfile:
        """Create the configured robot profile."""
        self.calls["profile"] += 1
        return _robot_profile(self.returned_profile_id)

    def create_atomic_action_engine(
        self,
        profile: RobotSkillProfile,
    ) -> AtomicActionEngine:
        """Create an engine for exactly the supplied profile."""
        self.calls["engine"] += 1
        return _engine(profile)

    def create_planning_observation_provider(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        clock: EnvironmentStepClock,
    ) -> GymPlanningObservationProvider:
        """Create a callback-backed Gym observation port."""
        self.calls["observation"] += 1
        scene_provider = scene_registry.make_scene_provider(batch_size=_BATCH_SIZE)

        def capture(task_state: TaskState) -> PlanningContext:
            self.observation_samples += 1
            env_ids = torch.arange(_BATCH_SIZE, dtype=torch.long)
            timestamp = clock.now()
            return PlanningContext(
                robot=RobotObservation(
                    timestamp=timestamp,
                    qpos=engine.robot.get_qpos(),
                    qvel=engine.robot.get_qvel(),
                ),
                task=task_state,
                scene=scene_provider.snapshot(
                    timestamp=timestamp,
                    env_ids=env_ids,
                ),
                env_ids=env_ids,
            )

        return GymPlanningObservationProvider(capture)

    def create_effect_evidence_providers(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        observation_provider: PlanningObservationPort,
    ) -> tuple[EffectEvidenceProvider, ...]:
        """Return the fake environment's explicit evidence-provider set."""
        del scene_registry, engine, observation_provider
        self.calls["evidence"] += 1
        return ()


class _ParallelArticulationFactory(_FakeEnvironmentFactory):
    """Expose two robot resources and one shared articulation write target."""

    scene_registry_id = "parallel_articulation_scene"
    robot_profile_id = "parallel_articulation_robot"

    def create_scene_registry(self) -> SceneRegistry:
        self.calls["scene"] += 1
        return _parallel_articulation_scene_registry()

    def create_robot_skill_profile(self) -> RobotSkillProfile:
        self.calls["profile"] += 1
        return _parallel_articulation_profile()

    def create_atomic_action_engine(
        self,
        profile: RobotSkillProfile,
    ) -> AtomicActionEngine:
        self.calls["engine"] += 1
        return _parallel_articulation_engine(profile)


class _DynamicCollisionFactory(_FakeEnvironmentFactory):
    """Expose a safe dynamic scene backed by an unsupported planner."""

    def __init__(self) -> None:
        super().__init__()
        self.pose_provider = _PoseProvider()
        self.last_engine: AtomicActionEngine | None = None

    def create_scene_registry(self) -> SceneRegistry:
        self.calls["scene"] += 1
        return _scene_registry(
            dynamic_collision=True,
            pose_provider=self.pose_provider,
        )

    def create_robot_skill_profile(self) -> RobotSkillProfile:
        self.calls["profile"] += 1
        return _robot_profile(
            safe_motion_policy=MotionPolicy(strategy="motion_gen"),
        )

    def create_atomic_action_engine(
        self,
        profile: RobotSkillProfile,
    ) -> AtomicActionEngine:
        self.calls["engine"] += 1
        engine = _engine(profile)
        engine.motion_generator.supports_dynamic_collision_world = False
        self.last_engine = engine
        return engine


class _FakeDeclarativeEnvironment(ExpertProgramEnvironmentMixin):
    """Environment surface requiring no task-level motion implementation."""

    def __init__(self, adapter: ExpertProgramEnvironmentAdapter) -> None:
        self._adapter = adapter

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the reusable environment adapter."""
        return self._adapter


class _AcceptParallelSafety:
    """Accept test-only merged commands after static preflight succeeds."""

    def validate(
        self,
        *,
        branch_frames: object,
        merged_frame: object,
    ) -> None:
        del branch_frames, merged_frame


class _PresetCheckingPostPolicyPort:
    """Pure test port that rejects policies outside its preset table."""

    def __init__(self, preset_ids: tuple[str, ...]) -> None:
        self._preset_ids = frozenset(preset_ids)
        self.validated_presets: list[str] = []

    def validate_policy(self, policy: object, *, segment: object) -> None:
        del segment
        cfg = getattr(policy, "cfg")
        preset = getattr(cfg, "preset")
        self.validated_presets.append(preset)
        if preset not in self._preset_ids:
            raise KeyError(f"Unknown settle preset {preset!r}.")

    def actions(
        self,
        policy: object,
        *,
        segment: object,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        del policy, segment, active_mask
        raise AssertionError("Preflight must not request post-policy actions.")


def _program(
    *,
    robot_profile: str = "fake_robot",
    scene_registry: str = "fake_scene",
    runtime_preset: str = "safe",
    schema_version: int = EXPERT_PROGRAM_SCHEMA_VERSION,
    node: ProgramNodeCfg | None = None,
    targets: dict[str, CyclicPoseTargetCfg] | None = None,
) -> ExpertProgramCfg:
    """Build one minimal declarative pick program."""
    return ExpertProgramCfg(
        schema_version=schema_version,
        program_id="fake_pick",
        integration=ExpertProgramIntegrationCfg(
            robot_profile=robot_profile,
            scene_registry=scene_registry,
            runtime_preset=runtime_preset,
        ),
        program=(InvokeCfg(call=PickCfg(object="cube")) if node is None else node),
        targets={} if targets is None else targets,
    )


def _program_with_later_parallel_conflict() -> ExpertProgramCfg:
    """Build an early sequential call followed by conflicting branch claims."""
    return _program(
        schema_version=EXPERT_PROGRAM_SCHEMA_VERSION_V2,
        node=SequenceCfg(
            items=(
                InvokeCfg(call=PickCfg(object="cube")),
                ParallelCfg(
                    branches=(
                        InvokeCfg(call=PickCfg(object="cube")),
                        InvokeCfg(call=PickCfg(object="cube")),
                    ),
                    barrier=BarrierCfg(name="conflicting_join"),
                ),
            )
        ),
    )


def _program_with_later_segment_hooks(
    *,
    post: tuple[WaitStablePostCfg, ...] = (),
    validators: tuple[ObjectNearTargetValidatorCfg, ...] = (),
) -> ExpertProgramCfg:
    """Build a valid pick/place flow whose hooks live on the later segment."""
    return _program(
        node=SequenceCfg(
            items=(
                SegmentCfg(
                    name="pick",
                    steps=InvokeCfg(call=PickCfg(object="cube")),
                ),
                SegmentCfg(
                    name="place",
                    steps=InvokeCfg(
                        call=PlaceCfg(
                            object="cube",
                            at=TargetRefCfg(target="drop"),
                        )
                    ),
                    post=post,
                    validators=validators,
                ),
            )
        ),
        targets={
            "drop": CyclicPoseTargetCfg(
                values=(
                    PoseCfg(
                        position=(0.4, 0.1, 0.2),
                        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    ),
                )
            )
        },
    )


def _parallel_articulation_program() -> ExpertProgramCfg:
    """Operate one joint from disjoint resources in two parallel branches."""
    return ExpertProgramCfg(
        schema_version=EXPERT_PROGRAM_SCHEMA_VERSION_V2,
        program_id="conflicting_drawer_operations",
        integration=ExpertProgramIntegrationCfg(
            robot_profile="parallel_articulation_robot",
            scene_registry="parallel_articulation_scene",
            runtime_preset="safe",
        ),
        targets={},
        program=ParallelCfg(
            branches=tuple(
                InvokeCfg(
                    call=OperateArticulationCfg(
                        articulation="drawer",
                        target="open",
                        resources={"primary": resource_id},
                    )
                )
                for resource_id in ("left", "right")
            ),
            barrier=BarrierCfg(name="drawer_join"),
        ),
    )


def test_mixin_compiles_and_assembles_bridge_without_task_motion_code() -> None:
    """One adapter property implements both EmbodiedEnv integration hooks."""
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    env = _FakeDeclarativeEnvironment(adapter)

    compiled = env.compile_expert_program(_program())

    assert factory.calls == Counter(scene=1)
    bridge = env.create_expert_program_bridge(compiled)
    assert isinstance(bridge, AtomicDemoBridge)
    assert factory.calls == Counter(
        scene=2,
        profile=1,
        engine=1,
        observation=1,
        evidence=1,
    )
    segment_iterator = bridge.iter_segments()
    segment = next(segment_iterator)
    assert segment.name == "invoke:pick"
    assert segment.failure_policy == "row_independent"
    segment_iterator.close()


def test_later_sequential_resource_error_fails_before_observation_or_action() -> None:
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    compiled = adapter.compile(
        _program(
            node=SequenceCfg(
                items=(
                    InvokeCfg(call=PickCfg(object="cube")),
                    InvokeCfg(
                        call=PickCfg(
                            object="cube",
                            resources={"primary": "missing"},
                        )
                    ),
                )
            )
        )
    )

    with pytest.raises(SemanticValidationError) as error:
        adapter.create_bridge(compiled)

    assert error.value.diagnostic.code == "unknown_resource"
    assert factory.calls == Counter(scene=2, profile=1, engine=1)
    assert factory.observation_samples == 0


def test_later_post_policy_requires_port_before_runtime_assembly() -> None:
    """A later hook cannot defer its missing-port error until segment execution."""
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    compiled = adapter.compile(
        _program_with_later_segment_hooks(
            post=(WaitStablePostCfg(entity="cube", preset="fast"),),
        )
    )

    with pytest.raises(DemoBridgeError, match="SegmentPostPolicyPort"):
        adapter.create_bridge(compiled)

    assert factory.calls == Counter(scene=1)
    assert factory.observation_samples == 0


def test_later_validator_requires_port_before_runtime_assembly() -> None:
    """A later validator must have an installed pure-validation boundary."""
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    compiled = adapter.compile(
        _program_with_later_segment_hooks(
            validators=(
                ObjectNearTargetValidatorCfg(
                    object="cube",
                    target="drop",
                ),
            ),
        )
    )

    with pytest.raises(DemoBridgeError, match="SegmentValidatorPort"):
        adapter.create_bridge(compiled)

    assert factory.calls == Counter(scene=1)
    assert factory.observation_samples == 0


def test_later_unknown_settle_preset_fails_during_pure_preflight() -> None:
    """Every declared preset is checked before semantic or live runtime assembly."""
    factory = _FakeEnvironmentFactory()
    post_policy_port = _PresetCheckingPostPolicyPort(("fast",))
    adapter = ExpertProgramEnvironmentAdapter(
        factory,
        step_dt=_STEP_DT,
        post_policy_port=post_policy_port,
    )
    compiled = adapter.compile(
        _program_with_later_segment_hooks(
            post=(WaitStablePostCfg(entity="cube", preset="missing"),),
        )
    )

    with pytest.raises(KeyError, match="Unknown settle preset 'missing'"):
        adapter.create_bridge(compiled)

    assert post_policy_port.validated_presets == ["missing"]
    assert factory.calls == Counter(scene=1)
    assert factory.observation_samples == 0


def test_preflight_preserves_pick_target_lookahead_across_explicit_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    config = _program(
        node=SequenceCfg(
            items=(
                SegmentCfg(
                    name="pick",
                    steps=InvokeCfg(call=PickCfg(object="cube")),
                ),
                SegmentCfg(
                    name="place",
                    steps=InvokeCfg(
                        call=PlaceCfg(
                            object="cube",
                            at=TargetRefCfg(target="drop"),
                        )
                    ),
                ),
            )
        ),
        targets={
            "drop": CyclicPoseTargetCfg(
                values=(
                    PoseCfg(
                        position=(0.4, 0.1, 0.2),
                        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    ),
                )
            )
        },
    )
    workflows: list[object] = []
    original_analyze = SemanticSkillCompiler.analyze

    def record_analyze(
        compiler: SemanticSkillCompiler,
        calls: object,
        **kwargs: object,
    ) -> object:
        workflow = original_analyze(compiler, calls, **kwargs)
        workflows.append(workflow)
        return workflow

    monkeypatch.setattr(SemanticSkillCompiler, "analyze", record_analyze)

    adapter.create_bridge(adapter.compile(config))

    assert len(workflows) == 1
    workflow = workflows[0]
    assert len(workflow.calls) == 2  # type: ignore[attr-defined]
    downstream = workflow.calls[0].downstream_object_targets  # type: ignore[attr-defined]
    assert len(downstream) == 1
    assert downstream[0].pose is not None
    torch.testing.assert_close(
        downstream[0].pose.position,
        torch.tensor((0.4, 0.1, 0.2)),
    )
    assert factory.observation_samples == 0


def test_parallel_program_requires_safety_validator_before_first_action() -> None:
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    compiled = adapter.compile(_program_with_later_parallel_conflict())

    with pytest.raises(ValueError, match="ParallelCommandSafetyValidator"):
        adapter.create_bridge(compiled)

    assert factory.observation_samples == 0


def test_later_parallel_claim_conflict_fails_during_whole_program_preflight() -> None:
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(
        factory,
        step_dt=_STEP_DT,
        parallel_safety_validator=_AcceptParallelSafety(),
    )
    compiled = adapter.compile(_program_with_later_parallel_conflict())

    with pytest.raises(ValueError, match="overlapping resource claims"):
        adapter.create_bridge(compiled)

    assert factory.observation_samples == 0


def test_parallel_symbolic_write_conflict_fails_before_observation_or_action() -> None:
    factory = _ParallelArticulationFactory()
    adapter = ExpertProgramEnvironmentAdapter(
        factory,
        step_dt=_STEP_DT,
        parallel_safety_validator=_AcceptParallelSafety(),
    )
    compiled = adapter.compile(_parallel_articulation_program())
    materialized = compiled.materialize()
    parallel_block = tuple(materialized.iter_segments())[0].parallel_block
    assert parallel_block is not None
    expected_path = parallel_block.branches[1].source_path

    with pytest.raises(SemanticValidationError) as error:
        adapter.create_bridge(compiled)

    diagnostic = error.value.diagnostic
    assert diagnostic.code == "parallel_symbolic_write_conflict"
    assert diagnostic.path == expected_path
    assert "articulation_joint['drawer', 'drawer_slide']" in diagnostic.message
    assert factory.observation_samples == 0


def test_runtime_assembly_shares_exact_bound_components() -> None:
    """Compiler, runtime, clock, sink, scene, and profile form one ownership graph."""
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)

    assembly = adapter.assemble_runtime(_program().integration)

    assert assembly.compiler.integration.scene_registry is assembly.scene_registry
    assert assembly.compiler.integration.manifest is assembly.manifest
    assert assembly.compiler.integration.engine is assembly.engine
    assert assembly.compiler.integration.manifest.runtime_preset == "safe"
    assert assembly.compiler.integration.robot_profile.engine is assembly.engine
    assert assembly.runtime.compiler is assembly.compiler
    assert assembly.runtime.clock is assembly.clock
    assert assembly.command_sink.clock is assembly.clock
    assert assembly.clock.step_dt == pytest.approx(_STEP_DT)
    assert assembly.evidence_collector.registry.providers == {}


def test_safe_dynamic_collision_fails_before_observation_planning_or_command() -> None:
    factory = _DynamicCollisionFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    compiled = adapter.compile(_program(runtime_preset="safe"))

    with pytest.raises(SemanticValidationError) as error:
        adapter.create_bridge(compiled)

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
    assert factory.calls == Counter(scene=2, profile=1, engine=1)
    assert factory.pose_provider.calls == 0
    assert factory.observation_samples == 0
    assert factory.last_engine is not None
    factory.last_engine.motion_generator.generate.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("robot_profile", "other_robot", "selects robot_profile"),
        ("scene_registry", "other_scene", "selects scene_registry"),
    ),
)
def test_integration_id_mismatch_fails_before_live_factory_access(
    field: str,
    value: str,
    match: str,
) -> None:
    """Static selection drift never reaches simulation or motion factories."""
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    options = {field: value}

    with pytest.raises(ValueError, match=match):
        adapter.compile(_program(**options))

    assert factory.calls == Counter()


def test_robot_profile_factory_drift_fails_before_engine_creation() -> None:
    """The declared profile ID must match the concrete factory output."""
    factory = _FakeEnvironmentFactory(returned_profile_id="different_robot")
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)

    with pytest.raises(ValueError, match="profile declaration drifted"):
        adapter.assemble_runtime(_program().integration)

    assert factory.calls == Counter(scene=1, profile=1)


def test_factory_selection_declaration_drift_fails_before_live_access() -> None:
    """A mutable factory cannot silently change IDs after adapter creation."""
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    factory.scene_registry_id = "changed_scene"

    with pytest.raises(ValueError, match="scene registry declaration drifted"):
        adapter.compile(_program())

    assert factory.calls == Counter()


def test_unknown_runtime_preset_fails_during_manifest_assembly() -> None:
    """Runtime preset names are validated against the selected robot profile."""
    factory = _FakeEnvironmentFactory()
    adapter = ExpertProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)

    with pytest.raises(ValueError, match="Unknown runtime preset"):
        adapter.assemble_runtime(_program(runtime_preset="unregistered").integration)


def test_factory_protocol_is_required() -> None:
    """Loose objects cannot enter the production assembly boundary."""
    with pytest.raises(TypeError, match="ExpertProgramEnvironmentFactory"):
        ExpertProgramEnvironmentAdapter(object(), step_dt=_STEP_DT)
