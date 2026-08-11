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

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AssembleGoal,
    AtomicAction,
    AtomicActionEngine,
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    CommandAcknowledgement,
    ControlPartCommandProfile,
    EntityState,
    ExecutionRunnerCfg,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    GraspGoal,
    HeldObjectState,
    JointPositionTarget,
    PickUp,
    PickUpOptions,
    Place as AtomicPlace,
    PlaceGoal,
    PlaceOptions,
    PlanningContext,
    RobotObservation,
    RuntimeCommandFrame,
    RuntimeEndpointTarget,
    StateDelta,
    TaskState,
    TimedCommandSequence,
)
from embodichain.lab.sim.skills import (
    ControlPartEndpoint,
    GRASP_AFFORDANCE_CAPABILITY,
    Pick,
    Place,
    ResourceBinding,
    RobotResource,
    RobotSkillProfile,
    SceneAffordanceRef,
    SceneEntityRegistration,
    SceneManifest,
    SceneObjectRef,
    SceneRegistry,
    SemanticExecutionStatus,
    SemanticEffectVerifier,
    SemanticIntegrationManifest,
    SemanticPose,
    SemanticSkillRuntime,
    SemanticTaskStatus,
    SkillPolicyPreset,
    builtin_semantic_call_catalog,
)

_MOTION_CAPABILITIES = frozenset(
    {
        BATCH_INVERSE_KINEMATICS_CAPABILITY,
        CARTESIAN_POSE_CAPABILITY,
        FORWARD_KINEMATICS_CAPABILITY,
    }
)


class _PoseProvider:
    def __init__(self, pose: torch.Tensor) -> None:
        self.pose = pose

    def observe(self, *, timestamp: float, env_ids: torch.Tensor) -> EntityState:
        del timestamp, env_ids
        return EntityState(self.pose)


class _InstantPick(AtomicAction):
    skill_id = PickUp.skill_id
    GoalType = GraspGoal
    OptionsType = PickUpOptions
    binding_contract = PickUp.binding_contract

    def _plan(self, request, context):
        goal = self.require_goal(request)
        endpoint = request.binding.endpoint("primary", "motion")
        endpoint.require_target(JointPositionTarget)
        held = HeldObjectState(
            semantics=goal.semantics,
            object_to_eef=torch.eye(4).repeat(context.batch_size, 1, 1),
            grasp_xpos=torch.eye(4).repeat(context.batch_size, 1, 1),
        )
        return self.build_command_plan(
            request,
            context,
            success=True,
            commands=TimedCommandSequence((), context.env_ids),
            expected_effects=StateDelta(
                held_object_updates={endpoint.task_state_key: held}
            ),
        )


class _InstantPlace(AtomicAction):
    skill_id = AtomicPlace.skill_id
    GoalType = (PlaceGoal, AssembleGoal)
    OptionsType = PlaceOptions
    binding_contract = AtomicPlace.binding_contract

    def _plan(self, request, context):
        self.require_goal(request)
        endpoint = request.binding.endpoint("primary", "motion")
        endpoint.require_target(JointPositionTarget)
        return self.build_command_plan(
            request,
            context,
            success=True,
            commands=TimedCommandSequence((), context.env_ids),
            expected_effects=StateDelta(
                held_object_updates={endpoint.task_state_key: None}
            ),
        )


class _ExecutionPorts:
    def __init__(self, registry: SceneRegistry, robot: Mock) -> None:
        self.registry = registry
        self.robot = robot
        self.env_ids = torch.tensor([0, 1], dtype=torch.long)
        self.scene_provider = registry.make_scene_provider(batch_size=2)
        self.time = 0.0
        self.hold_calls = 0
        self.cancel_calls = 0

    def observe(self, task_state: TaskState) -> PlanningContext:
        qpos = self.robot.get_qpos()
        return PlanningContext(
            robot=RobotObservation(
                timestamp=self.time,
                qpos=qpos,
                qvel=torch.zeros_like(qpos),
            ),
            task=task_state,
            scene=self.scene_provider.snapshot(
                timestamp=self.time,
                env_ids=self.env_ids,
            ),
            env_ids=self.env_ids,
            control_dt=0.01,
        )

    def send(
        self,
        command: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        del command, timeout
        return CommandAcknowledgement.accepted_ack()

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        del targets, context, timeout
        self.hold_calls += 1
        return CommandAcknowledgement.accepted_ack()

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        del targets, timeout
        self.cancel_calls += 1
        return CommandAcknowledgement.accepted_ack()

    def now(self) -> float:
        return self.time

    def sleep(self, duration: float) -> None:
        self.time += duration


def _scene_registry() -> SceneRegistry:
    cube = SceneObjectRef("cube")
    grasp = SceneAffordanceRef("cube_grasp")
    return SceneRegistry(
        (
            SceneEntityRegistration(
                ref=cube,
                state_provider=_PoseProvider(torch.eye(4).repeat(2, 1, 1)),
                semantic_type="cube",
                default_affordances={GRASP_AFFORDANCE_CAPABILITY: grasp},
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
        )
    )


def _profile(
    *,
    runner_cfg: ExecutionRunnerCfg | None = None,
) -> RobotSkillProfile:
    return RobotSkillProfile(
        profile_id="runtime_test_robot",
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
        defaults={
            "pick_up": ResourceBinding({"primary": "manipulator"}),
            "place": ResourceBinding({"primary": "manipulator"}),
        },
        presets={"safe": SkillPolicyPreset("safe", runner_cfg=runner_cfg)},
        default_preset="safe",
    )


def _robot() -> Mock:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 2
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_qpos.return_value = torch.zeros(2, 2)
    robot.get_qvel.return_value = torch.zeros(2, 2)
    robot.get_joint_ids.side_effect = lambda name: {"arm": [0], "hand": [1]}[name]
    robot.get_solver.return_value = object()
    return robot


def _runtime(
    *,
    verifier: SemanticEffectVerifier | None = None,
    profile_runner_cfg: ExecutionRunnerCfg | None = None,
    runtime_runner_cfg: ExecutionRunnerCfg | None = None,
) -> tuple[SemanticSkillRuntime, _ExecutionPorts]:
    registry = _scene_registry()
    profile = _profile(runner_cfg=profile_runner_cfg)
    robot = _robot()
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    engine = AtomicActionEngine(generator, skill_profile=profile)
    engine.register(_InstantPick(), replace=True)
    engine.register(_InstantPlace(), replace=True)
    manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=profile,
        call_catalog=builtin_semantic_call_catalog(),
    )
    ports = _ExecutionPorts(registry, robot)
    runtime = SemanticSkillRuntime.bind(
        manifest=manifest,
        scene_registry=registry,
        engine=engine,
        observation_provider=ports,
        command_sink=ports,
        clock=ports,
        effect_verifier=verifier,
        runner_cfg=runtime_runner_cfg,
    )
    return runtime, ports


def _successful_verifier(call, request, context) -> torch.Tensor:
    del call, request
    return torch.ones(context.batch_size, dtype=torch.bool)


def _pick_place_calls() -> tuple[Pick, Place]:
    cube = SceneObjectRef("cube")
    return (
        Pick(object=cube),
        Place(
            object=cube,
            at=SemanticPose((0.4, 0.0, 0.2), (1.0, 0.0, 0.0, 0.0)),
        ),
    )


def test_runtime_runs_jit_grounded_workflow_to_verified_completion() -> None:
    observed_calls: list[str] = []

    def verifier(call, request, context) -> torch.Tensor:
        del request
        observed_calls.append(call.semantic_id)
        return torch.ones(context.batch_size, dtype=torch.bool)

    runtime, ports = _runtime(verifier=verifier)

    result = runtime.run(_pick_place_calls(), task_id="pick_place")

    assert result.status is SemanticTaskStatus.SUCCEEDED
    assert result.eligible_mask.tolist() == [True, True]
    assert result.task_state.held_objects == {}
    assert observed_calls == ["pick", "place"]
    assert [record.skill_id for record in result.segments[0].calls] == [
        "pick_up",
        "place",
    ]
    assert ports.hold_calls == 2
    assert runtime.active_task is None


def test_task_preserves_verified_state_across_dynamic_segments() -> None:
    runtime, _ = _runtime(verifier=_successful_verifier)
    cube = SceneObjectRef("cube")
    task = runtime.open_task("dynamic_delivery")

    pick_result = task.run_segment((Pick(object=cube),), segment_id="acquire")
    assert pick_result.status is SemanticExecutionStatus.COMPLETED
    assert task.task_state.held_object_mask("manipulator").tolist() == [True, True]

    place_result = task.run_segment(
        (
            Place(
                object=cube,
                at=SemanticPose((0.5, 0.0, 0.2), (1.0, 0.0, 0.0, 0.0)),
            ),
        ),
        segment_id="deliver",
    )
    assert place_result.status is SemanticExecutionStatus.COMPLETED
    assert task.task_state.held_objects == {}

    result = task.finish()
    assert result.status is SemanticTaskStatus.SUCCEEDED
    assert [segment.segment_id for segment in result.segments] == [
        "acquire",
        "deliver",
    ]


def test_manual_execution_blocks_until_effect_mask_is_submitted() -> None:
    runtime, _ = _runtime()
    execution = runtime.start(
        (Pick(object=SceneObjectRef("cube")),),
        task_id="manual_pick",
    )

    blocked = execution.run_until_blocked()
    assert blocked.status is SemanticExecutionStatus.WAITING_FOR_EFFECT
    assert blocked.pending_effect is not None
    assert execution.task_result is None

    completed = blocked
    for _ in range(10):
        if completed.status is SemanticExecutionStatus.COMPLETED:
            break
        effect_success = (
            torch.tensor([True, True]) if execution.pending_effect is not None else None
        )
        completed = execution.step(effect_success=effect_success)
        if (
            completed.runner_step is not None
            and completed.runner_step.wait_duration > 0
        ):
            runtime.clock.sleep(completed.runner_step.wait_duration)
    assert completed.status is SemanticExecutionStatus.COMPLETED
    assert execution.task_result is not None
    assert execution.task_result.status is SemanticTaskStatus.SUCCEEDED
    assert runtime.active_task is None


def test_manual_execution_rejects_effect_before_verification_boundary() -> None:
    runtime, _ = _runtime()
    execution = runtime.start(
        (Pick(object=SceneObjectRef("cube")),),
        task_id="premature_effect",
    )

    with pytest.raises(RuntimeError, match="pending effect verification"):
        execution.step(effect_success=torch.tensor([True, True]))

    execution.cancel()
    assert runtime.active_task is None


def test_execution_stages_same_call_revision_through_runner_boundary() -> None:
    runtime, _ = _runtime(verifier=_successful_verifier)
    cube = SceneObjectRef("cube")
    execution = runtime.start((Pick(object=cube),), task_id="revised_pick")

    execution.revise_current(Pick(object=cube))
    completed = execution.run_until_blocked(
        effect_verifier=_successful_verifier,
    )

    assert completed.status is SemanticExecutionStatus.COMPLETED
    assert execution.task_result is not None
    assert execution.task_result.segments[0].calls[0].invocation_revision == 1


def test_effect_failures_produce_partial_task_success_after_bounded_retries() -> None:
    runtime, _ = _runtime(
        verifier=lambda call, request, context: torch.tensor([True, False])
    )

    result = runtime.run(
        (Pick(object=SceneObjectRef("cube")),),
        task_id="partial_pick",
    )

    assert result.status is SemanticTaskStatus.PARTIAL_SUCCESS
    assert result.eligible_mask.tolist() == [True, False]
    assert result.task_state.held_object_mask("manipulator").tolist() == [True, False]


def test_runtime_rejects_concurrent_tasks() -> None:
    runtime, _ = _runtime()
    task = runtime.open_task("first")

    with pytest.raises(RuntimeError, match="already owns this runtime"):
        runtime.open_task("second")

    result = task.cancel()
    assert result.status is SemanticTaskStatus.CANCELLED
    assert runtime.active_task is None


def test_blocking_run_requires_effect_verifier_before_owning_runtime() -> None:
    runtime, _ = _runtime()

    with pytest.raises(ValueError, match="requires an effect_verifier"):
        runtime.run((Pick(object=SceneObjectRef("cube")),))

    assert runtime.active_task is None


def test_verifier_exception_fails_safely_and_releases_runtime() -> None:
    def failing_verifier(call, request, context) -> torch.Tensor:
        del call, request, context
        raise RuntimeError("camera unavailable")

    runtime, ports = _runtime(verifier=failing_verifier)

    result = runtime.run(
        (Pick(object=SceneObjectRef("cube")),),
        task_id="failed_verification",
    )

    assert result.status is SemanticTaskStatus.FAILED
    assert result.segments[0].status is SemanticExecutionStatus.FAILED
    assert "camera unavailable" in (result.message or "")
    assert ports.cancel_calls == 1
    assert runtime.active_task is None


def test_failed_dynamic_segment_closes_terminal_task_ownership() -> None:
    runtime, _ = _runtime(
        verifier=lambda call, request, context: torch.zeros(
            context.batch_size,
            dtype=torch.bool,
        )
    )
    task = runtime.open_task("terminal_dynamic_failure")

    segment = task.run_segment((Pick(object=SceneObjectRef("cube")),))

    assert segment.status is SemanticExecutionStatus.FAILED
    assert task.result is not None
    assert task.result.status is SemanticTaskStatus.FAILED
    assert runtime.active_task is None


def test_from_simulation_builds_ports_and_filters_agent_visible_calls() -> None:
    registry = _scene_registry()
    profile = _profile()
    robot = _robot()
    simulation = Mock()
    simulation.sim_config.physics_dt = 0.01
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub_planner"
    generator.collision_world_info = None

    runtime = SemanticSkillRuntime.from_simulation(
        simulation=simulation,
        robot=robot,
        motion_generator=generator,
        scene_registry=registry,
        robot_profile=profile,
        control_dt=0.04,
    )

    assert set(runtime.available_calls) == {"pick", "place"}
    assert runtime.observation_provider is runtime.command_sink
    assert runtime.clock is runtime.observation_provider
    assert runtime.observation_provider.control_dt == pytest.approx(0.04)


def test_runtime_uses_skill_preset_runner_cfg_without_global_override() -> None:
    runtime, ports = _runtime(
        verifier=_successful_verifier,
        profile_runner_cfg=ExecutionRunnerCfg(hold_on_completion=False),
    )

    result = runtime.run((Pick(object=SceneObjectRef("cube")),))

    assert result.status is SemanticTaskStatus.SUCCEEDED
    assert ports.hold_calls == 0


def test_runtime_runner_cfg_overrides_skill_preset() -> None:
    runtime, ports = _runtime(
        verifier=_successful_verifier,
        profile_runner_cfg=ExecutionRunnerCfg(hold_on_completion=False),
        runtime_runner_cfg=ExecutionRunnerCfg(hold_on_completion=True),
    )

    result = runtime.run((Pick(object=SceneObjectRef("cube")),))

    assert result.status is SemanticTaskStatus.SUCCEEDED
    assert ports.hold_calls == 1
