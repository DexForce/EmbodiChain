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

"""Completion-trace audit across semantic execution and the Gym bridge."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import torch

from embodichain.lab.gym.envs.expert_program.bridge import (
    AtomicDemoBridge,
    BufferedGymCommandSink,
    EnvironmentStepClock,
    RuntimeCommandFrameEncoder,
)
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    AtomicAction,
    AtomicActionEngine,
    EndpointCommand,
    EntityState,
    JointPositionPayload,
    JointPositionTarget,
    MotionPolicy,
    PlannerDiagnostics,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    RuntimeCommandFrame,
    SceneSnapshot,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    TaskState,
    TimedCommandSequence,
)
from embodichain.lab.sim.skills.calls import RegisteredSemanticCall
from embodichain.lab.sim.skills.compiler import SemanticSkillCompiler
from embodichain.lab.sim.skills.runtime import SkillRuntime, SkillStatus
from embodichain.lab.sim.skills.scene import SceneRegistry

STEP_DT = 0.02
BATCH_SIZE = 2
ROBOT_DOF = 5
ENV_IDS = torch.tensor([7, 3], dtype=torch.long)
INITIAL_SCENE_VERSION = 41
REPLANNED_SCENE_VERSION = 42
INITIAL_COLLISION_REVISIONS = (5, 7)
REPLANNED_COLLISION_REVISIONS = (6, 8)


@dataclass(frozen=True, slots=True)
class _TraceGoal:
    """Test goal for a deterministic two-phase runtime command sequence."""

    goal_kind: ClassVar[str] = "completion_trace"


class _TraceAction(AtomicAction[_TraceGoal, ActionOptions]):
    """Emit named segments and preserve distinct diagnostics on every replan."""

    skill_id: ClassVar[str] = "completion_trace"
    GoalType: ClassVar[type] = _TraceGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(SkillEndpointRequirement(endpoint_id="motion"),),
            ),
        )
    )

    def __init__(self) -> None:
        super().__init__()
        self.plan_count = 0

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[_TraceGoal, ActionOptions],
    ) -> tuple[str, ...]:
        del request
        return ("trace_target",)

    def _plan(
        self,
        request: ResolvedActionRequest[_TraceGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        self.require_goal(request)
        generation = self.plan_count
        self.plan_count += 1
        target = request.binding.endpoint(
            "primary",
            "motion",
        ).require_target(JointPositionTarget)
        frames = tuple(
            RuntimeCommandFrame(
                commands=(
                    EndpointCommand(
                        target=target,
                        payload=JointPositionPayload(
                            torch.full(
                                (context.batch_size, len(target.joint_ids)),
                                float(generation + phase_index + 1),
                                device=context.robot.qpos.device,
                                dtype=context.robot.qpos.dtype,
                            )
                        ),
                    ),
                ),
                active_mask=torch.ones(
                    context.batch_size,
                    dtype=torch.bool,
                    device=context.robot.qpos.device,
                ),
                env_ids=context.env_ids,
                hold_duration=torch.full(
                    (context.batch_size,),
                    STEP_DT,
                    device=context.robot.qpos.device,
                    dtype=context.robot.qpos.dtype,
                ),
            )
            for phase_index in range(2)
        )
        return self.build_command_plan(
            request,
            context,
            success=True,
            commands=TimedCommandSequence(frames, context.env_ids),
            replannable=True,
            diagnostics=PlannerDiagnostics(
                backend="completion_trace_planner",
                messages=(f"installed generation {generation}",),
                metadata={
                    "generation": generation,
                    "quality": {"accepted": True, "score": generation + 0.25},
                },
            ),
            segment_lengths={"approach": 1, "commit": 1},
            scene_dependency_monitor_until={"trace_target": 2},
        )


class _TraceObservationProvider:
    """Move the scene once and report accepted commands as observed state."""

    def __init__(self, clock: EnvironmentStepClock) -> None:
        self.clock = clock
        self.calls = 0

    def observe(self, task_state: TaskState) -> PlanningContext:
        self.calls += 1
        replanned_scene = self.calls >= 2
        pose = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
        if replanned_scene:
            pose[:, 0, 3] = 0.25
        qpos = torch.full(
            (BATCH_SIZE, ROBOT_DOF),
            float(min(max(self.calls - 1, 0), 3)),
        )
        timestamp = self.clock.now()
        return PlanningContext(
            robot=RobotObservation(
                timestamp=timestamp,
                qpos=qpos,
                qvel=torch.zeros_like(qpos),
            ),
            task=task_state,
            scene=SceneSnapshot(
                timestamp=timestamp,
                version=(
                    REPLANNED_SCENE_VERSION
                    if replanned_scene
                    else INITIAL_SCENE_VERSION
                ),
                entities={"trace_target": EntityState(pose)},
                collision_world_revision=(
                    REPLANNED_COLLISION_REVISIONS
                    if replanned_scene
                    else INITIAL_COLLISION_REVISIONS
                ),
            ),
            env_ids=ENV_IDS,
        )


class _StaticQposProvider:
    """Supply full robot state to the bridge's transport encoder."""

    def current_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        assert torch.equal(env_ids, ENV_IDS)
        return torch.zeros(BATCH_SIZE, ROBOT_DOF)


class _UnusedEvidenceCollector:
    """Satisfy the runtime port; this action declares no physical effect."""

    def collect(
        self,
        spec: object,
        *,
        timestamp: float,
        observation_revision: int,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, object]:
        del spec, timestamp, observation_revision, env_ids
        raise AssertionError("The completion trace must not request effect evidence.")


@dataclass(frozen=True, slots=True)
class _TraceWorkflow:
    """Minimal analyzed workflow retained by the production runtime."""

    workflow_id: str
    calls: tuple[RegisteredSemanticCall, ...]


@dataclass(frozen=True, slots=True)
class _TraceIntegration:
    """Production engine and registry exposed through the compiler boundary."""

    engine: AtomicActionEngine
    scene_registry: SceneRegistry


@dataclass(frozen=True, slots=True)
class _TraceGroundedCall:
    """One grounded invocation with no external effect-verification boundary."""

    analyzed: object
    invocation: ActionInvocation
    eligible_mask: torch.Tensor
    effect_spec: None = None
    effect_monitor: None = None


class _TraceCompiler(SemanticSkillCompiler):
    """Keep semantic boundaries real while making lowering deterministic."""

    def __init__(self, engine: AtomicActionEngine) -> None:
        self._trace_integration = _TraceIntegration(engine, SceneRegistry())

    @property
    def integration(self) -> _TraceIntegration:
        return self._trace_integration

    def analyze(
        self,
        calls: tuple[RegisteredSemanticCall, ...],
        *,
        workflow_id: str = "semantic_workflow",
        path: tuple[object, ...] = ("workflow",),
    ) -> _TraceWorkflow:
        del path
        return _TraceWorkflow(workflow_id, tuple(calls))

    def ground(
        self,
        workflow: _TraceWorkflow,
        call_index: int,
        context: PlanningContext,
        *,
        eligible_mask: torch.Tensor | None = None,
        revision: int = 0,
        path: tuple[object, ...] = ("workflow",),
    ) -> _TraceGroundedCall:
        del context, path
        assert eligible_mask is not None
        binding = self.integration.engine.bind_control_parts(
            _TraceAction.skill_id,
            {"primary": {"motion": "arm"}},
        )
        invocation = ActionInvocation(
            skill_id=_TraceAction.skill_id,
            goal=_TraceGoal(),
            binding=binding,
            motion_policy=MotionPolicy(
                planner="completion_trace_planner",
                sample_count=9,
                control_dt=STEP_DT,
            ),
            recovery_policy=RecoveryPolicy(
                max_replans=2,
                max_action_retries=1,
                action_timeout=1.0,
            ),
            invocation_id=f"{workflow.workflow_id}:{call_index}",
            revision=revision,
        )
        analyzed = SimpleNamespace(
            bound=SimpleNamespace(
                robot_profile=SimpleNamespace(profile_id="completion_trace_robot"),
                preset=SimpleNamespace(
                    preset_id="completion_trace_preset",
                    schema_version=1,
                    motion_policy=invocation.motion_policy,
                    recovery_policy=invocation.recovery_policy,
                ),
            )
        )
        return _TraceGroundedCall(
            analyzed=analyzed,
            invocation=invocation,
            eligible_mask=eligible_mask.clone(),
        )


@dataclass(frozen=True, slots=True)
class _CompiledCall:
    """Program-owned semantic call and stable call index."""

    call_index: int
    call: RegisteredSemanticCall


@dataclass(frozen=True, slots=True)
class _CompiledSegment:
    """One logical program segment consumed by the production bridge."""

    calls: tuple[_CompiledCall, ...]
    segment_index: int = 0
    segment_id: str = "completion-segment"
    name: str = "completion-audit"
    source_path: tuple[object, ...] = ("program", "steps", 0)
    post_policies: tuple[object, ...] = ()
    validators: tuple[object, ...] = ()
    parallel_block: None = None
    implicit: bool = False


@dataclass(frozen=True, slots=True)
class _ProgramAnalysis:
    """Sequential look-ahead window selected for one bridge segment."""

    calls: tuple[RegisteredSemanticCall, ...]
    execution_prefix_length: int


class _CompiledProgram:
    """Single-segment compiled-program port for the completion audit."""

    schema_version = 2
    program_id = "completion-audit-program"

    def __init__(self, segment: _CompiledSegment) -> None:
        self.segment = segment

    def iter_segments(self):
        yield self.segment

    def sequential_execution_analysis(self, segment_index: int) -> _ProgramAnalysis:
        assert segment_index == self.segment.segment_index
        return _ProgramAnalysis(
            tuple(compiled.call for compiled in self.segment.calls),
            len(self.segment.calls),
        )


def _runtime_and_bridge() -> tuple[AtomicDemoBridge, _TraceAction]:
    """Assemble real execution/runtime/bridge layers around deterministic ports."""
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = ROBOT_DOF
    robot.control_parts = {"arm": object()}
    robot.get_joint_ids.return_value = (1, 3)
    robot.get_qpos.return_value = torch.zeros(BATCH_SIZE, ROBOT_DOF)
    robot.get_qvel.return_value = torch.zeros(BATCH_SIZE, ROBOT_DOF)
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "completion_trace_planner"
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = _TraceAction()
    engine.register(action)

    clock = EnvironmentStepClock(STEP_DT)
    sink = BufferedGymCommandSink(
        RuntimeCommandFrameEncoder(_StaticQposProvider()),
        clock,
    )
    runtime = SkillRuntime.from_components(
        _TraceCompiler(engine),
        _TraceObservationProvider(clock),
        sink,
        _UnusedEvidenceCollector(),
        task_state=TaskState.empty(BATCH_SIZE, "cpu"),
        clock=clock,
    )
    call = RegisteredSemanticCall(call_id="audit.completion_metadata")
    segment = _CompiledSegment((_CompiledCall(0, call),))
    bridge = AtomicDemoBridge(_CompiledProgram(segment), runtime, sink, clock)
    return bridge, action


def test_completion_trace_preserves_every_plan_generation_as_json_metadata() -> None:
    """A real scene replan remains complete after SkillResult and bridge snapshots."""
    bridge, action = _runtime_and_bridge()
    demo_segment = next(bridge.iter_segments())

    emitted_actions = tuple(demo_segment.actions)
    accepted = demo_segment.validator()
    metadata = demo_segment.metadata

    serialized = json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert json.loads(serialized) == metadata
    assert emitted_actions
    assert accepted.tolist() == [True, True]
    assert action.plan_count == 2

    assert metadata["expert_program_schema_version"] == 2
    assert metadata["expert_program_id"] == "completion-audit-program"
    assert metadata["program_segment_id"] == "completion-segment"
    assert metadata["program_segment_index"] == 0
    assert metadata["program_segment_source_path"] == ["program", "steps", 0]
    assert metadata["program_segment_implicit"] is False
    assert metadata["semantic_call_indices"] == [0]
    assert metadata["post_policy_count"] == 0
    assert metadata["validator_count"] == 0
    assert metadata["parallel"] is False
    assert metadata["validation"]["accepted_mask"] == [True, True]

    runtime_trace = metadata["runtime"]
    assert runtime_trace["kind"] == "skill_result"
    assert runtime_trace["status"] == SkillStatus.COMPLETED.value
    call_trace = runtime_trace["calls"][0]
    assert call_trace["active_plan_attempt_generation"] == 1
    attempts = call_trace["plan_attempts"]
    assert [attempt["attempt_generation"] for attempt in attempts] == [0, 1]
    assert [attempt["trigger"] for attempt in attempts] == [
        "action_planned",
        "replanned",
    ]
    assert [attempt["planned_scene_version"] for attempt in attempts] == [
        INITIAL_SCENE_VERSION,
        REPLANNED_SCENE_VERSION,
    ]
    assert [attempt["planned_collision_world_revision"] for attempt in attempts] == [
        list(INITIAL_COLLISION_REVISIONS),
        list(REPLANNED_COLLISION_REVISIONS),
    ]
    assert all(
        attempt["scene_dependency_monitor_until"] == {"trace_target": 2}
        for attempt in attempts
    )
    assert all(
        attempt["trajectory_segments"]
        == [
            {"name": "approach", "start": 0, "stop": 1, "waypoint_count": 1},
            {"name": "commit", "start": 1, "stop": 2, "waypoint_count": 1},
        ]
        for attempt in attempts
    )
    assert [attempt["recovery_counters"] for attempt in attempts] == [
        {"action_retries": [0, 0], "replans": [0, 0]},
        {"action_retries": [0, 0], "replans": [1, 1]},
    ]
    assert [attempt["planner_diagnostics"] for attempt in attempts] == [
        {
            "backend": "completion_trace_planner",
            "messages": ["installed generation 0"],
            "metadata": {
                "generation": 0,
                "quality": {"accepted": True, "score": 0.25},
            },
        },
        {
            "backend": "completion_trace_planner",
            "messages": ["installed generation 1"],
            "metadata": {
                "generation": 1,
                "quality": {"accepted": True, "score": 1.25},
            },
        },
    ]
    event_kinds = [event["kind"] for event in call_trace["events"]]
    assert runtime_trace["events"] == call_trace["events"]
    assert "dynamic_goal_changed" in event_kinds
    assert "replanned" in event_kinds
    assert event_kinds[-3:] == [
        "trajectory_completed",
        "action_completed",
        "session_completed",
    ]
