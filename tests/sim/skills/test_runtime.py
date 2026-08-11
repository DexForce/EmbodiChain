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

"""Tests for canonical semantic-skill execution and its public facade."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from types import MethodType, SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

import embodichain.lab.sim.skills.runtime as runtime_module
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ActionOptions,
    ActionPlan,
    Affordance,
    ArticulationJointState,
    AtomicAction,
    AtomicActionEngine,
    CommandAcknowledgement,
    EffectVerificationRequirement,
    EffectVerificationRequest,
    EndpointBinding,
    EndpointTrackingChannelBinding,
    EndpointTrackingFeedbackAddress,
    HeldObjectGuardRequest,
    HeldObjectState,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    SceneSnapshot,
    SkillBindingContract,
    StateDelta,
    TaskState,
    TimedCommandSequence,
    TrackingFeedbackSourceRef,
    TrackingProjectorRef,
)
from embodichain.lab.sim.atomic_actions.tracking import TrackingPolicy
from embodichain.lab.sim.skills.calls import HandOver, Place, RegisteredSemanticCall
from embodichain.lab.sim.skills.compiler import (
    GroundedHeldObjectGuard,
    HeldObjectGuardBaseline,
    SemanticSkillCompiler,
)
from embodichain.lab.sim.skills.effects import (
    ArticulationJointStateExpectation,
    BinaryEffectClause,
    BinaryEvidenceKind,
    ControlPartEvidenceAddress,
    EffectEvidenceBatch,
    EffectEvidenceSourceRef,
    EffectExpectationDecision,
    EffectMonitor,
    EffectMonitorDecision,
    HeldObjectRelation,
    HeldObjectStateExpectation,
    JOINT_STATE_EFFECT_CHANNEL,
    JointStateEffectClause,
    SemanticEffectKind,
    SemanticEffectSpec,
)
from embodichain.lab.sim.skills.runtime import (
    AtomicSkills,
    SkillEndpointBindingTrace,
    SkillRuntime,
    SkillStatus,
)
from embodichain.lab.sim.skills.parallel import ParallelTimingPolicy
from embodichain.lab.sim.skills.parallel_runtime import ParallelSkillRuntime
from embodichain.lab.sim.skills.profiles import ResourceClaim
from embodichain.lab.sim.skills.scene import SceneObjectRef, SceneRegistry

BATCH_SIZE = 2


class _Clock:
    """Deterministic execution clock."""

    def __init__(self) -> None:
        self.time = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.time

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.time += duration


class _ObservationProvider:
    """Return a new timestamped context on every external observation."""

    def __init__(self) -> None:
        self.calls = 0
        self.task_states: list[TaskState] = []

    def observe(self, task_state: TaskState) -> PlanningContext:
        self.calls += 1
        self.task_states.append(task_state)
        timestamp = float(self.calls)
        return PlanningContext(
            robot=RobotObservation(
                timestamp=timestamp,
                qpos=torch.zeros(BATCH_SIZE, 1),
                qvel=torch.zeros(BATCH_SIZE, 1),
            ),
            task=task_state,
            scene=SceneSnapshot(timestamp=timestamp, version=self.calls),
            env_ids=torch.arange(BATCH_SIZE, dtype=torch.long),
        )


class _CommandSink:
    """Accept every command while recording safe-stop operations."""

    def __init__(self) -> None:
        self.sent = 0
        self.held = 0
        self.cancelled = 0

    def send(self, command: object, *, timeout: float) -> CommandAcknowledgement:
        del command, timeout
        self.sent += 1
        return CommandAcknowledgement.accepted_ack()

    def hold(
        self,
        targets: tuple[object, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        del targets, context, timeout
        self.held += 1
        return CommandAcknowledgement.accepted_ack()

    def cancel(
        self,
        targets: tuple[object, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        del targets, timeout
        self.cancelled += 1
        return CommandAcknowledgement.accepted_ack()


class _Collector:
    """Fake acquisition boundary; the test monitor owns decisions."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, float, torch.Tensor]] = []

    def collect(
        self,
        spec: SemanticEffectSpec,
        *,
        timestamp: float,
        observation_revision: int,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, EffectEvidenceBatch]:
        assert env_ids is not None
        self.calls.append((observation_revision, timestamp, env_ids.clone()))
        del spec
        return {}


class _DecisionMonitor(EffectMonitor):
    """Return one deterministic row-local physical-effect decision."""

    def __init__(self, spec: SemanticEffectSpec, decision: EffectMonitorDecision):
        self._spec = spec
        self._decision = decision
        self.calls = 0
        self.requests: list[EffectVerificationRequest] = []

    @property
    def spec(self) -> SemanticEffectSpec:
        return self._spec.snapshot()

    def observe(
        self,
        request: EffectVerificationRequest,
        evidence: dict[str, EffectEvidenceBatch],
    ) -> EffectMonitorDecision:
        del evidence
        self.calls += 1
        self.requests.append(request.snapshot())
        return EffectMonitorDecision(
            self._decision.success_mask,
            self._decision.failure_mask,
            self._decision.expectation_decisions,
        )


@dataclass(frozen=True, slots=True, eq=False)
class _EffectGoal:
    """Test-only goal carrying plan success and symbolic target value."""

    goal_kind: ClassVar[str] = "runtime_test_effect"

    plan_success: torch.Tensor
    target_position: float


class _EffectAction(AtomicAction[_EffectGoal, ActionOptions]):
    """Zero-frame action with an explicit verified articulation effect."""

    skill_id: ClassVar[str] = "runtime_test_effect"
    GoalType: ClassVar[type] = _EffectGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract()

    def __init__(self) -> None:
        super().__init__()
        self.plan_count = 0

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[_EffectGoal, ActionOptions],
    ) -> tuple[str, ...]:
        del request
        return ("fixture",)

    def _plan(
        self,
        request: ResolvedActionRequest[_EffectGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        self.plan_count += 1
        position = torch.full(
            (context.batch_size, 1),
            goal.target_position,
            dtype=context.robot.qpos.dtype,
            device=context.robot.qpos.device,
        )
        return self.build_command_plan(
            request,
            context,
            success=goal.plan_success,
            commands=TimedCommandSequence((), context.env_ids),
            expected_effects=StateDelta(
                articulation_joint_updates={
                    ("fixture", "joint"): ArticulationJointState(position)
                }
            ),
            effect_verification=EffectVerificationRequirement("semantic_effect"),
            replannable=False,
            scene_dependency_monitor_until={"fixture": 0},
        )


@dataclass(frozen=True, slots=True)
class _Workflow:
    workflow_id: str
    calls: tuple[RegisteredSemanticCall, ...]


@dataclass(frozen=True, slots=True)
class _Grounded:
    analyzed: object
    invocation: ActionInvocation
    effect_spec: SemanticEffectSpec
    effect_monitor: EffectMonitor
    eligible_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class _Integration:
    engine: AtomicActionEngine
    scene_registry: SceneRegistry


class _Compiler(SemanticSkillCompiler):
    """Semantic compiler test double retaining the production call boundaries."""

    def __init__(
        self,
        engine: AtomicActionEngine,
        decisions: tuple[EffectMonitorDecision, ...],
        plan_success: tuple[torch.Tensor, ...],
    ) -> None:
        self._test_integration = _Integration(engine, SceneRegistry())
        self._decisions = decisions
        self._plan_success = plan_success
        self.analyze_count = 0
        self.ground_count = 0
        self.ground_timestamps: list[float] = []
        self.ground_task_masks: list[torch.Tensor | None] = []
        self.invocations: list[ActionInvocation] = []
        self.monitors: list[_DecisionMonitor] = []

    @property
    def integration(self) -> _Integration:
        return self._test_integration

    def analyze(
        self,
        calls: tuple[RegisteredSemanticCall, ...],
        *,
        workflow_id: str = "semantic_workflow",
        path: tuple[object, ...] = ("workflow",),
    ) -> _Workflow:
        del path
        self.analyze_count += 1
        return _Workflow(workflow_id, tuple(calls))

    def ground(
        self,
        workflow: _Workflow,
        call_index: int,
        context: PlanningContext,
        *,
        eligible_mask: torch.Tensor | None = None,
        revision: int = 0,
        path: tuple[object, ...] = ("workflow",),
    ) -> _Grounded:
        del path
        assert eligible_mask is not None
        self.ground_count += 1
        self.ground_timestamps.append(context.robot.timestamp)
        state = context.task.get_articulation_joint_state("fixture", "joint")
        self.ground_task_masks.append(None if state is None else state.env_mask.clone())
        call = workflow.calls[call_index]
        invocation = ActionInvocation(
            skill_id=_EffectAction.skill_id,
            goal=_EffectGoal(
                self._plan_success[call_index].clone(),
                float(call_index + 1),
            ),
            binding=self.integration.engine.bind_control_parts(
                _EffectAction.skill_id,
                {},
            ),
            motion_policy=MotionPolicy(
                planner="runtime_test",
                sample_count=7,
                control_dt=0.02,
                velocity_limit=0.4,
                acceleration_limit=0.8,
            ),
            tracking_policy=TrackingPolicy.timed(),
            recovery_policy=RecoveryPolicy(
                max_replans=0,
                max_action_retries=0,
                action_timeout=100.0,
            ),
            invocation_id=f"{workflow.workflow_id}:{call_index}",
            revision=revision,
        )
        target = torch.full((BATCH_SIZE, 1), float(call_index + 1))
        expectation = ArticulationJointStateExpectation(
            "joint_target",
            "fixture",
            "joint",
            target,
        )
        source = EffectEvidenceSourceRef(
            "test.provider",
            "1",
            ControlPartEvidenceAddress("virtual", JOINT_STATE_EFFECT_CHANNEL),
        )
        spec = SemanticEffectSpec(
            semantic_id=call.semantic_id,
            effect_kind=SemanticEffectKind.ARTICULATION,
            skill_id=invocation.skill_id,
            invocation_id=invocation.invocation_id,
            invocation_revision=invocation.revision,
            env_ids=context.env_ids,
            state_expectations=(expectation,),
            clauses=(
                JointStateEffectClause(
                    "joint_position",
                    expectation.expectation_id,
                    source,
                    target,
                ),
            ),
        )
        monitor = _DecisionMonitor(spec, self._decisions[call_index])
        self.invocations.append(invocation)
        self.monitors.append(monitor)
        analyzed = SimpleNamespace(
            bound=SimpleNamespace(
                robot_profile=SimpleNamespace(profile_id="runtime_test_profile"),
                binding=SimpleNamespace(action_binding=invocation.binding),
                linked=SimpleNamespace(
                    descriptor=SimpleNamespace(skill_id=invocation.skill_id)
                ),
                preset=SimpleNamespace(
                    preset_id="runtime_test_preset",
                    schema_version=1,
                    motion_policy=invocation.motion_policy,
                    recovery_policy=invocation.recovery_policy,
                ),
            )
        )
        return _Grounded(
            analyzed,
            invocation,
            spec,
            monitor,
            eligible_mask.clone(),
        )


@dataclass(slots=True)
class _System:
    runtime: SkillRuntime
    compiler: _Compiler
    engine: AtomicActionEngine
    action: _EffectAction
    observation: _ObservationProvider
    sink: _CommandSink
    collector: _Collector
    clock: _Clock


def _mask(*values: bool) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def _call(name: str) -> RegisteredSemanticCall:
    return RegisteredSemanticCall(call_id=f"test.{name}")


def _system(
    decisions: tuple[EffectMonitorDecision, ...],
    *,
    plan_success: tuple[torch.Tensor, ...] | None = None,
) -> _System:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 1
    robot.control_parts = {}
    robot.get_qpos.return_value = torch.zeros(BATCH_SIZE, 1)
    robot.get_qvel.return_value = torch.zeros(BATCH_SIZE, 1)
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "runtime_test"
    engine = AtomicActionEngine(generator, load_builtins=False)
    action = _EffectAction()
    engine.register(action)
    selected_plan_success = plan_success or tuple(_mask(True, True) for _ in decisions)
    compiler = _Compiler(engine, decisions, selected_plan_success)
    observation = _ObservationProvider()
    sink = _CommandSink()
    collector = _Collector()
    clock = _Clock()
    runtime = SkillRuntime.from_components(
        compiler,
        observation,
        sink,
        collector,
        task_state=TaskState.empty(BATCH_SIZE, "cpu"),
        clock=clock,
    )
    return _System(
        runtime,
        compiler,
        engine,
        action,
        observation,
        sink,
        collector,
        clock,
    )


def test_runtime_analyzes_once_and_uses_one_fresh_session_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(
        (
            EffectMonitorDecision(_mask(True, True), _mask(False, False)),
            EffectMonitorDecision(_mask(True, True), _mask(False, False)),
        )
    )
    session_calls = 0
    runner_calls = 0
    original_start = system.engine.start
    original_runner = runtime_module.ExecutionRunner

    def counted_start(self: AtomicActionEngine, *args: object, **kwargs: object):
        nonlocal session_calls
        del self
        session_calls += 1
        return original_start(*args, **kwargs)

    system.engine.start = MethodType(counted_start, system.engine)

    def counted_runner(*args: object, **kwargs: object):
        nonlocal runner_calls
        runner_calls += 1
        return original_runner(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "ExecutionRunner", counted_runner)
    result = system.runtime.run((_call("first"), _call("second")))

    assert result.status is SkillStatus.COMPLETED
    assert system.compiler.analyze_count == 1
    assert system.compiler.ground_count == 2
    assert session_calls == 2
    assert runner_calls == 2
    assert system.action.plan_count == 2
    assert len(result.calls) == 2
    assert len(system.collector.calls) == 2
    assert system.compiler.ground_timestamps[1] > system.compiler.ground_timestamps[0]
    assert system.observation.calls == 4


def test_runtime_analyzes_downstream_calls_but_executes_only_requested_prefix() -> None:
    system = _system(
        (
            EffectMonitorDecision(_mask(True, True), _mask(False, False)),
            EffectMonitorDecision(_mask(True, True), _mask(False, False)),
        )
    )
    calls = (_call("current_segment"), _call("downstream_segment"))

    result = system.runtime.run(calls, execution_prefix_length=1)

    assert result.status is SkillStatus.COMPLETED
    assert system.compiler.analyze_count == 1
    assert system.compiler.ground_count == 1
    assert len(system.compiler.invocations) == 1
    assert len(result.calls) == 1
    assert result.calls[0].semantic_id == "test.current_segment"


@pytest.mark.parametrize("prefix_length", (0, 3, True, 1.5))
def test_runtime_rejects_invalid_execution_prefix_before_analysis(
    prefix_length: object,
) -> None:
    system = _system(
        (
            EffectMonitorDecision(_mask(True, True), _mask(False, False)),
            EffectMonitorDecision(_mask(True, True), _mask(False, False)),
        )
    )

    with pytest.raises((TypeError, ValueError), match="execution_prefix_length"):
        system.runtime.start(
            (_call("first"), _call("second")),
            execution_prefix_length=prefix_length,  # type: ignore[arg-type]
        )

    assert system.compiler.analyze_count == 0
    assert system.observation.calls == 0


def test_runtime_keeps_partial_rows_at_the_shared_call_barrier() -> None:
    system = _system(
        (
            EffectMonitorDecision(_mask(True, False), _mask(False, True)),
            EffectMonitorDecision(_mask(True, False), _mask(False, False)),
        )
    )
    result = system.runtime.run(_call("first"), _call("second"))

    assert result.status is SkillStatus.COMPLETED
    assert torch.equal(result.success_mask, _mask(True, False))
    assert torch.equal(result.failure_mask, _mask(False, True))
    assert torch.equal(result.calls[0].completed_mask, _mask(True, False))
    assert torch.equal(result.calls[0].failed_mask, _mask(False, True))
    assert torch.equal(result.calls[1].entered_mask, _mask(True, False))
    assert torch.equal(system.compiler.ground_task_masks[1], _mask(True, False))
    joint = result.task_state.get_articulation_joint_state("fixture", "joint")
    assert joint is not None
    assert torch.equal(joint.env_mask, _mask(True, False))
    assert torch.allclose(joint.position[0], torch.tensor([2.0]))
    assert len(result.failures) == 1


def test_nonblocking_step_routes_effect_feedback_through_collector() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    result = system.runtime.start(_call("stepwise"))

    assert result.status is SkillStatus.RUNNING
    while not result.terminal:
        if result.wait_duration:
            system.clock.sleep(result.wait_duration)
        result = system.runtime.step()

    assert result.status is SkillStatus.COMPLETED
    assert len(result.effects) == 1
    assert len(result.calls[0].effects) == 1
    assert system.collector.calls[0][0] == 0
    assert torch.equal(system.collector.calls[0][2], torch.tensor([0, 1]))
    assert system.compiler.monitors[0].requests[0].verification_id == 0


def test_runtime_preserves_per_expectation_effect_outcomes_in_trace() -> None:
    expectation = EffectExpectationDecision(
        expectation_id="joint_target",
        satisfied_mask=_mask(True, True),
        contradicted_mask=_mask(False, False),
        inverse_satisfied_mask=_mask(False, False),
    )
    system = _system(
        (
            EffectMonitorDecision(
                _mask(True, True),
                _mask(False, False),
                (expectation,),
            ),
        )
    )

    result = system.runtime.run(_call("expectation_trace"))

    assert result.status is SkillStatus.COMPLETED
    assert len(result.effects) == 1
    recorded = result.effects[0].expectation_decisions
    assert len(recorded) == 1
    assert recorded[0].expectation_id == "joint_target"
    assert result.to_metadata()["effects"][0]["decision"]["expectations"] == [
        {
            "expectation_id": "joint_target",
            "satisfied_mask": [True, True],
            "contradicted_mask": [False, False],
            "inverse_satisfied_mask": [False, False],
        }
    ]


@pytest.mark.parametrize(
    ("call", "expected_invalidation", "expected_retry"),
    (
        (
            Place(
                object=SceneObjectRef("cube"),
                inside=SceneObjectRef("bin"),
            ),
            _mask(False, True),
            _mask(True, False),
        ),
        (
            HandOver(object=SceneObjectRef("cube")),
            _mask(False, True),
            _mask(False, False),
        ),
    ),
)
def test_terminal_failure_policy_only_retains_strongly_proven_source_attachment(
    call: Place | HandOver,
    expected_invalidation: torch.Tensor,
    expected_retry: torch.Tensor,
) -> None:
    failure = _mask(True, True)
    source = EffectExpectationDecision(
        expectation_id="source",
        satisfied_mask=_mask(False, False),
        contradicted_mask=failure,
        inverse_satisfied_mask=_mask(True, False),
    )
    destination = EffectExpectationDecision(
        expectation_id="destination",
        satisfied_mask=_mask(False, False),
        contradicted_mask=failure,
        inverse_satisfied_mask=_mask(False, False),
    )
    grounded = SimpleNamespace(analyzed=SimpleNamespace(call=call))

    invalidation, retry = SkillRuntime._terminal_failure_policy(
        grounded,
        failure,
        (source, destination),
    )

    assert torch.equal(invalidation, expected_invalidation)
    assert torch.equal(retry, expected_retry)


def test_in_flight_guard_collects_live_evidence_and_builds_loss_reconciliation() -> (
    None
):
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    semantics = ObjectSemantics(
        affordance=Affordance(),
        geometry={},
        label="object",
        entity_id="cube",
    )
    poses = torch.eye(4).repeat(BATCH_SIZE, 1, 1)
    held = HeldObjectState(
        semantics=semantics,
        object_to_eef=poses,
        grasp_xpos=poses,
        env_mask=_mask(True, True),
    )
    task_state = TaskState(
        batch_size=BATCH_SIZE,
        device="cpu",
        held_objects={"arm": held},
    )
    expectation = HeldObjectStateExpectation(
        expectation_id="source",
        relation=HeldObjectRelation.ATTACHED,
        object_id="cube",
        slot_id="primary",
        resource_id="arm",
        task_state_key="arm",
    )
    spec = SemanticEffectSpec(
        semantic_id="carry",
        effect_kind=SemanticEffectKind.ATTACH,
        skill_id="carry",
        invocation_id="workflow:0",
        invocation_revision=0,
        env_ids=torch.arange(BATCH_SIZE, dtype=torch.long),
        state_expectations=(expectation,),
        clauses=(
            BinaryEffectClause(
                clause_id="source.constraint",
                expectation_id="source",
                source=EffectEvidenceSourceRef(
                    "test.provider",
                    "1",
                    ControlPartEvidenceAddress("hand", "constraint"),
                ),
                evidence_kind=BinaryEvidenceKind.CONSTRAINT,
                expected=True,
            ),
        ),
    )
    monitor = _DecisionMonitor(
        spec,
        EffectMonitorDecision(_mask(False, True), _mask(True, False)),
    )
    guard = GroundedHeldObjectGuard(
        guard_id="source_attached",
        active_segments=("carry",),
        baseline=HeldObjectGuardBaseline.VERIFIED_TASK_STATE,
        effect_spec=spec,
        effect_monitor=monitor,
        invalidation_task_state_keys=("arm",),
        retry_action=False,
    )
    system.runtime._grounded = SimpleNamespace(
        analyzed=SimpleNamespace(effect_monitor_ref=None),
        effect_guards=(guard,),
    )
    system.runtime._runner = SimpleNamespace(
        session=SimpleNamespace(task_state=task_state)
    )
    system.runtime._current_call_index = 0
    context = system.observation.observe(task_state)
    request = HeldObjectGuardRequest(
        verification_id=0,
        skill_id="carry",
        invocation_id="workflow:0",
        invocation_revision=0,
        invocation_index=0,
        attempt_generation=0,
        next_waypoint_index=1,
        segment_name="carry",
        env_mask=_mask(True, True),
        allowed_held_object_relations=(("arm", "cube"),),
        allowed_coordinated_held_object_relations=(),
        deadline=10.0,
    )

    result = system.runtime._held_object_guard_verifier(context, request)

    assert result is not None
    assert torch.equal(result.failure_mask, _mask(True, False))
    assert torch.equal(result.retry_mask, _mask(False, False))
    assert result.state_invalidation.held_object_updates == {"arm": None}
    assert len(system.runtime._effect_traces) == 1
    trace = system.runtime._effect_traces[0]
    assert trace.boundary_kind == "in_flight_guard"
    assert trace.guard_id == "source_attached"
    assert trace.segment_name == "carry"
    assert torch.equal(system.collector.calls[0][2], torch.tensor([0, 1]))


def test_result_metadata_is_json_safe_and_contains_typed_runtime_trace() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))

    result = system.runtime.run(_call("metadata"))
    metadata = result.to_metadata()

    json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert metadata["schema_version"] == 1
    assert metadata["kind"] == "skill_result"
    call = metadata["calls"][0]
    assert call["semantic_id"] == "test.metadata"
    assert call["call"]["arguments"]["call_id"] == "test.metadata"
    assert call["active_plan_attempt_generation"] == 0
    attempt = call["plan_attempts"][0]
    assert attempt["trigger"] == "action_planned"
    assert attempt["planned_scene_version"] == 1
    assert attempt["planned_collision_world_revision"] == [0, 0]
    assert attempt["scene_dependencies"] == ["fixture"]
    assert attempt["scene_dependency_monitor_until"] == {"fixture": 0}
    typed_attempt = result.calls[0].plan_attempts[0]
    assert typed_attempt.scene_dependency_monitor_until == {"fixture": 0}
    assert typed_attempt.snapshot().scene_dependency_monitor_until == {"fixture": 0}
    resolved = call["resolved_core_policy"]
    assert resolved["profile_id"] == "runtime_test_profile"
    assert resolved["preset"] == {
        "preset_id": "runtime_test_preset",
        "schema_version": 1,
    }
    assert resolved["motion_policy"]["strategy"] == "ik_interp"
    assert resolved["motion_policy"]["planner"] == "runtime_test"
    assert resolved["motion_policy"]["sample_count"] == 7
    assert resolved["tracking_policy"] == {
        "in_flight": None,
        "terminal": {"mode": "timed", "settle_duration": 0.0},
    }
    assert resolved["recovery_policy"]["max_replans"] == 0
    assert resolved["endpoints"] == []
    assert attempt["resolved_core_policy"] == resolved
    assert attempt["tracking_policy"] == resolved["tracking_policy"]
    assert attempt["tracking_contract"] is None
    assert "feedback_mode" not in attempt
    assert result.calls[0].resolved_core_policy.preset_id == "runtime_test_preset"
    effect = call["effects"][0]
    assert effect["boundary"] == {"kind": "terminal"}
    assert effect["effect_spec"]["semantic_id"] == "test.metadata"
    assert effect["monitor"]["monitor_id"].endswith("._DecisionMonitor")
    assert effect["evidence"] == {}

    metadata["masks"]["success"][0] = False
    assert system.runtime.result.to_metadata()["masks"]["success"] == [True, True]


@pytest.mark.parametrize("waypoint_index", (-1, 1, True, 1.5))
def test_plan_attempt_trace_rejects_invalid_scene_dependency_monitor_cutoff(
    waypoint_index: object,
) -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    result = system.runtime.run(_call("trace_cutoff"))
    attempt = result.calls[0].plan_attempts[0]

    with pytest.raises(ValueError, match="waypoint indices"):
        replace(
            attempt,
            scene_dependency_monitor_until={
                "fixture": waypoint_index  # type: ignore[dict-item]
            },
        )


def test_plan_attempt_trace_rejects_monitor_cutoff_for_non_dependency() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    result = system.runtime.run(_call("trace_dependency"))
    attempt = result.calls[0].plan_attempts[0]

    with pytest.raises(ValueError, match="keys must be scene dependencies"):
        replace(
            attempt,
            scene_dependency_monitor_until={"other": 0},
        )


def test_endpoint_binding_trace_records_only_stable_binding_choices() -> None:
    target = JointPositionTarget("left_arm_control", (3, 1))
    binding = EndpointBinding(
        slot_id="primary",
        endpoint_id="motion",
        resource_id="left_arm",
        adapter_id="control_part",
        target=target,
        task_state_key="left_arm_state",
        capabilities=frozenset({"cartesian_pose", "joint_position"}),
        claim_tokens=frozenset({"arm_workspace", "left_side"}),
        joint_ids=(3, 1),
        tracking_channels={
            "joint.position": EndpointTrackingChannelBinding(
                channel_id="joint.position",
                source=TrackingFeedbackSourceRef(
                    provider_id="planning_context.robot",
                    revision="1",
                    address=EndpointTrackingFeedbackAddress(
                        target=target,
                        channel_id="joint.position",
                    ),
                ),
                projector=TrackingProjectorRef(
                    projector_id="joint_position_payload",
                    revision="1",
                ),
            )
        },
    )

    trace = SkillEndpointBindingTrace.from_binding(binding)
    metadata = trace.to_metadata()

    json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert metadata["resource_id"] == "left_arm"
    assert metadata["adapter_id"] == "control_part"
    assert metadata["transport_id"] == "robot.joint_position"
    assert metadata["target_id"] == "left_arm_control"
    assert metadata["capabilities"] == ["cartesian_pose", "joint_position"]
    assert metadata["claim_tokens"] == ["arm_workspace", "left_side"]
    assert metadata["joint_ids"] == [3, 1]
    assert "target" not in metadata
    tracking = metadata["tracking_channels"][0]
    target_fingerprint = [
        {
            "__type__": (
                "embodichain.lab.sim.atomic_actions.bindings." "JointPositionTarget"
            )
        },
        "robot.joint_position",
        "left_arm_control",
        [3, 1],
    ]
    address_fingerprint = [target_fingerprint, "joint.position"]
    assert tracking["feedback_source"]["address_fingerprint"] == address_fingerprint
    assert tracking["route_fingerprint"] == [
        "joint.position",
        ["planning_context.robot", "1", address_fingerprint],
        "joint_position_payload",
        "1",
    ]
    tracking["feedback_source"]["address_fingerprint"][0][1] = "mutated"
    assert (
        trace.to_metadata()["tracking_channels"][0]["feedback_source"][
            "address_fingerprint"
        ][0][1]
        == "robot.joint_position"
    )


def test_preparation_failure_keeps_resolved_policy_without_plan_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    monkeypatch.setattr(
        system.engine,
        "start",
        Mock(side_effect=RuntimeError("planner unavailable")),
    )

    result = system.runtime.start(_call("planning_failure"))
    metadata = result.to_metadata()

    assert result.status is SkillStatus.FAILED
    assert len(result.calls) == 1
    assert result.calls[0].plan_attempts == ()
    assert result.calls[0].resolved_core_policy.preset_id == "runtime_test_preset"
    assert metadata["calls"][0]["active_plan_attempt_generation"] is None
    assert (
        metadata["calls"][0]["resolved_core_policy"]["motion_policy"]["planner"]
        == "runtime_test"
    )


def test_cancel_inherits_runner_cancel_then_hold_safe_stop() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    system.runtime.start(_call("cancel"))

    result = system.runtime.cancel("operator stop")

    assert result.status is SkillStatus.CANCELLED
    assert torch.equal(result.cancelled_mask, _mask(True, True))
    assert not result.eligible_mask.any()
    assert system.sink.cancelled == 1
    assert system.sink.held == 1
    assert result.calls[0].status.value == "cancelled"


def test_facade_varargs_and_programmatic_iterable_share_runtime_path() -> None:
    decisions = (
        EffectMonitorDecision(_mask(True, True), _mask(False, False)),
        EffectMonitorDecision(_mask(True, True), _mask(False, False)),
    )
    iterable_system = _system(decisions)
    facade_system = _system(decisions)
    calls = (_call("first"), _call("second"))

    iterable_result = iterable_system.runtime.run(calls)
    facade_result = AtomicSkills(facade_system.runtime).run(*calls)

    assert iterable_result.status is facade_result.status
    assert torch.equal(iterable_result.success_mask, facade_result.success_mask)
    assert [trace.skill_id for trace in iterable_result.calls] == [
        trace.skill_id for trace in facade_result.calls
    ]
    assert iterable_system.compiler.analyze_count == 1
    assert facade_system.compiler.analyze_count == 1
    assert [item.skill_id for item in iterable_system.compiler.invocations] == [
        item.skill_id for item in facade_system.compiler.invocations
    ]


def test_from_env_requires_an_explicit_runtime_provider() -> None:
    class AttributeBag:
        compiler = object()
        robot = object()
        scene = object()

    with pytest.raises(TypeError, match="no semantic-skill integration adapter"):
        AtomicSkills.from_env(AttributeBag())


def test_from_env_delegates_preset_to_installed_provider() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))

    class Provider:
        def __init__(self) -> None:
            self.presets: list[str] = []

        def create_skill_runtime(self, *, preset: str) -> SkillRuntime:
            self.presets.append(preset)
            return system.runtime

    provider = Provider()
    skills = AtomicSkills.from_env(provider, preset="precise")

    assert skills.runtime is system.runtime
    assert provider.presets == ["precise"]


def test_result_snapshots_do_not_expose_runtime_masks() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    result = system.runtime.run(_call("owned"))

    result.success_mask.zero_()
    result.calls[0].completed_mask.zero_()
    fresh = system.runtime.result

    assert torch.equal(fresh.success_mask, _mask(True, True))
    assert torch.equal(fresh.calls[0].completed_mask, _mask(True, True))


def test_fork_creates_an_independent_lane_on_the_shared_clock() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    lane_sink = _CommandSink()

    lane = system.runtime.fork(lane_sink)

    assert lane is not system.runtime
    assert lane.compiler is system.runtime.compiler
    assert lane.clock is system.runtime.clock
    assert lane.status is SkillStatus.IDLE
    assert lane_sink.sent == 0


def test_runner_failure_does_not_relabel_peer_cancelled_rows() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    system.runtime.start(_call("row_failure"))
    system.runtime.deactivate_rows(_mask(True, False), reason="peer branch failed")

    def fail_observation(task_state: TaskState) -> PlanningContext:
        del task_state
        raise RuntimeError("observation unavailable")

    system.observation.observe = fail_observation
    result = system.runtime.step()
    if result.wait_duration > 0.0:
        system.clock.sleep(result.wait_duration)
        result = system.runtime.step()

    assert result.status is SkillStatus.FAILED
    assert torch.equal(result.cancelled_mask, _mask(True, False))
    assert torch.equal(result.failure_mask, _mask(False, True))


def test_deactivate_all_rows_safe_stops_immediately_before_due_time() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    system.runtime.start(_call("deactivate_all"))

    result = system.runtime.deactivate_rows(
        _mask(True, True),
        reason="parallel peer failed",
    )

    assert result.status is SkillStatus.CANCELLED
    assert torch.equal(result.cancelled_mask, _mask(True, True))
    assert system.sink.cancelled == 1
    assert system.sink.held == 1


def test_parallel_factory_analyzes_claims_and_forks_owned_shared_clock_lanes() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))

    def analyze_claims(
        self: _Compiler,
        calls: tuple[RegisteredSemanticCall, ...],
        *,
        workflow_id: str,
        path: tuple[object, ...] = ("workflow",),
    ) -> object:
        del self, workflow_id, path
        analyzed = []
        for call_index, call in enumerate(calls):
            joint_id = 0 if call.call_id.endswith("left") else 1
            analyzed.append(
                SimpleNamespace(
                    index=call_index,
                    symbolic_writes=frozenset(),
                    opaque_symbolic_effect=False,
                    bound=SimpleNamespace(
                        binding=SimpleNamespace(
                            claim=ResourceClaim(
                                frozenset({f"resource_{joint_id}"}),
                                (joint_id,),
                            )
                        )
                    ),
                )
            )
        return SimpleNamespace(calls=tuple(analyzed))

    class AcceptSafety:
        def validate(self, *, branch_frames: object, merged_frame: object) -> None:
            del branch_frames, merged_frame

    system.compiler.analyze = MethodType(analyze_claims, system.compiler)
    parallel = ParallelSkillRuntime.from_template(
        system.runtime,
        {
            "left": (_call("left"),),
            "right": (_call("right"),),
        },
        system.sink,
        ParallelTimingPolicy(0.1),
        AcceptSafety(),
        timeout_steps=5,
    )

    assert parallel.clock is system.runtime.clock
    assert parallel.branch_claims["left"].joint_ids == (0,)
    assert parallel.branch_claims["right"].joint_ids == (1,)

    changed = StateDelta(
        articulation_joint_updates={
            ("template", "joint"): ArticulationJointState(torch.ones(2, 1))
        }
    ).apply(system.runtime.task_state, _mask(True, True))
    system.runtime.adopt_verified_task_state(changed)
    assert all(
        result.task_state.get_articulation_joint_state("template", "joint") is None
        for result in parallel.result.branch_results.values()
    )
