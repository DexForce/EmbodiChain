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

"""Tests for canonical Semantic Call execution and its public facade."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from types import MethodType, SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest
import torch

import embodichain.lab.task_program.runtime.executor as runtime_module
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
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
    EntityState,
    ExecutionEventKind,
    ExecutionRunnerCfg,
    FORWARD_KINEMATICS_CAPABILITY,
    HeldObjectGuardRequest,
    HeldObjectState,
    JointPositionTarget,
    MotionPolicy,
    ObjectSemantics,
    PhaseEffectGateRequest,
    PlanningContext,
    RecoveryPolicy,
    ResolvedActionRequest,
    RobotObservation,
    SceneSnapshot,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
    StateDelta,
    TaskState,
    TimedCommandSequence,
    TrackingFeedbackSourceRef,
    TrackingProjectorRef,
)
from embodichain.lab.sim.atomic_actions.tracking import TrackingPolicy
from embodichain.lab.task_program.semantics.calls import (
    HandOver,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallSpec,
)
from embodichain.lab.task_program.compiler.lowering import (
    GroundedHeldObjectGuard,
    GroundedPhaseEffectGate,
    HeldObjectGuardBaseline,
    SemanticCallCompiler,
)
from embodichain.lab.task_program.semantics.effects import (
    ArticulationJointStateExpectation,
    BinaryEffectClause,
    BinaryEvidenceKind,
    CONSTRAINT_EFFECT_CHANNEL,
    ControlPartEvidenceAddress,
    EffectEvidenceBatch,
    EffectEvidenceSourceRef,
    EffectExpectationDecision,
    EffectMonitor,
    EffectMonitorDecision,
    EffectMonitorParam,
    HeldObjectRelation,
    HeldObjectStateExpectation,
    JOINT_STATE_EFFECT_CHANNEL,
    JointStateEffectClause,
    SemanticEffectKind,
    SemanticEffectSpec,
)
from embodichain.lab.task_program.semantics.integration import (
    SemanticDiagnostic,
    SemanticValidationError,
)
from embodichain.lab.task_program.runtime.executor import SemanticCallExecutor
from embodichain.lab.task_program.runtime.results import (
    SkillEndpointBindingTrace,
    SemanticExecutionStatus,
    SkillWorkflowRecoveryRole,
)
from embodichain.lab.task_program.runtime.parallel import ParallelTimingPolicy
from embodichain.lab.task_program.runtime.parallel_executor import (
    ParallelSemanticExecutor,
)
from embodichain.lab.task_program.semantics.profiles import (
    EffectAssurance,
    ResourceClaim,
    WorkflowRecoveryPolicy,
)
from embodichain.lab.task_program.semantics.scene import SceneObjectRef, SceneRegistry

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
        self.entities: dict[str, EntityState] = {}

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
            scene=SceneSnapshot(
                timestamp=timestamp,
                version=self.calls,
                entities=self.entities,
            ),
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
        if not decision.expectation_decisions:
            raise ValueError("Test monitors require explicit expectation decisions.")
        self._spec = spec
        self._decision = decision
        self.calls = 0
        self.requests: list[EffectVerificationRequest] = []

    @property
    def spec(self) -> SemanticEffectSpec:
        return self._spec.snapshot()

    @property
    def resolved_params(self) -> dict[str, EffectMonitorParam]:
        return {}

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
class _WorkflowEffectGoal:
    """Test-only held-object effect for workflow recovery."""

    goal_kind: ClassVar[str] = "runtime_test_workflow_effect"

    object_id: str
    attach: bool

    def __post_init__(self) -> None:
        if type(self.object_id) is not str or not self.object_id:
            raise ValueError("object_id must be a non-empty string.")
        if type(self.attach) is not bool:
            raise TypeError("attach must be exactly bool.")


class _WorkflowEffectAction(AtomicAction[_WorkflowEffectGoal, ActionOptions]):
    """Zero-frame action that commits only a verified held-object effect."""

    skill_id: ClassVar[str] = "runtime_test_workflow_primary"
    GoalType: ClassVar[type] = _WorkflowEffectGoal
    source_slot: ClassVar[str] = "primary"
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(SkillEndpointRequirement(endpoint_id="motion"),),
            ),
        )
    )

    def _plan(
        self,
        request: ResolvedActionRequest[_WorkflowEffectGoal, ActionOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        endpoint = request.binding.endpoint(self.source_slot, "motion")
        task_state_key = endpoint.task_state_key
        assert task_state_key is not None
        if goal.attach:
            poses = (
                torch.eye(
                    4,
                    dtype=context.robot.qpos.dtype,
                    device=context.robot.qpos.device,
                )
                .unsqueeze(0)
                .repeat(context.batch_size, 1, 1)
            )
            effect: HeldObjectState | None = HeldObjectState(
                semantics=ObjectSemantics(
                    affordance=Affordance(),
                    geometry={},
                    label=goal.object_id,
                    entity_id=goal.object_id,
                ),
                object_to_eef=poses,
                grasp_xpos=poses,
                env_mask=torch.ones(
                    context.batch_size,
                    dtype=torch.bool,
                    device=context.robot.qpos.device,
                ),
            )
        else:
            effect = None
        return self.build_command_plan(
            request,
            context,
            success=torch.ones(
                context.batch_size,
                dtype=torch.bool,
                device=context.robot.qpos.device,
            ),
            commands=TimedCommandSequence((), context.env_ids),
            expected_effects=StateDelta(
                held_object_updates={task_state_key: effect},
            ),
            effect_verification=EffectVerificationRequirement("semantic_effect"),
            replannable=False,
        )


class _WorkflowSourceEffectAction(_WorkflowEffectAction):
    """Held-object effect addressed through a hand-over source slot."""

    skill_id: ClassVar[str] = "runtime_test_workflow_source"
    source_slot: ClassVar[str] = "source"
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="source",
                endpoints=(SkillEndpointRequirement(endpoint_id="motion"),),
            ),
        )
    )


@dataclass(frozen=True, slots=True, eq=False)
class _WorkflowEffectDecision:
    """One queued physical-effect result for a grounded recovery call."""

    success_mask: torch.Tensor
    failure_mask: torch.Tensor
    inverse_satisfied_mask: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "success_mask",
            "failure_mask",
            "inverse_satisfied_mask",
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != torch.bool or value.shape != (BATCH_SIZE,):
                raise ValueError(f"{name} must be bool with shape ({BATCH_SIZE},).")
            object.__setattr__(self, name, value.clone())


@dataclass(frozen=True, slots=True)
class _Workflow:
    workflow_id: str
    calls: tuple[SemanticCallSpec, ...]


@dataclass(frozen=True, slots=True)
class _Grounded:
    analyzed: object
    invocation: ActionInvocation
    effect_spec: SemanticEffectSpec | None
    effect_monitor: EffectMonitor | None
    eligible_mask: torch.Tensor
    effect_guards: tuple[GroundedHeldObjectGuard, ...] = ()
    effect_gates: tuple[GroundedPhaseEffectGate, ...] = ()


@dataclass(frozen=True, slots=True)
class _Integration:
    engine: AtomicActionEngine
    scene_registry: SceneRegistry

    @property
    def robot_profile(self) -> object:
        """Expose the production capability surface used by the facade."""
        return SimpleNamespace(skills=self.engine.skills)


class _Compiler(SemanticCallCompiler):
    """Semantic compiler test double retaining the production call boundaries."""

    def __init__(
        self,
        engine: AtomicActionEngine,
        decisions: tuple[EffectMonitorDecision, ...],
        plan_success: tuple[torch.Tensor, ...],
        runner_cfg: ExecutionRunnerCfg,
        *,
        install_effect_monitor: bool,
    ) -> None:
        self._test_integration = _Integration(engine, SceneRegistry())
        self._decisions = tuple(
            EffectMonitorDecision(
                decision.success_mask,
                decision.failure_mask,
                decision.expectation_decisions
                or (
                    EffectExpectationDecision(
                        expectation_id="joint_target",
                        satisfied_mask=decision.success_mask,
                        contradicted_mask=decision.failure_mask,
                        inverse_satisfied_mask=torch.zeros_like(decision.failure_mask),
                    ),
                ),
            )
            for decision in decisions
        )
        self._plan_success = plan_success
        self._runner_cfg = runner_cfg
        self._install_effect_monitor = install_effect_monitor
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
        calls: tuple[SemanticCallSpec, ...],
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
                sample_count=7,
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
        self.invocations.append(invocation)
        monitor: _DecisionMonitor | None = None
        if self._install_effect_monitor:
            monitor = _DecisionMonitor(spec, self._decisions[call_index])
            self.monitors.append(monitor)
        analyzed = SimpleNamespace(
            call=call,
            effect_assurance=(
                EffectAssurance.VERIFIED
                if self._install_effect_monitor
                else EffectAssurance.PROJECTED
            ),
            effect_monitor_ref=None,
            bound=SimpleNamespace(
                robot_profile=SimpleNamespace(profile_id="runtime_test_profile"),
                binding=SimpleNamespace(action_binding=invocation.binding),
                linked=SimpleNamespace(
                    descriptor=SimpleNamespace(skill_id=invocation.skill_id)
                ),
                preset=SimpleNamespace(
                    preset_id="runtime_test_preset",
                    motion_policy=invocation.motion_policy,
                    recovery_policy=invocation.recovery_policy,
                    runner_cfg=self._runner_cfg,
                ),
            ),
        )
        return _Grounded(
            analyzed,
            invocation,
            spec if monitor is not None else None,
            monitor,
            eligible_mask.clone(),
        )


class _WorkflowRecoveryCompiler(SemanticCallCompiler):
    """Ground queued physical outcomes through real execution sessions."""

    def __init__(
        self,
        engine: AtomicActionEngine,
        decisions: tuple[_WorkflowEffectDecision, ...],
        *,
        max_recovery_attempts: int,
    ) -> None:
        self._test_integration = _Integration(engine, SceneRegistry())
        self._decisions = decisions
        self._workflow_policy = WorkflowRecoveryPolicy(max_recovery_attempts)
        self.analysis_windows: list[tuple[str, ...]] = []
        self.grounded_calls: list[SemanticCallSpec] = []
        self.grounded_masks: list[torch.Tensor] = []
        self.invocations: list[ActionInvocation] = []

    @property
    def integration(self) -> _Integration:
        return self._test_integration

    def analyze(
        self,
        calls: tuple[SemanticCallSpec, ...],
        *,
        workflow_id: str = "semantic_workflow",
        path: tuple[object, ...] = ("workflow",),
    ) -> _Workflow:
        del path
        self.analysis_windows.append(tuple(call.semantic_id for call in calls))
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
        if eligible_mask is None:
            raise ValueError("eligible_mask is required by this compiler.")
        decision_index = len(self.grounded_calls)
        if decision_index >= len(self._decisions):
            raise RuntimeError("No queued workflow-effect decision remains.")
        decision = self._decisions[decision_index]
        call = workflow.calls[call_index]
        if type(call) is Pick:
            action_type = _WorkflowEffectAction
            source_slot = "primary"
            expectation_id = "destination"
            relation = HeldObjectRelation.ATTACHED
            effect_kind = SemanticEffectKind.ATTACH
            attach = True
            expected_binary = True
        elif type(call) is Place:
            action_type = _WorkflowEffectAction
            source_slot = "primary"
            expectation_id = "source"
            relation = HeldObjectRelation.DETACHED
            effect_kind = SemanticEffectKind.RELEASE
            attach = False
            expected_binary = False
        elif type(call) is HandOver:
            action_type = _WorkflowSourceEffectAction
            source_slot = "source"
            expectation_id = "source"
            relation = HeldObjectRelation.DETACHED
            effect_kind = SemanticEffectKind.RELEASE
            attach = False
            expected_binary = False
        else:
            raise TypeError(
                "Workflow-recovery test compiler accepts Pick, Place, or HandOver."
            )
        object_id = call.object.entity_id
        binding = ActionBinding(
            owner_id=self.integration.engine.binding_owner_id,
            endpoints=(
                EndpointBinding(
                    slot_id=source_slot,
                    endpoint_id="motion",
                    resource_id="left_actor",
                    adapter_id="test",
                    target=JointPositionTarget("virtual", (0,)),
                    task_state_key="left_gripper",
                    capabilities=frozenset({FORWARD_KINEMATICS_CAPABILITY}),
                    joint_ids=(0,),
                ),
            ),
        )
        motion_policy = MotionPolicy(sample_count=7)
        tracking_policy = TrackingPolicy.timed()
        recovery_policy = RecoveryPolicy(
            max_replans=0,
            max_action_retries=0,
            action_timeout=100.0,
        )
        invocation = ActionInvocation(
            skill_id=action_type.skill_id,
            goal=_WorkflowEffectGoal(object_id=object_id, attach=attach),
            binding=binding,
            motion_policy=motion_policy,
            tracking_policy=tracking_policy,
            recovery_policy=recovery_policy,
            invocation_id=f"{workflow.workflow_id}:{decision_index}",
            revision=revision,
        )
        expectation = HeldObjectStateExpectation(
            expectation_id=expectation_id,
            relation=relation,
            object_id=object_id,
            slot_id=source_slot,
            resource_id="left_actor",
            task_state_key="left_gripper",
        )
        spec = SemanticEffectSpec(
            semantic_id=call.semantic_id,
            effect_kind=effect_kind,
            skill_id=invocation.skill_id,
            invocation_id=invocation.invocation_id,
            invocation_revision=invocation.revision,
            env_ids=context.env_ids,
            state_expectations=(expectation,),
            clauses=(
                BinaryEffectClause(
                    clause_id=f"{expectation_id}.constraint",
                    expectation_id=expectation_id,
                    source=EffectEvidenceSourceRef(
                        "test.provider",
                        "1",
                        ControlPartEvidenceAddress(
                            "virtual",
                            CONSTRAINT_EFFECT_CHANNEL,
                        ),
                    ),
                    evidence_kind=BinaryEvidenceKind.CONSTRAINT,
                    expected=expected_binary,
                ),
            ),
        )
        monitor = _DecisionMonitor(
            spec,
            EffectMonitorDecision(
                success_mask=decision.success_mask,
                failure_mask=decision.failure_mask,
                expectation_decisions=(
                    EffectExpectationDecision(
                        expectation_id=expectation_id,
                        satisfied_mask=decision.success_mask,
                        contradicted_mask=decision.failure_mask,
                        inverse_satisfied_mask=decision.inverse_satisfied_mask,
                    ),
                ),
            ),
        )
        analyzed = SimpleNamespace(
            call=call,
            effect_assurance=EffectAssurance.VERIFIED,
            effect_monitor_ref=None,
            bound=SimpleNamespace(
                robot_profile=SimpleNamespace(profile_id="runtime_test_profile"),
                binding=SimpleNamespace(action_binding=binding),
                linked=SimpleNamespace(
                    descriptor=SimpleNamespace(skill_id=invocation.skill_id)
                ),
                preset=SimpleNamespace(
                    preset_id="runtime_test_recovery_preset",
                    motion_policy=motion_policy,
                    tracking_policy=tracking_policy,
                    recovery_policy=recovery_policy,
                    workflow_recovery_policy=self._workflow_policy,
                    runner_cfg=ExecutionRunnerCfg(
                        minimum_cycle_time=0.0,
                        hold_on_completion=False,
                    ),
                ),
            ),
        )
        self.grounded_calls.append(call)
        self.grounded_masks.append(eligible_mask.clone())
        self.invocations.append(invocation)
        return _Grounded(
            analyzed=analyzed,
            invocation=invocation,
            effect_spec=spec,
            effect_monitor=monitor,
            eligible_mask=eligible_mask.clone(),
        )


@dataclass(slots=True)
class _System:
    runtime: SemanticCallExecutor
    compiler: _Compiler
    engine: AtomicActionEngine
    action: _EffectAction
    observation: _ObservationProvider
    sink: _CommandSink
    collector: _Collector
    clock: _Clock


@dataclass(slots=True)
class _WorkflowRecoverySystem:
    runtime: SemanticCallExecutor
    compiler: _WorkflowRecoveryCompiler
    engine: AtomicActionEngine
    observation: _ObservationProvider
    sink: _CommandSink
    collector: _Collector
    clock: _Clock


def _mask(*values: bool) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def _workflow_decision(
    success_mask: torch.Tensor,
    failure_mask: torch.Tensor,
    *,
    inverse_satisfied_mask: torch.Tensor | None = None,
) -> _WorkflowEffectDecision:
    return _WorkflowEffectDecision(
        success_mask=success_mask,
        failure_mask=failure_mask,
        inverse_satisfied_mask=(
            torch.zeros_like(failure_mask)
            if inverse_satisfied_mask is None
            else inverse_satisfied_mask
        ),
    )


def _call(name: str) -> RegisteredSemanticCall:
    return RegisteredSemanticCall(call_id=f"test.{name}")


def _system(
    decisions: tuple[EffectMonitorDecision, ...],
    *,
    plan_success: tuple[torch.Tensor, ...] | None = None,
    preset_runner_cfg: ExecutionRunnerCfg | None = None,
    runtime_runner_cfg: ExecutionRunnerCfg | None = None,
    install_effect_monitor: bool = True,
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
    selected_runner_cfg = (
        ExecutionRunnerCfg() if preset_runner_cfg is None else preset_runner_cfg
    )
    compiler = _Compiler(
        engine,
        decisions,
        selected_plan_success,
        selected_runner_cfg,
        install_effect_monitor=install_effect_monitor,
    )
    observation = _ObservationProvider()
    sink = _CommandSink()
    collector = _Collector()
    clock = _Clock()
    runtime = SemanticCallExecutor(
        compiler,
        observation,
        sink,
        collector,
        task_state=TaskState.empty(BATCH_SIZE, "cpu"),
        clock=clock,
        runner_cfg=runtime_runner_cfg,
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


def _workflow_recovery_system(
    decisions: tuple[_WorkflowEffectDecision, ...],
    *,
    max_recovery_attempts: int = 2,
    task_state: TaskState | None = None,
) -> _WorkflowRecoverySystem:
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
    engine.register(_WorkflowEffectAction())
    engine.register(_WorkflowSourceEffectAction())
    compiler = _WorkflowRecoveryCompiler(
        engine,
        decisions,
        max_recovery_attempts=max_recovery_attempts,
    )
    observation = _ObservationProvider()
    sink = _CommandSink()
    collector = _Collector()
    clock = _Clock()
    runtime = SemanticCallExecutor(
        compiler,
        observation,
        sink,
        collector,
        task_state=(
            TaskState.empty(BATCH_SIZE, "cpu") if task_state is None else task_state
        ),
        clock=clock,
    )
    return _WorkflowRecoverySystem(
        runtime=runtime,
        compiler=compiler,
        engine=engine,
        observation=observation,
        sink=sink,
        collector=collector,
        clock=clock,
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

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert system.compiler.analyze_count == 1
    assert system.compiler.ground_count == 2
    assert session_calls == 2
    assert runner_calls == 2
    assert system.action.plan_count == 2
    assert len(result.calls) == 2
    assert len(system.collector.calls) == 2
    assert system.compiler.ground_timestamps[1] > system.compiler.ground_timestamps[0]
    assert system.observation.calls == 4


def test_runtime_projects_planned_effect_when_grounded_call_has_no_monitor() -> None:
    system = _system(
        (EffectMonitorDecision(_mask(True, True), _mask(False, False)),),
        install_effect_monitor=False,
    )

    result = system.runtime.run(_call("trajectory_only"))

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert result.effects == ()
    assert system.collector.calls == []
    joint = result.task_state.get_articulation_joint_state("fixture", "joint")
    assert joint is not None
    assert torch.equal(joint.env_mask, _mask(True, True))
    assert torch.allclose(joint.position, torch.ones(BATCH_SIZE, 1))


def test_runtime_uses_selected_preset_runner_cfg_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset_runner_cfg = ExecutionRunnerCfg(command_timeout=3.0)
    system = _system(
        (EffectMonitorDecision(_mask(True, True), _mask(False, False)),),
        preset_runner_cfg=preset_runner_cfg,
    )
    captured_cfgs: list[ExecutionRunnerCfg] = []
    original_runner = runtime_module.ExecutionRunner

    def capture_runner(*args: object, **kwargs: object):
        cfg = kwargs["cfg"]
        assert isinstance(cfg, ExecutionRunnerCfg)
        captured_cfgs.append(cfg)
        return original_runner(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "ExecutionRunner", capture_runner)

    result = system.runtime.run(_call("preset_runner"))

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert len(captured_cfgs) == 1
    assert captured_cfgs[0].command_timeout == 3.0


def test_runtime_runner_cfg_override_wins_over_selected_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(
        (EffectMonitorDecision(_mask(True, True), _mask(False, False)),),
        preset_runner_cfg=ExecutionRunnerCfg(command_timeout=3.0),
        runtime_runner_cfg=ExecutionRunnerCfg(command_timeout=5.0),
    )
    captured_cfgs: list[ExecutionRunnerCfg] = []
    original_runner = runtime_module.ExecutionRunner

    def capture_runner(*args: object, **kwargs: object):
        cfg = kwargs["cfg"]
        assert isinstance(cfg, ExecutionRunnerCfg)
        captured_cfgs.append(cfg)
        return original_runner(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "ExecutionRunner", capture_runner)

    result = system.runtime.run(_call("runner_override"))

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert len(captured_cfgs) == 1
    assert captured_cfgs[0].command_timeout == 5.0


def test_runtime_analyzes_downstream_calls_but_executes_only_requested_prefix() -> None:
    system = _system(
        (
            EffectMonitorDecision(_mask(True, True), _mask(False, False)),
            EffectMonitorDecision(_mask(True, True), _mask(False, False)),
        )
    )
    calls = (_call("current_segment"), _call("downstream_segment"))

    result = system.runtime.run(calls, execution_prefix_length=1)

    assert result.status is SemanticExecutionStatus.COMPLETED
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

    assert result.status is SemanticExecutionStatus.COMPLETED
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


def test_runtime_reacquires_a_lost_source_with_a_real_pick_then_retries() -> None:
    system = _workflow_recovery_system(
        (
            _workflow_decision(_mask(True, True), _mask(False, False)),
            _workflow_decision(_mask(True, False), _mask(False, True)),
            _workflow_decision(_mask(False, True), _mask(False, False)),
            _workflow_decision(_mask(False, True), _mask(False, False)),
        )
    )
    cube = SceneObjectRef("cube")

    result = system.runtime.run(
        Pick(object=cube),
        Place(object=cube, inside=SceneObjectRef("bin")),
    )

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert torch.equal(result.success_mask, _mask(True, True))
    assert torch.equal(result.failure_mask, _mask(False, False))
    assert len(result.calls) == 2
    assert [call.semantic_id for call in system.compiler.grounded_calls] == [
        "pick",
        "place",
        "pick",
        "place",
    ]
    assert [mask.tolist() for mask in system.compiler.grounded_masks] == [
        [True, True],
        [True, True],
        [False, True],
        [False, True],
    ]
    assert system.compiler.analysis_windows == [
        ("pick", "place"),
        ("pick", "place"),
        ("place",),
    ]
    assert [trace.role for trace in result.workflow_recoveries] == [
        SkillWorkflowRecoveryRole.REACQUIRE,
        SkillWorkflowRecoveryRole.RETRY_REACQUIRED,
    ]
    assert all(
        torch.equal(trace.entered_mask, _mask(False, True))
        for trace in result.workflow_recoveries
    )
    assert result.workflow_recoveries[0].call is not None
    assert result.workflow_recoveries[0].call.semantic_id == "pick"
    assert result.workflow_recoveries[1].call is not None
    assert result.workflow_recoveries[1].call.semantic_id == "place"
    assert any(
        event.kind is ExecutionEventKind.RECOVERY_REQUIRED
        and torch.equal(event.env_mask, _mask(False, True))
        for event in result.events
    )
    metadata = result.to_metadata()
    json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert [entry["role"] for entry in metadata["workflow_recoveries"]] == [
        "reacquire",
        "retry_reacquired",
    ]
    assert metadata["workflow_recoveries"][0]["source_resource_id"] == "left_actor"
    assert metadata["workflow_recoveries"][0]["source_task_state_key"] == (
        "left_gripper"
    )
    assert result.task_state.get_held_object("left_gripper") is None


def test_runtime_reconciles_completed_pick_with_observed_attachment_pose() -> None:
    """The next semantic call receives the physical, not projected, grasp frame."""
    system = _workflow_recovery_system(
        (_workflow_decision(_mask(True, True), _mask(False, False)),)
    )
    object_poses = torch.eye(4).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    object_poses[0, :3, 3] = torch.tensor([0.5, 0.1, 0.2])
    object_poses[1, :3, 3] = torch.tensor([-0.2, 0.3, 0.4])
    observed_relations = torch.eye(4).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    observed_relations[0, :3, 3] = torch.tensor([0.1, 0.02, 0.3])
    observed_relations[1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    observed_relations[1, :3, 3] = torch.tensor([0.0, -0.1, 0.2])
    endpoint_poses = torch.bmm(object_poses, observed_relations)
    system.observation.entities["cube"] = EntityState(object_poses)
    system.engine.robot.compute_fk.return_value = endpoint_poses

    result = system.runtime.run(Pick(object=SceneObjectRef("cube")))

    assert result.status is SemanticExecutionStatus.COMPLETED
    held = result.task_state.get_held_object("left_gripper")
    assert held is not None
    torch.testing.assert_close(held.object_to_eef, observed_relations)
    torch.testing.assert_close(held.grasp_xpos, endpoint_poses)
    system.engine.robot.compute_fk.assert_called_once()
    call = system.engine.robot.compute_fk.call_args
    assert call.kwargs["name"] == "virtual"
    assert call.kwargs["env_ids"] == [0, 1]
    assert call.kwargs["to_matrix"] is True
    torch.testing.assert_close(call.kwargs["qpos"], torch.zeros(BATCH_SIZE, 1))


def test_runtime_retries_directly_when_verified_source_relation_remains() -> None:
    poses = torch.eye(4).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    initial_state = TaskState(
        batch_size=BATCH_SIZE,
        device="cpu",
        held_objects={
            "left_gripper": HeldObjectState(
                semantics=ObjectSemantics(
                    affordance=Affordance(),
                    geometry={},
                    label="cube",
                    entity_id="cube",
                ),
                object_to_eef=poses,
                grasp_xpos=poses,
                env_mask=_mask(True, True),
            )
        },
    )
    system = _workflow_recovery_system(
        (
            _workflow_decision(
                _mask(True, False),
                _mask(False, True),
                inverse_satisfied_mask=_mask(False, True),
            ),
            _workflow_decision(_mask(False, True), _mask(False, False)),
        ),
        task_state=initial_state,
    )

    result = system.runtime.run(
        HandOver(
            object=SceneObjectRef("cube"),
            resources={"source": "left_actor", "destination": "right_actor"},
        )
    )

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert torch.equal(result.success_mask, _mask(True, True))
    assert [call.semantic_id for call in system.compiler.grounded_calls] == [
        "hand_over",
        "hand_over",
    ]
    assert [mask.tolist() for mask in system.compiler.grounded_masks] == [
        [True, True],
        [False, True],
    ]
    assert len(result.workflow_recoveries) == 1
    recovery = result.workflow_recoveries[0]
    assert recovery.role is SkillWorkflowRecoveryRole.RETRY_RETAINED
    assert recovery.attempt_index == 1
    assert torch.equal(recovery.entered_mask, _mask(False, True))
    assert result.task_state.get_held_object("left_gripper") is None


def test_runtime_partitions_retained_and_lost_source_rows_in_one_barrier() -> None:
    poses = torch.eye(4).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    initial_state = TaskState(
        batch_size=BATCH_SIZE,
        device="cpu",
        held_objects={
            "left_gripper": HeldObjectState(
                semantics=ObjectSemantics(
                    affordance=Affordance(),
                    geometry={},
                    label="cube",
                    entity_id="cube",
                ),
                object_to_eef=poses,
                grasp_xpos=poses,
                env_mask=_mask(True, True),
            )
        },
    )
    system = _workflow_recovery_system(
        (
            _workflow_decision(
                _mask(False, False),
                _mask(True, True),
                inverse_satisfied_mask=_mask(True, False),
            ),
            _workflow_decision(_mask(True, False), _mask(False, False)),
            _workflow_decision(_mask(False, True), _mask(False, False)),
            _workflow_decision(_mask(False, True), _mask(False, False)),
        ),
        task_state=initial_state,
    )

    result = system.runtime.run(
        HandOver(
            object=SceneObjectRef("cube"),
            resources={"source": "left_actor", "destination": "right_actor"},
        )
    )

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert torch.equal(result.success_mask, _mask(True, True))
    assert [call.semantic_id for call in system.compiler.grounded_calls] == [
        "hand_over",
        "hand_over",
        "pick",
        "hand_over",
    ]
    assert [mask.tolist() for mask in system.compiler.grounded_masks] == [
        [True, True],
        [True, False],
        [False, True],
        [False, True],
    ]
    assert [trace.role for trace in result.workflow_recoveries] == [
        SkillWorkflowRecoveryRole.RETRY_RETAINED,
        SkillWorkflowRecoveryRole.REACQUIRE,
        SkillWorkflowRecoveryRole.RETRY_REACQUIRED,
    ]
    assert [trace.attempt_index for trace in result.workflow_recoveries] == [1, 1, 1]
    assert result.task_state.get_held_object("left_gripper") is None


def test_runtime_bounds_reacquisition_attempts_per_failed_row() -> None:
    system = _workflow_recovery_system(
        (
            _workflow_decision(_mask(True, True), _mask(False, False)),
            _workflow_decision(_mask(True, False), _mask(False, True)),
            _workflow_decision(_mask(False, False), _mask(False, True)),
            _workflow_decision(_mask(False, False), _mask(False, True)),
        ),
        max_recovery_attempts=2,
    )
    cube = SceneObjectRef("cube")

    result = system.runtime.run(
        Pick(object=cube),
        Place(object=cube, inside=SceneObjectRef("bin")),
    )

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert torch.equal(result.success_mask, _mask(True, False))
    assert torch.equal(result.failure_mask, _mask(False, True))
    assert [call.semantic_id for call in system.compiler.grounded_calls] == [
        "pick",
        "place",
        "pick",
        "pick",
    ]
    assert [trace.role for trace in result.workflow_recoveries] == [
        SkillWorkflowRecoveryRole.REACQUIRE,
        SkillWorkflowRecoveryRole.REACQUIRE,
    ]
    assert [trace.attempt_index for trace in result.workflow_recoveries] == [1, 2]
    assert len(result.failures) == 1
    assert "exhausted" in result.failures[0].message


def test_runtime_leaves_external_recovery_disabled_at_zero_budget() -> None:
    system = _workflow_recovery_system(
        (
            _workflow_decision(_mask(True, True), _mask(False, False)),
            _workflow_decision(_mask(True, False), _mask(False, True)),
        ),
        max_recovery_attempts=0,
    )
    cube = SceneObjectRef("cube")

    result = system.runtime.run(
        Pick(object=cube),
        Place(object=cube, inside=SceneObjectRef("bin")),
    )

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert torch.equal(result.success_mask, _mask(True, False))
    assert torch.equal(result.failure_mask, _mask(False, True))
    assert result.workflow_recoveries == ()
    assert [call.semantic_id for call in system.compiler.grounded_calls] == [
        "pick",
        "place",
    ]


def test_runtime_resolves_workflow_policy_only_after_typed_core_handoff() -> None:
    system = _workflow_recovery_system(
        (
            _workflow_decision(_mask(True, True), _mask(False, False)),
            _workflow_decision(_mask(True, True), _mask(False, False)),
        )
    )
    system.compiler._workflow_policy = object()  # type: ignore[assignment]
    cube = SceneObjectRef("cube")

    result = system.runtime.run(
        Pick(object=cube),
        Place(object=cube, inside=SceneObjectRef("bin")),
    )

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert result.workflow_recoveries == ()


def test_cancel_during_reacquisition_safe_stops_every_barrier_row() -> None:
    system = _workflow_recovery_system(
        (
            _workflow_decision(_mask(True, True), _mask(False, False)),
            _workflow_decision(_mask(True, False), _mask(False, True)),
            _workflow_decision(_mask(False, True), _mask(False, False)),
        )
    )
    cube = SceneObjectRef("cube")
    result = system.runtime.start(
        Pick(object=cube),
        Place(object=cube, inside=SceneObjectRef("bin")),
    )
    while len(system.compiler.grounded_calls) < 3:
        if result.wait_duration:
            system.clock.sleep(result.wait_duration)
        result = system.runtime.step()

    held_before_cancel = system.sink.held
    result = system.runtime.cancel("caller stopped recovery")

    assert result.status is SemanticExecutionStatus.CANCELLED
    assert torch.equal(result.cancelled_mask, _mask(True, True))
    assert len(result.workflow_recoveries) == 1
    assert result.workflow_recoveries[0].role is SkillWorkflowRecoveryRole.REACQUIRE
    assert result.workflow_recoveries[0].call is not None
    assert (
        result.workflow_recoveries[0].call.status
        is runtime_module.RunnerStatus.CANCELLED
    )
    assert system.sink.cancelled == 1
    assert system.sink.held == held_before_cancel + 1


def test_deactivating_a_waiting_row_does_not_cancel_active_reacquisition() -> None:
    system = _workflow_recovery_system(
        (
            _workflow_decision(_mask(True, True), _mask(False, False)),
            _workflow_decision(_mask(True, False), _mask(False, True)),
            _workflow_decision(_mask(False, True), _mask(False, False)),
            _workflow_decision(_mask(False, True), _mask(False, False)),
        )
    )
    cube = SceneObjectRef("cube")
    result = system.runtime.start(
        Pick(object=cube),
        Place(object=cube, inside=SceneObjectRef("bin")),
    )
    while len(system.compiler.grounded_calls) < 3:
        if result.wait_duration:
            system.clock.sleep(result.wait_duration)
        result = system.runtime.step()

    result = system.runtime.deactivate_rows(
        _mask(True, False),
        reason="parallel peer failed",
    )
    while not result.terminal:
        if result.wait_duration:
            system.clock.sleep(result.wait_duration)
        result = system.runtime.step()

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert torch.equal(result.cancelled_mask, _mask(True, False))
    assert torch.equal(result.success_mask, _mask(False, True))
    assert torch.equal(result.failure_mask, _mask(False, False))
    assert [trace.role for trace in result.workflow_recoveries] == [
        SkillWorkflowRecoveryRole.REACQUIRE,
        SkillWorkflowRecoveryRole.RETRY_REACQUIRED,
    ]


def test_nonblocking_step_routes_effect_feedback_through_collector() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    result = system.runtime.start(_call("stepwise"))

    assert result.status is SemanticExecutionStatus.RUNNING
    while not result.terminal:
        if result.wait_duration:
            system.clock.sleep(result.wait_duration)
        result = system.runtime.step()

    assert result.status is SemanticExecutionStatus.COMPLETED
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

    assert result.status is SemanticExecutionStatus.COMPLETED
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

    invalidation, retry = SemanticCallExecutor._terminal_failure_policy(
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
        EffectMonitorDecision(
            _mask(False, True),
            _mask(True, False),
            (
                EffectExpectationDecision(
                    "source",
                    _mask(False, True),
                    _mask(True, False),
                    _mask(False, False),
                ),
            ),
        ),
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


def test_phase_effect_gate_uses_independent_monitor_and_records_boundary_trace() -> (
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
    expectation = HeldObjectStateExpectation(
        expectation_id="destination",
        relation=HeldObjectRelation.ATTACHED,
        object_id="cube",
        slot_id="primary",
        resource_id="arm",
        task_state_key="arm",
    )
    spec = SemanticEffectSpec(
        semantic_id="pick",
        effect_kind=SemanticEffectKind.ATTACH,
        skill_id="pick_up",
        invocation_id="workflow:0",
        invocation_revision=0,
        env_ids=torch.arange(BATCH_SIZE, dtype=torch.long),
        state_expectations=(expectation,),
        clauses=(
            BinaryEffectClause(
                clause_id="destination.constraint",
                expectation_id="destination",
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
    terminal_monitor = _DecisionMonitor(
        spec,
        EffectMonitorDecision(
            _mask(True, True),
            _mask(False, False),
            (
                EffectExpectationDecision(
                    "destination",
                    _mask(True, True),
                    _mask(False, False),
                    _mask(False, False),
                ),
            ),
        ),
    )
    gate_monitor = _DecisionMonitor(
        spec,
        EffectMonitorDecision(
            _mask(False, True),
            _mask(True, False),
            (
                EffectExpectationDecision(
                    "destination",
                    _mask(False, True),
                    _mask(True, False),
                    _mask(False, False),
                ),
            ),
        ),
    )
    gate = GroundedPhaseEffectGate(
        gate_id="destination_acquired",
        segment_name="lift",
        effect_spec=spec,
        effect_monitor=gate_monitor,
        retry_action=True,
    )
    system.runtime._grounded = SimpleNamespace(
        analyzed=SimpleNamespace(effect_monitor_ref=None),
        effect_monitor=terminal_monitor,
        effect_gates=(gate,),
    )
    system.runtime._runner = SimpleNamespace(
        session=SimpleNamespace(
            active_plan=SimpleNamespace(
                expected_effects=StateDelta(held_object_updates={"arm": None}),
                effect_candidates=StateDelta(held_object_updates={"arm": held}),
            )
        )
    )
    system.runtime._current_call_index = 0
    context = system.observation.observe(TaskState.empty(BATCH_SIZE, "cpu"))
    request = PhaseEffectGateRequest(
        verification_id=7,
        gate_id="destination_acquired",
        skill_id="pick_up",
        invocation_id="workflow:0",
        invocation_revision=0,
        invocation_index=0,
        attempt_generation=3,
        next_waypoint_index=4,
        segment_name="lift",
        requested_at=0.0,
        deadline=10.0,
        env_mask=_mask(True, True),
    )

    result = system.runtime._phase_effect_gate_verifier(context, request)

    assert result.verification_id == 7
    assert result.gate_id == "destination_acquired"
    assert result.attempt_generation == 3
    assert result.next_waypoint_index == 4
    assert torch.equal(result.success_mask, _mask(False, True))
    assert torch.equal(result.failure_mask, _mask(True, False))
    assert torch.equal(result.retry_mask, _mask(True, False))
    assert terminal_monitor.calls == 0
    assert gate_monitor.calls == 1
    assert gate_monitor.requests[0].terminal_segment == "lift"
    candidate = gate_monitor.requests[0].expected_effects.held_object_updates["arm"]
    assert isinstance(candidate, HeldObjectState)
    assert candidate.semantics.entity_id == "cube"
    trace = system.runtime._effect_traces[0]
    assert trace.boundary_kind == "phase_effect_gate"
    assert trace.guard_id is None
    assert trace.gate_id == "destination_acquired"
    assert trace.segment_name == "lift"
    assert torch.equal(system.collector.calls[0][2], torch.tensor([0, 1]))


def test_runtime_invokes_step_observer() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    observed_steps = []

    result = system.runtime.run(
        (_call("canonical"),),
        workflow_id="canonical",
        on_step=observed_steps.append,
    )

    assert result.status is SemanticExecutionStatus.COMPLETED
    assert observed_steps


def test_result_metadata_is_json_safe_and_contains_typed_runtime_trace() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))

    result = system.runtime.run(_call("metadata"))
    metadata = result.to_metadata()

    json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert metadata["schema_version"] == 2
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
    assert attempt["planner_diagnostics"]["backend"] == "runtime_test"
    typed_attempt = result.calls[0].plan_attempts[0]
    assert typed_attempt.scene_dependency_monitor_until == {"fixture": 0}
    assert typed_attempt.snapshot().scene_dependency_monitor_until == {"fixture": 0}
    resolved = call["resolved_core_policy"]
    assert resolved["profile_id"] == "runtime_test_profile"
    assert resolved["preset"] == {"preset_id": "runtime_test_preset"}
    assert resolved["motion_policy"]["strategy"] == "ik_interp"
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
    assert metadata["workflow_recoveries"] == []

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

    assert result.status is SemanticExecutionStatus.FAILED
    assert len(result.calls) == 1
    assert result.calls[0].plan_attempts == ()
    assert result.calls[0].resolved_core_policy.preset_id == "runtime_test_preset"
    assert metadata["calls"][0]["active_plan_attempt_generation"] is None
    assert (
        metadata["calls"][0]["resolved_core_policy"]["motion_policy"]["sample_count"]
        == 7
    )
    assert result.failures[0].code == "semantic_call_preparation_failed"
    assert result.failures[0].phase == "preparation"
    assert result.failures[0].diagnostic is None
    assert metadata["failures"][0]["code"] == "semantic_call_preparation_failed"
    assert metadata["failures"][0]["phase"] == "preparation"
    assert metadata["failures"][0]["diagnostic"] is None


def test_preparation_failure_preserves_structured_semantic_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    diagnostic = SemanticDiagnostic(
        "unknown_entity",
        ("workflow", 0, "object"),
        "The referenced object is unavailable.",
        ("cube",),
    )
    monkeypatch.setattr(
        system.compiler,
        "ground",
        Mock(side_effect=SemanticValidationError(diagnostic)),
    )

    result = system.runtime.start(_call("invalid_reference"))
    metadata = result.to_metadata()

    assert result.status is SemanticExecutionStatus.FAILED
    assert len(result.failures) == 1
    assert result.failures[0].code == "unknown_entity"
    assert result.failures[0].phase == "preparation"
    assert result.failures[0].diagnostic == diagnostic
    assert metadata["failures"][0]["diagnostic"] == {
        "code": "unknown_entity",
        "path": ["workflow", 0, "object"],
        "rendered_path": "workflow[0].object",
        "message": "The referenced object is unavailable.",
        "candidates": ["cube"],
    }


def test_cancel_inherits_runner_cancel_then_hold_safe_stop() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    system.runtime.start(_call("cancel"))

    result = system.runtime.cancel("operator stop")

    assert result.status is SemanticExecutionStatus.CANCELLED
    assert torch.equal(result.cancelled_mask, _mask(True, True))
    assert not result.eligible_mask.any()
    assert system.sink.cancelled == 1
    assert system.sink.held == 1
    assert result.calls[0].status.value == "cancelled"


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
    assert lane.status is SemanticExecutionStatus.IDLE
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

    assert result.status is SemanticExecutionStatus.FAILED
    assert torch.equal(result.cancelled_mask, _mask(True, False))
    assert torch.equal(result.failure_mask, _mask(False, True))


def test_deactivate_all_rows_safe_stops_immediately_before_due_time() -> None:
    system = _system((EffectMonitorDecision(_mask(True, True), _mask(False, False)),))
    system.runtime.start(_call("deactivate_all"))

    result = system.runtime.deactivate_rows(
        _mask(True, True),
        reason="parallel peer failed",
    )

    assert result.status is SemanticExecutionStatus.CANCELLED
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
    parallel = ParallelSemanticExecutor.from_template(
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
