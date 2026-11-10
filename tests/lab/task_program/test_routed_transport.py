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

"""Routed transport uses the real lowering, skill, guard and terminal lifecycle."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_skills import (
    MoveHeldObjectOptions,
    RecoveryPolicy,
    ExecutionRunnerCfg,
    StateDelta,
    Affordance,
    ObjectSemantics,
)
from embodichain.lab.sim.atomic_skills.tracking import (
    TrackingPolicy,
    TimedTerminalAcceptance,
)
from embodichain.lab.sim.motion.planners import PlanResult
from embodichain.lab.task_program.compiler.lowering import SemanticCallCompiler
from embodichain.lab.task_program.integrations.simulation.targets import (
    AbsolutePoseTarget,
)
from embodichain.lab.task_program.integrations.simulation.transport import (
    HeldObjectTransportRoute,
    RoutedMoveHeldObjectLowererFactory,
)
from embodichain.lab.task_program.runtime.executor import SemanticCallExecutor
from embodichain.lab.task_program.runtime.results import SemanticExecutionStatus
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    SemanticCallDescriptor,
    builtin_semantic_call_catalog,
)
from embodichain.lab.task_program.semantics.profiles import EffectAssurance
from embodichain.lab.task_program.semantics.integration import (
    SceneManifest,
    SemanticIntegrationManifest,
)
from embodichain.lab.task_program.semantics.effects import (
    EffectMonitorDecision,
    EffectExpectationDecision,
    EffectMonitorRef,
    EffectMonitorRegistry,
)
from test_semantic_compiler import (
    _engine,
    _profile,
    _preset,
    _scene_registry,
    _held_context,
)
from test_semantic_executor import _Clock, _CommandSink, _Collector, _DecisionMonitor


def _decision(*values: bool | None) -> EffectMonitorDecision:
    satisfied = torch.tensor([value is True for value in values])
    failed = torch.tensor([value is False for value in values])
    return EffectMonitorDecision(
        satisfied,
        failed,
        (EffectExpectationDecision("primary", satisfied, failed, failed),),
    )


def _transport_system(monkeypatch: pytest.MonkeyPatch, outcome: str) -> SimpleNamespace:
    registry, _ = _scene_registry()
    factory = RoutedMoveHeldObjectLowererFactory(
        (
            HeldObjectTransportRoute(
                "cube",
                "destination",
                AbsolutePoseTarget((0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ),
        )
    )
    preset = _preset(
        "safe",
        skill_option_templates={factory.call_id: MoveHeldObjectOptions()},
        effect_assurance=(
            EffectAssurance.PROJECTED
            if outcome == "projected"
            else EffectAssurance.VERIFIED
        ),
        effect_monitors=(
            {}
            if outcome == "projected"
            else {factory.call_id: EffectMonitorRef("builtin.composite_effect", "1")}
        ),
        runner_cfg=ExecutionRunnerCfg(minimum_cycle_time=0.05),
        tracking_policy=TrackingPolicy(
            in_flight=None, terminal=TimedTerminalAcceptance()
        ),
        recovery_policy=RecoveryPolicy(
            max_skill_retries=0, max_replans=0, skill_timeout=0.5
        ),
    )
    profile = _profile(preset=preset)
    engine = _engine(profile)
    dt = torch.tensor([[0.0, 0.1, 0.1]]).repeat(2, 1)
    engine.motion_generator.generate = Mock(
        return_value=PlanResult(
            success=torch.tensor([True, True]), positions=torch.zeros(2, 3, 1), dt=dt
        )
    )
    engine.robot.get_qvel_limits.return_value = torch.full((2, 1), 100.0)
    engine.robot.compute_fk.side_effect = lambda qpos, **kwargs: torch.eye(4).repeat(
        qpos.shape[0], 1, 1
    )
    manifest = SemanticIntegrationManifest(
        scene=SceneManifest.from_registry(registry),
        robot_profile=profile,
        call_catalog=builtin_semantic_call_catalog().with_descriptor(
            SemanticCallDescriptor(
                call_id=factory.call_id,
                spec_type=RegisteredSemanticCall,
                target_descriptor=factory.target_descriptor,
            )
        ),
    )
    bound = manifest.bind(registry, engine)
    lowerer = factory.create(
        simulation=None, robot=engine.robot, engine=engine, scene_registry=registry
    )
    compiler = SemanticCallCompiler(bound, registered_lowerers=(lowerer,))
    created = []

    def create_monitor(self, spec, ref):
        monitor = _DecisionMonitor(spec, _decision(True, True))
        observe = monitor.observe

        def selected_rows(request, evidence):
            decision = observe(request, evidence)
            selected = request.env_mask
            return EffectMonitorDecision(
                decision.success_mask & selected,
                decision.failure_mask & selected,
                tuple(
                    replace(
                        item,
                        satisfied_mask=item.satisfied_mask & selected,
                        contradicted_mask=item.contradicted_mask & selected,
                        inverse_satisfied_mask=item.inverse_satisfied_mask & selected,
                    )
                    for item in decision.expectation_decisions
                ),
            )

        monkeypatch.setattr(monitor, "observe", selected_rows)
        created.append(monitor)
        return monitor

    monkeypatch.setattr(EffectMonitorRegistry, "create", create_monitor)
    original_ground = compiler.ground
    grounded_calls = []

    def record_ground(*args, **kwargs):
        grounded = original_ground(*args, **kwargs)
        grounded_calls.append(grounded)
        if outcome == "terminal_loss":
            grounded.effect_monitor._decision = _decision(False, True)
        elif outcome == "missing":
            grounded.effect_monitor._decision = _decision(None, True)
        for guard in grounded.effect_guards:
            if outcome == "lost":
                guard.effect_monitor._decision = _decision(False, True)
            elif outcome == "missing":
                guard.effect_monitor._decision = _decision(None, True)
        return grounded

    monkeypatch.setattr(compiler, "ground", record_ground)
    entity = "table" if outcome == "wrong_object" else "cube"
    semantics = ObjectSemantics(affordance=Affordance(), geometry={}, entity_id=entity)
    context = _held_context(
        registry,
        semantics,
        torch.eye(4).repeat(2, 1, 1),
        env_mask=torch.tensor([True, False]) if outcome == "partial_input" else None,
    )
    clock, sink, collector = _Clock(), _CommandSink(), _Collector()
    observation = SimpleNamespace(
        observe=lambda task: replace(
            context,
            robot=replace(context.robot, timestamp=clock.now()),
            task=task,
            control_dt=0.1,
        )
    )
    runtime = SemanticCallExecutor(
        compiler,
        observation,
        sink,
        collector,
        task_state=context.task,
        clock=clock,
        plan_transform_factory=(
            SimpleNamespace(
                create_plan_transform=lambda *args, **kwargs: lambda request, current, plan: replace(
                    plan, expected_effects=StateDelta(), effect_verification=None
                )
            )
            if outcome == "stripped_effects"
            else None
        ),
    )
    call = RegisteredSemanticCall(
        call_id=factory.call_id,
        arguments={"object": "cube", "target": "destination"},
        resources={"primary": "manipulator"},
    )
    return SimpleNamespace(
        runtime=runtime,
        call=call,
        sink=sink,
        collector=collector,
        engine=engine,
        grounded=grounded_calls,
        monitors=created,
    )


@pytest.mark.parametrize("outcome", ["wrong_object", "partial_input", "projected"])
def test_routed_transport_rejects_incompatible_input_before_motion(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    system = _transport_system(monkeypatch, outcome)
    result = system.runtime.run(system.call, max_steps=100)
    assert result.status is SemanticExecutionStatus.FAILED
    assert system.sink.sent == 0
    system.engine.motion_generator.generate.assert_not_called()


@pytest.mark.parametrize("outcome", ["success", "lost", "missing", "terminal_loss"])
def test_routed_transport_executes_retention_and_terminal_verification(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    system = _transport_system(monkeypatch, outcome)
    result = system.runtime.run(system.call, max_steps=100)
    assert system.grounded
    grounded = system.grounded[0]
    assert grounded.invocation.goal.object_id == "cube"
    assert len(grounded.effect_guards) == 1 and grounded.effect_gates == ()
    guard = grounded.effect_guards[0]
    assert guard.active_segments == ("transport",) and not guard.retry_action
    assert guard.effect_monitor.calls > 0
    assert grounded.effect_monitor.calls > 0
    if outcome == "success":
        assert result.status is SemanticExecutionStatus.COMPLETED
        assert result.success_mask.tolist() == [True, True]
    else:
        # A completed workflow may contain failed rows; per-row masks own truth.
        assert result.status is SemanticExecutionStatus.COMPLETED
        assert result.success_mask.tolist() == [False, True]
        assert result.failure_mask.tolist() == [True, False]
    held = result.task_state.get_held_object("manipulator")
    assert held is not None and held.semantics.entity_id == "cube"
    if outcome in {"lost", "terminal_loss"}:
        assert held.env_mask.tolist() == [False, True]


def test_routed_transport_cannot_drop_terminal_verification_in_plan_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _transport_system(monkeypatch, "stripped_effects")
    result = system.runtime.run(system.call, max_steps=100)
    assert result.status is SemanticExecutionStatus.FAILED
    assert system.sink.sent == 0
    assert "terminal effect verification" in (result.message or "")


def test_routed_transport_checks_only_eligible_input_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _transport_system(monkeypatch, "partial_input")
    result = system.runtime.run(
        system.call, eligible_mask=torch.tensor([True, False]), max_steps=100
    )
    assert result.status is SemanticExecutionStatus.COMPLETED
    assert result.success_mask.tolist() == [True, False]
    assert result.failure_mask.tolist() == [False, False]
    assert result.calls[0].entered_mask.tolist() == [True, False]
    assert system.sink.sent > 0
