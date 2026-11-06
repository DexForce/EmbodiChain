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

"""Observed source contracts through the ordinary engine and runner."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.data_pipeline.real2sim import (
    RobotDemonstration,
    DemoSkillSpec,
    DemoSkillPhase,
    DemoSkillEvent,
    DemoSkillConstraint,
    JointSpec,
    build_demo_skill,
)
from embodichain.lab.sim.atomic_actions import (
    DemoSkillGoal,
    ExecuteDemoSkill,
    DemoSkillMotionProvider,
    DemoSkillObservation,
    AtomicActionEngine,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
    ActionInvocation,
    MotionPolicy,
    RecoveryPolicy,
    ExecutionRunner,
    ExecutionRunnerCfg,
    CommandAcknowledgement,
    RunnerStatus,
    ControlPartCommandProfile,
    DemoSkillMonitor,
)

DT = 0.02


def _asset():
    return build_demo_skill(
        RobotDemonstration(
            "teleop",
            "1",
            "whole-task",
            ("a", "h"),
            ("rad", "m"),
            np.zeros((6, 2)),
            np.arange(6) * DT,
            "s",
            "a" * 64,
            "b" * 64,
        ),
        DemoSkillSpec(
            asset_id="task.test",
            instruction="Execute a specialized task",
            phases=(DemoSkillPhase("specialized", 0, 6),),
            events=(DemoSkillEvent("checkpoint", 3, "checkpoint"),),
            constraints=(
                DemoSkillConstraint(
                    "observed_success", "metric", "terminal", minimum=1, maximum=1
                ),
            ),
        ),
        joints=(JointSpec("a", "revolute"), JointSpec("h", "prismatic")),
        control_dt=DT,
    )


def _context(t=0.0, row=0):
    return PlanningContext(
        RobotObservation(t, torch.zeros(1, 2), torch.zeros(1, 2)),
        TaskState.empty(batch_size=1, device="cpu"),
        SceneSnapshot.empty(),
        torch.tensor([row]),
        control_dt=DT,
    )


def _setup(row=0, asset=None):
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 2
    robot.joint_names = ("a", "h")
    robot.control_parts = {"custom_motion": object(), "custom_hand": object()}
    robot.get_qpos.side_effect = lambda name=None: torch.zeros(2, 1 if name else 2)
    robot.get_qvel.return_value = torch.zeros(2, 2)
    robot.get_joint_ids.side_effect = lambda name=None: (
        [0] if name == "custom_motion" else ([1] if name == "custom_hand" else [0, 1])
    )
    robot.get_qpos_limits.return_value = torch.tensor([[[-2.0, 2.0]], [[-2.0, 2.0]]])
    robot.get_qvel_limits.return_value = torch.full((2, 1), 10.0)
    robot.body_data = SimpleNamespace(
        qpos_limits=torch.tensor(
            [[[-2.0, 2.0], [-2.0, 2.0]], [[-2.0, 2.0], [-2.0, 2.0]]]
        )
    )
    robot.mimic_ids = ()
    generator = Mock()
    generator.robot = robot
    generator.device = robot.device
    generator.planner.cfg.planner_type = "stub"
    engine = AtomicActionEngine(
        generator,
        env_ids=(row,),
        control_profiles={
            "custom_hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(1), grasp=torch.ones(1)
            )
        },
    )
    asset = _asset() if asset is None else asset
    evidence = SimpleNamespace(
        observe=lambda context: DemoSkillObservation(
            context.robot.timestamp, row, {}, metrics={"observed_success": 1.0}
        )
    )
    provider = DemoSkillMotionProvider(
        asset,
        role_bindings={},
        robot_geometry_sha256="a" * 64,
        role_geometry_sha256={},
        evidence_provider=evidence,
        motion_generator=generator,
    )
    invocation = ActionInvocation(
        "execute_demo_skill",
        DemoSkillGoal(asset.content_sha256, {}),
        binding=engine.bind_control_parts(
            "execute_demo_skill",
            {"primary": {"motion": "custom_motion", "grasp": "custom_hand"}},
        ),
        motion_policy=MotionPolicy(sample_count=6),
        recovery_policy=RecoveryPolicy(max_replans=0, max_action_retries=0),
    )
    return engine, provider, invocation, evidence


def test_specialized_motion_is_one_nonreplannable_atomic_plan_on_selected_row():
    engine, provider, invocation, _ = _setup(row=1)
    plan = engine.plan(invocation, _context(row=1), nominal_source=provider)
    assert plan.effect_verification.kind == "demo_skill"
    assert not plan.replannable
    assert [segment.name for segment in plan.segments] == ["specialized"]
    assert plan.joint_trajectory.env_ids.tolist() == [1]
    assert plan.diagnostics.metadata["demo_skill"]["asset_id"] == "task.test"
    engine.motion_generator.generate.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "source",
        "semantic",
        "explicit_success",
        "explicit_failure",
        "explicit_unresolved",
        "explicit_mismatched",
    ],
)
def test_runner_preserves_source_and_semantic_verification_boundaries(failure):
    engine, provider, invocation, evidence = _setup()
    current = [0.0]
    context = _context()
    transform = None
    verifier = None
    if failure == "semantic" or failure.startswith("explicit_"):
        from embodichain.lab.sim.atomic_actions import (
            StateDelta,
            ArticulationJointState,
            EffectVerificationResult,
        )

        transform = lambda resolved, ctx, plan: replace(
            plan,
            expected_effects=StateDelta(
                articulation_joint_updates={
                    ("hinge", "joint"): ArticulationJointState(torch.ones(1))
                }
            ),
        )

        if failure == "semantic":

            def verifier(ctx, request):
                return EffectVerificationResult(
                    verification_id=request.verification_id,
                    success_mask=torch.zeros_like(request.env_mask),
                    failure_mask=request.env_mask,
                    invalidation_mask=torch.zeros_like(request.env_mask),
                    retry_mask=torch.zeros_like(request.env_mask),
                )

    session = engine.start(
        (invocation,), context, nominal_source=provider, plan_transform=transform
    )

    class Observations:
        def observe(self, task_state):
            return replace(_context(current[0]), task=task_state)

    class Clock:
        def now(self):
            return current[0]

        def sleep(self, duration):
            current[0] += duration

    sink = SimpleNamespace(
        send=Mock(return_value=CommandAcknowledgement.accepted_ack()),
        hold=Mock(return_value=CommandAcknowledgement.accepted_ack()),
        cancel=Mock(return_value=CommandAcknowledgement.accepted_ack()),
    )
    if failure == "source":
        evidence.observe = lambda context: DemoSkillObservation(
            context.robot.timestamp, 0, {}, metrics={"observed_success": 0.0}
        )
    runner = ExecutionRunner(
        session, Observations(), sink, clock=Clock(), cfg=ExecutionRunnerCfg()
    )
    supplied = None
    for _ in range(30):
        pending = session.pending_effect
        if failure.startswith("explicit_") and pending is not None:
            supplied = EffectVerificationResult(
                verification_id=pending.verification_id
                + (failure == "explicit_mismatched"),
                success_mask=torch.full_like(
                    pending.env_mask,
                    failure in {"explicit_success", "explicit_mismatched"},
                ),
                failure_mask=torch.full_like(
                    pending.env_mask, failure == "explicit_failure"
                ),
                invalidation_mask=torch.zeros_like(pending.env_mask),
                retry_mask=torch.zeros_like(pending.env_mask),
            )
        result = runner.step(effect_verifier=verifier, effect_result=supplied)
        if result.status != RunnerStatus.RUNNING or supplied is not None:
            break
        current[0] += max(result.wait_duration, DT)
    expected_status = (
        RunnerStatus.COMPLETED
        if failure in {"none", "explicit_success"}
        else (
            RunnerStatus.RUNNING
            if failure == "explicit_unresolved"
            else RunnerStatus.FAILED
        )
    )
    assert result.status == expected_status, result.message
    assert provider.monitor.terminal_passed is (failure != "source")
    state = session.task_state.get_articulation_joint_state("hinge", "joint")
    if failure == "explicit_success":
        torch.testing.assert_close(state.position, torch.ones(1, 1))
    elif failure in {
        "semantic",
        "explicit_failure",
        "explicit_unresolved",
        "explicit_mismatched",
    }:
        assert state is None
    if failure == "explicit_unresolved":
        current[0] += max(result.wait_duration, DT)
        waiting = runner.step()
        assert waiting.status is RunnerStatus.RUNNING
        assert session.task_state.get_articulation_joint_state("hinge", "joint") is None
        # A later correlated approval can finish the same pending boundary.
        current[0] += max(waiting.wait_duration, DT)
        pending = session.pending_effect
        result = runner.step(
            effect_result=replace(
                supplied,
                verification_id=pending.verification_id,
                success_mask=pending.env_mask,
            )
        )
        assert result.status is RunnerStatus.COMPLETED, result.message
        assert (
            session.task_state.get_articulation_joint_state("hinge", "joint")
            is not None
        )
    if expected_status is RunnerStatus.FAILED:
        assert sink.cancel.called and sink.hold.called


def test_monitor_missing_contact_is_not_evidence_of_no_contact():
    asset = _asset()
    spec = asset.spec
    spec.constraints = (
        DemoSkillConstraint(
            "separate", "no_contact", "terminal", subject="tcp", other="world"
        ),
    )
    artifact = build_demo_skill(
        asset.demonstration, spec, joints=asset.joint_specs, control_dt=DT
    )
    monitor = DemoSkillMonitor(
        artifact,
        SimpleNamespace(
            observe=lambda c: DemoSkillObservation(c.robot.timestamp, 0, {})
        ),
    )
    monitor.observe(_context(), None)
    for index in range(6):
        if index == 5:
            with pytest.raises(ValueError, match="unavailable"):
                monitor.observe(_context((index + 1) * DT), index, terminal=True)
        else:
            monitor.observe(_context((index + 1) * DT), index)


@pytest.mark.parametrize("operator", ["joint_residual", "retime"])
def test_demo_skill_expansion_preserves_internal_event_and_gripper(operator):
    from embodichain.lab.task_program.runtime.expansion import TaskProgramPlanRequest
    from embodichain.lab.task_program.semantics.calls import RegisteredSemanticCall
    from embodichain.lab.task_program.integrations.expansion import (
        TaskProgramSameGridExpansionFactory,
        TaskProgramMappedRetimeFactory,
    )

    source = replace(
        _asset().demonstration,
        qpos=np.column_stack((np.sin(np.linspace(0, np.pi, 40)) * 0.01, np.zeros(40))),
        timestamps=np.arange(40) * DT,
    )
    asset = build_demo_skill(
        source,
        DemoSkillSpec(
            asset_id="continuous",
            instruction="follow task path",
            phases=(
                DemoSkillPhase(
                    "specialized", 0, 40, allowed_operators=("joint_residual", "retime")
                ),
            ),
            events=(DemoSkillEvent("checkpoint", 20, "checkpoint"),),
            constraints=(
                DemoSkillConstraint(
                    "observed_success", "metric", "terminal", minimum=1, maximum=1
                ),
            ),
            max_duration_scale=1.5,
        ),
        joints=(JointSpec("a", "revolute"), JointSpec("h", "prismatic")),
        control_dt=DT,
    )
    engine, provider, invocation, _ = _setup(asset=asset)
    context = _context()
    invocation = replace(
        invocation, motion_policy=MotionPolicy(sample_count=40, acceleration_limit=50)
    )
    plan = engine.plan(invocation, context, nominal_source=provider)
    request = TaskProgramPlanRequest(
        workflow_id="demo/segment",
        workflow_call_index=0,
        analysis_call_index=0,
        call=RegisteredSemanticCall("task.test", {}),
        invocation=invocation,
    )
    common = {
        "velocity_limits": (10, 10),
        "acceleration_limits": (50, 50),
        "call_indices": (0,),
        "diagnostic_unchecked": True,
    }
    factory = (
        TaskProgramMappedRetimeFactory(duration_scale=1.3, **common)
        if operator == "retime"
        else TaskProgramSameGridExpansionFactory(
            operator="joint_residual",
            seed=7,
            geometry_family_id="scene",
            initial_state_id="initial",
            joint_offset_scale=0.003,
            **common,
        )
    )
    transformed = factory.create_plan_transform(request, engine=engine)(
        engine.resolve(invocation), context, plan
    )
    provider.bind_execution_plan(transformed)
    event = provider.monitor.spec.events[0].sample_index
    torch.testing.assert_close(
        transformed.joint_trajectory.positions[:, event - 1 : event + 2],
        plan.joint_trajectory.positions[:, 19:22],
        atol=1e-6,
        rtol=0,
    )
    assert torch.count_nonzero(transformed.joint_trajectory.positions[:, :, 1]) == 0
    if operator == "retime":
        assert (
            transformed.joint_trajectory.waypoint_count
            > plan.joint_trajectory.waypoint_count
        )
        assert event > 20
    else:
        assert not torch.equal(
            transformed.joint_trajectory.positions, plan.joint_trajectory.positions
        )


def test_contact_loss_is_latched_before_terminal_success():
    from embodichain.data_pipeline.real2sim import DemoSkillRole

    original = _asset()
    spec = original.spec
    spec.roles = (DemoSkillRole("tool", "c" * 64, tuple(np.eye(4).flat)),)
    spec.phases = (DemoSkillPhase("specialized", 0, 6, "contact"),)
    spec.constraints = (
        *spec.constraints,
        DemoSkillConstraint(
            "contact_maintained", "contact", "throughout", subject="tcp", other="tool"
        ),
    )
    asset = build_demo_skill(
        original.demonstration, spec, joints=original.joint_specs, control_dt=DT
    )
    contact = [True]
    evidence = SimpleNamespace(
        observe=lambda context: DemoSkillObservation(
            context.robot.timestamp,
            0,
            {},
            contacts={("tcp", "tool"): contact[0]},
            metrics={"observed_success": 1.0},
        )
    )
    monitor = DemoSkillMonitor(asset, evidence)
    monitor.observe(_context(), None)
    monitor.observe(_context(DT), 0)
    contact[0] = False
    with pytest.raises(ValueError, match="contact_maintained"):
        monitor.observe(_context(2 * DT), 1)
    contact[0] = True
    with pytest.raises(ValueError, match="contact_maintained"):
        monitor.observe(_context(3 * DT), 2)
    assert not monitor.terminal_passed
