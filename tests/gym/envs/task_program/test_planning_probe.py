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

"""Configured probe diagnostics use the production compiler and one mocked plan."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.gym.envs.base_env import BaseEnv
from embodichain.lab.sim.atomic_actions import RecoveryPolicy, TrackingPolicy
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    PlanningFailure,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    RuntimeCommandFrame,
    TimedCommandSequence,
)
from embodichain.lab.task_program import (
    BarrierCfg,
    InvokeCfg,
    PickCfg,
    ParallelCfg,
    PlaceCfg,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    CyclicPoseTargetCfg,
    PoseCfg,
)
from embodichain.lab.task_program.integrations import (
    InvocationPolicyBinding,
    task_program_fingerprint,
)
from embodichain.lab.task_program.integrations.environment import (
    TaskProgramEnvironmentAdapter,
)
from embodichain.lab.task_program.integrations.simulation.planning_probe import (
    probe_configured_task_program_initial_plan,
)
from embodichain.lab.task_program.runtime.executor import SemanticCallExecutor
from embodichain.lab.task_program.compiler.lowering import SemanticCallCompiler

from .test_environment import _InvocationPolicyFactory, _program, _STEP_DT


def _environment():
    env = object.__new__(BaseEnv)
    env.cfg = SimpleNamespace(seed=17)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.step = lambda *a, **k: pytest.fail("probe stepped environment")
    env.reset = lambda *a, **k: pytest.fail("probe reset environment")
    return env


def _snapshot(env):
    return dict(
        python=random.getstate(),
        numpy=deepcopy(np.random.get_state()),
        torch=torch.get_rng_state().clone(),
        component=env.get_generator("existing").get_state().clone(),
        gym=deepcopy(env.np_random.bit_generator.state),
    )


def _assert_random_state(env, before):
    assert random.getstate() == before["python"]
    state = np.random.get_state()
    assert state[0] == before["numpy"][0]
    np.testing.assert_array_equal(state[1], before["numpy"][1])
    assert state[2:] == before["numpy"][2:]
    assert torch.equal(torch.get_rng_state(), before["torch"])
    assert torch.equal(env.get_generator("existing").get_state(), before["component"])
    assert env.np_random.bit_generator.state == before["gym"]


@pytest.mark.parametrize("fail", [False, True])
def test_environment_rng_scope_restores_lazy_and_existing_generators(monkeypatch, fail):
    env = _environment()
    generator = env.get_generator("existing")
    env.np_random = np.random.default_rng(91)
    before = _snapshot(env)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "device_count", lambda: pytest.fail("RNG capture initialized CUDA")
    )
    created = None
    try:
        with env.preserve_random_state():
            random.random()
            np.random.rand()
            torch.rand(3)
            torch.rand(3, generator=generator)
            env.np_random.random(3)
            created = env.get_generator("probe-created")
            if fail:
                raise RuntimeError("diagnostic failed")
    except RuntimeError as error:
        assert fail and str(error) == "diagnostic failed"
    _assert_random_state(env, before)
    assert env.get_generator("existing") is generator
    assert env.get_generator("probe-created") is not created


def test_environment_rng_scope_forgets_probe_created_gym_generator():
    env = _environment()
    with env.preserve_random_state():
        created = env.np_random
        created.random(3)
    assert env.np_random is not created


class _ProbeFactory(_InvocationPolicyFactory):
    def __init__(self, env, path, *, failure=None):
        super().__init__(effect_monitors={})
        self.env, self.path, self.failure = env, path, failure
        self.planned, self.closed = [], []

    def random_draw(self):
        random.random()
        np.random.rand()
        torch.rand(1)
        torch.rand(1, generator=self.env.get_generator("existing"))
        self.env.np_random.random()
        self.env.get_generator("probe-created")

    def create_atomic_action_engine(self, profile, **kwargs):
        engine = super().create_atomic_action_engine(profile)
        robot = engine.robot
        robot.cfg = SimpleNamespace(fpath=str(self.path))
        robot.joint_names = ["arm_joint", "hand_joint"]
        robot.get_joint_ids.side_effect = lambda name=None: {
            None: [0, 1],
            "arm": [0],
            "hand": [1],
        }[name]
        robot.get_qvel_limits.side_effect = lambda name=None: torch.full(
            (2, 2 if name is None else 1), 2.0
        )

        def plan(request, context):
            self.random_draw()
            self.planned.append((request, context))
            if self.failure == "planning":
                raise RuntimeError("planner stopped")
            positions = torch.tensor(
                [
                    [[0.0, 0.0], [0.01, 0.01], [0.02, 0.02]],
                    [[0.0, 0.0], [0.2, 0.2], [0.4, 0.4]],
                ]
            )
            dt = torch.full((2, 3), 0.04)
            dt[:, 0] = 0
            ids = context.env_ids
            trajectory = TimedTrajectory(positions, None, None, dt, ids)
            frames = tuple(
                RuntimeCommandFrame((), torch.ones(2, dtype=torch.bool), ids, dt[:, i])
                for i in range(3)
            )
            success = (
                torch.tensor([False, False])
                if self.failure == "failed-plan"
                else torch.tensor([True, True])
            )
            if self.failure in {"failed-plan", "missing-trajectory"}:
                trajectory = None
                frames = ()
            return ActionPlan(
                skill_id=request.skill_id,
                plan_success=success,
                commands=TimedCommandSequence(frames, ids),
                recovery_policy=RecoveryPolicy(),
                tracking_policy=TrackingPolicy.timed(),
                planned_scene_version=context.scene.version,
                planned_collision_world_revision=context.scene.collision_world_revision,
                diagnostics=PlannerDiagnostics(
                    "probe-fixture",
                    failure=(
                        PlanningFailure("probe_failed", retryable=False)
                        if self.failure == "failed-plan"
                        else None
                    ),
                ),
                joint_trajectory=trajectory,
                invocation_id=request.invocation_id,
            )

        engine.plan = plan
        return engine

    def close_runtime_resources(self):
        self.random_draw()
        self.closed.append("adapter")


def _deployment(tmp_path, monkeypatch, *, failure=None, lookahead=False):
    env = _environment()
    env.np_random = np.random.default_rng(11)
    env.get_generator("existing")
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="probe"><joint name="arm_joint" type="revolute"><limit velocity="1"/></joint><joint name="hand_joint" type="revolute"><limit velocity="1"/></joint></robot>'
    )
    factory = _ProbeFactory(env, path, failure=failure)
    adapter = TaskProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    first = SegmentCfg(name="first", steps=InvokeCfg(call=PickCfg(object="cube")))
    config = _program(
        node=(
            SequenceCfg(
                items=(
                    first,
                    SegmentCfg(
                        name="later",
                        steps=InvokeCfg(
                            call=PlaceCfg(object="cube", at=TargetRefCfg(target="drop"))
                        ),
                    ),
                )
            )
            if lookahead
            else SequenceCfg(items=(first,))
        ),
        targets=(
            {
                "drop": CyclicPoseTargetCfg(
                    values=(
                        PoseCfg(
                            position=(0.4, 0.1, 0.2),
                            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                        ),
                    )
                )
            }
            if lookahead
            else None
        ),
    )
    compiled = adapter.compile(config)
    first = next(compiled.iter_segments())
    invocation = (
        f"{compiled.program_id}/{first.segment_id}:{first.calls[0].segment_call_index}"
    )
    binding = InvocationPolicyBinding(
        task_program_fingerprint(compiled), invocation, "careful"
    )
    original_assembly = adapter._create_bridge_with_assembly

    def assemble(*args, **kwargs):
        bridge, assembly = original_assembly(*args, **kwargs)
        close = bridge.close_runtime_resources

        def closed():
            close()
            factory.closed.append("bridge")

        bridge.close_runtime_resources = closed
        return bridge, assembly

    monkeypatch.setattr(adapter, "_create_bridge_with_assembly", assemble)

    def create(target):
        assert target is env
        factory.random_draw()
        return adapter

    deployment = SimpleNamespace(
        integration=SimpleNamespace(
            adapter_factory=SimpleNamespace(create_adapter=create),
            registration=SimpleNamespace(invocation_policy_bindings=(binding,)),
        ),
        load_program=lambda: config,
    )
    monkeypatch.setattr(
        SemanticCallExecutor,
        "start",
        lambda *a, **k: pytest.fail("probe started execution"),
    )
    monkeypatch.setattr(
        SemanticCallExecutor,
        "step",
        lambda *a, **k: pytest.fail("probe dispatched execution"),
    )
    return env, deployment, compiled, factory, invocation


@pytest.mark.parametrize("lookahead", [False, True])
def test_probe_uses_exact_occurrence_policy_and_returns_only_proposal_data(
    tmp_path, monkeypatch, lookahead
):
    env, deployment, program, factory, invocation = _deployment(
        tmp_path, monkeypatch, lookahead=lookahead
    )
    before = _snapshot(env)
    analyzed = []
    analyze = SemanticCallCompiler.analyze

    def record_analyze(self, *args, **kwargs):
        analyzed.append(kwargs["workflow_id"])
        return analyze(self, *args, **kwargs)

    monkeypatch.setattr(SemanticCallCompiler, "analyze", record_analyze)
    report = probe_configured_task_program_initial_plan(
        env, deployment, compiled_program=program
    )
    _assert_random_state(env, before)
    assert len(factory.planned) == 1
    assert len(analyzed) == 1
    request, context = factory.planned[0]
    assert request.invocation_id == invocation
    assert request.motion_policy.sample_count == 81
    assert factory.observation_samples == 1
    assert factory.closed == ["bridge", "adapter"]
    assert report["plan_success"] == [True, False]
    assert report["final_command_velocity"]["valid_mask"] == [True, False]
    assert report["scope"] == "initial_call" and report["planning_source"] == "probe"
    assert report["task_success"] is None
    assert report["analysis_call_count"] == (2 if lookahead else 1)
    assert report["unplanned_call_indices"] == ([1] if lookahead else [])
    assert report["initial_downstream_target_count"] == int(lookahead)
    assert report["planned_grasp_candidates"] == []
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("failure", ["planning", "missing-trajectory", "failed-plan"])
def test_probe_closes_services_and_restores_rng_when_planning_fails(
    tmp_path, monkeypatch, failure
):
    env, deployment, _, factory, _ = _deployment(tmp_path, monkeypatch, failure=failure)
    before = _snapshot(env)
    if failure == "failed-plan":
        report = probe_configured_task_program_initial_plan(env, deployment)
        assert report["plan_success"] == [False, False]
        assert report["final_command_velocity"]["diagnostics"] == []
    else:
        with pytest.raises(
            (RuntimeError, ValueError), match="planner stopped|final joint trajectory"
        ):
            probe_configured_task_program_initial_plan(env, deployment)
    _assert_random_state(env, before)
    assert factory.closed == ["bridge", "adapter"]
    assert len(factory.planned) == 1


def test_probe_rejects_different_compiled_program_before_planning(
    tmp_path, monkeypatch
):
    env, deployment, _, factory, _ = _deployment(tmp_path, monkeypatch)
    other_config = _program()
    other_config.program_id = "another-program"
    other_adapter = TaskProgramEnvironmentAdapter(
        _InvocationPolicyFactory(effect_monitors={}), step_dt=_STEP_DT
    )
    other_program = other_adapter.compile(other_config)
    other_adapter.close()
    before = _snapshot(env)
    with pytest.raises(ValueError, match="differs from its deployment"):
        probe_configured_task_program_initial_plan(
            env, deployment, compiled_program=other_program
        )
    _assert_random_state(env, before)
    assert factory.planned == [] and factory.closed == ["adapter"]


def test_probe_rejects_initial_parallel_analysis_without_observing(
    tmp_path, monkeypatch
):
    env, deployment, _, factory, _ = _deployment(tmp_path, monkeypatch)
    deployment.load_program = lambda: _program(
        node=ParallelCfg(
            branches=(
                InvokeCfg(call=PickCfg(object="cube")),
                InvokeCfg(call=PickCfg(object="cube")),
            ),
            barrier=BarrierCfg(name="join"),
        )
    )
    deployment.integration.registration.invocation_policy_bindings = ()
    before = _snapshot(env)
    with pytest.raises(ValueError, match="non-empty sequential workflow"):
        probe_configured_task_program_initial_plan(env, deployment)
    _assert_random_state(env, before)
    assert factory.planned == [] and factory.observation_samples == 0
    assert factory.closed == ["adapter"]


@pytest.mark.parametrize("registration_matches", [True, False])
def test_probe_borrows_matching_initialized_adapter_without_closing_its_host(
    tmp_path, monkeypatch, registration_matches
):
    from unittest.mock import Mock

    env, deployment, _, factory, _ = _deployment(tmp_path, monkeypatch)
    adapter = deployment.integration.adapter_factory.create_adapter(env)
    borrowed = Mock(spec=TaskProgramEnvironmentAdapter, wraps=adapter)
    borrowed._registration = SimpleNamespace(
        fingerprint_payload=lambda: {
            "owner": "host" if registration_matches else "another"
        }
    )
    deployment.integration.registration.fingerprint_payload = lambda: {"owner": "host"}
    env.task_program_adapter = borrowed
    deployment.integration.adapter_factory.create_adapter = lambda target: pytest.fail(
        "probe recreated an initialized host's registered native observers"
    )
    before = _snapshot(env)
    try:
        if registration_matches:
            report = probe_configured_task_program_initial_plan(env, deployment)
            assert report["task_success"] is None
            assert factory.closed == ["bridge"]
            assert len(factory.planned) == 1
        else:
            with pytest.raises(
                ValueError, match="differs from its deployment registration"
            ):
                probe_configured_task_program_initial_plan(env, deployment)
            assert factory.closed == [] and factory.planned == []
        _assert_random_state(env, before)
        borrowed.close.assert_not_called()
        assert not adapter._closed
    finally:
        with env.preserve_random_state():
            adapter.close()


def test_probe_refuses_an_active_execution_bridge_without_rebinding_or_closing_it(
    tmp_path, monkeypatch
):
    env, deployment, _, factory, _ = _deployment(tmp_path, monkeypatch)
    active = SimpleNamespace(
        close_runtime_resources=lambda: pytest.fail("borrowed bridge closed")
    )
    env.active_task_program_bridge = active
    before = _snapshot(env)
    with pytest.raises(ValueError, match="unselected execution bridge"):
        probe_configured_task_program_initial_plan(env, deployment)
    _assert_random_state(env, before)
    assert env.active_task_program_bridge is active
    assert factory.closed == [] and factory.planned == []
