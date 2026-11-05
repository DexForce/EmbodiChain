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

"""Passive initial-plan diagnostics observe the installed runtime plan only."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.lab.sim.atomic_actions import TaskState
from embodichain.lab.sim.atomic_actions.execution import ExecutionSession
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.task_program.integrations.environment import (
    TaskProgramEnvironmentAdapter,
)

from .test_environment import _program, _STEP_DT
from .test_planning_probe import _ProbeFactory, _environment


class _ExecutionFactory(_ProbeFactory):
    def create_atomic_action_engine(self, profile, **kwargs):
        engine = super().create_atomic_action_engine(profile, **kwargs)
        planned = engine.plan
        self.transforms = []

        def plan(request, context, **options):
            self.transforms.append(options.get("plan_transform"))
            return planned(request, context)

        engine._plan_request = plan
        return engine


def _runtime(tmp_path: Path, monkeypatch, observer):
    env = _environment()
    env.np_random = np.random.default_rng(19)
    path = tmp_path / "robot.urdf"
    path.write_text(
        '<robot name="probe"><joint name="arm_joint" type="revolute"><limit velocity="1"/></joint><joint name="hand_joint" type="revolute"><limit velocity="1"/></joint></robot>'
    )
    factory = _ExecutionFactory(env, path)
    adapter = TaskProgramEnvironmentAdapter(factory, step_dt=_STEP_DT)
    program = adapter.compile(_program())
    captured = []
    original = adapter._create_bridge_with_assembly

    def assembly(*args, **kwargs):
        bridge, runtime = original(*args, **kwargs)
        captured.append((bridge, runtime))
        return bridge, runtime

    monkeypatch.setattr(adapter, "_create_bridge_with_assembly", assembly)
    bridge = adapter.create_bridge(
        program, **({"initial_plan_observer": observer} if observer is not None else {})
    )
    return env, factory, adapter, bridge, captured[0][1], program


@pytest.mark.parametrize("observe", [False, True])
def test_initial_observer_uses_one_actual_complete_plan_before_dispatch(
    tmp_path, monkeypatch, observe
):
    records = []
    env, factory, adapter, bridge, assembly, program = _runtime(
        tmp_path, monkeypatch, records.append if observe else None
    )
    snapshots = []
    sessions = []
    evidence = ExecutionSession.active_plan_proposal_evidence
    snapshot = ActionPlan.snapshot

    def counted_snapshot(self):
        snapshots.append(True)
        return snapshot(self)

    monkeypatch.setattr(ActionPlan, "snapshot", counted_snapshot)
    if not observe:
        monkeypatch.setattr(
            ExecutionSession,
            "active_plan_proposal_evidence",
            lambda *a: pytest.fail("default runtime requested optional diagnostics"),
        )
    else:

        def read_evidence(self):
            sessions.append(self)
            return evidence(self)

        monkeypatch.setattr(
            ExecutionSession, "active_plan_proposal_evidence", read_evidence
        )
    try:
        first = next(program.iter_segments())
        result = assembly.runtime.start(
            tuple(call.call for call in first.calls),
            workflow_id=f"{program.program_id}/{first.segment_id}",
        )
        assert result.current_call_index == 0
        assert len(factory.planned) == 1
        assert factory.transforms == [None]
        assert snapshots == [True]
        assert assembly.command_sink.pending_count == 0
        if observe:
            assert len(records) == 1
            report = records[0]
            assert report["planning_source"] == "execution"
            assert (
                report["invocation_id"] == f"{program.program_id}/{first.segment_id}:0"
            )
            assert report["task_success"] is None
            assert report["plan_success"] == [True, False]
            json.dumps(report, allow_nan=False)
            report["plan_success"][0] = False
            assert evidence(sessions[0])["plan_success"] == [True, False]
            assert factory.planned[0][0].invocation_id == report["invocation_id"]
        else:
            assert records == []
    finally:
        bridge.close_runtime_resources()
        adapter.close()


def test_optional_observer_does_not_change_planning_rng_or_installed_samples(
    tmp_path, monkeypatch
):
    outcomes = []
    for observe in (False, True):
        folder = tmp_path / str(observe)
        folder.mkdir()
        random.seed(37)
        np.random.seed(37)
        torch.manual_seed(37)
        records = []
        env, factory, adapter, bridge, assembly, program = _runtime(
            folder, monkeypatch, records.append if observe else None
        )
        try:
            first = next(program.iter_segments())
            assembly.runtime.start(
                tuple(call.call for call in first.calls),
                workflow_id=f"{program.program_id}/{first.segment_id}",
            )
            outcomes.append(
                (
                    random.getstate(),
                    np.random.get_state(),
                    torch.get_rng_state(),
                    factory.planned[0][0].motion_policy,
                    records,
                )
            )
        finally:
            bridge.close_runtime_resources()
            adapter.close()
    assert outcomes[0][0] == outcomes[1][0]
    assert outcomes[0][1][0] == outcomes[1][1][0]
    np.testing.assert_array_equal(outcomes[0][1][1], outcomes[1][1][1])
    assert outcomes[0][1][2:] == outcomes[1][1][2:]
    assert torch.equal(outcomes[0][2], outcomes[1][2])
    assert outcomes[0][3] == outcomes[1][3]
    assert len(outcomes[1][4]) == 1 and outcomes[0][4] == []


def test_atomic_preview_plan_is_not_published_as_the_actual_initial_plan(
    tmp_path, monkeypatch
):
    records = []
    env, factory, adapter, bridge, assembly, program = _runtime(
        tmp_path, monkeypatch, records.append
    )
    try:
        first = next(program.iter_segments())
        calls = tuple(call.call for call in first.calls)
        workflow_id = f"{program.program_id}/{first.segment_id}"
        workflow = assembly.compiler.analyze(calls, workflow_id=workflow_id)
        context = assembly.observation_provider.observe(
            TaskState(batch_size=2, device="cpu")
        )
        grounded = assembly.compiler.ground(
            workflow, 0, context, invocation_id=f"{workflow_id}:0"
        )
        assembly.engine.plan(grounded.invocation, context)
        assert records == []
        assembly.runtime.start(calls, workflow_id=workflow_id)
        assert len(factory.planned) == 2
        assert len(records) == 1 and records[0]["planning_source"] == "execution"
        assert assembly.command_sink.pending_count == 0
    finally:
        bridge.close_runtime_resources()
        adapter.close()


def test_gym_bridge_selection_forwards_optional_data_observer(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from .test_environment import _FakeDeclarativeEnvironment

    _, factory, adapter, bridge, _, program = _runtime(tmp_path, monkeypatch, None)
    host = _FakeDeclarativeEnvironment(adapter)
    observer = lambda value: None
    create = Mock(return_value=bridge)
    monkeypatch.setattr(adapter, "create_bridge", create)
    try:
        segments = host.create_demo_segments(
            task_program=program, initial_plan_observer=observer
        )
        assert segments is not None and host.active_task_program_bridge is bridge
        assert create.call_args.kwargs["initial_plan_observer"] is observer
        host.create_demo_segments(task_program=program)
        assert "initial_plan_observer" not in create.call_args.kwargs
    finally:
        bridge.close_runtime_resources()
        adapter.close()


def test_observer_output_failure_stops_before_dispatch_without_replanning(
    tmp_path, monkeypatch
):
    records = []

    def fail_output(data):
        records.append(data)
        raise RuntimeError("diagnostic output failed")

    _, factory, adapter, bridge, assembly, program = _runtime(
        tmp_path, monkeypatch, fail_output
    )
    try:
        first = next(program.iter_segments())
        result = assembly.runtime.start(
            tuple(call.call for call in first.calls),
            workflow_id=f"{program.program_id}/{first.segment_id}",
        )
        assert result.status.value == "failed"
        assert len(records) == 1 and len(factory.planned) == 1
        assert factory.transforms == [None]
        assert assembly.command_sink.pending_count == 0
    finally:
        bridge.close_runtime_resources()
        adapter.close()
