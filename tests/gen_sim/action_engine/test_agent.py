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

"""Action Agent compilation, preflight, and report boundary tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import embodichain.gen_sim.action_engine.agent as module
from embodichain.gen_sim.action_engine.agent import ActionAgent
from embodichain.gen_sim.action_engine.capabilities import (
    AtomicCapabilityRegistry,
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import seed_graph_hash
from embodichain.gen_sim.action_engine.runtime import ExecutionResult
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph

from .task_fixtures import make_task_spec


def _bindings(requirements: dict) -> dict[str, str]:
    return {
        item["role_id"]: f"scene_{item['role_id']}" for item in requirements["objects"]
    }


def _task_of_type(task_type: str) -> tuple[dict, dict]:
    return make_task_spec(task_type)


def _registry_with_planning_only(action_name: str) -> AtomicCapabilityRegistry:
    source = build_atomic_capability_registry()
    registry = AtomicCapabilityRegistry()
    for name in source.names():
        capability = source.get(name)
        if name == action_name:
            capability = replace(
                capability,
                runtime_available=False,
                unavailable_reason="test-only unavailable runtime",
            )
        registry.register(capability)
    return registry


def test_plan_hash_matches_direct_seed_graph_instantiation(monkeypatch) -> None:
    task, requirements = make_task_spec("E1")
    bindings = _bindings(requirements)
    grounded_plan = {
        "task_spec": task,
        "role_bindings": {"role_bindings": bindings},
    }
    monkeypatch.setattr(
        module,
        "_validate_grounded_plan",
        lambda value: dict(value),
    )

    graph = ActionAgent().plan(grounded_plan)
    direct = instantiate_seed_graph(task, bindings)

    assert seed_graph_hash(graph) == seed_graph_hash(direct)


def test_planning_only_graph_is_rejected_before_executor_construction() -> None:
    task, requirements = _task_of_type("E6")
    bindings = _bindings(requirements)
    registry = _registry_with_planning_only("PullArticulatedPart")
    graph = instantiate_seed_graph(task, bindings, registry=registry)
    constructed = False

    def executor_factory(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("preflight must reject before executor construction")

    report = ActionAgent(
        registry=registry,
        executor_factory=executor_factory,
    ).execute(
        graph,
        SimpleNamespace(num_envs=2),
        known_uids=set(bindings.values()),
        run_id="preflight-test",
    )

    assert report.status == "rejected"
    assert report.action_count == 0
    assert "planning-only" in (report.error or "")
    assert not constructed


def test_execution_report_is_strictly_json_serializable(tmp_path: Path) -> None:
    task, requirements = make_task_spec("E1")
    bindings = _bindings(requirements)
    graph = instantiate_seed_graph(task, bindings)

    class FakeExecutor:
        def __init__(self, program, env, **kwargs) -> None:
            self.program = program
            self.env = env

        def run(self, **kwargs) -> ExecutionResult:
            return ExecutionResult(
                actions=[torch.ones((2, 3), dtype=torch.float32)],
                success=torch.tensor([True, False]),
                semantic_success={
                    "task_01": torch.tensor([True, False]),
                },
                record_dir=str(tmp_path),
                retry_count=1,
                retry_counts=[0, 1],
                failure_events=[
                    {
                        "failure_type": "plan_failed",
                        "env_ids": torch.tensor([1]),
                    }
                ],
            )

    report = ActionAgent(executor_factory=FakeExecutor).execute(
        graph,
        SimpleNamespace(num_envs=2),
        known_uids=set(bindings.values()),
        run_id="json-test",
    )
    payload = report.as_mapping()

    assert report.status == "failed"
    assert payload["environments"][0]["semantic_success"] == {"task_01": True}
    assert payload["environments"][1]["semantic_success"] == {"task_01": False}
    assert [item["retry_count"] for item in payload["environments"]] == [0, 1]
    assert "actions" not in payload
    json.dumps(payload, allow_nan=False)
    assert (
        json.loads((tmp_path / "execution_report.json").read_text(encoding="utf-8"))
        == payload
    )
    trajectory = torch.load(tmp_path / "executed_trajectory.pt", weights_only=True)
    assert torch.equal(trajectory["actions"][0], torch.ones((2, 3)))
    trajectory_manifest = json.loads(
        (tmp_path / "executed_trajectory.json").read_text(encoding="utf-8")
    )
    assert trajectory_manifest["actions"][0]["shape"] == [2, 3]


def test_existing_execution_result_can_be_reported_without_reexecution() -> None:
    task, requirements = make_task_spec("E1")
    bindings = _bindings(requirements)
    graph = instantiate_seed_graph(task, bindings)
    result = ExecutionResult(
        actions=[torch.zeros((1, 2), dtype=torch.float32)],
        success=torch.tensor([True]),
        semantic_success={"task_01": torch.tensor([True])},
    )

    report = ActionAgent().report_execution_result(
        result,
        action_graph=graph,
        run_id="legacy-run",
        episode_index=3,
    )

    assert report.status == "succeeded"
    assert report.episode_id == "3"
    assert report.action_count == 1


def test_runtime_exception_is_reported_as_aborted() -> None:
    task, requirements = make_task_spec("E1")
    bindings = _bindings(requirements)
    graph = instantiate_seed_graph(task, bindings)

    def fail_executor(*_args, **_kwargs):
        raise RuntimeError("simulator stopped")

    report = ActionAgent(executor_factory=fail_executor).execute(
        graph,
        SimpleNamespace(num_envs=1),
        known_uids=set(bindings.values()),
        run_id="aborted-test",
    )

    assert report.status == "aborted"
    assert report.action_count == 0
    assert report.error == "RuntimeError: simulator stopped"


def test_preflight_raises_for_planning_only_graph() -> None:
    task, requirements = _task_of_type("E8")
    bindings = _bindings(requirements)
    registry = _registry_with_planning_only("TurnKnob")

    with pytest.raises(ValueError, match="planning-only"):
        ActionAgent(registry=registry).preflight(
            instantiate_seed_graph(task, bindings, registry=registry),
            known_uids=set(bindings.values()),
        )
