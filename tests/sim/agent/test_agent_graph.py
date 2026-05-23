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

import ast
import hashlib
import json
import sys
import types
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def extract_json_object(content):
    if isinstance(content, str):
        return json.loads(content)
    return dict(content)


def normalize_json_content(content):
    return json.dumps(extract_json_object(content), indent=2)


def stable_json_hash(content):
    payload = json.dumps(
        content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_compile_agent_namespace():
    source_path = (
        REPO_ROOT / "embodichain" / "agents" / "hierarchy" / "compile_agent.py"
    )
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    nodes = [
        node
        for node in module.body
        if (isinstance(node, ast.ClassDef) and node.name == "CompileAgent")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "COMPILED_GRAPH_SCHEMA_VERSION"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name
            in {
                "_canonicalize_recovery_spec_with_llm",
                "_build_runtime_recovery_planner",
                "_call_runtime_recovery_agent",
                "_heuristic_runtime_recovery_edges",
                "_heuristic_runtime_recovery_spec",
                "_infer_recovery_object_name",
                "_default_recovery_robot_for_object",
                "_collect_runtime_state",
                "_sanitize_runtime_recovery_spec",
                "_sanitize_runtime_recovery_step",
                "_runtime_recovery_signature",
                "_runtime_recovery_repeats_strategy",
                "_runtime_recovery_step_type",
                "_first_runtime_branch_edges",
                "_task_graph_with_runtime_edge",
                "_edge_to_dict",
                "_action_to_jsonable",
                "_monitor_from_failure_context",
                "_jsonable",
                "_empty_recovery_spec",
                "_runtime_kwargs",
                "_stable_json_hash",
            }
        )
    ]
    namespace = {
        "AgentBase": object,
        "Any": Any,
        "hashlib": hashlib,
        "json": json,
        "Path": Path,
        "partial": partial,
        "deepcopy": deepcopy,
        "database_agent_prompt_dir": Path("/tmp"),
        "extract_json_object": extract_json_object,
        "normalize_json_content": normalize_json_content,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace


def _load_compile_agent_class():
    namespace = _load_compile_agent_namespace()
    return namespace["CompileAgent"]


def _load_agent_graph_namespace(fake_execute):
    source_path = REPO_ROOT / "embodichain" / "lab" / "sim" / "agent" / "agent_graph.py"
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    body = [
        node
        for node in module.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "embodichain.lab.sim.agent.edge_action_executor"
        )
    ]

    class FakeEdgeActionExecutor:
        def execute(self, *, edge, **kwargs):
            return fake_execute(
                left_arm_action=edge.left_arm_action,
                right_arm_action=edge.right_arm_action,
                monitor_sequences=edge.monitor_sequences,
                **kwargs,
            )

    namespace = {
        "defaultdict": defaultdict,
        "dataclass": dataclass,
        "field": field,
        "Any": Any,
        "EdgeActionExecutor": FakeEdgeActionExecutor,
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace


def _load_generate_and_execute_action_list():
    source_path = REPO_ROOT / "embodichain" / "lab" / "scripts" / "run_env.py"
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    function_node = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "generate_and_execute_action_list"
    )
    namespace = {
        "tqdm": types.SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable),
        "log_warning": lambda *args, **kwargs: None,
        "log_info": lambda *args, **kwargs: None,
    }
    exec(
        compile(
            ast.Module(body=[function_node], type_ignores=[]), str(source_path), "exec"
        ),
        namespace,
    )
    return namespace["generate_and_execute_action_list"]


def test_agent_graph_can_recover_multiple_nominal_edges(monkeypatch) -> None:
    calls = []

    def fake_execute(right_arm_action=None, return_result=False, **kwargs):
        calls.append(right_arm_action)
        monitor_index = 0 if right_arm_action in {"fail_1", "fail_2"} else None
        return {
            "actions": [right_arm_action],
            "monitor_index": monitor_index,
            "monitor_name": "triggered" if monitor_index is not None else None,
            "step_index": 0 if monitor_index is not None else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]

    graph = agent_task_graph(start="v0_start", goal="v2_done")
    graph.add_node("v0_start")
    graph.add_node("v1_ready")
    graph.add_node("v2_done")
    graph.add_edge("e01", "v0_start", "v1_ready", right_arm_action="fail_1")
    graph.add_edge("e12", "v1_ready", "v2_done", right_arm_action="fail_2")
    graph.add_edge(
        "r01", "v0_start", "v1_ready", right_arm_action="recover_1", is_recovery=True
    )
    graph.add_edge(
        "r12", "v1_ready", "v2_done", right_arm_action="recover_2", is_recovery=True
    )
    graph.add_recovery("e01", monitor_index=0, recovery_edges=["r01"])
    graph.add_recovery("e12", monitor_index=0, recovery_edges=["r12"])

    actions = graph.run(env=object())

    assert actions.already_executed is True
    assert actions.actions == ["fail_1", "recover_1", "fail_2", "recover_2"]
    assert calls == ["fail_1", "recover_1", "fail_2", "recover_2"]


def test_agent_graph_can_use_runtime_recovery_planner() -> None:
    calls = []
    triggered = {"value": False}

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        calls.append(right_arm_action)
        monitor_index = (
            0
            if right_arm_action == "fail_nominal"
            and monitor_sequences
            and not triggered["value"]
            else None
        )
        if monitor_index is not None:
            triggered["value"] = True
        return {
            "actions": [right_arm_action],
            "monitor_index": monitor_index,
            "monitor_name": (
                "monitor_object_fallen" if monitor_index is not None else None
            ),
            "step_index": 0 if monitor_index is not None else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]
    agent_graph_edge = graph_namespace["AgentGraphEdge"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )

    def runtime_planner(edge, monitor_index, monitor_name, **kwargs):
        assert edge.id == "e01"
        assert monitor_index == 0
        assert monitor_name == "monitor_object_fallen"
        return [
            agent_graph_edge(
                id="rt01",
                source="v0_start",
                target="v0_start",
                right_arm_action="runtime_upright",
                is_recovery=True,
            )
        ]

    actions = graph.run(
        env=object(),
        runtime_recovery_planner=runtime_planner,
    )

    assert actions.actions == ["fail_nominal", "runtime_upright", "fail_nominal"]
    assert calls == ["fail_nominal", "runtime_upright", "fail_nominal"]


def test_agent_graph_prefers_runtime_recovery_over_static_branch() -> None:
    calls = []
    captured_contexts = []
    triggered = {"value": False}

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        calls.append(right_arm_action)
        monitor_index = (
            0
            if right_arm_action == "fail_nominal"
            and monitor_sequences
            and not triggered["value"]
            else None
        )
        if monitor_index is not None:
            triggered["value"] = True
        return {
            "actions": [right_arm_action],
            "monitor_index": monitor_index,
            "monitor_name": (
                "monitor_object_held" if monitor_index is not None else None
            ),
            "step_index": 3 if monitor_index is not None else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]
    agent_graph_edge = graph_namespace["AgentGraphEdge"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )
    graph.add_edge(
        "r_static",
        "v0_start",
        "v1_done",
        right_arm_action="static_recovery",
        is_recovery=True,
    )
    graph.add_recovery("e01", monitor_index=0, recovery_edges=["r_static"])

    def runtime_planner(**kwargs):
        captured_contexts.append(kwargs)
        return [
            agent_graph_edge(
                id="rt01",
                source="v0_start",
                target="v0_start",
                right_arm_action="runtime_recovery",
                is_recovery=True,
            )
        ]

    actions = graph.run(
        env=object(),
        runtime_recovery_planner=runtime_planner,
        prefer_runtime_llm_recovery=True,
    )

    assert actions.actions == ["fail_nominal", "runtime_recovery", "fail_nominal"]
    assert calls == ["fail_nominal", "runtime_recovery", "fail_nominal"]
    assert captured_contexts[0]["failure_context"] == {
        "edge_id": "e01",
        "origin_edge_id": "e01",
        "edge_source": "v0_start",
        "edge_target": "v1_done",
        "edge_is_recovery": False,
        "monitor_index": 0,
        "monitor_name": "monitor_object_held",
        "step_index": 3,
        "failure_class": "monitor:monitor_object_held:0",
        "failure_reason": "Monitor 'monitor_object_held' triggered.",
    }
    assert captured_contexts[0]["recovery_history"] == []


def test_agent_graph_passes_recovery_history_to_runtime_planner() -> None:
    calls = []
    planner_calls = []
    nominal_failures = 0
    runtime_failures = 0

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        nonlocal nominal_failures, runtime_failures

        calls.append(right_arm_action)
        monitor_index = None
        if right_arm_action == "fail_nominal" and monitor_sequences:
            nominal_failures += 1
            monitor_index = 0 if nominal_failures == 1 else None
        elif right_arm_action == "runtime_recovery" and monitor_sequences:
            runtime_failures += 1
            monitor_index = 0 if runtime_failures == 1 else None
        return {
            "actions": [right_arm_action],
            "monitor_index": monitor_index,
            "monitor_name": (
                "monitor_object_held" if monitor_index is not None else None
            ),
            "step_index": 0 if monitor_index is not None else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]
    agent_graph_edge = graph_namespace["AgentGraphEdge"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )

    def runtime_planner(edge, recovery_history, **kwargs):
        planner_calls.append(
            {
                "edge_id": edge.id,
                "failure_context": kwargs["failure_context"],
                "recovery_history": list(recovery_history),
            }
        )
        if edge.id == "e01":
            return [
                agent_graph_edge(
                    id="rt01",
                    source="v0_start",
                    target="v0_start",
                    right_arm_action="runtime_recovery",
                    monitor_sequences=[["monitor"]],
                    is_recovery=True,
                )
            ]
        return [
            agent_graph_edge(
                id="rt_fix",
                source="v0_start",
                target="v0_start",
                right_arm_action="runtime_fix",
                is_recovery=True,
            )
        ]

    actions = graph.run(
        env=object(),
        runtime_recovery_planner=runtime_planner,
        prefer_runtime_llm_recovery=True,
    )

    assert actions.actions == [
        "fail_nominal",
        "runtime_recovery",
        "runtime_fix",
        "fail_nominal",
    ]
    assert planner_calls[0]["edge_id"] == "e01"
    assert planner_calls[0]["recovery_history"] == []
    assert planner_calls[0]["failure_context"]["origin_edge_id"] == "e01"
    assert planner_calls[0]["failure_context"]["failure_class"] == (
        "monitor:monitor_object_held:0"
    )
    assert planner_calls[1]["edge_id"] == "rt01"
    assert planner_calls[1]["failure_context"]["origin_edge_id"] == "e01"
    assert planner_calls[1]["failure_context"]["failure_class"] == (
        "monitor:monitor_object_held:0"
    )
    assert planner_calls[1]["recovery_history"][0]["strategy"] == "runtime"
    assert planner_calls[1]["recovery_history"][0]["edge_id"] == "e01"
    assert planner_calls[1]["recovery_history"][0]["origin_edge_id"] == "e01"
    assert planner_calls[1]["recovery_history"][0]["attempt_key"] == (
        "e01:monitor:monitor_object_held:0"
    )
    assert planner_calls[1]["recovery_history"][0]["recovery_edge_ids"] == ["rt01"]


def test_agent_graph_falls_back_to_static_when_runtime_returns_no_edges() -> None:
    calls = []

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        calls.append(right_arm_action)
        return {
            "actions": [right_arm_action],
            "monitor_index": 0 if right_arm_action == "fail_nominal" else None,
            "monitor_name": (
                "monitor_object_held" if right_arm_action == "fail_nominal" else None
            ),
            "step_index": 0 if right_arm_action == "fail_nominal" else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )
    graph.add_edge(
        "r_static",
        "v0_start",
        "v1_done",
        right_arm_action="static_recovery",
        is_recovery=True,
    )
    graph.add_recovery("e01", monitor_index=0, recovery_edges=["r_static"])

    actions = graph.run(
        env=object(),
        runtime_recovery_planner=lambda **kwargs: None,
        prefer_runtime_llm_recovery=True,
    )

    assert actions.actions == ["fail_nominal", "static_recovery"]
    assert calls == ["fail_nominal", "static_recovery"]


def test_agent_graph_limits_total_runtime_recovery_attempts() -> None:
    calls = []

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        calls.append(right_arm_action)
        return {
            "actions": [right_arm_action],
            "monitor_index": 0 if monitor_sequences else None,
            "monitor_name": "monitor_object_fallen" if monitor_sequences else None,
            "step_index": 0 if monitor_sequences else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]
    agent_graph_edge = graph_namespace["AgentGraphEdge"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )

    def runtime_planner(edge, **kwargs):
        return [
            agent_graph_edge(
                id=f"rt_{edge.id}",
                source=edge.source,
                target=edge.source,
                right_arm_action="runtime_upright",
                monitor_sequences=[["monitor"]],
                is_recovery=True,
            )
        ]

    with pytest.raises(RuntimeError, match="Runtime recovery exceeded total retry"):
        graph.run(
            env=object(),
            runtime_recovery_planner=runtime_planner,
            runtime_recovery_max_total_attempts=1,
        )

    assert calls == ["fail_nominal", "runtime_upright"]


def test_agent_graph_limits_runtime_recovery_by_origin_failure_class() -> None:
    calls = []

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        calls.append(right_arm_action)
        return {
            "actions": [right_arm_action],
            "monitor_index": 0 if monitor_sequences else None,
            "monitor_name": "monitor_object_held" if monitor_sequences else None,
            "step_index": 0 if monitor_sequences else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]
    agent_graph_edge = graph_namespace["AgentGraphEdge"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )

    def runtime_planner(edge, **kwargs):
        return [
            agent_graph_edge(
                id=f"rt_{edge.id}",
                source=edge.source,
                target=edge.source,
                right_arm_action="runtime_upright",
                monitor_sequences=[["monitor"]],
                is_recovery=True,
            )
        ]

    with pytest.raises(
        RuntimeError,
        match="e01:monitor:monitor_object_held:0",
    ):
        graph.run(
            env=object(),
            runtime_recovery_planner=runtime_planner,
            runtime_recovery_max_monitor_attempts=1,
        )

    assert calls == ["fail_nominal", "runtime_upright"]


def test_agent_graph_recovers_inside_recovery_path_and_resumes_continuation() -> None:
    calls = []

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        calls.append(right_arm_action)
        monitor_index = (
            0
            if right_arm_action in {"fail_nominal", "fail_recovery"}
            and monitor_sequences
            else None
        )
        return {
            "actions": [right_arm_action],
            "monitor_index": monitor_index,
            "monitor_name": "triggered" if monitor_index is not None else None,
            "step_index": 0 if monitor_index is not None else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]

    graph = agent_task_graph(start="v0_start", goal="v2_done")
    graph.add_node("v0_start")
    graph.add_node("rn_recovery_started")
    graph.add_node("rn_recovery_checked")
    graph.add_node("v1_ready")
    graph.add_node("v2_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_ready",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )
    graph.add_edge("e12", "v1_ready", "v2_done", right_arm_action="finish_nominal")
    graph.add_edge(
        "r01_start",
        "v0_start",
        "rn_recovery_started",
        right_arm_action="start_recovery",
        is_recovery=True,
    )
    graph.add_edge(
        "r01_checked",
        "rn_recovery_started",
        "rn_recovery_checked",
        right_arm_action="fail_recovery",
        monitor_sequences=[["monitor"]],
        is_recovery=True,
    )
    graph.add_edge(
        "r01_finish",
        "rn_recovery_checked",
        "v1_ready",
        right_arm_action="finish_recovery",
        is_recovery=True,
    )
    graph.add_edge(
        "r01_fix_checked",
        "rn_recovery_started",
        "rn_recovery_checked",
        right_arm_action="fix_recovery",
        is_recovery=True,
    )
    graph.add_recovery(
        "e01",
        monitor_index=0,
        recovery_edges=["r01_start", "r01_checked", "r01_finish"],
    )
    graph.add_recovery(
        "r01_checked", monitor_index=0, recovery_edges=["r01_fix_checked"]
    )

    actions = graph.run(env=object())

    assert actions.actions == [
        "fail_nominal",
        "start_recovery",
        "fail_recovery",
        "fix_recovery",
        "finish_recovery",
        "finish_nominal",
    ]
    assert calls == actions.actions


def test_agent_graph_reuses_recovery_edge_until_it_succeeds() -> None:
    calls = []
    recovery_attempts = 0

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        nonlocal recovery_attempts

        calls.append(right_arm_action)
        monitor_index = None
        if right_arm_action == "fail_nominal":
            monitor_index = 0
        elif right_arm_action == "reusable_recovery":
            recovery_attempts += 1
            if recovery_attempts < 3:
                monitor_index = 0

        return {
            "actions": [right_arm_action],
            "monitor_index": monitor_index if monitor_sequences else None,
            "monitor_name": "triggered" if monitor_index is not None else None,
            "step_index": 0 if monitor_index is not None else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]

    graph = agent_task_graph(start="v0_start", goal="v2_done")
    graph.add_node("v0_start")
    graph.add_node("v1_ready")
    graph.add_node("v2_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_ready",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )
    graph.add_edge("e12", "v1_ready", "v2_done", right_arm_action="finish")
    graph.add_edge(
        "r01_reusable",
        "v0_start",
        "v1_ready",
        right_arm_action="reusable_recovery",
        monitor_sequences=[["monitor"]],
        is_recovery=True,
    )
    graph.add_recovery("e01", monitor_index=0, recovery_edges=["r01_reusable"])
    graph.add_recovery("r01_reusable", monitor_index=0, recovery_edges=["r01_reusable"])

    actions = graph.run(env=object())

    assert actions.actions == [
        "fail_nominal",
        "reusable_recovery",
        "reusable_recovery",
        "reusable_recovery",
        "finish",
    ]
    assert calls == actions.actions


def test_agent_graph_limits_static_recovery_monitor_attempts() -> None:
    calls = []

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        calls.append(right_arm_action)
        return {
            "actions": [right_arm_action],
            "monitor_index": 0 if monitor_sequences else None,
            "monitor_name": "triggered" if monitor_sequences else None,
            "step_index": 0 if monitor_sequences else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )
    graph.add_edge(
        "r01_reusable",
        "v0_start",
        "v1_done",
        right_arm_action="reusable_recovery",
        monitor_sequences=[["monitor"]],
        is_recovery=True,
    )
    graph.add_recovery("e01", monitor_index=0, recovery_edges=["r01_reusable"])
    graph.add_recovery("r01_reusable", monitor_index=0, recovery_edges=["r01_reusable"])

    with pytest.raises(RuntimeError, match="Static recovery exceeded monitor retry"):
        graph.run(env=object(), recovery_max_monitor_attempts=1)

    assert calls == ["fail_nominal", "reusable_recovery", "reusable_recovery"]


def test_agent_graph_limits_static_recovery_total_attempts() -> None:
    calls = []

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        calls.append(right_arm_action)
        return {
            "actions": [right_arm_action],
            "monitor_index": 0 if monitor_sequences else None,
            "monitor_name": "triggered" if monitor_sequences else None,
            "step_index": 0 if monitor_sequences else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )
    graph.add_edge(
        "r01_reusable",
        "v0_start",
        "v1_done",
        right_arm_action="reusable_recovery",
        monitor_sequences=[["monitor"]],
        is_recovery=True,
    )
    graph.add_recovery("e01", monitor_index=0, recovery_edges=["r01_reusable"])
    graph.add_recovery("r01_reusable", monitor_index=0, recovery_edges=["r01_reusable"])

    with pytest.raises(RuntimeError, match="Static recovery exceeded total retry"):
        graph.run(env=object(), recovery_max_total_attempts=2)

    assert calls == ["fail_nominal", "reusable_recovery", "reusable_recovery"]


def test_agent_graph_self_loop_recovery_retries_without_duplicate_success() -> None:
    calls = []
    self_loop_attempts = 0

    def fake_execute(right_arm_action=None, monitor_sequences=None, **kwargs):
        nonlocal self_loop_attempts

        calls.append(right_arm_action)
        monitor_index = None
        if right_arm_action == "fail_nominal":
            monitor_index = 0
        elif right_arm_action == "self_loop_recovery":
            self_loop_attempts += 1
            if self_loop_attempts == 1:
                monitor_index = 0

        return {
            "actions": [right_arm_action],
            "monitor_index": monitor_index if monitor_sequences else None,
            "monitor_name": "triggered" if monitor_index is not None else None,
            "step_index": 0 if monitor_index is not None else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_execute)
    agent_task_graph = graph_namespace["AgentTaskGraph"]

    graph = agent_task_graph(start="v0_start", goal="v1_done")
    graph.add_node("v0_start")
    graph.add_node("v1_done")
    graph.add_edge(
        "e01",
        "v0_start",
        "v1_done",
        right_arm_action="fail_nominal",
        monitor_sequences=[["monitor"]],
    )
    graph.add_edge(
        "r01_restore",
        "v0_start",
        "v0_start",
        right_arm_action="self_loop_recovery",
        monitor_sequences=[["monitor"]],
        is_recovery=True,
    )
    graph.add_edge(
        "r01_finish",
        "v0_start",
        "v1_done",
        right_arm_action="finish_recovery",
        is_recovery=True,
    )
    graph.add_recovery(
        "e01", monitor_index=0, recovery_edges=["r01_restore", "r01_finish"]
    )
    graph.add_recovery("r01_restore", monitor_index=0, recovery_edges=["r01_restore"])

    actions = graph.run(env=object())

    assert actions.actions == [
        "fail_nominal",
        "self_loop_recovery",
        "self_loop_recovery",
        "finish_recovery",
    ]
    assert calls == actions.actions


def test_compile_agent_generate_writes_compiled_graph_bundle(tmp_path: Path) -> None:
    compile_agent_cls = _load_compile_agent_class()
    agent = object.__new__(compile_agent_cls)
    agent.prompt_name = "compile_agent_graph"
    agent.task_name = "Task"

    task_graph = {
        "task": "grasp",
        "start": "v0_start",
        "goal": "v1_grasped",
        "nodes": [
            {"id": "v0_start", "semantic": "Cup is on table."},
            {"id": "v1_grasped", "semantic": "Cup is grasped."},
        ],
        "edges": [
            {
                "id": "e01_grasp",
                "source": "v0_start",
                "target": "v1_grasped",
                "left_arm_action": {
                    "fn": "grasp",
                    "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                },
                "right_arm_action": None,
            }
        ],
    }
    recovery_spec = {
        "task": "grasp",
        "recovery_bindings": [
            {
                "edge_id": "e01_grasp",
                "failure_name": "cup_moved",
                "monitors": [
                    {"type": "object_moved", "objects": ["cup"], "threshold": 0.02}
                ],
                "recovery": [
                    {
                        "type": "regrasp",
                        "robot_name": "left_arm",
                        "obj_name": "cup",
                        "pre_grasp_dis": 0.1,
                        "force_valid": True,
                    }
                ],
                "merge": "target",
                "repeat_until_success": True,
            }
        ],
    }

    path, _, content = agent.generate(
        log_dir=tmp_path,
        regenerate=True,
        task_graph=task_graph,
        recovery_spec=recovery_spec,
        recovery_enabled=True,
    )

    bundle = json.loads(content)

    assert path.name == "agent_compiled_graph.json"
    assert bundle["task_graph"] == task_graph
    assert bundle["recovery_spec"] == recovery_spec
    assert bundle["metadata"] == {
        "recovery_enabled": True,
        "schema_version": "recovery_bindings_atomic_v3",
        "task_graph_hash": stable_json_hash(task_graph),
        "raw_recovery_spec_hash": stable_json_hash(recovery_spec),
        "recovery_spec_hash": stable_json_hash(recovery_spec),
    }
    assert bundle["recovery_graph"]["recovery_nodes"] == []
    assert [edge["id"] for edge in bundle["recovery_graph"]["recovery_edges"]] == [
        "re_e01_grasp_1_regrasp_cup"
    ]
    assert [
        branch["edge_id"] for branch in bundle["recovery_graph"]["recovery_branches"]
    ] == ["e01_grasp", "re_e01_grasp_1_regrasp_cup"]
    assert json.loads(path.read_text()) == bundle


def test_compile_agent_uses_one_llm_call_for_ambiguous_recovery_spec(
    tmp_path: Path,
) -> None:
    compile_agent_cls = _load_compile_agent_class()

    task_graph = {
        "task": "grasp",
        "start": "v0_start",
        "goal": "v1_grasped",
        "nodes": [
            {"id": "v0_start", "semantic": "Cup is on table."},
            {"id": "v1_grasped", "semantic": "Cup is grasped."},
        ],
        "edges": [
            {
                "id": "e01_grasp",
                "source": "v0_start",
                "target": "v1_grasped",
                "left_arm_action": {
                    "fn": "grasp",
                    "kwargs": {"robot_name": "left_arm", "obj_name": "cup"},
                },
                "right_arm_action": None,
            }
        ],
    }
    raw_recovery_spec = {
        "task": "grasp",
        "recovery_bindings": [
            {
                "edge_id": "e01_grasp",
                "failure_name": "cup_moved",
                "monitor_intent": "detect when the cup moves during grasping",
                "recovery_intent": "grasp the cup again",
            }
        ],
    }
    canonical_recovery_spec = {
        "task": "grasp",
        "recovery_bindings": [
            {
                "edge_id": "e01_grasp",
                "failure_name": "cup_moved",
                "monitors": [{"type": "object_moved", "objects": ["cup"]}],
                "recovery": [{"type": "regrasp"}],
                "merge": "target",
                "repeat_until_success": True,
            }
        ],
    }

    class FakeLLM:
        def __init__(self) -> None:
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            return types.SimpleNamespace(content=json.dumps(canonical_recovery_spec))

    fake_llm = FakeLLM()
    agent = object.__new__(compile_agent_cls)
    agent.prompt_name = "compile_agent_graph"
    agent.task_name = "Task"
    agent.llm = fake_llm

    _, _, content = agent.generate(
        log_dir=tmp_path,
        regenerate=True,
        task_graph=task_graph,
        recovery_spec=raw_recovery_spec,
        recovery_enabled=True,
    )

    bundle = json.loads(content)

    assert len(fake_llm.calls) == 1
    assert bundle["recovery_spec"]["recovery_bindings"][0]["monitors"] == [
        {"type": "object_moved", "objects": ["cup"], "threshold": 0.02}
    ]
    assert bundle["recovery_spec"]["recovery_bindings"][0]["recovery"] == [
        {
            "type": "regrasp",
            "robot_name": "left_arm",
            "obj_name": "cup",
            "pre_grasp_dis": 0.1,
            "force_valid": False,
        }
    ]
    assert bundle["metadata"]["raw_recovery_spec_hash"] == stable_json_hash(
        raw_recovery_spec
    )
    assert bundle["metadata"]["recovery_spec_hash"] == stable_json_hash(
        bundle["recovery_spec"]
    )


def test_compile_agent_executes_compiled_json_graph(
    tmp_path: Path, monkeypatch
) -> None:
    code_path = tmp_path / "agent_compiled_graph.json"
    code_path.write_text('{"task_graph": {}, "recovery_graph": {}}')
    env = object()
    calls = []

    class FakeGraph:
        def run(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(already_executed=True, actions=["done"])

    def compile_agent_graph_from_file(path, env=None):
        assert path == code_path
        assert env is not None
        return FakeGraph()

    lab_module = types.ModuleType("embodichain.lab")
    sim_module = types.ModuleType("embodichain.lab.sim")
    agent_module = types.ModuleType("embodichain.lab.sim.agent")
    graph_spec_module = types.ModuleType("embodichain.lab.sim.agent.graph_spec")

    lab_module.__path__ = []
    sim_module.__path__ = []
    agent_module.__path__ = []
    graph_spec_module.compile_agent_graph_from_file = compile_agent_graph_from_file

    monkeypatch.setitem(sys.modules, "embodichain.lab", lab_module)
    monkeypatch.setitem(sys.modules, "embodichain.lab.sim", sim_module)
    monkeypatch.setitem(sys.modules, "embodichain.lab.sim.agent", agent_module)
    monkeypatch.setitem(
        sys.modules, "embodichain.lab.sim.agent.graph_spec", graph_spec_module
    )

    compile_agent_cls = _load_compile_agent_class()
    agent = object.__new__(compile_agent_cls)
    agent.prompt_kwargs = {"task_graph": {"content": "prompt only"}}

    actions = agent.act(code_path, env=env, task_graph="prompt only")

    assert actions.already_executed is True
    assert actions.actions == ["done"]
    assert calls == [{"env": env}]


def test_compile_agent_act_wires_runtime_recovery_agent(
    tmp_path: Path, monkeypatch
) -> None:
    code_path = tmp_path / "agent_compiled_graph.json"
    code_path.write_text(
        json.dumps(
            {
                "task_graph": {
                    "task": "runtime",
                    "start": "v0_start",
                    "goal": "v1_done",
                    "nodes": [],
                    "edges": [],
                },
                "recovery_graph": {},
            }
        )
    )
    env = object()
    calls = []

    class FakeGraph:
        def run(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(already_executed=True, actions=["done"])

    def compile_agent_graph_from_file(path, env=None):
        assert path == code_path
        assert env is not None
        return FakeGraph()

    lab_module = types.ModuleType("embodichain.lab")
    sim_module = types.ModuleType("embodichain.lab.sim")
    agent_module = types.ModuleType("embodichain.lab.sim.agent")
    graph_spec_module = types.ModuleType("embodichain.lab.sim.agent.graph_spec")

    lab_module.__path__ = []
    sim_module.__path__ = []
    agent_module.__path__ = []
    graph_spec_module.compile_agent_graph_from_file = compile_agent_graph_from_file

    monkeypatch.setitem(sys.modules, "embodichain.lab", lab_module)
    monkeypatch.setitem(sys.modules, "embodichain.lab.sim", sim_module)
    monkeypatch.setitem(sys.modules, "embodichain.lab.sim.agent", agent_module)
    monkeypatch.setitem(
        sys.modules, "embodichain.lab.sim.agent.graph_spec", graph_spec_module
    )

    compile_agent_cls = _load_compile_agent_class()
    agent = object.__new__(compile_agent_cls)
    agent.prompt_kwargs = {}

    actions = agent.act(
        code_path,
        env=env,
        runtime_llm_recovery=True,
        runtime_recovery_agent=object(),
        prefer_runtime_llm_recovery=False,
    )

    assert actions.already_executed is True
    assert actions.actions == ["done"]
    assert "runtime_recovery_agent" not in calls[0]
    assert callable(calls[0]["runtime_recovery_planner"])
    assert callable(calls[0]["runtime_state_collector"])
    assert calls[0]["prefer_runtime_llm_recovery"] is True


def test_runtime_recovery_normalizer_expands_regrasp_both() -> None:
    namespace = _load_compile_agent_namespace()
    sanitize_runtime_recovery_spec = namespace["_sanitize_runtime_recovery_spec"]

    normalized = sanitize_runtime_recovery_spec(
        {
            "task": "runtime",
            "recovery_bindings": [
                {
                    "edge_id": "e02",
                    "failure_name": "runtime_repair",
                    "monitors": [
                        {
                            "type": "hold_lost",
                            "robot_name": "left_arm",
                            "obj_name": "fork",
                        }
                    ],
                    "recovery": [
                        {
                            "type": "regrasp_both",
                            "arms": {
                                "right_arm": "spoon",
                                "left_arm": "fork",
                            },
                            "pre_grasp_dis": 0.12,
                            "force_valid": True,
                        }
                    ],
                    "merge": "target",
                }
            ],
        },
        current_edge_id="e02",
    )

    assert normalized is not None
    binding = normalized["recovery_bindings"][0]
    assert binding["merge"] == "source"
    assert binding["recovery"] == [
        {
            "type": "regrasp",
            "robot_name": "left_arm",
            "obj_name": "fork",
            "pre_grasp_dis": 0.12,
            "force_valid": True,
        },
        {
            "type": "regrasp",
            "robot_name": "right_arm",
            "obj_name": "spoon",
            "pre_grasp_dis": 0.12,
            "force_valid": True,
        },
    ]


def test_runtime_recovery_normalizer_resolves_replay_edge() -> None:
    namespace = _load_compile_agent_namespace()
    sanitize_runtime_recovery_spec = namespace["_sanitize_runtime_recovery_spec"]

    nominal_edges = {
        "e01": {
            "id": "e01",
            "source": "v0_start",
            "target": "v1_mid",
            "left_arm_action": {
                "kind": "atomic_action",
                "name": "move",
                "cfg_class": "MoveActionCfg",
                "cfg": {"control_part": "left_arm"},
                "target": {
                    "kind": "eef_orientation",
                    "direction": "down",
                },
                "runtime_kwargs": {},
            },
            "right_arm_action": None,
        }
    }

    normalized = sanitize_runtime_recovery_spec(
        {
            "task": "runtime",
            "recovery_bindings": [
                {
                    "edge_id": "e02",
                    "monitors": [
                        {
                            "type": "hold_lost",
                            "robot_name": "left_arm",
                            "obj_name": "fork",
                        }
                    ],
                    "recovery": [{"type": "replay_edge", "edge_id": "e01"}],
                }
            ],
        },
        current_edge_id="e02",
        nominal_edges=nominal_edges,
    )

    assert normalized is not None
    binding = normalized["recovery_bindings"][0]
    assert binding["merge"] == "target"
    assert binding["recovery"] == [
        {
            "type": "action",
            "name": "replay_e01",
            "left_arm_action": nominal_edges["e01"]["left_arm_action"],
            "right_arm_action": None,
        }
    ]


def test_runtime_recovery_normalizer_rejects_unknown_step_type() -> None:
    namespace = _load_compile_agent_namespace()
    sanitize_runtime_recovery_spec = namespace["_sanitize_runtime_recovery_spec"]

    normalized = sanitize_runtime_recovery_spec(
        {
            "task": "runtime",
            "recovery_bindings": [
                {
                    "edge_id": "e02",
                    "monitors": [{"type": "hold_lost"}],
                    "recovery": [{"type": "teleport_object"}],
                }
            ],
        },
        current_edge_id="e02",
    )

    assert normalized is None


def test_runtime_recovery_planner_accepts_regrasp_both() -> None:
    namespace = _load_compile_agent_namespace()
    build_runtime_recovery_planner = namespace["_build_runtime_recovery_planner"]

    task_graph = {
        "task": "runtime",
        "start": "v0_start",
        "goal": "v1_done",
        "nodes": [
            {"id": "v0_start", "semantic": ""},
            {"id": "v1_done", "semantic": ""},
        ],
        "edges": [
            {
                "id": "e02",
                "source": "v0_start",
                "target": "v1_done",
                "left_arm_action": {
                    "kind": "atomic_action",
                    "name": "move",
                    "cfg_class": "MoveActionCfg",
                    "cfg": {"control_part": "left_arm"},
                    "target": {
                        "kind": "eef_orientation",
                        "direction": "down",
                    },
                    "runtime_kwargs": {},
                },
                "right_arm_action": {
                    "kind": "atomic_action",
                    "name": "move",
                    "cfg_class": "MoveActionCfg",
                    "cfg": {"control_part": "right_arm"},
                    "target": {
                        "kind": "eef_orientation",
                        "direction": "down",
                    },
                    "runtime_kwargs": {},
                },
            }
        ],
    }

    class FakeRecoveryAgent:
        def get_composed_observations(self, **kwargs):
            return kwargs

        def generate_runtime(self, **kwargs):
            return json.dumps(
                {
                    "task": "runtime",
                    "recovery_bindings": [
                        {
                            "edge_id": "e02",
                            "monitors": [
                                {
                                    "type": "hold_lost",
                                    "robot_name": "left_arm",
                                    "obj_name": "fork",
                                }
                            ],
                            "recovery": [
                                {
                                    "type": "regrasp_both",
                                    "arms": {
                                        "left_arm": "fork",
                                        "right_arm": "spoon",
                                    },
                                    "pre_grasp_dis": 0.12,
                                }
                            ],
                        }
                    ],
                }
            )

    planner = build_runtime_recovery_planner(
        FakeRecoveryAgent(),
        task_graph=task_graph,
        use_llm=True,
    )
    edge = types.SimpleNamespace(
        id="e02",
        source="v0_start",
        target="v1_done",
        left_arm_action=task_graph["edges"][0]["left_arm_action"],
        right_arm_action=task_graph["edges"][0]["right_arm_action"],
        is_recovery=False,
    )

    recovery_edges = planner(
        graph=object(),
        edge=edge,
        monitor_index=0,
        monitor_name="monitor_object_held",
        step_index=0,
        env=object(),
        runtime_kwargs={},
        failure_context={
            "edge_id": "e02",
            "edge_source": "v0_start",
            "edge_target": "v1_done",
            "edge_is_recovery": False,
            "monitor_index": 0,
            "monitor_name": "monitor_object_held",
            "step_index": 0,
            "failure_reason": "Monitor 'monitor_object_held' triggered.",
        },
        runtime_state={},
        recovery_history=[],
    )

    assert recovery_edges is not None
    assert len(recovery_edges) == 2
    assert recovery_edges[-1].target == "v0_start"


def test_runtime_recovery_planner_rejects_repeated_origin_strategy() -> None:
    namespace = _load_compile_agent_namespace()
    build_runtime_recovery_planner = namespace["_build_runtime_recovery_planner"]
    sanitize_runtime_recovery_spec = namespace["_sanitize_runtime_recovery_spec"]
    runtime_recovery_signature = namespace["_runtime_recovery_signature"]

    task_graph = {
        "task": "runtime",
        "start": "v0_start",
        "goal": "v1_done",
        "nodes": [
            {"id": "v0_start", "semantic": ""},
            {"id": "v1_done", "semantic": ""},
        ],
        "edges": [
            {
                "id": "e02",
                "source": "v0_start",
                "target": "v1_done",
                "left_arm_action": None,
                "right_arm_action": None,
            }
        ],
    }
    raw_recovery_spec = {
        "task": "runtime",
        "recovery_bindings": [
            {
                "edge_id": "e02",
                "monitors": [{"type": "hold_lost"}],
                "recovery": [
                    {"type": "move_to_safe_pose", "arms": ["left_arm"]},
                    {
                        "type": "regrasp",
                        "robot_name": "left_arm",
                        "obj_name": "fork",
                    },
                ],
            }
        ],
    }
    normalized_recovery_spec = sanitize_runtime_recovery_spec(
        raw_recovery_spec,
        current_edge_id="e02",
    )
    repeated_signature = runtime_recovery_signature(normalized_recovery_spec)

    class FakeRecoveryAgent:
        def get_composed_observations(self, **kwargs):
            return kwargs

        def generate_runtime(self, **kwargs):
            return json.dumps(raw_recovery_spec)

    planner = build_runtime_recovery_planner(
        FakeRecoveryAgent(),
        task_graph=task_graph,
        use_llm=True,
    )
    edge = types.SimpleNamespace(
        id="e02",
        source="v0_start",
        target="v1_done",
        left_arm_action=None,
        right_arm_action=None,
        is_recovery=False,
    )

    recovery_edges = planner(
        graph=object(),
        edge=edge,
        monitor_index=0,
        monitor_name="monitor_object_held",
        step_index=0,
        env=object(),
        runtime_kwargs={},
        failure_context={
            "edge_id": "e02",
            "origin_edge_id": "e02",
            "monitor_index": 0,
            "monitor_name": "monitor_object_held",
            "step_index": 0,
            "failure_class": "monitor:monitor_object_held:0",
            "failure_reason": "Monitor 'monitor_object_held' triggered.",
        },
        runtime_state={},
        recovery_history=[
            {
                "strategy": "runtime",
                "edge_id": "e02",
                "origin_edge_id": "e02",
                "failure_class": "monitor:monitor_object_held:0",
                "recovery_signature": repeated_signature,
            }
        ],
    )

    assert recovery_edges is None


def test_run_env_does_not_replay_already_executed_agent_graph_actions() -> None:
    generate_and_execute_action_list = _load_generate_and_execute_action_list()

    class ExecutedActions:
        already_executed = True

        def __len__(self):
            return 1

        def __iter__(self):
            return iter(["already_done"])

    class Env:
        def __init__(self) -> None:
            self.step_calls = []

        def get_wrapper_attr(self, name):
            assert name == "create_demo_action_list"
            return lambda **kwargs: ExecutedActions()

        def step(self, action):
            self.step_calls.append(action)
            return None, None, None, None, None

    env = Env()

    valid = generate_and_execute_action_list(env, idx=0, debug_mode=False)

    assert valid is True
    assert env.step_calls == []
