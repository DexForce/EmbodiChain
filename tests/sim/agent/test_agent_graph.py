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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def _load_compile_agent_class():
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
        "database_agent_prompt_dir": Path("/tmp"),
        "extract_json_object": extract_json_object,
        "normalize_json_content": normalize_json_content,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return namespace["CompileAgent"]


def _load_agent_graph_namespace(fake_drive):
    source_path = REPO_ROOT / "embodichain" / "lab" / "sim" / "agent" / "agent_graph.py"
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    body = [
        node
        for node in module.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module == "embodichain.lab.sim.agent.atom_actions"
        )
    ]
    namespace = {
        "defaultdict": defaultdict,
        "dataclass": dataclass,
        "field": field,
        "Any": Any,
        "drive": fake_drive,
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

    def fake_drive(right_arm_action=None, return_result=False, **kwargs):
        calls.append(right_arm_action)
        monitor_index = 0 if right_arm_action in {"fail_1", "fail_2"} else None
        return {
            "actions": [right_arm_action],
            "monitor_index": monitor_index,
            "monitor_name": "triggered" if monitor_index is not None else None,
            "step_index": 0 if monitor_index is not None else None,
        }

    graph_namespace = _load_agent_graph_namespace(fake_drive)
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


def test_agent_graph_recovers_inside_recovery_path_and_resumes_continuation() -> None:
    calls = []

    def fake_drive(right_arm_action=None, monitor_sequences=None, **kwargs):
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

    graph_namespace = _load_agent_graph_namespace(fake_drive)
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

    def fake_drive(right_arm_action=None, monitor_sequences=None, **kwargs):
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

    graph_namespace = _load_agent_graph_namespace(fake_drive)
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


def test_agent_graph_self_loop_recovery_retries_without_duplicate_success() -> None:
    calls = []
    self_loop_attempts = 0

    def fake_drive(right_arm_action=None, monitor_sequences=None, **kwargs):
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

    graph_namespace = _load_agent_graph_namespace(fake_drive)
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
        "schema_version": "recovery_bindings_v1",
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
