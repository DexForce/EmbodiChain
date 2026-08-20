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

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import embodichain.gen_sim.action_engine.runtime.executor as executor_module
from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.runtime import (
    DynamicRecoveryController,
    ProgramExecutor,
    RuntimeGraph,
    build_upright_recovery,
    classify_failure,
    load_execution_program,
)
from embodichain.gen_sim.action_engine.runtime.executor import _EdgeResult
from embodichain.gen_sim.action_engine.runtime.grounding import ActionGrounder
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph

from ..task_fixtures import make_task_spec


def _graph(task_type: str) -> dict:
    task, requirements = make_task_spec(task_type)
    bindings = {
        item["role_id"]: f"uid_{item['role_id']}" for item in requirements["objects"]
    }
    return instantiate_seed_graph(task, bindings)


def _handover_then_place_graph() -> dict:
    task = {
        "schema_version": "action_engine_task_spec_v2",
        "task_id": "handover_then_place_recovery",
        "level": "L3",
        "instruction": "Hand the yellow can from the left arm to the right arm.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E4",
                "params": {
                    "object_role": "yellow_can",
                    "transfer_arm": "left_arm",
                    "receive_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_02",
                "task_type": "E1",
                "params": {
                    "object_role": "yellow_can",
                    "target_role": "purple_can",
                    "relation": "right_of",
                },
                "depends_on": ["task_01"],
                "role": "primary",
            },
        ],
        "success": {
            "op": "all",
            "terms": [
                {"type": "handover_complete", "task_instance_id": "task_01"},
                {"type": "semantic_goal", "task_instance_id": "task_02"},
            ],
        },
        "oracle": {"task_order": ["task_01", "task_02"]},
        "metadata": {},
    }
    return instantiate_seed_graph(
        task,
        {
            "yellow_can": "interact_yellow_can",
            "purple_can": "interact_purple_can",
        },
    )


class _RecoveryRecorder:
    def __init__(self) -> None:
        self.recovery_events: list[dict[str, Any]] = []
        self.edge_events: list[dict[str, Any]] = []

    def recovery(self, **event: Any) -> None:
        self.recovery_events.append(event)

    def edge(self, edge_id: str, step: Any, **event: Any) -> None:
        self.edge_events.append({"edge_id": edge_id, "step_id": step.id, **event})

    def step(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def register_step(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _local_recovery_harness(
    graph: dict[str, Any],
    *,
    num_envs: int,
    max_transitions: int = 100,
    max_revisions: int = 8,
) -> tuple[ProgramExecutor, Any, Any, list[tuple[str, str, list[bool]]]]:
    program = load_execution_program(graph, require_executable=True)
    step = next(item for item in program.semantic_steps if item.id == "task_01")
    failed_node = next(
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == step.id and node["atomic_action"] == "HandOver"
    )
    edge = next(
        item
        for item in program.edges
        if item.actions[0].get("seed_node_id") == failed_node["id"]
    )
    executor = object.__new__(ProgramExecutor)
    executor.runtime_graph = RuntimeGraph(
        graph,
        num_envs=num_envs,
        max_revisions=max_revisions,
    )
    executor.env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        robot=SimpleNamespace(get_qpos=lambda: torch.zeros((num_envs, 4))),
    )
    executor.capability_registry = None
    executor.steps = {item.id: item for item in program.semantic_steps}
    executor.edges = {item.id: item for item in program.edges}
    executor.step_by_edge = {
        edge_id: item for item in program.semantic_steps for edge_id in item.edge_ids
    }
    executor._assignments = {step.id: ["left_arm"] * num_envs}
    executor._candidate_cache = {}
    executor._candidate_failures = {}
    executor._candidate_diagnostics = {}
    executor._object_states = {}
    executor._step_states = {}
    executor._object_owners = {}
    executor._arm_owners = {
        "left_arm": [None] * num_envs,
        "right_arm": [None] * num_envs,
    }
    executor._targets = {}
    executor.record_runtime = False
    executor.max_transitions = max_transitions
    executor._transition_count = 0
    executor.retry_count = 0
    call_log: list[tuple[str, str, list[bool]]] = []

    def execute_edge(current_edge: Any, current_step: Any, *, failed: torch.Tensor):
        call_log.append((current_step.id, current_edge.id, failed.tolist()))
        return _EdgeResult([], failed.clone(), [])

    def ensure_assignment(current_step: Any, failed: torch.Tensor) -> None:
        actor = current_step.actor
        assignment = (
            str(actor["arm"]) if actor.get("mode") == "required" else "right_arm"
        )
        executor._assignments[current_step.id] = [
            None if bool(failed[index]) else assignment for index in range(num_envs)
        ]

    executor._execute_edge_with_retries = execute_edge
    executor._ensure_assignment = ensure_assignment
    executor._clear_recovery_rows = lambda *_args, **_kwargs: None
    executor._verify_step = lambda _step, failed: (
        failed.clone(),
        ~failed,
        torch.zeros((num_envs, 3)),
    )
    return executor, step, edge, call_log


def test_runtime_graph_retries_twice_then_requests_recovery() -> None:
    graph = _graph("E4")
    runtime = RuntimeGraph(graph, num_envs=2, max_retries=2)
    handover = next(
        node for node in graph["nodes"] if node["atomic_action"] == "HandOver"
    )
    failed = torch.tensor([True, False])
    holds = torch.tensor([True, True])

    first = runtime.record_failure(handover["id"], failed, precondition_holds=holds)
    second = runtime.record_failure(handover["id"], failed, precondition_holds=holds)
    third = runtime.record_failure(handover["id"], failed, precondition_holds=holds)

    assert first.retry.tolist() == [True, False]
    assert second.retry.tolist() == [True, False]
    assert third.recover.tolist() == [True, False]
    assert runtime.seed_graph == graph


def test_recovery_insertion_revises_runtime_graph_not_seed_graph() -> None:
    graph = _graph("E4")
    runtime = RuntimeGraph(graph, num_envs=1)
    failed_node = next(
        node for node in graph["nodes"] if node["atomic_action"] == "HandOver"
    )
    recovery_source = _graph("E2")
    source_group = recovery_source["task_groups"][0]
    recovery_group_id = "recovery_upright_01"
    recovery_nodes = []
    id_map = {
        node["id"]: f"recovery_{index:02d}"
        for index, node in enumerate(recovery_source["nodes"], start=1)
    }
    for node in recovery_source["nodes"]:
        item = deepcopy(node)
        item["id"] = id_map[node["id"]]
        item["object_uid"] = failed_node["object_uid"]
        item["target_binding"] = deepcopy(item["target_binding"])
        if item["target_binding"].get("kind") == "object":
            item["target_binding"]["object"] = failed_node["object_uid"]
        item["depends_on"] = [id_map.get(dep, dep) for dep in node["depends_on"]]
        recovery_nodes.append(item)
    recovery_group = deepcopy(source_group)
    recovery_group.update(
        {
            "id": recovery_group_id,
            "role": "recovery",
            "object_uid": failed_node["object_uid"],
            "node_ids": [node["id"] for node in recovery_nodes],
            "depends_on": [],
            "parent_task_instance_id": failed_node["task_instance_id"],
        }
    )
    recovery_group["success"] = {
        "type": "object_upright",
        "object": failed_node["object_uid"],
    }

    patched = runtime.insert_recovery_subgraph(
        failed_node_id=failed_node["id"],
        recovery_nodes=recovery_nodes,
        recovery_group=recovery_group,
        failure_type="object_fallen",
    )

    assert graph == runtime.seed_graph
    assert any(group["id"] == recovery_group_id for group in patched["task_groups"])
    assert not any(
        node["task_instance_id"] == failed_node["task_instance_id"]
        and node["target_binding"].get("source") == "handover"
        for node in patched["nodes"]
    )
    assert runtime.revisions[0].kind == "insert_recovery"
    assert (
        classify_failure("PickUp", planning_succeeded=True, held_after=False)
        == "grasp_missed"
    )


def test_recovery_rejects_downstream_contract_that_requires_actor_switch() -> None:
    graph = _handover_then_place_graph()
    runtime = RuntimeGraph(graph, num_envs=1)
    handover = next(
        node for node in graph["nodes"] if node["atomic_action"] == "HandOver"
    )
    cleanup_ids = {
        node["id"]
        for node in graph["nodes"]
        if node["task_instance_id"] == handover["task_instance_id"]
        and node["role"] == "cleanup"
    }
    assert cleanup_ids

    with pytest.raises(ValueError, match="without changing.*actor"):
        runtime.insert_default_recovery(
            failed_node_id=handover["id"],
            failure_type="object_fallen",
        )

    assert runtime.graph == graph
    assert runtime.revisions == []


def test_failed_group_resume_recovery_places_before_prefix_replay() -> None:
    graph = _handover_then_place_graph()
    runtime = RuntimeGraph(graph, num_envs=1)
    handover = next(
        node for node in graph["nodes"] if node["atomic_action"] == "HandOver"
    )

    patched = runtime.insert_default_recovery(
        failed_node_id=handover["id"],
        failure_type="object_fallen",
        resume_failed_group=True,
    )

    group_id = runtime.revisions[-1].inserted_group_ids[0]
    group = next(item for item in patched["task_groups"] if item["id"] == group_id)
    nodes = [node for node in patched["nodes"] if node["id"] in group["node_ids"]]
    original_cleanup = {
        node["id"]
        for node in graph["nodes"]
        if node["task_instance_id"] == handover["task_instance_id"]
        and node["role"] == "cleanup"
    }
    assert group["goal"]["terminal_behavior"] == "place"
    assert [node["atomic_action"] for node in nodes] == [
        "PickUp",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert original_cleanup <= {node["id"] for node in patched["nodes"]}


@pytest.mark.parametrize(
    "actor",
    (
        {"mode": "required", "arm": "left_arm"},
        {"mode": "required", "arm": "right_arm"},
        {"mode": "auto"},
    ),
)
def test_upright_recovery_inherits_failed_group_actor(actor: dict[str, Any]) -> None:
    graph = _graph("E2")
    failed_group = graph["task_groups"][0]
    failed_group["actor"] = deepcopy(actor)
    for node in graph["nodes"]:
        if node["task_instance_id"] == failed_group["id"]:
            node["actor"] = deepcopy(actor)

    nodes, recovery_group = build_upright_recovery(
        graph,
        failed_node_id=failed_group["node_ids"][0],
        revision=1,
        resume_failed_group=True,
    )

    assert recovery_group["actor"] == actor
    assert all(node["actor"] == actor for node in nodes)


def test_local_recovery_replays_failed_group_prefix_and_preserves_seed_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _handover_then_place_graph()
    original = deepcopy(graph)
    executor, step, edge, calls = _local_recovery_harness(graph, num_envs=1)
    recorder = _RecoveryRecorder()
    monkeypatch.setattr(
        executor_module,
        "evaluate_predicate",
        lambda *_args, **_kwargs: torch.tensor([False]),
    )

    result = executor._recover_object_fallen(
        edge,
        step,
        _EdgeResult(
            [],
            torch.tensor([True]),
            [],
            executed=torch.tensor([True]),
        ),
        inherited_failed=torch.tensor([False]),
        fallen_transition=torch.tensor([True]),
        recorder=recorder,
    )

    replayed = [edge_id for step_id, edge_id, _failed in calls if step_id == step.id]
    expected_prefix = list(step.edge_ids[: step.edge_ids.index(edge.id) + 1])
    assert result.failed.tolist() == [False]
    assert replayed == expected_prefix
    assert executor.runtime_graph.seed_graph == original
    assert graph == original
    assert [event["status"] for event in recorder.recovery_events] == [
        "started",
        "succeeded",
    ]
    recovery_edges = [
        event
        for event in recorder.edge_events
        if event["step_id"].startswith("recovery_e2_")
    ]
    replay_edges = [
        event for event in recorder.edge_events if event["step_id"] == step.id
    ]
    assert recovery_edges
    assert all(event["phase"] == "recovery" for event in recovery_edges)
    assert replay_edges
    assert all(event["phase"] == "replay" for event in replay_edges)


def test_local_recovery_only_executes_and_rebinds_failed_vector_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _handover_then_place_graph()
    executor, step, edge, calls = _local_recovery_harness(graph, num_envs=2)
    recorder = _RecoveryRecorder()
    monkeypatch.setattr(
        executor_module,
        "evaluate_predicate",
        lambda *_args, **_kwargs: torch.tensor([True, False]),
    )

    result = executor._recover_object_fallen(
        edge,
        step,
        _EdgeResult(
            [],
            torch.tensor([False, True]),
            [],
            executed=torch.tensor([False, True]),
        ),
        inherited_failed=torch.tensor([False, False]),
        fallen_transition=torch.tensor([False, True]),
        recorder=recorder,
    )

    assert result.failed.tolist() == [False, False]
    assert all(failed == [True, False] for _step_id, _edge_id, failed in calls)
    assert executor._assignments[step.id] == ["left_arm", "left_arm"]
    assert executor.runtime_graph.revisions[-1].active_env_ids == (1,)
    assert all(
        event["active"].tolist() == [False, True] for event in recorder.recovery_events
    )


def test_local_recovery_failure_does_not_replay_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _handover_then_place_graph()
    executor, step, edge, calls = _local_recovery_harness(graph, num_envs=1)
    executor._verify_step = lambda _step, failed: (
        torch.ones_like(failed),
        torch.zeros_like(failed),
        torch.zeros((1, 3)),
    )
    recorder = _RecoveryRecorder()
    monkeypatch.setattr(
        executor_module,
        "evaluate_predicate",
        lambda *_args, **_kwargs: torch.tensor([False]),
    )

    result = executor._recover_object_fallen(
        edge,
        step,
        _EdgeResult(
            [],
            torch.tensor([True]),
            [],
            executed=torch.tensor([True]),
        ),
        inherited_failed=torch.tensor([False]),
        fallen_transition=torch.tensor([True]),
        recorder=recorder,
    )

    assert result.failed.tolist() == [True]
    assert not any(step_id == step.id for step_id, _edge_id, _failed in calls)
    assert recorder.recovery_events[-1]["status"] == "failed"


def test_local_recovery_budget_exhaustion_terminates_with_original_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _handover_then_place_graph()
    executor, step, edge, calls = _local_recovery_harness(
        graph,
        num_envs=1,
        max_transitions=0,
    )
    recorder = _RecoveryRecorder()
    monkeypatch.setattr(
        executor_module,
        "evaluate_predicate",
        lambda *_args, **_kwargs: torch.tensor([False]),
    )

    result = executor._recover_object_fallen(
        edge,
        step,
        _EdgeResult(
            [],
            torch.tensor([True]),
            [],
            executed=torch.tensor([True]),
        ),
        inherited_failed=torch.tensor([False]),
        fallen_transition=torch.tensor([True]),
        recorder=recorder,
    )

    assert result.failed.tolist() == [True]
    assert calls == []
    assert recorder.recovery_events[-1]["status"] == "failed"
    assert "max_transitions" in recorder.recovery_events[-1]["error"]


def test_non_fallen_failure_does_not_create_recovery_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _handover_then_place_graph()
    executor, step, edge, calls = _local_recovery_harness(graph, num_envs=1)
    recorder = _RecoveryRecorder()
    monkeypatch.setattr(
        executor_module,
        "evaluate_predicate",
        lambda *_args, **_kwargs: torch.tensor([True]),
    )

    result = executor._recover_object_fallen(
        edge,
        step,
        _EdgeResult(
            [],
            torch.tensor([True]),
            [],
            executed=torch.tensor([True]),
        ),
        inherited_failed=torch.tensor([False]),
        fallen_transition=torch.tensor([False]),
        recorder=recorder,
    )

    assert result.failed.tolist() == [True]
    assert calls == []
    assert executor.runtime_graph.revisions == []
    assert recorder.recovery_events == []


def test_initially_fallen_planning_failure_does_not_trigger_recovery() -> None:
    graph = _handover_then_place_graph()
    executor, step, edge, calls = _local_recovery_harness(graph, num_envs=1)
    recorder = _RecoveryRecorder()

    result = executor._recover_object_fallen(
        edge,
        step,
        _EdgeResult(
            [],
            torch.tensor([True]),
            [],
            executed=torch.tensor([False]),
        ),
        inherited_failed=torch.tensor([False]),
        fallen_transition=torch.tensor([False]),
        recorder=recorder,
    )

    assert result.failed.tolist() == [True]
    assert calls == []
    assert executor.runtime_graph.revisions == []
    assert recorder.recovery_events == []


def test_failure_provenance_distinguishes_planning_from_execution_caused_fall() -> None:
    graph = _handover_then_place_graph()
    executor, step, edge, _ = _local_recovery_harness(graph, num_envs=1)
    executor.adapter = SimpleNamespace(capabilities=build_atomic_capability_registry())
    failed = torch.tensor([True])

    planning = executor._failure_events(
        edge,
        step,
        failed,
        postcondition=False,
        executed=torch.tensor([False]),
        fallen_transition=torch.tensor([False]),
    )
    execution = executor._failure_events(
        edge,
        step,
        failed,
        postcondition=False,
        executed=torch.tensor([True]),
        fallen_transition=torch.tensor([True]),
    )

    assert [event["failure_type"] for event in planning] == ["search_exhausted"]
    assert [event["failure_type"] for event in execution] == ["object_fallen"]


def test_offline_and_online_dynamic_replanners_are_route_isolated() -> None:
    for mode in ("offline_dynamic", "online_dynamic"):
        graph = _graph("E4")
        runtime = RuntimeGraph(graph, num_envs=1)
        calls = []

        def replanner(**kwargs):
            calls.append((mode, kwargs["failure_type"]))
            return kwargs["graph"]

        controller = DynamicRecoveryController(
            runtime,
            mode=mode,
            offline_replanner=replanner if mode == "offline_dynamic" else None,
            online_replanner=replanner if mode == "online_dynamic" else None,
        )
        failed_node = next(
            node for node in graph["nodes"] if node["atomic_action"] == "HandOver"
        )
        directive = controller.handle_failure(
            failed_node_id=failed_node["id"],
            failure_type="postcondition_failed",
        )
        completed = [group["id"] for group in graph["task_groups"]]
        controller.replan(
            directive,
            completed_group_ids=completed,
            recovery_succeeded=False,
        )

        assert calls == [(mode, "postcondition_failed")]
        assert runtime.revisions[-1].kind == "replan_suffix"


def test_dynamic_recovery_consumes_per_environment_failure_events() -> None:
    graph = _graph("E4")
    runtime = RuntimeGraph(graph, num_envs=2)
    controller = DynamicRecoveryController(
        runtime,
        mode="offline_dynamic",
        offline_replanner=lambda **kwargs: kwargs["graph"],
    )
    failed_node = next(
        node for node in graph["nodes"] if node["atomic_action"] == "HandOver"
    )
    result = SimpleNamespace(
        failure_events=[
            {
                "node_id": failed_node["id"],
                "failure_type": "object_fallen",
                "env_ids": [1],
            }
        ]
    )

    directive = controller.handle_execution_result(result)

    assert directive.active_env_ids == (1,)
    assert runtime.revisions[-1].active_env_ids == (1,)


def test_runtime_graph_stops_at_revision_and_recovery_budgets() -> None:
    graph = _graph("E4")
    failed_node = next(
        node for node in graph["nodes"] if node["atomic_action"] == "HandOver"
    )

    with pytest.raises(RuntimeError, match="revision budget"):
        RuntimeGraph(graph, num_envs=1, max_revisions=0).insert_default_recovery(
            failed_node_id=failed_node["id"],
            failure_type="object_fallen",
        )
    with pytest.raises(RuntimeError, match="recovery-action budget"):
        RuntimeGraph(graph, num_envs=1, max_recovery_actions=0).insert_default_recovery(
            failed_node_id=failed_node["id"],
            failure_type="object_fallen",
        )


def test_visual_constraint_grounding_reads_fresh_camera_depth() -> None:
    class Sensor:
        def __init__(self) -> None:
            self.depth = torch.ones((1, 4, 4, 1))

        def get_data(self):
            return {"depth": self.depth}

        def get_intrinsics(self):
            return torch.tensor([[[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]])

        def get_arena_pose(self, *, to_matrix: bool):
            assert to_matrix
            return torch.eye(4).unsqueeze(0)

    sensor = Sensor()
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        sim=SimpleNamespace(get_sensor=lambda uid: sensor if uid == "front" else None),
        get_current_xpos_agent=lambda: (
            torch.eye(4).unsqueeze(0),
            torch.eye(4).unsqueeze(0),
        ),
    )
    grounder = object.__new__(ActionGrounder)
    grounder.env = env
    binding = {"camera_uid": "front", "normalized_keypoint": [0.5, 0.5]}

    first = grounder._visual_target(binding, "left_arm")
    sensor.depth.fill_(2.0)
    second = grounder._visual_target(
        {"camera_uid": "front", "normalized_bbox": [0.4, 0.4, 0.6, 0.6]},
        "left_arm",
    )

    assert first[0, 2, 3].item() == 1.0
    assert second[0, 2, 3].item() == 2.0
