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

import pytest
import torch

from embodichain.gen_sim.action_engine.runtime import (
    DynamicRecoveryController,
    RuntimeGraph,
    classify_failure,
)
from embodichain.gen_sim.action_engine.runtime.grounding import ActionGrounder
from embodichain.gen_sim.action_engine.tasks import TaskFactory, instantiate_seed_graph


def _graph(task_type: str) -> dict:
    factory = TaskFactory(13, executable_only=True)
    for index in range(100):
        task, requirements = factory.generate("L1", index)
        if task["task_instances"][0]["task_type"] == task_type:
            bindings = {
                item["role_id"]: f"uid_{item['role_id']}"
                for item in requirements["objects"]
            }
            return instantiate_seed_graph(task, bindings)
    raise AssertionError(f"No {task_type} graph generated.")


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


def test_handover_recovery_replaces_cleanup_suffix_before_downstream_work() -> None:
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

    patched = runtime.insert_default_recovery(
        failed_node_id=handover["id"],
        failure_type="object_fallen",
    )

    recovery_group_id = runtime.revisions[-1].inserted_group_ids[0]
    recovery_group = next(
        group for group in patched["task_groups"] if group["id"] == recovery_group_id
    )
    recovery_terminal = recovery_group["node_ids"][-1]
    failed_group = next(
        group
        for group in patched["task_groups"]
        if group["id"] == handover["task_instance_id"]
    )
    downstream_group = next(
        group for group in patched["task_groups"] if group["id"] == "task_02"
    )
    downstream_nodes = [
        node for node in patched["nodes"] if node["id"] in downstream_group["node_ids"]
    ]

    assert cleanup_ids.isdisjoint({node["id"] for node in patched["nodes"]})
    assert cleanup_ids.isdisjoint(failed_group["node_ids"])
    assert downstream_group["depends_on"] == [recovery_group_id]
    assert all(recovery_terminal in node["depends_on"] for node in downstream_nodes)
    assert all(cleanup_ids.isdisjoint(node["depends_on"]) for node in downstream_nodes)


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
