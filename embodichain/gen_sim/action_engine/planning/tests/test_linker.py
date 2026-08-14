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

import pytest

from embodichain.gen_sim.action_engine.domain import seed_graph_hash
from embodichain.gen_sim.action_engine.planning.linker import (
    link_seed_graph,
    link_task_dependencies,
)
from embodichain.gen_sim.action_engine.protocol import (
    SEED_GRAPH_SCHEMA,
    TASK_SPEC_SCHEMA,
)
from embodichain.gen_sim.action_engine.runtime.loader import load_execution_program
from embodichain.gen_sim.action_engine.tasks.recipes import instantiate_seed_graph


def _handover_task() -> dict:
    return {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "handover_then_place",
        "level": "L3",
        "instruction": "Stand both cans, hand over the purple can, then place it.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E2",
                "params": {
                    "object_role": "purple",
                    "required_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_02",
                "task_type": "E2",
                "params": {
                    "object_role": "orange",
                    "required_arm": "left_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_03",
                "task_type": "E4",
                "params": {
                    "object_role": "purple",
                    "transfer_arm": "right_arm",
                    "receive_arm": "left_arm",
                },
                "depends_on": ["task_02"],
                "role": "primary",
            },
            {
                "id": "task_04",
                "task_type": "E1",
                "params": {
                    "object_role": "purple",
                    "target_role": "orange",
                    "relation": "left_of",
                    "required_arm": "left_arm",
                },
                "depends_on": ["task_03"],
                "role": "primary",
            },
        ],
        "success": {"type": "all_complete"},
        "oracle": {},
        "metadata": {},
    }


def _handover_graph() -> dict:
    return instantiate_seed_graph(
        _handover_task(),
        {"purple": "purple_can", "orange": "orange_can"},
    )


def _unlink_for_rebuild(graph: dict) -> None:
    graph["metadata"].pop("action_contract_linker", None)
    for group in graph["task_groups"]:
        group.pop("contract", None)


def test_task_linker_preserves_parallel_arms_and_waits_for_both_before_handover() -> (
    None
):
    linked = link_task_dependencies(
        _handover_task(),
        {"purple": "purple_can", "orange": "orange_can"},
    )
    by_id = {item["id"]: item for item in linked["task_instances"]}

    assert by_id["task_01"]["depends_on"] == []
    assert by_id["task_02"]["depends_on"] == []
    assert by_id["task_03"]["depends_on"] == ["task_02", "task_01"]


def test_same_object_e2_handover_gets_direct_causal_edge_through_a_chain() -> None:
    task = _handover_task()
    task["task_instances"][1]["depends_on"] = ["task_01"]
    linked = link_task_dependencies(
        task,
        {"purple": "purple_can", "orange": "orange_can"},
    )
    handover = next(
        item for item in linked["task_instances"] if item["id"] == "task_03"
    )

    assert handover["depends_on"] == ["task_02", "task_01"]


def test_handover_ownership_flows_through_home_terminal_barrier() -> None:
    graph = _handover_graph()
    groups = {group["id"]: group for group in graph["task_groups"]}
    nodes = {node["id"]: node for node in graph["nodes"]}
    handover_group = groups["task_03"]
    terminal_id = handover_group["contract"]["terminal_node_ids"][0]
    terminal = nodes[terminal_id]
    receiver_entry = nodes[groups["task_04"]["contract"]["entry_node_ids"][0]]
    handover = next(
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_03" and node["atomic_action"] == "HandOver"
    )

    assert terminal["atomic_action"] == "MoveJoints"
    assert terminal["contract"]["completion"] == "terminal_barrier"
    assert terminal["contract"]["failure_policy"] == "best_effort"
    retreat = next(
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_03"
        and node["atomic_action"] == "MoveEndEffector"
    )
    assert retreat["contract"]["failure_policy"] == "safety_required"
    assert terminal_id in receiver_entry["depends_on"]
    assert {
        (effect["op"], effect["atom"]["predicate"], effect["atom"].get("arm"))
        for effect in handover["contract"]["effects"]
    } >= {
        ("delete", "object_held", "right_arm"),
        ("add", "object_held", "left_arm"),
    }


def test_linker_is_idempotent_and_hash_stable() -> None:
    graph = _handover_graph()
    relinked = link_seed_graph(
        graph,
        task_order=["task_01", "task_02", "task_03", "task_04"],
        known_objects={"purple_can", "orange_can", "table"},
    )

    assert relinked == graph
    assert seed_graph_hash(relinked) == seed_graph_hash(graph)


def test_linker_rejects_missing_cleanup_wrong_holder_and_duplicate_pickup() -> None:
    missing_cleanup = deepcopy(_handover_graph())
    home = next(
        node
        for node in missing_cleanup["nodes"]
        if node["task_instance_id"] == "task_03"
        and node["atomic_action"] == "MoveJoints"
    )
    missing_cleanup["nodes"].remove(home)
    next(group for group in missing_cleanup["task_groups"] if group["id"] == "task_03")[
        "node_ids"
    ].remove(home["id"])
    for node in missing_cleanup["nodes"]:
        node["depends_on"] = [
            dependency for dependency in node["depends_on"] if dependency != home["id"]
        ]
    _unlink_for_rebuild(missing_cleanup)
    with pytest.raises(ValueError, match="terminal barrier"):
        link_seed_graph(missing_cleanup)

    wrong_holder = deepcopy(_handover_graph())
    staging = next(
        node
        for node in wrong_holder["nodes"]
        if node["task_instance_id"] == "task_03"
        and node["atomic_action"] == "MoveHeldObject"
    )
    staging["actor"] = {"mode": "required", "arm": "left_arm"}
    staging.pop("contract")
    _unlink_for_rebuild(wrong_holder)
    with pytest.raises(ValueError, match="no producer|unavailable state"):
        link_seed_graph(wrong_holder)

    duplicate_pickup = deepcopy(_handover_graph())
    pickup = next(
        node
        for node in duplicate_pickup["nodes"]
        if node["task_instance_id"] == "task_01" and node["atomic_action"] == "PickUp"
    )
    staging = next(
        node
        for node in duplicate_pickup["nodes"]
        if node["task_instance_id"] == "task_01"
        and node["atomic_action"] == "MoveHeldObject"
    )
    repeated = deepcopy(pickup)
    repeated["id"] = "task_01__duplicate_pickup"
    repeated["depends_on"] = [pickup["id"]]
    repeated.pop("contract")
    staging["depends_on"] = [repeated["id"]]
    group = next(
        group for group in duplicate_pickup["task_groups"] if group["id"] == "task_01"
    )
    pickup_index = group["node_ids"].index(pickup["id"])
    group["node_ids"].insert(pickup_index + 1, repeated["id"])
    duplicate_pickup["nodes"].insert(
        duplicate_pickup["nodes"].index(pickup) + 1, repeated
    )
    _unlink_for_rebuild(duplicate_pickup)
    with pytest.raises(ValueError, match="requires unavailable state"):
        link_seed_graph(duplicate_pickup)


def test_readers_remain_parallel_and_writer_waits_for_both() -> None:
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "read_write",
        "level": "L3",
        "instruction": "Inspect a shared target, then manipulate it.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "read_left",
                "task_type": "E1",
                "params": {
                    "object_role": "a",
                    "target_role": "target",
                    "required_arm": "left_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "read_right",
                "task_type": "E1",
                "params": {
                    "object_role": "b",
                    "target_role": "target",
                    "required_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "write_target",
                "task_type": "E2",
                "params": {
                    "object_role": "target",
                    "required_arm": "left_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
        ],
        "success": {},
        "oracle": {},
        "metadata": {},
    }
    linked = link_task_dependencies(
        task,
        {"a": "object_a", "b": "object_b", "target": "shared_target"},
    )
    by_id = {item["id"]: item for item in linked["task_instances"]}

    assert by_id["read_left"]["depends_on"] == []
    assert by_id["read_right"]["depends_on"] == []
    assert by_id["write_target"]["depends_on"] == ["read_left", "read_right"]


def test_explicit_distinct_arm_allocation_keeps_auto_groups_parallel() -> None:
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "allocated_auto",
        "level": "L2",
        "instruction": "Stand both objects upright in parallel.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "first",
                "task_type": "E2",
                "params": {"object_role": "first_object"},
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "second",
                "task_type": "E2",
                "params": {"object_role": "second_object"},
                "depends_on": [],
                "role": "primary",
            },
        ],
        "success": {},
        "oracle": {},
        "metadata": {
            "allocation_groups": [
                {
                    "id": "distinct_pair",
                    "task_instance_ids": ["first", "second"],
                    "arm_constraint": "distinct_arms",
                }
            ]
        },
    }
    bindings = {"first_object": "first_uid", "second_object": "second_uid"}
    linked = link_task_dependencies(task, bindings)
    graph = instantiate_seed_graph(linked, bindings)

    assert all(not item["depends_on"] for item in linked["task_instances"])
    assert all(not group["depends_on"] for group in graph["task_groups"])


def test_v2_and_resolver_mismatch_require_regeneration() -> None:
    with pytest.raises(
        ValueError, match="lacks persisted Action Contracts.*regenerate"
    ):
        load_execution_program({"schema_version": "action_engine_seed_graph_v2"})

    graph = _handover_graph()
    graph["nodes"][0]["contract"]["claims"][0]["access"] = "shared_read"
    with pytest.raises(
        ValueError, match="does not match the current capability resolver"
    ):
        load_execution_program(graph)


def test_seed_graph_schema_is_v3() -> None:
    assert _handover_graph()["schema_version"] == SEED_GRAPH_SCHEMA
