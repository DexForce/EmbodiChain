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

"""Contract-driven direct-payload propagation tests."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from textwrap import dedent

import pytest

import embodichain.gen_sim.action_engine.domain.v2 as domain_v2_module
import embodichain.gen_sim.action_engine.tasks.recipes as recipes_module
import embodichain.gen_sim.action_engine.planning.linker as linker_module
import embodichain.gen_sim.action_engine.runtime.executor as executor_module
from embodichain.gen_sim.action_engine.domain import task_contract
from embodichain.gen_sim.action_engine.protocol import TASK_SPEC_SCHEMA
from embodichain.gen_sim.action_engine.runtime import load_execution_program
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph

from ..task_fixtures import make_task_spec


def _loaded_carrier_task(
    *,
    consumer_type: str = "E5",
    terminal_behavior: str = "hold",
) -> tuple[dict, dict[str, str]]:
    consumer_params = {
        "object_role": "tray",
        "direction": "up",
        "terminal_behavior": terminal_behavior,
    }
    if consumer_type == "E2":
        consumer_params = {
            "object_role": "tray",
            "orientation_goal": "upright",
            "support_role": "table",
            "upright_local_axis": "long_axis",
        }
    elif consumer_type == "E6":
        consumer_params = {
            "object_role": "tray",
            "target_state": "open",
        }
    task = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "loaded-carrier",
        "level": "L3",
        "instruction": "Place two objects in a tray, then move the loaded tray.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E1",
                "params": {
                    "object_role": "cube",
                    "target_role": "tray",
                    "relation": "inside",
                    "required_arm": "left_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_02",
                "task_type": "E1",
                "params": {
                    "object_role": "apple",
                    "target_role": "tray",
                    "relation": "inside",
                    "required_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "task_03",
                "task_type": consumer_type,
                "params": consumer_params,
                "depends_on": [],
                "role": "primary",
            },
        ],
        "success": {"op": "all", "terms": []},
        "oracle": {},
        "metadata": {},
    }
    return task, {
        "cube": "interact_cube",
        "apple": "interact_apple",
        "tray": "interact_tray",
    }


def test_loaded_e5_propagates_payloads_through_goal_binding_and_claims() -> None:
    task, bindings = _loaded_carrier_task()

    graph = instantiate_seed_graph(task, bindings)

    carrier_group = next(
        group for group in graph["task_groups"] if group["id"] == "task_03"
    )
    payloads = [
        {"object": "interact_cube", "slot": "center"},
        {"object": "interact_apple", "slot": "center"},
    ]
    assert carrier_group["goal"]["payloads"] == payloads
    assert carrier_group["depends_on"] == ["task_01", "task_02"]
    pick = next(
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_03"
        and node["atomic_action"] == "CoordinatedPickment"
    )
    assert pick["target_binding"]["payloads"] == payloads
    payload_claims = {
        claim["resource"]
        for claim in pick["contract"]["claims"]
        if claim["resource"].startswith("object:")
    }
    assert payload_claims == {
        "object:interact_tray",
        "object:interact_cube",
        "object:interact_apple",
    }
    assert graph["metadata"]["direct_payload_links"] == [
        {
            "producer": "task_01",
            "consumer": "task_03",
            "carrier": "interact_tray",
            "payload": "interact_cube",
            "relation": "direct_support",
        },
        {
            "producer": "task_02",
            "consumer": "task_03",
            "carrier": "interact_tray",
            "payload": "interact_apple",
            "relation": "direct_support",
        },
    ]


def test_standalone_e5_binds_task_owned_payload_without_a_producer() -> None:
    task, _requirements = make_task_spec("E5")
    task["task_instances"][0]["params"]["payload_roles"] = ["payload"]

    graph = instantiate_seed_graph(
        task,
        {"object_01": "tray_uid", "payload": "can_uid"},
    )
    node = graph["nodes"][0]

    assert graph["metadata"]["direct_payload_links"] == []
    assert graph["task_groups"][0]["goal"]["payloads"] == [
        {"object": "can_uid", "slot": "center"}
    ]
    assert node["target_binding"]["payloads"] == [
        {"object": "can_uid", "slot": "center"}
    ]
    assert "object:can_uid" in {
        claim["resource"] for claim in node["contract"]["claims"]
    }


def test_loaded_carrier_graph_and_payload_order_are_deterministic() -> None:
    task, bindings = _loaded_carrier_task()

    first = instantiate_seed_graph(task, bindings)
    second = instantiate_seed_graph(task, bindings)

    assert first == second


def test_loaded_e5_place_keeps_payload_contract_through_synchronized_release() -> None:
    task, bindings = _loaded_carrier_task(terminal_behavior="place")

    graph = instantiate_seed_graph(task, bindings)

    group = next(item for item in graph["task_groups"] if item["id"] == "task_03")
    nodes = [node for node in graph["nodes"] if node["task_instance_id"] == "task_03"]
    assert len(group["goal"]["payloads"]) == 2
    assert [node["atomic_action"] for node in nodes] == [
        "CoordinatedPickment",
        "MoveJoints",
        "MoveJoints",
        "MoveEndEffector",
        "MoveEndEffector",
        "MoveJoints",
        "MoveJoints",
    ]
    assert len({node.get("sync_group") for node in nodes[1:3]}) == 1
    assert all(node.get("sync_group") is None for node in nodes[3:])


def test_consumer_without_direct_payload_capability_is_rejected() -> None:
    task, bindings = _loaded_carrier_task(consumer_type="E2")

    with pytest.raises(ValueError, match="does not accept direct payloads"):
        instantiate_seed_graph(task, bindings)


def test_test_consumer_contract_needs_no_propagation_type_change() -> None:
    task, bindings = _loaded_carrier_task(consumer_type="E6")
    custom_consumer = replace(
        task_contract("E6"),
        accepts_direct_payloads=True,
        moves_primary_object=True,
    )

    propagated, links = recipes_module._propagate_direct_payloads(
        task,
        bindings,
        contract_resolver=lambda task_type: (
            custom_consumer if task_type == "E6" else task_contract(task_type)
        ),
    )

    assert propagated["task_instances"][2]["params"]["payload_roles"] == [
        "cube",
        "apple",
    ]
    assert propagated["task_instances"][2]["depends_on"] == ["task_01", "task_02"]
    assert [link["producer"] for link in links] == ["task_01", "task_02"]


def test_payload_infrastructure_does_not_branch_on_task_numbers() -> None:
    functions = (
        recipes_module._propagate_direct_payloads,
        linker_module._task_claims,
        linker_module._task_primary_object,
        domain_v2_module._validate_ownership_transitions,
        executor_module.ProgramExecutor._capture_payloads,
        executor_module.ProgramExecutor._verify_payloads,
    )
    task_numbers = {f"E{index}" for index in range(1, 10)}

    for function in functions:
        tree = ast.parse(dedent(inspect.getsource(function)))
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert literals.isdisjoint(task_numbers), function.__qualname__


def test_e2_incoming_hold_reads_only_generic_ownership_state() -> None:
    holder = recipes_module._incoming_held_arm(
        "E2",
        "object_uid",
        ["anonymous_producer"],
        {"anonymous_producer": ("object_uid", "right_arm")},
    )
    tree = ast.parse(dedent(inspect.getsource(recipes_module._incoming_held_arm)))
    task_numbers = {f"E{index}" for index in range(1, 10)}
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert holder == "right_arm"
    assert literals.isdisjoint(task_numbers)


@pytest.mark.parametrize("task_type", [f"E{index}" for index in range(1, 10)])
def test_standalone_tasks_gain_no_payload_dependency(task_type: str) -> None:
    task, _requirements = make_task_spec(task_type)
    params = task["task_instances"][0]["params"]
    role_names = {
        value
        for key, value in params.items()
        if (key.endswith("_role") or key.endswith("_roles"))
        and isinstance(value, str)
        and value != "table"
    }
    bindings = {role: f"runtime_{role}" for role in role_names}

    graph = instantiate_seed_graph(task, bindings)
    program = load_execution_program(graph)

    assert graph["task_groups"][0]["depends_on"] == []
    assert graph["metadata"]["direct_payload_links"] == []
    assert len(program.semantic_steps) == 1
    assert program.semantic_steps[0].id == "task_01"
