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

"""Representative tests for the explicitly selected legacy rule adapter."""

from __future__ import annotations

from embodichain.gen_sim.action_engine.tasks import (
    instantiate_seed_graph,
    plan_grounded_task_spec,
)


def _scene() -> list[dict]:
    return [
        {
            "runtime_uid": "purple_can",
            "uid": "purple_can",
            "role": "rigid_object",
            "description": "A purple soda can.",
            "init_pos": [0.0, -0.25, 0.7],
        },
        {
            "runtime_uid": "orange_can",
            "uid": "orange_can",
            "role": "rigid_object",
            "description": "An orange soda can.",
            "init_pos": [0.0, 0.2, 0.7],
        },
    ]


def test_handles_mixed_language_pronouns_and_handover() -> None:
    grounded = plan_grounded_task_spec(
        "mixed_language",
        "Use right arm to upright the purple can, then transfer it to left arm, "
        "then put it left of the orange can.",
        _scene(),
        robot_profile="ur10",
    )

    instances = grounded.task_spec["task_instances"]
    assert [item["task_type"] for item in instances] == ["E2", "E4", "E1"]
    assert instances[1]["params"]["transfer_arm"] == "right_arm"
    assert instances[1]["params"]["receive_arm"] == "left_arm"
    assert instances[2]["params"]["relation"] == "left_of"


def test_consumes_transfer_arm_retreat_as_handover_cleanup() -> None:
    grounded = plan_grounded_task_spec(
        "handover_retreat",
        "用右臂扶正紫色易拉罐，然后用右臂递给左臂，然后右臂撤回，"
        "然后将其放到橘色易拉罐的左边。",
        _scene(),
        robot_profile="ur10",
    )
    graph = instantiate_seed_graph(grounded.task_spec, grounded.role_bindings)

    assert [item["task_type"] for item in grounded.task_spec["task_instances"]] == [
        "E2",
        "E4",
        "E1",
    ]
    handover_nodes = [
        node for node in graph["nodes"] if node["task_instance_id"] == "task_02"
    ]
    assert [node["atomic_action"] for node in handover_nodes] == [
        "PickUp",
        "MoveHeldObject",
        "HandOver",
        "MoveEndEffector",
        "MoveJoints",
    ]
    placement = next(
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_03"
        and node["atomic_action"] == "MoveHeldObject"
    )
    assert placement["depends_on"] == [handover_nodes[-1]["id"]]


def test_keeps_target_side_distinct_from_relation_side() -> None:
    grounded = plan_grounded_task_spec(
        "target_side",
        "Put the purple can on the right can.",
        _scene(),
        robot_profile="ur10",
    )

    bindings = grounded.role_bindings
    instance = grounded.task_spec["task_instances"][0]
    assert bindings[instance["params"]["object_role"]] == "purple_can"
    assert bindings[instance["params"]["target_role"]] == "orange_can"


def test_resolves_explicit_multi_object_count() -> None:
    grounded = plan_grounded_task_spec(
        "two_cans",
        "扶正两个易拉罐。",
        _scene(),
        robot_profile="ur10",
    )

    assert [item["task_type"] for item in grounded.task_spec["task_instances"]] == [
        "E2",
        "E2",
    ]


def test_does_not_treat_uid_digits_as_quantity() -> None:
    scene = [
        {
            "runtime_uid": "can_10",
            "uid": "can_10",
            "role": "rigid_object",
            "description": "A soda can.",
            "init_pos": [0.0, 0.1, 0.7],
        }
    ]
    grounded = plan_grounded_task_spec(
        "uid_digits",
        "扶正 can_10。",
        scene,
        robot_profile="ur10",
    )
    assert len(grounded.task_spec["task_instances"]) == 1
    assert grounded.role_bindings["object_01"] == "can_10"


def test_keeps_chinese_target_side_as_selector() -> None:
    scene = [
        {
            "runtime_uid": "purple_can",
            "uid": "purple_can",
            "role": "rigid_object",
            "description": "紫色易拉罐",
            "init_pos": [0.0, 0.0, 0.7],
        },
        {
            "runtime_uid": "orange_left",
            "uid": "orange_left",
            "role": "rigid_object",
            "description": "橘色易拉罐",
            "init_pos": [0.0, -0.25, 0.7],
        },
        {
            "runtime_uid": "orange_right",
            "uid": "orange_right",
            "role": "rigid_object",
            "description": "橘色易拉罐",
            "init_pos": [0.0, 0.25, 0.7],
        },
    ]
    grounded = plan_grounded_task_spec(
        "target_side_zh",
        "把紫色易拉罐放到左边的橘色易拉罐上。",
        scene,
        robot_profile="ur10",
    )
    instance = grounded.task_spec["task_instances"][0]
    assert instance["params"]["relation"] == "on"
    assert grounded.role_bindings[instance["params"]["target_role"]] == "orange_left"
