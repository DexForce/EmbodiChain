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

"""Task-first generation, instantiation, and scene hand-off contracts."""

from __future__ import annotations

from copy import deepcopy

from embodichain.gen_sim.action_engine.tasks import (
    ground_instruction_draft,
    instantiate_seed_graph,
)
from tests.gen_sim.action_engine.task_fixtures import (
    TASK2_1_HISTORICAL_ROLE_BINDINGS,
    TASK2_1_HISTORICAL_SCENE_FINGERPRINT,
    make_task2_1_historical_spec,
)


def _selector(
    kind: str = "none",
    *,
    reference: str = "",
    step_id: str = "",
    quantifier: str = "one",
) -> dict:
    return {
        "kind": kind,
        "step_id": step_id,
        "reference": reference,
        "quantifier": quantifier,
        "count": 0,
    }


def _intent_step(
    step_id: str,
    task_type: str,
    object_selector: dict,
    **updates,
) -> dict:
    step = {
        "id": step_id,
        "task_type": task_type,
        "object": object_selector,
        "target": _selector(),
        "relation": "none",
        "required_arm": "auto",
        "transfer_arm": "none",
        "receive_arm": "none",
        "orientation_goal": "upright" if task_type == "E2" else "none",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "none",
        "depends_on": [],
    }
    step.update(updates)
    return step


def _ground_draft(
    task_id: str,
    instruction: str,
    scene_objects: list[dict],
    steps: list[dict],
    bindings: dict[str, list[str]],
):
    return ground_instruction_draft(
        task_id,
        instruction,
        {"steps": steps},
        scene_objects,
        robot_profile="ur10",
        reference_bindings=bindings,
    )


def _historical_task2_1_graph() -> dict:
    return instantiate_seed_graph(
        make_task2_1_historical_spec(),
        TASK2_1_HISTORICAL_ROLE_BINDINGS,
    )


def _actions_by_task_group(graph: dict) -> dict[str, list[str]]:
    nodes = {node["id"]: node for node in graph["nodes"]}
    return {
        group["id"]: [nodes[node_id]["atomic_action"] for node_id in group["node_ids"]]
        for group in graph["task_groups"]
    }


def test_historical_task2_1_fixture_preserves_ten_step_semantics() -> None:
    task = make_task2_1_historical_spec()

    assert [instance["task_type"] for instance in task["task_instances"]] == [
        "E2",
        "E2",
        "E4",
        "E1",
        "E4",
        "E1",
        "E1",
        "E1",
        "E4",
        "E1",
    ]
    assert [term["task_instance_id"] for term in task["success"]["terms"]] == [
        instance["id"] for instance in task["task_instances"]
    ]
    assert TASK2_1_HISTORICAL_SCENE_FINGERPRINT["config_sha256"] == (
        "1042967c9b7021f518e82ace62aa824015a3ad50639fa8a326b7dc0474277481"
    )


def test_historical_task2_1_uses_axis_align_and_explicit_handover_arms() -> None:
    task = make_task2_1_historical_spec()
    graph = _historical_task2_1_graph()
    actions = _actions_by_task_group(graph)

    expected_orient_actions = [
        "AxisAlign",
        "MoveEndEffector",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert actions["task_01"] == expected_orient_actions
    assert actions["task_02"] == expected_orient_actions
    assert [
        (
            instance["params"]["transfer_arm"],
            instance["params"]["receive_arm"],
        )
        for instance in task["task_instances"]
        if instance["task_type"] == "E4"
    ] == [
        ("left_arm", "right_arm"),
        ("right_arm", "left_arm"),
        ("left_arm", "right_arm"),
    ]


def test_historical_task2_1_handover_continuations_preserve_receiver_hold() -> None:
    graph = _historical_task2_1_graph()
    actions = _actions_by_task_group(graph)
    groups = {group["id"]: group for group in graph["task_groups"]}

    for group_id, expected_arm in (
        ("task_04", "right_arm"),
        ("task_06", "left_arm"),
        ("task_10", "right_arm"),
    ):
        assert actions[group_id][0] == "MoveHeldObject"
        assert "PickUp" not in actions[group_id]
        assert groups[group_id]["contract"]["entry_requires"] == [
            {
                "predicate": "object_held",
                "object_uid": groups[group_id]["object_uid"],
                "arm": expected_arm,
            }
        ]


def test_historical_task2_1_reacquires_objects_after_release() -> None:
    graph = _historical_task2_1_graph()
    actions = _actions_by_task_group(graph)

    assert actions["task_05"][0] == "PickUp"
    assert actions["task_07"][0] == "PickUp"
    assert actions["task_08"][0] == "PickUp"
    assert actions["task_09"][0] == "PickUp"


def test_orient_then_handover_releases_then_reacquires_with_role_side_pickup() -> None:
    task = {
        "schema_version": "action_engine_task_spec_v2",
        "task_id": "orient_then_handover",
        "level": "L3",
        "instruction": "test-instruction-orient-handover",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "orient",
                "task_type": "E2",
                "params": {
                    "object_role": "can",
                    "required_arm": "left_arm",
                },
                "depends_on": [],
                "role": "primary",
            },
            {
                "id": "handover",
                "task_type": "E4",
                "params": {
                    "object_role": "can",
                    "transfer_arm": "right_arm",
                    "receive_arm": "left_arm",
                },
                "depends_on": ["orient"],
                "role": "primary",
            },
        ],
        "success": {"type": "handover_complete"},
        "oracle": {},
        "metadata": {},
    }

    graph = instantiate_seed_graph(task, {"can": "interact_can"})
    orient_nodes = [
        node for node in graph["nodes"] if node["task_instance_id"] == "orient"
    ]
    handover_nodes = [
        node for node in graph["nodes"] if node["task_instance_id"] == "handover"
    ]
    orient = next(group for group in graph["task_groups"] if group["id"] == "orient")

    assert [node["atomic_action"] for node in orient_nodes] == [
        "AxisAlign",
        "MoveEndEffector",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert orient_nodes[0]["motion_policy"] == {"modifiers": []}
    assert [node["role"] for node in orient_nodes] == [
        "primary",
        "cleanup",
        "cleanup",
        "cleanup",
    ]
    assert orient_nodes[1]["target_binding"] == {
        "kind": "policy_pose",
        "source": "release",
        "operation": "lift_clear",
    }
    assert orient_nodes[1]["motion_policy"] == {"modifiers": []}
    assert orient_nodes[2]["target_binding"] == {
        "kind": "policy_pose",
        "source": "release",
        "operation": "retreat_after_lift",
    }
    assert orient_nodes[3]["target_binding"] == {
        "kind": "joint_state",
        "source": "initial",
        "operation": "e2_home",
    }
    assert orient_nodes[1]["depends_on"] == [orient_nodes[0]["id"]]
    assert orient_nodes[2]["depends_on"] == [orient_nodes[1]["id"]]
    assert orient_nodes[3]["depends_on"] == [orient_nodes[2]["id"]]
    assert [node["atomic_action"] for node in handover_nodes] == [
        "PickUp",
        "MoveHeldObject",
        "HandOver",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert handover_nodes[0]["motion_policy"] == {
        "modifiers": [{"type": "handover_role", "mode": "transfer"}]
    }
    assert handover_nodes[0]["depends_on"] == [orient["node_ids"][-1]]
    assert orient["actor"] == {"mode": "required", "arm": "left_arm"}
    assert handover_nodes[0]["actor"] == {"mode": "required", "arm": "right_arm"}
    assert orient_nodes[-1]["contract"]["completion"] == "terminal_barrier"
    assert orient_nodes[0]["contract"]["failure_policy"] == "task_required"
    assert orient_nodes[1]["contract"]["failure_policy"] == "safety_required"
    assert orient_nodes[2]["contract"]["failure_policy"] == "safety_required"
    assert orient_nodes[-1]["contract"]["failure_policy"] == "safety_required"
    assert orient_nodes[1]["contract"]["requires"] == [
        {"predicate": "arm_free", "arm": "left_arm"}
    ]
    assert orient_nodes[2]["contract"]["requires"] == [
        {"predicate": "arm_clear", "arm": "left_arm"}
    ]
    assert any(
        effect["atom"]["predicate"] == "arm_home"
        for effect in orient["contract"]["exit_effects"]
    )
    assert any(
        requirement["predicate"] == "object_free"
        for requirement in handover_nodes[0]["contract"]["requires"]
    )


def test_pour_recipe_establishes_hold_and_requires_observable_contents() -> None:
    task = {
        "schema_version": "action_engine_task_spec_v2",
        "task_id": "pour_contents",
        "level": "L1",
        "instruction": "Pour the ball from the cup into the bin.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "pour",
                "task_type": "E3",
                "params": {
                    "source_role": "cup",
                    "target_role": "bin",
                    "content_roles": ["ball"],
                    "required_arm": "right_arm",
                },
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "poured"},
        "oracle": {},
        "metadata": {},
    }

    graph = instantiate_seed_graph(
        task,
        {"cup": "source_cup", "bin": "target_bin", "ball": "content_ball"},
    )
    nodes = graph["nodes"]
    group = graph["task_groups"][0]

    assert [node["atomic_action"] for node in nodes] == [
        "PickUp",
        "MoveHeldObject",
        "Pour",
    ]
    assert nodes[1]["depends_on"] == [nodes[0]["id"]]
    assert nodes[2]["depends_on"] == [nodes[1]["id"]]
    assert nodes[2]["contract"]["completion"] == "terminal_barrier"
    assert nodes[2]["contract"]["failure_policy"] == "task_required"
    assert group["success"]["contents"] == [{"object": "content_ball"}]
    assert nodes[2]["target_binding"]["payloads"] == [{"object": "content_ball"}]


def test_handover_to_place_uses_receiver_hold_without_repickup() -> None:
    task = {
        "schema_version": "action_engine_task_spec_v2",
        "task_id": "handover_then_place",
        "level": "L3",
        "instruction": "test-instruction-handover-place",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "task_01",
                "task_type": "E4",
                "params": {
                    "object_role": "yellow_can",
                    "transfer_arm": "left_arm",
                    "receive_arm": "right_arm",
                    "orientation_goal": "preserve",
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

    graph = instantiate_seed_graph(
        task,
        {
            "yellow_can": "interact_yellow_can",
            "purple_can": "interact_purple_can",
        },
    )

    handover = next(group for group in graph["task_groups"] if group["id"] == "task_01")
    placement = next(
        group for group in graph["task_groups"] if group["id"] == "task_02"
    )
    placement_nodes = [
        node for node in graph["nodes"] if node["task_instance_id"] == "task_02"
    ]

    assert [
        node["atomic_action"]
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_01"
    ] == [
        "PickUp",
        "MoveHeldObject",
        "HandOver",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert handover["actor"] == {"mode": "required", "arm": "left_arm"}
    assert graph["nodes"][0]["motion_policy"] == {
        "modifiers": [{"type": "handover_role", "mode": "transfer"}]
    }
    assert graph["nodes"][1]["target_binding"]["kind"] == "handover_staging"
    assert graph["nodes"][2]["motion_policy"] == {"modifiers": []}
    handover_retreat = graph["nodes"][3]
    assert handover_retreat["actor"] == {"mode": "required", "arm": "left_arm"}
    assert handover_retreat["target_binding"] == {
        "kind": "policy_pose",
        "source": "handover",
        "operation": "retreat",
    }
    assert handover_retreat["motion_policy"] == {"modifiers": []}
    assert handover_retreat["depends_on"] == [graph["nodes"][2]["id"]]
    handover_home = graph["nodes"][4]
    assert handover_home["actor"] == {"mode": "required", "arm": "left_arm"}
    assert handover_home["target_binding"] == {
        "kind": "joint_state",
        "source": "initial",
        "operation": "handover_home",
    }
    assert handover_home["motion_policy"] == {"modifiers": []}
    assert handover_home["depends_on"] == [handover_retreat["id"]]
    assert [node["atomic_action"] for node in placement_nodes] == [
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert placement["actor"] == {"mode": "required", "arm": "right_arm"}
    assert placement_nodes[0]["precondition"] == {
        "type": "object_held",
        "object": "interact_yellow_can",
        "arm": "right_arm",
    }
    assert placement_nodes[0]["depends_on"] == [handover["node_ids"][-1]]


def test_structured_draft_grounds_handover_then_receiver_placement() -> None:
    scene = [
        {
            "runtime_uid": "table",
            "role": "background",
            "description": "A table.",
            "init_pos": [0.0, 0.0, 0.0],
        },
        {
            "runtime_uid": "interact_yellow_can",
            "role": "rigid_object",
            "description": "A yellow soda can.",
            "init_pos": [0.0, -0.25, 0.75],
        },
        {
            "runtime_uid": "interact_purple_can",
            "role": "rigid_object",
            "description": "A purple soda can.",
            "init_pos": [0.0, 0.25, 0.75],
        },
    ]

    planned = _ground_draft(
        "handover_then_place",
        "test-instruction-handover-place",
        scene,
        [
            _intent_step(
                "handover",
                "E4",
                _selector("scene_ref", reference="object-alpha"),
                transfer_arm="left_arm",
                receive_arm="right_arm",
            ),
            _intent_step(
                "place",
                "E1",
                _selector("step_result", step_id="handover"),
                target=_selector("scene_ref", reference="object-beta"),
                relation="right_of",
                required_arm="right_arm",
            ),
        ],
        {
            "handover.object": ["interact_yellow_can"],
            "place.target": ["interact_purple_can"],
        },
    )
    graph = instantiate_seed_graph(planned.task_spec, planned.role_bindings)

    assert planned.task_spec["level"] == "L3"
    assert [item["task_type"] for item in planned.task_spec["task_instances"]] == [
        "E4",
        "E1",
    ]
    assert planned.role_bindings == {
        "object_01": "interact_yellow_can",
        "object_02": "interact_purple_can",
    }
    assert [node["atomic_action"] for node in graph["nodes"]] == [
        "PickUp",
        "MoveHeldObject",
        "HandOver",
        "MoveEndEffector",
        "MoveJoints",
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert graph["task_groups"][1]["goal"]["relation"] == "right_of"
    assert graph["task_groups"][1]["goal"]["relation_frame"] == "robot"


def test_seed_graph_adds_missing_same_object_e2_handover_dependency() -> None:
    scene = [
        {
            "runtime_uid": "purple_can",
            "role": "rigid_object",
            "description": "A purple soda can.",
            "init_pos": [0.0, -0.25, 0.7],
        },
        {
            "runtime_uid": "orange_can",
            "role": "rigid_object",
            "description": "An orange soda can.",
            "init_pos": [0.0, 0.2, 0.7],
        },
    ]
    planned = _ground_draft(
        "missing_same_object_edge",
        "test-instruction-multi-step",
        scene,
        [
            _intent_step(
                "orient_purple",
                "E2",
                _selector("scene_ref", reference="object-alpha"),
                required_arm="right_arm",
            ),
            _intent_step(
                "orient_orange",
                "E2",
                _selector("scene_ref", reference="object-beta"),
                required_arm="left_arm",
                depends_on=["orient_purple"],
            ),
            _intent_step(
                "handover_purple",
                "E4",
                _selector("step_result", step_id="orient_purple"),
                transfer_arm="right_arm",
                receive_arm="left_arm",
                depends_on=["orient_orange"],
            ),
            _intent_step(
                "place_purple",
                "E1",
                _selector("step_result", step_id="handover_purple"),
                target=_selector("scene_ref", reference="object-beta"),
                relation="left_of",
                required_arm="left_arm",
            ),
        ],
        {
            "orient_purple.object": ["purple_can"],
            "orient_orange.object": ["orange_can"],
            "place_purple.target": ["orange_can"],
        },
    )
    underconstrained = deepcopy(planned.task_spec)
    underconstrained["task_instances"][2]["depends_on"] = ["task_02"]

    graph = instantiate_seed_graph(underconstrained, planned.role_bindings)
    handover = next(group for group in graph["task_groups"] if group["id"] == "task_03")
    purple = next(group for group in graph["task_groups"] if group["id"] == "task_01")
    staging = next(
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_03"
        and node["atomic_action"] == "MoveHeldObject"
    )
    assert handover["depends_on"] == ["task_02", "task_01"]
    assert [
        node["atomic_action"]
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_03"
    ] == ["PickUp", "MoveHeldObject", "HandOver", "MoveEndEffector", "MoveJoints"]
    pickup = next(
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_03" and node["atomic_action"] == "PickUp"
    )
    assert purple["node_ids"][-1] in pickup["depends_on"]
    assert staging["depends_on"] == [pickup["id"]]


def test_structured_draft_treats_table_as_support_in_generic_line_task() -> None:
    scene = [
        {
            "runtime_uid": "table",
            "role": "background",
            "description": "A table.",
            "init_pos": [0.0, 0.0, 0.0],
        },
        {
            "runtime_uid": "interact_red_can",
            "role": "rigid_object",
            "description": "A red soda can.",
            "init_pos": [0.0, -0.25, 0.75],
        },
        {
            "runtime_uid": "interact_blue_cup",
            "role": "rigid_object",
            "description": "A blue cup.",
            "init_pos": [0.0, 0.25, 0.75],
        },
    ]

    planned = _ground_draft(
        "arrange_line",
        "test-instruction-line",
        scene,
        [
            _intent_step(
                "line",
                "E1",
                _selector(
                    "scene_ref",
                    reference="object-set",
                    quantifier="all",
                ),
                layout="line",
            )
        ],
        {"line.object": ["interact_red_can", "interact_blue_cup"]},
    )
    graph = instantiate_seed_graph(planned.task_spec, planned.role_bindings)

    assert planned.task_spec["level"] == "L2"
    assert set(planned.role_bindings.values()) == {
        "interact_red_can",
        "interact_blue_cup",
    }
    assert all(group["operator"] == "arrange_line" for group in graph["task_groups"])
