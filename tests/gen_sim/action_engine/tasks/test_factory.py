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

import pytest

from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.tasks import (
    TaskFactory,
    ground_instruction_draft,
    instantiate_seed_graph,
    validate_scene_handoff,
)
from embodichain.gen_sim.action_engine.runtime.motion_policy import (
    resolve_motion_policy,
)


def _bindings(requirements: dict) -> dict[str, str]:
    return {
        item["role_id"]: f"scene_{item['role_id']}" for item in requirements["objects"]
    }


def _scene(requirements: dict, *, with_camera: bool = True) -> dict:
    return {
        "objects": [
            {
                "uid": f"scene_{item['role_id']}",
                "category": item["category"],
                "affordances": item["affordances"],
                "initial_state": item["initial_state"],
                "attributes": item["attributes"],
            }
            for item in requirements["objects"]
        ],
        "cameras": (
            [
                {
                    "uid": "front_camera",
                    "modalities": ["rgb", "depth"],
                    "coverage": "all_interaction_objects",
                }
            ]
            if with_camera
            else []
        ),
        "satisfied_spatial_constraints": requirements["spatial_constraints"],
    }


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


def test_fixed_seed_batch_of_one_thousand_is_reproducible_and_valid() -> None:
    first = TaskFactory(1729).generate_batch(1000)
    second = TaskFactory(1729).generate_batch(1000)

    assert first.tasks == second.tasks
    assert first.scene_requirements == second.scene_requirements
    assert len({task["task_id"] for task in first.tasks}) == 1000
    assert (
        len(
            {
                repr(
                    {
                        key: value
                        for key, value in task.items()
                        if key not in {"task_id", "metadata"}
                    }
                )
                for task in first.tasks
            }
        )
        == 1000
    )
    registry = build_atomic_capability_registry()
    for task, requirements in zip(first.tasks, first.scene_requirements):
        graph = instantiate_seed_graph(task, _bindings(requirements))
        assert graph["task_id"] == task["task_id"]
        for node in graph["nodes"]:
            if registry.get(node["atomic_action"]).runtime_available:
                resolve_motion_policy(
                    "dual_ur10",
                    node["atomic_action"],
                    node["motion_policy"],
                )
    assert {task["level"] for task in first.tasks} == {"L1", "L2", "L3", "L4"}


def test_executable_only_never_emits_planning_only_task_types() -> None:
    batch = TaskFactory(31, executable_only=True).generate_batch(200)
    emitted = {
        instance["task_type"]
        for task in batch.tasks
        for instance in task["task_instances"]
    }

    assert emitted <= {"E1", "E2", "E4", "E5", "E9"}


@pytest.mark.parametrize("level", ["L1", "L2", "L3", "L4"])
def test_scene_handoff_instantiates_direct_atomic_action_graph(level: str) -> None:
    task, requirements = TaskFactory(9, executable_only=True).generate(level, 4)
    bindings = _bindings(requirements)
    handoff = validate_scene_handoff(requirements, _scene(requirements), bindings)
    graph = instantiate_seed_graph(task, handoff.role_bindings)

    assert graph["task_id"] == task["task_id"]
    assert graph["level"] == level
    assert {group["id"] for group in graph["task_groups"]} == {
        instance["id"] for instance in task["task_instances"]
    }
    assert all("atomic_action" in node for node in graph["nodes"])
    assert not any("target_pose" in node for node in graph["nodes"])


def test_scene_handoff_rejects_affordance_or_camera_mismatch() -> None:
    _, requirements = TaskFactory(2).generate("L4", 1)
    bindings = _bindings(requirements)
    scene = _scene(requirements, with_camera=False)
    with pytest.raises(ValueError, match="cameras"):
        validate_scene_handoff(requirements, scene, bindings)

    scene = _scene(requirements)
    broken = deepcopy(scene)
    broken["objects"][0]["affordances"] = []
    with pytest.raises(ValueError, match="lacks affordances"):
        validate_scene_handoff(requirements, broken, bindings)


def test_planning_only_graph_is_generated_but_runtime_preflight_rejects_it() -> None:
    factory = TaskFactory(10)
    for index in range(100):
        task, requirements = factory.generate("L1", index)
        if task["task_instances"][0]["task_type"] in {"E3", "E6", "E7", "E8"}:
            break
    else:
        raise AssertionError("Expected a planning-only task in deterministic sample.")
    graph = instantiate_seed_graph(task, _bindings(requirements))

    from embodichain.gen_sim.action_engine.runtime.loader import load_execution_program

    with pytest.raises(ValueError, match="planning-only"):
        load_execution_program(graph)


def test_orient_then_handover_releases_then_reacquires_with_role_side_pickup() -> None:
    task = {
        "schema_version": "action_engine_task_spec_v2",
        "task_id": "orient_then_handover",
        "level": "L3",
        "instruction": "扶正易拉罐后递给另一只手。",
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
        "PickUp",
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]
    assert orient_nodes[0]["motion_policy"] == {
        "modifiers": [{"type": "orientation", "mode": "upright"}]
    }
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
    assert orient_nodes[-2]["contract"]["failure_policy"] == "safety_required"
    assert orient_nodes[-1]["contract"]["failure_policy"] == "best_effort"
    assert not any(
        effect["atom"]["predicate"] == "arm_home"
        for effect in orient["contract"]["exit_effects"]
    )
    assert any(
        requirement["predicate"] == "object_free"
        for requirement in handover_nodes[0]["contract"]["requires"]
    )


def test_handover_to_place_uses_receiver_hold_without_repickup() -> None:
    task = {
        "schema_version": "action_engine_task_spec_v2",
        "task_id": "handover_then_place",
        "level": "L3",
        "instruction": ("用左臂拿起黄色易拉罐并交给右臂，然后放到紫色易拉罐右边。"),
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
        "用左臂把左侧的黄色易拉罐交接到右臂上，然后放到右边紫色易拉罐右边",
        scene,
        [
            _intent_step(
                "handover",
                "E4",
                _selector("scene_ref", reference="黄色易拉罐"),
                transfer_arm="left_arm",
                receive_arm="right_arm",
            ),
            _intent_step(
                "place",
                "E1",
                _selector("step_result", step_id="handover"),
                target=_selector("scene_ref", reference="紫色易拉罐"),
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
        "用右臂把紫色易拉罐扶正，然后用左臂把橘色罐头扶正，然后用右臂把紫色罐头递给左臂，"
        "然后左臂将其放到橘色易拉罐的左边",
        scene,
        [
            _intent_step(
                "orient_purple",
                "E2",
                _selector("scene_ref", reference="紫色易拉罐"),
                required_arm="right_arm",
            ),
            _intent_step(
                "orient_orange",
                "E2",
                _selector("scene_ref", reference="橘色易拉罐"),
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
                target=_selector("scene_ref", reference="橘色易拉罐"),
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
        "把桌面上的东西摆成一排",
        scene,
        [
            _intent_step(
                "line",
                "E1",
                _selector(
                    "scene_ref",
                    reference="桌面上的东西",
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
