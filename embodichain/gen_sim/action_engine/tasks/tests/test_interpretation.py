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

import embodichain.gen_sim.action_engine.tasks.interpretation as interpretation_module
import embodichain.gen_sim.task_engine.interpretation as task_interpretation_module
from embodichain.gen_sim.action_engine.tasks.assembly import SceneInventory
from embodichain.gen_sim.action_engine.tasks import (
    INSTRUCTION_INTENT_SCHEMA,
    instantiate_seed_graph,
    interpret_and_ground_task_spec,
    plan_grounded_task_spec,
    validate_instruction_intent,
)


def _selector(kind: str = "none", **values):
    legacy_kind = kind
    if kind == "selector":
        kind = "scene_ref"
    reference = values.pop("reference", "")
    if legacy_kind == "selector":
        uid = str(values.pop("uid", "")).strip()
        legacy_terms = [
            str(values.pop(field, "")).strip()
            for field in ("side", "color", "category")
        ]
        reference = reference or uid
        if not reference:
            reference = " ".join(
                term for term in legacy_terms if term not in {"", "none"}
            )
    result = {
        "kind": kind,
        "step_id": "",
        "reference": reference,
        "quantifier": "one",
        "count": 0,
    }
    result.update(values)
    return result


def _grounding(**bindings):
    return {
        "bindings": [
            {
                "reference_id": reference_id,
                "status": "resolved",
                "uids": [uid] if isinstance(uid, str) else list(uid),
                "confidence": 1.0,
            }
            for reference_id, uid in bindings.items()
        ]
    }


def _grounding_caller(**bindings):
    response = _grounding(**bindings)
    return lambda **_kwargs: deepcopy(response)


def _step(step_id: str, task_type: str, object_selector: dict, **values):
    result = {
        "id": step_id,
        "task_type": task_type,
        "object": object_selector,
        "target": _selector(),
        "relation": "none",
        "required_arm": "auto",
        "transfer_arm": "none",
        "receive_arm": "none",
        "orientation_goal": "upright" if task_type == "E2" else "preserve",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "none",
        "depends_on": [],
    }
    result.update(values)
    return result


def _scene():
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


def _scene_with_table():
    return [
        {
            "runtime_uid": "table",
            "uid": "table",
            "role": "background",
            "description": "A table.",
            "init_pos": [0.0, 0.0, 0.0],
        },
        *_scene(),
    ]


def _scene_export_style_scene():
    return [
        {
            "runtime_uid": "table",
            "uid": "table",
            "role": "background",
            "description": "A light grey dining table.",
            "init_pos": [0.0, 0.0, 0.0],
        },
        {
            "runtime_uid": "carrot_001",
            "uid": "carrot_001",
            "role": "rigid_object",
            "category": "carrot",
            "description": (
                "A single orange carrot with a green top located at the top left "
                "of the table."
            ),
            "init_pos": [0.28, 0.47, 1.06],
        },
        {
            "runtime_uid": "cutting_board_001",
            "uid": "cutting_board_001",
            "role": "rigid_object",
            "category": "cutting_board",
            "description": (
                "A rectangular cutting board located in the upper middle-left "
                "area of the table."
            ),
            "init_pos": [0.14, 0.21, 1.07],
        },
        {
            "runtime_uid": "peeler_001",
            "uid": "peeler_001",
            "role": "rigid_object",
            "category": "vegetable_peeler",
            "description": "A black-handled vegetable peeler.",
            "init_pos": [-0.13, -0.61, 1.07],
        },
    ]


def _payload_scene():
    return [
        {
            "runtime_uid": "glue_stick",
            "uid": "glue_stick",
            "role": "object",
            "category": "glue_stick",
            "description": "A solid glue stick.",
            "init_pos": [0.0, -0.2, 0.7],
        },
        {
            "runtime_uid": "paper_cup",
            "uid": "paper_cup",
            "role": "object",
            "category": "cup",
            "description": "A paper cup.",
            "init_pos": [0.0, 0.0, 0.7],
        },
        {
            "runtime_uid": "popcorn_bucket",
            "uid": "popcorn_bucket",
            "role": "object",
            "category": "bucket",
            "description": "A popcorn bucket.",
            "init_pos": [0.0, 0.25, 0.7],
        },
    ]


def _handover_intent():
    return {
        "steps": [
            _step(
                "orient",
                "E2",
                _selector("scene_ref", reference="紫色易拉罐"),
                required_arm="right_arm",
            ),
            _step(
                "handover",
                "E4",
                _selector("step_result", step_id="orient"),
                transfer_arm="right_arm",
                receive_arm="left_arm",
                depends_on=["orient"],
            ),
            _step(
                "place",
                "E1",
                _selector("step_result", step_id="handover"),
                target=_selector("scene_ref", reference="橘色易拉罐"),
                relation="left_of",
                required_arm="left_arm",
                depends_on=["handover"],
            ),
        ]
    }


def _two_object_handover_intent_with_missing_place_target():
    return {
        "steps": [
            _step(
                "orient_purple",
                "E2",
                _selector("scene_ref", reference="紫色易拉罐"),
                required_arm="right_arm",
            ),
            _step(
                "orient_orange",
                "E2",
                _selector("scene_ref", reference="橘色易拉罐"),
                required_arm="left_arm",
                depends_on=["orient_purple"],
            ),
            _step(
                "handover",
                "E4",
                _selector("step_result", step_id="orient_purple"),
                transfer_arm="right_arm",
                receive_arm="left_arm",
                depends_on=["orient_orange"],
            ),
            _step(
                "place",
                "E1",
                _selector("step_result", step_id="handover"),
                relation="left_of",
                required_arm="left_arm",
                depends_on=["handover"],
            ),
        ]
    }


def test_llm_intent_handles_handover_pronoun_and_elliptical_place() -> None:
    calls = []

    def caller(**kwargs):
        calls.append(kwargs)
        return _handover_intent()

    grounded = interpret_and_ground_task_spec(
        "handover_task",
        "用右臂扶正紫色易拉罐，然后递给左臂，然后将其橘色罐头的左边。",
        _scene(),
        robot_profile="ur10",
        model="test-model",
        caller=caller,
        grounding_caller=_grounding_caller(
            **{
                "orient.object": "purple_can",
                "place.target": "orange_can",
            }
        ),
    )
    graph = instantiate_seed_graph(grounded.task_spec, grounded.role_bindings)

    assert [item["task_type"] for item in grounded.task_spec["task_instances"]] == [
        "E2",
        "E4",
        "E1",
    ]
    assert grounded.role_bindings == {
        "object_01": "purple_can",
        "object_02": "orange_can",
    }
    placement_actions = [
        node["atomic_action"] for node in graph["nodes"] if node["task_type"] == "E1"
    ]
    orient_actions = [
        node["atomic_action"] for node in graph["nodes"] if node["task_type"] == "E2"
    ]
    handover_nodes = [node for node in graph["nodes"] if node["task_type"] == "E4"]
    assert orient_actions == [
        "PickUp",
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]
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
    assert any(
        requirement["predicate"] == "object_free"
        for requirement in handover_nodes[0]["contract"]["requires"]
    )
    assert placement_actions == [
        "MoveHeldObject",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]
    out_of_order = deepcopy(grounded.task_spec)
    out_of_order["task_instances"] = list(reversed(out_of_order["task_instances"]))
    reordered_graph = instantiate_seed_graph(
        out_of_order,
        grounded.role_bindings,
    )
    assert [group["task_type"] for group in reordered_graph["task_groups"]] == [
        "E2",
        "E4",
        "E1",
    ]
    assert "递给" in calls[0]["prompt"]
    assert calls[0]["model"] == "test-model"


def test_e5_relative_transport_emits_pickment_and_optional_release() -> None:
    scene = [
        {
            "runtime_uid": "table",
            "uid": "table",
            "role": "background",
            "category": "table",
            "description": "table",
            "init_pos": [0.0, 0.0, 0.0],
        },
        {
            "runtime_uid": "plastic_tray",
            "uid": "plastic_tray",
            "role": "object",
            "category": "tray",
            "description": "plastic tray",
            "init_pos": [0.0, 0.0, 0.7],
        },
        {
            "runtime_uid": "banana_left",
            "uid": "banana_left",
            "role": "object",
            "category": "banana",
            "description": "left banana",
            "init_pos": [0.0, 0.25, 0.7],
        },
    ]
    intent = {
        "steps": [
            _step(
                "move_tray",
                "E5",
                _selector("selector", uid="plastic_tray"),
                target=_selector("selector", uid="banana_left"),
                relation="behind",
                direction="none",
                terminal_behavior="hold",
            )
        ]
    }
    deterministic = plan_grounded_task_spec(
        task_name="dual_tray_deterministic",
        task_description="用双臂把桌上的盘子移动到左边香蕉的后面",
        scene_objects=scene,
        robot_profile="franka",
    )
    deterministic_instance = deterministic.task_spec["task_instances"][0]
    assert deterministic_instance["task_type"] == "E5"
    assert deterministic_instance["params"]["relation"] == "behind"

    grounded = interpret_and_ground_task_spec(
        "dual_tray",
        "用双臂把桌上的盘子移动到左边香蕉的后面",
        scene,
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: intent,
        grounding_caller=_grounding_caller(
            **{
                "move_tray.object": "plastic_tray",
                "move_tray.target": "banana_left",
            }
        ),
    )

    graph = instantiate_seed_graph(grounded.task_spec, grounded.role_bindings)
    assert [node["atomic_action"] for node in graph["nodes"]] == ["CoordinatedPickment"]
    assert graph["task_groups"][0]["operator"] == "coordinated_transport"
    assert graph["task_groups"][0]["goal"] == {
        "direction": "none",
        "terminal_behavior": "hold",
        "orientation_goal": "preserve",
        "orientation_axis": "none",
        "relation_frame": "robot",
        "reference_object": "banana_left",
        "reference_state": "live",
        "relation": "behind",
    }

    released_spec = deepcopy(grounded.task_spec)
    released_spec["task_instances"][0]["params"]["terminal_behavior"] = "place"
    released = instantiate_seed_graph(released_spec, grounded.role_bindings)
    assert [node["atomic_action"] for node in released["nodes"]] == [
        "CoordinatedPickment",
        "MoveJoints",
        "MoveJoints",
    ]
    release_nodes = released["nodes"][1:]
    assert all(
        node["depends_on"] == [released["nodes"][0]["id"]] for node in release_nodes
    )
    assert {node["actor"]["arm"] for node in release_nodes} == {
        "left_arm",
        "right_arm",
    }
    assert {node["control"] for node in release_nodes} == {"hand"}
    assert len({node["sync_group"] for node in release_nodes}) == 1
    assert all(node["precondition"] == {} for node in release_nodes)
    assert {
        node["target_binding"]["coordinated_release_role"] for node in release_nodes
    } == {"participant", "commit"}
    contracts = {
        node["target_binding"]["coordinated_release_role"]: node["contract"]
        for node in release_nodes
    }
    coordinated_hold = {
        "predicate": "object_coordinated_held",
        "object_uid": "plastic_tray",
    }
    assert contracts["participant"]["requires"] == [coordinated_hold]
    assert contracts["participant"]["effects"] == []
    assert contracts["commit"]["requires"] == [coordinated_hold]
    assert {
        (
            effect["op"],
            effect["atom"]["predicate"],
            effect["atom"].get("arm"),
        )
        for effect in contracts["commit"]["effects"]
    } == {
        ("delete", "object_coordinated_held", None),
        ("add", "object_free", None),
        ("add", "arm_free", "left_arm"),
        ("add", "arm_free", "right_arm"),
    }
    from embodichain.gen_sim.action_engine.runtime import load_execution_program

    program = load_execution_program(released)
    assert [
        action["atomic_action_class"]
        for edge in program.edges
        for action in edge.actions
    ] == ["CoordinatedPickment", "MoveJoints", "MoveJoints"]
    assert len(program.edges[-1].actions) == 2

    in_place_spec = deepcopy(released_spec)
    in_place_params = in_place_spec["task_instances"][0]["params"]
    in_place_params.pop("target_role")
    in_place_params.update({"direction": "none", "relation": "none"})
    in_place = instantiate_seed_graph(in_place_spec, grounded.role_bindings)
    assert [node["atomic_action"] for node in in_place["nodes"]] == [
        "CoordinatedPickment",
        "MoveJoints",
        "MoveJoints",
    ]
    assert "reference_object" not in in_place["task_groups"][0]["goal"]


def test_e5_accepts_generic_rigid_object_without_exported_affordances() -> None:
    scene = [
        {
            "runtime_uid": "interact_wooden_block",
            "uid": "interact_wooden_block",
            "role": "rigid_object",
            "description": "A long rectangular wooden block.",
            "init_pos": [0.0, 0.0, 0.7],
        }
    ]
    intent = {
        "steps": [
            _step(
                "move_block",
                "E5",
                _selector("selector", uid="interact_wooden_block"),
                direction="left",
                terminal_behavior="hold",
            )
        ]
    }

    grounded = interpret_and_ground_task_spec(
        "dual_block",
        "用双臂把桌上的长方体往左移动",
        scene,
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: intent,
        grounding_caller=_grounding_caller(
            **{"move_block.object": "interact_wooden_block"}
        ),
    )

    instance = grounded.task_spec["task_instances"][0]
    assert instance["task_type"] == "E5"
    assert grounded.role_bindings[instance["params"]["object_role"]] == (
        "interact_wooden_block"
    )
    assert instance["params"]["direction"] == "left"


def test_task1_2_open_reference_generates_coordinated_pick_move_and_release() -> None:
    scene = [
        {
            "runtime_uid": "table",
            "uid": "table",
            "role": "background",
            "description": "A white table.",
            "init_pos": [0.0, 0.0, 0.0],
        },
        {
            "runtime_uid": "interact_apple",
            "uid": "interact_apple",
            "role": "rigid_object",
            "description": "A red apple.",
            "init_pos": [0.0, -0.2, 0.7],
        },
        {
            "runtime_uid": "interact_wooden_tray",
            "uid": "interact_wooden_tray",
            "role": "rigid_object",
            "description": "A long rectangular wooden tray.",
            "init_pos": [0.0, 0.0, 0.7],
        },
        {
            "runtime_uid": "interact_rubiks_cube",
            "uid": "interact_rubiks_cube",
            "role": "rigid_object",
            "description": "A Rubik's cube.",
            "init_pos": [0.0, 0.2, 0.7],
        },
    ]
    intent = {
        "steps": [
            _step(
                "move_block",
                "E5",
                _selector("scene_ref", reference="桌上的长方体"),
                direction="left",
                terminal_behavior="place",
            )
        ]
    }

    grounded = interpret_and_ground_task_spec(
        "task1_2",
        "用双臂把桌上的长方体往左移动并放下",
        scene,
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: intent,
        grounding_caller=_grounding_caller(
            **{"move_block.object": "interact_wooden_tray"}
        ),
    )
    graph = instantiate_seed_graph(grounded.task_spec, grounded.role_bindings)

    instance = grounded.task_spec["task_instances"][0]
    assert instance["params"]["direction"] == "left"
    assert instance["params"]["terminal_behavior"] == "place"
    assert grounded.task_spec["success"]["terms"] == [
        {"type": "semantic_goal", "task_instance_id": instance["id"]}
    ]
    assert [node["atomic_action"] for node in graph["nodes"]] == [
        "CoordinatedPickment",
        "MoveJoints",
        "MoveJoints",
    ]
    assert grounded.scene_requirements["objects"][0]["category"] == "rigid_object"
    assert grounded.task_spec["metadata"]["instruction_call_count"] == 1
    assert grounded.task_spec["metadata"]["scene_grounding_call_count"] == 1


def test_e5_pick_and_hold_defaults_missing_direction_to_up() -> None:
    scene = [
        {
            "runtime_uid": "table",
            "uid": "table",
            "role": "background",
            "description": "A wooden table.",
            "init_pos": [0.0, 0.0, 0.0],
        },
        {
            "runtime_uid": "wooden_tray",
            "uid": "wooden_tray",
            "role": "rigid_object",
            "description": "A shallow round wooden serving tray.",
            "init_pos": [0.0, 0.0, 0.7],
        },
    ]
    intent = {
        "steps": [
            _step(
                "lift_tray",
                "E5",
                _selector("scene_ref", reference="桌上的木盘"),
                required_arm="none",
                direction="none",
                terminal_behavior="hold",
            )
        ]
    }

    grounded = interpret_and_ground_task_spec(
        "lift_tray",
        "用双臂把桌上的木盘端起来",
        scene,
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: deepcopy(intent),
        grounding_caller=_grounding_caller(**{"lift_tray.object": "wooden_tray"}),
    )
    graph = instantiate_seed_graph(grounded.task_spec, grounded.role_bindings)

    instance = grounded.task_spec["task_instances"][0]
    assert instance["params"]["direction"] == "up"
    assert instance["params"]["terminal_behavior"] == "hold"
    assert [node["atomic_action"] for node in graph["nodes"]] == ["CoordinatedPickment"]
    assert grounded.task_spec["success"]["terms"] == [
        {"type": "held_by_both_grippers", "task_instance_id": instance["id"]}
    ]
    assert grounded.task_spec["metadata"]["instruction_call_count"] == 1
    assert grounded.task_spec["metadata"]["instruction_intent_normalizations"] == [
        {
            "path": "steps[0].direction",
            "from": "none",
            "to": "up",
            "reason": "e5_hold_defaults_to_lift",
        }
    ]

    deterministic = plan_grounded_task_spec(
        task_name="lift_tray_deterministic",
        task_description="用双臂把桌上的木盘端起来",
        scene_objects=scene,
        robot_profile="franka",
    )
    deterministic_instance = deterministic.task_spec["task_instances"][0]
    assert deterministic_instance["params"]["direction"] == "up"
    assert deterministic_instance["params"]["terminal_behavior"] == "hold"


@pytest.mark.parametrize(
    ("scene_update", "error"),
    (
        ({"affordances": ["rigid"]}, "missing affordances.*dual_graspable"),
        ({"role": "articulation"}, "requires .*rigid.object structure"),
    ),
)
def test_e5_rejects_explicitly_incompatible_scene_evidence(
    scene_update: dict,
    error: str,
) -> None:
    scene_object = {
        "runtime_uid": "candidate",
        "uid": "candidate",
        "role": "rigid_object",
        "description": "A candidate object.",
        "init_pos": [0.0, 0.0, 0.7],
        **scene_update,
    }
    intent = {
        "steps": [
            _step(
                "move_candidate",
                "E5",
                _selector("selector", uid="candidate"),
                direction="left",
                terminal_behavior="hold",
            )
        ]
    }

    with pytest.raises(ValueError, match=error):
        interpret_and_ground_task_spec(
            "invalid_dual_object",
            "用双臂把物体往左移动",
            [scene_object],
            robot_profile="franka",
            model="test-model",
            caller=lambda **_kwargs: intent,
            grounding_caller=_grounding_caller(
                **{"move_candidate.object": "candidate"}
            ),
        )


@pytest.mark.parametrize(
    ("scene_update", "should_succeed", "error"),
    (
        ({"role": "articulation"}, True, ""),
        (
            {"role": "articulation", "affordances": ["articulated"]},
            False,
            "missing affordances.*pullable",
        ),
        ({"role": "rigid_object"}, False, "requires articulation structure"),
    ),
)
def test_articulated_task_uses_structural_and_explicit_affordance_evidence(
    scene_update: dict,
    should_succeed: bool,
    error: str,
) -> None:
    scene_object = {
        "runtime_uid": "cabinet_part",
        "uid": "cabinet_part",
        "description": "A cabinet moving part.",
        "init_pos": [0.0, 0.0, 0.7],
        **scene_update,
    }
    intent = {
        "steps": [
            _step(
                "open_part",
                "E6",
                _selector("scene_ref", reference="柜子的活动部件"),
                target_state="open",
            )
        ]
    }

    invoke = lambda: interpret_and_ground_task_spec(
        "open_part",
        "打开柜子的活动部件。",
        [scene_object],
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: intent,
        grounding_caller=_grounding_caller(**{"open_part.object": "cabinet_part"}),
    )
    if should_succeed:
        assert invoke().task_spec["task_instances"][0]["task_type"] == "E6"
    else:
        with pytest.raises(ValueError, match=error):
            invoke()


def test_open_container_target_is_allowed_until_runtime_when_metadata_is_unknown() -> (
    None
):
    scene = [
        {
            "runtime_uid": "source_pitcher",
            "uid": "source_pitcher",
            "role": "rigid_object",
            "category": "ceramic_pitcher",
            "description": "A ceramic pitcher with water.",
            "init_pos": [0.0, -0.2, 0.7],
        },
        {
            "runtime_uid": "custom_receiver",
            "uid": "custom_receiver",
            "role": "rigid_object",
            "category": "handmade_vessel",
            "description": "A handmade receiving vessel.",
            "init_pos": [0.0, 0.2, 0.7],
        },
    ]
    intent = {
        "steps": [
            _step(
                "pour",
                "E3",
                _selector("scene_ref", reference="水壶"),
                target=_selector("scene_ref", reference="手工容器"),
                relation="above",
            )
        ]
    }

    grounded = interpret_and_ground_task_spec(
        "open_container",
        "把水壶里的水倒入手工容器。",
        scene,
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: intent,
        grounding_caller=_grounding_caller(
            **{
                "pour.object": "source_pitcher",
                "pour.target": "custom_receiver",
            }
        ),
    )
    assert grounded.task_spec["task_instances"][0]["task_type"] == "E3"

    explicit = deepcopy(scene)
    explicit[1]["affordances"] = ["support_surface"]
    with pytest.raises(ValueError, match="none support containment"):
        interpret_and_ground_task_spec(
            "explicit_non_container",
            "把水壶里的水倒入手工容器。",
            explicit,
            robot_profile="franka",
            model="test-model",
            caller=lambda **_kwargs: intent,
            grounding_caller=_grounding_caller(
                **{
                    "pour.object": "source_pitcher",
                    "pour.target": "custom_receiver",
                }
            ),
        )


def test_open_scene_reference_is_not_limited_by_fixed_selector_fields() -> None:
    intent = {
        "steps": [
            _step(
                "orient",
                "E2",
                _selector("scene_ref", reference="右边紫色易拉罐"),
            )
        ]
    }
    grounded = interpret_and_ground_task_spec(
        "open_reference",
        "扶正右边紫色易拉罐。",
        _scene(),
        robot_profile="ur10",
        caller=lambda **_kwargs: intent,
        grounding_caller=_grounding_caller(**{"orient.object": "purple_can"}),
    )
    assert grounded.role_bindings == {"object_01": "purple_can"}


def test_intent_rejects_atomic_actions_coordinates_and_extra_fields() -> None:
    intent = _handover_intent()
    intent["steps"][0]["atomic_action"] = "PickUp"
    with pytest.raises(ValueError, match="forbidden fields"):
        validate_instruction_intent(intent)

    intent = _handover_intent()
    intent["steps"][0]["object"]["target_pose"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="forbidden fields"):
        validate_instruction_intent(intent)


def test_invalid_intent_gets_one_repair_attempt() -> None:
    responses = [{"steps": []}, _handover_intent()]
    prompts = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return deepcopy(responses[len(prompts) - 1])

    grounded = interpret_and_ground_task_spec(
        "repair",
        "扶正后递给另一只手，再放到另一罐头左边。",
        _scene(),
        robot_profile="ur10",
        caller=caller,
        grounding_caller=_grounding_caller(
            **{
                "orient.object": "purple_can",
                "place.target": "orange_can",
            }
        ),
    )
    assert len(prompts) == 2
    assert "previous JSON was invalid" in prompts[1]
    assert grounded.task_spec["metadata"]["instruction_call_count"] == 2


def test_interpreter_normalizes_registry_inapplicable_e4_required_arm() -> None:
    intent = _handover_intent()
    intent["steps"][1]["required_arm"] = "right_arm"

    with pytest.raises(ValueError, match="uses transfer_arm/receive_arm"):
        validate_instruction_intent(intent)

    grounded = interpret_and_ground_task_spec(
        "normalized_handover",
        "用右臂扶正紫色易拉罐，然后用右臂递给左臂，再放到橘色易拉罐左边。",
        _scene(),
        robot_profile="ur10",
        caller=lambda **_kwargs: deepcopy(intent),
        grounding_caller=_grounding_caller(
            **{
                "orient.object": "purple_can",
                "place.target": "orange_can",
            }
        ),
    )

    assert grounded.task_spec["metadata"]["instruction_call_count"] == 1
    assert grounded.task_spec["metadata"]["instruction_intent_normalizations"] == [
        {
            "path": "steps[1].required_arm",
            "from": "right_arm",
            "to": "none",
            "reason": "inapplicable_for_E4",
        }
    ]


@pytest.mark.parametrize("use_step_result", [True, False])
def test_interpreter_resolves_same_arm_handover_from_adjacent_ownership(
    use_step_result: bool,
) -> None:
    handover_object = (
        _selector("step_result", step_id="orient_sprite")
        if use_step_result
        else _selector("scene_ref", reference="雪碧")
    )
    place_object = (
        _selector("step_result", step_id="handover_sprite")
        if use_step_result
        else _selector("scene_ref", reference="雪碧")
    )
    intent = {
        "steps": [
            _step(
                "orient_coke",
                "E2",
                _selector("scene_ref", reference="可乐"),
                required_arm="right_arm",
            ),
            _step(
                "orient_sprite",
                "E2",
                _selector("scene_ref", reference="雪碧"),
                required_arm="left_arm",
            ),
            _step(
                "handover_sprite",
                "E4",
                handover_object,
                required_arm="none",
                transfer_arm="left_arm",
                receive_arm="left_arm",
                depends_on=["orient_sprite"],
            ),
            _step(
                "place_sprite",
                "E1",
                place_object,
                target=_selector("step_result", step_id="orient_coke"),
                relation="on",
                required_arm="right_arm",
                depends_on=["orient_coke", "handover_sprite"],
            ),
        ]
    }

    with pytest.raises(ValueError, match="transfer and receive arms must differ"):
        validate_instruction_intent(intent)

    result = task_interpretation_module.interpret_instruction_draft(
        "用右臂把可乐摆正，同时用左臂把雪碧扶正，然后左臂把雪碧递给左臂，"
        "然后右臂把雪碧放到可乐上。",
        model="test-model",
        caller=lambda **_kwargs: deepcopy(intent),
    )

    handover = result.intent["steps"][2]
    assert (handover["transfer_arm"], handover["receive_arm"]) == (
        "left_arm",
        "right_arm",
    )
    assert result.attempts == 1
    assert result.normalizations == (
        {
            "path": "steps[2].receive_arm",
            "from": "left_arm",
            "to": "right_arm",
            "reason": "handover_arm_continuity",
        },
    )


def test_interpreter_does_not_guess_an_unconstrained_same_arm_handover() -> None:
    intent = {
        "steps": [
            _step(
                "handover",
                "E4",
                _selector("scene_ref", reference="雪碧"),
                transfer_arm="left_arm",
                receive_arm="left_arm",
            )
        ]
    }
    calls = 0

    def caller(**_kwargs):
        nonlocal calls
        calls += 1
        return deepcopy(intent)

    with pytest.raises(ValueError, match="after one repair.*arms must differ"):
        task_interpretation_module.interpret_instruction_draft(
            "左臂把雪碧递给左臂。",
            model="test-model",
            caller=caller,
        )

    assert calls == 2


def test_invalid_step_result_gets_repair_with_selector_rules() -> None:
    """A malformed cross-step selector should reach the structured repair call."""
    invalid_intent = _handover_intent()
    invalid_intent["steps"][1]["object"]["reference"] = "紫色易拉罐"
    responses = [invalid_intent, _handover_intent()]
    prompts: list[str] = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return deepcopy(responses[len(prompts) - 1])

    grounded = interpret_and_ground_task_spec(
        "repair_step_result",
        "用右臂扶正紫色易拉罐，然后递给左臂，再放到橘色易拉罐左边。",
        _scene(),
        robot_profile="ur10",
        caller=caller,
        grounding_caller=_grounding_caller(
            **{
                "orient.object": "purple_can",
                "place.target": "orange_can",
            }
        ),
    )

    assert len(prompts) == 2
    repair_prompt = prompts[1]
    for term in ("step_result", "step_id", "reference"):
        assert term in repair_prompt
    assert "none" in repair_prompt
    assert grounded.task_spec["metadata"]["instruction_call_count"] == 2


def test_repeated_missing_e1_target_fails_without_local_guessing() -> None:
    invalid_intent = _two_object_handover_intent_with_missing_place_target()
    prompts: list[str] = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return deepcopy(invalid_intent)

    with pytest.raises(ValueError, match="after one repair"):
        interpret_and_ground_task_spec(
            "missing_target",
            "用右臂把紫色易拉罐扶正，然后用左臂把橘色罐头扶正，然后用右臂把紫色罐头递给左臂，"
            "然后右臂撤回，然后左臂将其放到橘色易拉罐的左边。",
            _scene(),
            robot_profile="ur10",
            caller=caller,
        )

    assert len(prompts) == 2
    assert "Missing-target repair rule" in prompts[1]


def test_missing_target_completion_rejects_other_semantic_disagreement() -> None:
    invalid_intent = _two_object_handover_intent_with_missing_place_target()
    invalid_intent["steps"][-1]["required_arm"] = "right_arm"

    with pytest.raises(ValueError, match="after one repair"):
        interpret_and_ground_task_spec(
            "unsafe_target_completion",
            "用右臂把紫色易拉罐扶正，然后用左臂把橘色罐头扶正，然后用右臂把紫色罐头递给左臂，"
            "然后右臂撤回，然后左臂将其放到橘色易拉罐的左边。",
            _scene(),
            robot_profile="ur10",
            caller=lambda **_kwargs: deepcopy(invalid_intent),
        )


def test_second_invalid_intent_fails_without_rule_fallback() -> None:
    with pytest.raises(ValueError, match="after one repair"):
        interpret_and_ground_task_spec(
            "invalid",
            "递给。",
            _scene(),
            robot_profile="ur10",
            caller=lambda **_kwargs: {"steps": []},
        )


def test_intent_infers_pronoun_dependency_from_canonical_symbols() -> None:
    intent = {
        "steps": [
            _step(
                "handover",
                "E4",
                _selector("scene_ref", reference="紫色易拉罐"),
                transfer_arm="right_arm",
                receive_arm="left_arm",
            ),
            _step(
                "place",
                "E1",
                _selector("step_result", step_id="handover"),
                target=_selector("scene_ref", reference="橘色易拉罐"),
                relation="left_of",
                required_arm="left_arm",
                depends_on=["handover"],
            ),
        ]
    }

    grounded = interpret_and_ground_task_spec(
        "implicit_dependency",
        "递给左臂，再将其放在橘色罐头左边。",
        _scene(),
        robot_profile="ur10",
        caller=lambda **_kwargs: intent,
        grounding_caller=_grounding_caller(
            **{
                "handover.object": "purple_can",
                "place.target": "orange_can",
            }
        ),
    )

    instances = grounded.task_spec["task_instances"]
    assert [item["task_type"] for item in instances] == ["E4", "E1"]
    assert instances[1]["depends_on"] == [instances[0]["id"]]
    assert instances[1]["params"]["relation"] == "left_of"
    assert instances[1]["params"]["required_arm"] == "left_arm"


def test_scene_grounding_rejects_unknown_uid() -> None:
    intent = {
        "steps": [
            _step(
                "orient",
                "E2",
                _selector("scene_ref", reference="紫色易拉罐"),
            )
        ]
    }
    with pytest.raises(ValueError, match="after one repair.*unknown UIDs"):
        interpret_and_ground_task_spec(
            "unknown_uid",
            "扶正紫色易拉罐。",
            _scene(),
            robot_profile="ur10",
            caller=lambda **_kwargs: intent,
            grounding_caller=_grounding_caller(**{"orient.object": "invented_uid"}),
        )


def test_instruction_intent_rejects_legacy_selector_protocol() -> None:
    intent = _handover_intent()
    intent["steps"][0]["object"] = {
        "kind": "selector",
        "step_id": "",
        "uid": "purple_can",
        "category": "can",
        "color": "purple",
        "side": "none",
        "quantifier": "one",
        "count": 0,
    }
    with pytest.raises(ValueError, match="requires exactly fields"):
        validate_instruction_intent(intent)


def test_step_result_must_reference_a_preceding_step() -> None:
    intent = {
        "steps": [
            _step(
                "handover",
                "E4",
                _selector("step_result", step_id="orient"),
                transfer_arm="right_arm",
                receive_arm="left_arm",
            ),
            _step(
                "orient",
                "E2",
                _selector("scene_ref", reference="紫色易拉罐"),
            ),
        ]
    }
    with pytest.raises(ValueError, match="preceding step"):
        validate_instruction_intent(intent)


def test_step_result_selector_rejects_object_constraints() -> None:
    intent = _handover_intent()
    intent["steps"][1]["object"]["reference"] = "紫色易拉罐"

    with pytest.raises(ValueError, match="may identify only a prior step_id"):
        validate_instruction_intent(intent)


def test_instruction_intent_rejects_non_e_specific_parameters() -> None:
    invalid_e9 = _step(
        "press",
        "E9",
        _selector("selector", category="button"),
        target_state="activated",
        orientation_goal="upright",
    )
    with pytest.raises(ValueError, match="orientation_goal"):
        validate_instruction_intent({"steps": [invalid_e9]})

    invalid_line = _step(
        "line",
        "E1",
        _selector("selector", category="can", quantifier="all"),
        layout="line",
        relation="on",
    )
    with pytest.raises(ValueError, match="line arrangement cannot carry a relation"):
        validate_instruction_intent({"steps": [invalid_line]})


def test_implicit_e1_relation_requires_an_unambiguous_support_target() -> None:
    intent = {
        "steps": [
            _step(
                "place",
                "E1",
                _selector("selector", category="can", color="purple"),
                target=_selector("selector", category="can", color="orange"),
            )
        ]
    }

    with pytest.raises(ValueError, match="omitted relation"):
        interpret_and_ground_task_spec(
            "ambiguous_implicit_place",
            "把紫色罐放到橘色罐。",
            _scene(),
            robot_profile="ur10",
            caller=lambda **_kwargs: intent,
            grounding_caller=_grounding_caller(
                **{
                    "place.object": "purple_can",
                    "place.target": "orange_can",
                }
            ),
        )


def test_instruction_and_grounding_prompts_keep_their_boundaries() -> None:
    captured: dict[str, dict] = {}
    intent = {
        "steps": [
            _step(
                "place",
                "E1",
                _selector("selector", category="can", color="purple"),
                target=_selector("selector", uid="table", category="table"),
                relation="on",
            )
        ]
    }

    def caller(**kwargs):
        captured["intent"] = kwargs
        return intent

    def grounding_caller(**kwargs):
        captured["grounding"] = kwargs
        return _grounding(**{"place.object": "purple_can", "place.target": "table"})

    grounded = interpret_and_ground_task_spec(
        "onto_table",
        "Put the purple can on the table.",
        _scene_with_table(),
        robot_profile="ur10",
        caller=caller,
        grounding_caller=grounding_caller,
    )

    assert '"uid": "table"' not in captured["intent"]["prompt"]
    assert '"uid": "table"' in captured["grounding"]["prompt"]
    assert '"core_actions"' not in captured["intent"]["prompt"]
    assert captured["intent"]["schema"] == INSTRUCTION_INTENT_SCHEMA
    assert grounded.role_bindings["object_02"] == "table"


def test_instruction_intent_schema_declares_every_required_selector_field() -> None:
    selector_schema = INSTRUCTION_INTENT_SCHEMA["properties"]["steps"]["items"][
        "properties"
    ]["object"]

    assert set(selector_schema["required"]) == set(selector_schema["properties"])
    assert "quantifier" in selector_schema["properties"]


def test_grounding_prompt_redacts_nested_scene_geometry() -> None:
    scene = _scene()
    scene[0]["attributes"] = {
        "label": "purple",
        "geometry": {"position": [0.0, 0.0, 0.7], "note": "can"},
    }
    captured: dict[str, str] = {}

    def grounding_caller(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return _grounding(**{"orient.object": "purple_can"})

    interpret_and_ground_task_spec(
        "redacted_inventory",
        "扶正紫色易拉罐。",
        scene,
        robot_profile="ur10",
        caller=lambda **_kwargs: {
            "steps": [
                _step(
                    "orient",
                    "E2",
                    _selector("scene_ref", reference="紫色易拉罐"),
                )
            ]
        },
        grounding_caller=grounding_caller,
    )
    assert '"position"' not in captured["prompt"]
    assert '"label": "purple"' in captured["prompt"]


def test_default_llm_parser_requires_the_documented_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ACTION_ENGINE_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(task_interpretation_module, "_load_local_env", lambda: {})

    with pytest.raises(ValueError, match="text LLM model is required"):
        interpret_and_ground_task_spec(
            "missing_model",
            "扶正紫色易拉罐。",
            _scene(),
            robot_profile="ur10",
        )


def test_injected_caller_skips_production_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_model_resolution(_explicit: str | None) -> str | None:
        raise AssertionError(
            "injected callers must not resolve production model config"
        )

    monkeypatch.setattr(
        task_interpretation_module,
        "_instruction_model",
        unexpected_model_resolution,
    )

    grounded = interpret_and_ground_task_spec(
        "injected_caller",
        "扶正紫色易拉罐。",
        _scene(),
        robot_profile="ur10",
        caller=lambda **_kwargs: {
            "steps": [
                _step(
                    "orient",
                    "E2",
                    _selector("scene_ref", reference="紫色易拉罐"),
                )
            ]
        },
        grounding_caller=_grounding_caller(**{"orient.object": "purple_can"}),
    )

    assert grounded.task_spec["metadata"]["instruction_model"] == "injected_caller"


def test_mimo_instruction_caller_uses_json_mode_and_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MiMo-compatible endpoints must not use the lossy JSON-schema route."""
    import langchain_openai

    calls: list[dict] = []
    responses = [
        {
            "steps": [
                {
                    "id": "orient",
                    "task_type": "E2",
                    "object": _selector("scene_ref", reference="紫色易拉罐"),
                }
            ]
        },
        _handover_intent(),
        _grounding(
            **{
                "orient.object": "purple_can",
                "place.target": "orange_can",
            }
        ),
    ]

    class FakeRunnable:
        def invoke(self, messages):
            calls[-1]["messages"] = messages
            return deepcopy(responses.pop(0))

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append({"kwargs": kwargs})

        def with_structured_output(self, schema, **kwargs):
            calls[-1]["schema"] = schema
            calls[-1]["structured_kwargs"] = kwargs
            return FakeRunnable()

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        task_interpretation_module,
        "_load_llm_settings",
        lambda *, model: {
            "api_key": "test-key",
            "model": model or "mimo-v2.5",
            "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "default_query": {},
        },
    )

    grounded = interpret_and_ground_task_spec(
        "mimo_repair",
        "用右臂扶正紫色易拉罐，然后递给左臂，再放到橘色易拉罐左边。",
        _scene(),
        robot_profile="ur10",
        model="mimo-v2.5",
    )

    assert [item["task_type"] for item in grounded.task_spec["task_instances"]] == [
        "E2",
        "E4",
        "E1",
    ]
    assert len(calls) == 3
    for call in calls:
        assert call["structured_kwargs"] == {"method": "json_mode"}
        assert call["kwargs"]["max_completion_tokens"] == 4096
        assert call["kwargs"]["extra_body"] == {"thinking": {"type": "disabled"}}
    repair_messages = calls[1]["messages"]
    assert "previous JSON was invalid" in repair_messages[1].content


def test_instruction_prompt_contains_a_complete_shape_example() -> None:
    prompt = interpretation_module._instruction_prompt("扶正紫色易拉罐。")
    selector_rules = interpretation_module._instruction_selector_rules()
    assert '"target_setting": 0' in prompt
    assert '"depends_on": []' in prompt
    assert "every step has all 16 step keys" in prompt
    assert "step_result" in prompt
    assert "open scene_ref.reference" in prompt
    assert "Do not classify it or emit a scene UID" in prompt
    assert "step_result" in selector_rules
    assert "step_id" in selector_rules
    assert "reference" in selector_rules


def test_scene_export_spatial_descriptions_do_not_create_false_supports() -> None:
    index = SceneInventory(_scene_export_style_scene(), robot_profile="franka")

    assert [entity.uid for entity in index.support] == ["table"]
    assert {entity.uid for entity in index.movable} == {
        "carrot_001",
        "cutting_board_001",
        "peeler_001",
    }


def test_scene_export_exact_uids_ground_pick_and_place() -> None:
    intent = {
        "steps": [
            _step(
                "step_1",
                "E1",
                _selector("selector", uid="carrot_001"),
                target=_selector("selector", uid="cutting_board_001"),
                relation="on",
                required_arm="left_arm",
            )
        ]
    }

    grounded = interpret_and_ground_task_spec(
        "scene_export_pick_place",
        "先用左臂把胡萝卜放到砧板上",
        _scene_export_style_scene(),
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: intent,
        grounding_caller=_grounding_caller(
            **{
                "step_1.object": "carrot_001",
                "step_1.target": "cutting_board_001",
            }
        ),
    )

    assert set(grounded.role_bindings.values()) == {
        "carrot_001",
        "cutting_board_001",
    }
    assert grounded.task_spec["task_instances"][0]["params"]["required_arm"] == (
        "left_arm"
    )
    assert {item["category"] for item in grounded.scene_requirements["objects"]} == {
        "carrot",
        "cutting_board",
    }


def test_multi_object_handover_keeps_both_order_and_holder_dependencies() -> None:
    grounded = plan_grounded_task_spec(
        "multi_object_handover",
        "用右臂把紫色易拉罐扶正，然后用左臂把橘色罐头扶正，然后用右臂把紫色罐头递给左臂，"
        "然后右臂撤回，然后左臂将其放到橘色易拉罐的左边",
        _scene(),
        robot_profile="ur10",
    )

    instances = grounded.task_spec["task_instances"]
    assert instances[2]["depends_on"] == ["task_02", "task_01"]
    graph = instantiate_seed_graph(grounded.task_spec, grounded.role_bindings)
    handover = next(group for group in graph["task_groups"] if group["id"] == "task_03")
    handover_nodes = [
        node for node in graph["nodes"] if node["task_instance_id"] == "task_03"
    ]
    assert handover["depends_on"] == ["task_02", "task_01"]
    assert [node["atomic_action"] for node in handover_nodes] == [
        "PickUp",
        "MoveHeldObject",
        "HandOver",
        "MoveEndEffector",
        "MoveJoints",
    ]
    purple = next(group for group in graph["task_groups"] if group["id"] == "task_01")
    orange = next(group for group in graph["task_groups"] if group["id"] == "task_02")
    assert handover_nodes[0]["depends_on"] == [
        orange["node_ids"][-1],
        purple["node_ids"][-1],
    ]


def test_single_arm_e1_propagates_direct_payload_into_goal_and_contracts() -> None:
    intent = {
        "steps": [
            _step(
                "handover_glue",
                "E4",
                _selector("selector", uid="glue_stick"),
                required_arm="left_arm",
                transfer_arm="left_arm",
                receive_arm="right_arm",
            ),
            _step(
                "place_glue",
                "E1",
                _selector("step_result", step_id="handover_glue"),
                target=_selector("selector", uid="paper_cup"),
                relation="on",
                required_arm="right_arm",
                depends_on=["handover_glue"],
            ),
            _step(
                "place_cup",
                "E1",
                _selector("selector", uid="paper_cup"),
                target=_selector("selector", uid="popcorn_bucket"),
                relation="on",
                required_arm="right_arm",
                depends_on=["place_glue"],
            ),
        ]
    }
    grounded = interpret_and_ground_task_spec(
        "payload_chain",
        "用左臂把固体胶递给右臂，然后右臂将固体胶放到纸杯上，再然后右臂把纸杯放到爆米花桶上。",
        _payload_scene(),
        robot_profile="ur10",
        caller=lambda **_kwargs: deepcopy(intent),
        grounding_caller=_grounding_caller(
            **{
                "handover_glue.object": "glue_stick",
                "place_glue.target": "paper_cup",
                "place_cup.object": "paper_cup",
                "place_cup.target": "popcorn_bucket",
            }
        ),
    )

    graph = instantiate_seed_graph(grounded.task_spec, grounded.role_bindings)
    carrier_group = next(
        group for group in graph["task_groups"] if group["id"] == "task_03"
    )
    assert carrier_group["goal"]["payloads"] == [
        {"object": "glue_stick", "slot": "center"}
    ]
    carrier_nodes = [
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == carrier_group["id"]
        and node["atomic_action"] in {"PickUp", "MoveHeldObject", "Place"}
    ]
    assert carrier_nodes
    for node in carrier_nodes:
        assert node["target_binding"]["payloads"] == carrier_group["goal"]["payloads"]
        assert any(
            claim["resource"] == "object:glue_stick" and claim["access"] == "exclusive"
            for claim in node["contract"]["claims"]
        )


def test_seed_graph_repairs_missing_e2_handover_lifecycle_edge() -> None:
    grounded = plan_grounded_task_spec(
        "missing_lifecycle_edge",
        "用右臂把紫色易拉罐扶正，然后用左臂把橘色罐头扶正，然后用右臂把紫色罐头递给左臂，"
        "然后左臂将其放到橘色易拉罐的左边",
        _scene(),
        robot_profile="ur10",
    )
    underconstrained = deepcopy(grounded.task_spec)
    underconstrained["task_instances"][2]["depends_on"] = ["task_02"]

    graph = instantiate_seed_graph(underconstrained, grounded.role_bindings)
    handover = next(group for group in graph["task_groups"] if group["id"] == "task_03")
    purple = next(group for group in graph["task_groups"] if group["id"] == "task_01")
    assert handover["depends_on"] == ["task_02", "task_01"]
    pickup = next(
        node
        for node in graph["nodes"]
        if node["task_instance_id"] == "task_03" and node["atomic_action"] == "PickUp"
    )
    assert purple["node_ids"][-1] in pickup["depends_on"]
