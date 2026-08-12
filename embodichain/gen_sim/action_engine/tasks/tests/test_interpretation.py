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
from embodichain.gen_sim.action_engine.tasks import (
    INSTRUCTION_INTENT_SCHEMA,
    instantiate_seed_graph,
    interpret_and_ground_task_spec,
    plan_grounded_task_spec,
    validate_instruction_intent,
)


def _selector(kind: str = "none", **values):
    result = {
        "kind": kind,
        "step_id": "",
        "uid": "",
        "category": "none",
        "color": "none",
        "side": "none",
        "quantifier": "one",
        "count": 0,
    }
    result.update(values)
    return result


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
                _selector("selector", category="can", color="purple"),
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
                target=_selector("selector", category="can", color="orange"),
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
                _selector("selector", category="can", color="purple"),
                required_arm="right_arm",
            ),
            _step(
                "orient_orange",
                "E2",
                _selector("selector", category="can", color="orange"),
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


def test_selector_side_is_a_conjunctive_robot_frame_constraint() -> None:
    intent = {
        "steps": [
            _step(
                "orient",
                "E2",
                _selector(
                    "selector",
                    category="can",
                    color="purple",
                    side="right",
                ),
            )
        ]
    }
    with pytest.raises(ValueError, match="did not match"):
        interpret_and_ground_task_spec(
            "conflict",
            "扶正右边紫色易拉罐。",
            _scene(),
            robot_profile="ur10",
            caller=lambda **_kwargs: intent,
        )


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


def test_invalid_step_result_gets_repair_with_selector_rules() -> None:
    """A malformed cross-step selector should reach the structured repair call."""
    invalid_intent = _handover_intent()
    invalid_intent["steps"][1]["object"]["category"] = "can"
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
    )

    assert len(prompts) == 2
    repair_prompt = prompts[1]
    for term in ("step_result", "step_id", "uid", "category", "color", "side"):
        assert term in repair_prompt
    assert "none" in repair_prompt
    assert grounded.task_spec["metadata"]["instruction_call_count"] == 2


def test_repeated_missing_e1_target_gets_verified_local_completion() -> None:
    invalid_intent = _two_object_handover_intent_with_missing_place_target()
    prompts: list[str] = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return deepcopy(invalid_intent)

    grounded = interpret_and_ground_task_spec(
        "verified_target_completion",
        "用右臂把紫色易拉罐扶正，然后用左臂把橘色罐头扶正，然后用右臂把紫色罐头递给左臂，"
        "然后右臂撤回，然后左臂将其放到橘色易拉罐的左边。",
        _scene(),
        robot_profile="ur10",
        caller=caller,
    )

    placement = grounded.task_spec["task_instances"][-1]
    target_role = placement["params"]["target_role"]
    metadata = grounded.task_spec["metadata"]
    assert len(prompts) == 2
    assert "Missing-target repair rule" in prompts[1]
    assert grounded.role_bindings[target_role] == "orange_can"
    assert metadata["instruction_call_count"] == 2
    assert metadata["instruction_local_completion_count"] == 1
    assert metadata["instruction_local_completion_fields"] == ["steps[3].target"]
    assert (
        metadata["instruction_local_completion_basis"]
        == "deterministic_scene_grounding"
    )


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


def test_intent_normalizes_orange_alias_and_infers_pronoun_dependency() -> None:
    intent = {
        "steps": [
            _step(
                "handover",
                "E4",
                _selector("selector", category="can", color="purple"),
                transfer_arm="right_arm",
                receive_arm="left_arm",
            ),
            _step(
                "place",
                "E1",
                _selector("step_result", step_id="handover"),
                target=_selector("selector", category="can", color="橘色"),
                relation="左边",
                required_arm="左臂",
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
    )

    instances = grounded.task_spec["task_instances"]
    assert [item["task_type"] for item in instances] == ["E4", "E1"]
    assert instances[1]["depends_on"] == [instances[0]["id"]]
    assert instances[1]["params"]["relation"] == "left_of"
    assert instances[1]["params"]["required_arm"] == "left_arm"


def test_selector_rejects_unknown_uid_and_attribute_conflicts() -> None:
    unknown = {
        "steps": [
            _step(
                "orient",
                "E2",
                _selector("selector", uid="invented_uid"),
            )
        ]
    }
    with pytest.raises(ValueError, match="unknown scene UID"):
        interpret_and_ground_task_spec(
            "unknown_uid",
            "扶正它。",
            _scene(),
            robot_profile="ur10",
            caller=lambda **_kwargs: unknown,
        )

    conflict = {
        "steps": [
            _step(
                "orient",
                "E2",
                _selector(
                    "selector",
                    uid="purple_can",
                    category="can",
                    color="orange",
                ),
            )
        ]
    }
    with pytest.raises(ValueError, match="conflicts with UID.*color"):
        interpret_and_ground_task_spec(
            "attribute_conflict",
            "扶正紫色易拉罐。",
            _scene(),
            robot_profile="ur10",
            caller=lambda **_kwargs: conflict,
        )


def test_selector_uid_and_ordinal_side_are_conjunctive() -> None:
    intent = {
        "steps": [
            _step(
                "orient",
                "E2",
                _selector(
                    "selector",
                    uid="orange_can",
                    category="can",
                    side="leftmost",
                ),
            )
        ]
    }
    with pytest.raises(ValueError, match="not the unique robot-relative leftmost"):
        interpret_and_ground_task_spec(
            "ordinal_uid_conflict",
            "扶正最左边的橘色易拉罐。",
            _scene(),
            robot_profile="ur10",
            caller=lambda **_kwargs: intent,
        )


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
                _selector("selector", category="can", color="purple"),
            ),
        ]
    }
    with pytest.raises(ValueError, match="preceding step"):
        validate_instruction_intent(intent)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uid", "purple_can"),
        ("category", "can"),
        ("color", "purple"),
        ("side", "left"),
    ],
)
def test_step_result_selector_rejects_object_constraints(
    field: str, value: str
) -> None:
    intent = _handover_intent()
    intent["steps"][1]["object"][field] = value

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
        )


def test_prompt_inventory_includes_table_and_schema_is_strict() -> None:
    captured = {}
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
        captured.update(kwargs)
        return intent

    grounded = interpret_and_ground_task_spec(
        "onto_table",
        "Put the purple can on the table.",
        _scene_with_table(),
        robot_profile="ur10",
        caller=caller,
    )

    assert '"uid": "table"' in captured["prompt"]
    assert '"core_actions"' not in captured["prompt"]
    assert captured["schema"] == INSTRUCTION_INTENT_SCHEMA
    assert grounded.role_bindings["object_02"] == "table"


def test_instruction_intent_schema_declares_every_required_selector_field() -> None:
    selector_schema = INSTRUCTION_INTENT_SCHEMA["properties"]["steps"]["items"][
        "properties"
    ]["object"]

    assert set(selector_schema["required"]) == set(selector_schema["properties"])
    assert "quantifier" in selector_schema["properties"]


def test_instruction_prompt_redacts_nested_scene_geometry() -> None:
    scene = _scene()
    scene[0]["attributes"] = {
        "label": "purple",
        "geometry": {"position": [0.0, 0.0, 0.7], "note": "can"},
    }
    captured: dict[str, str] = {}

    def caller(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return {
            "steps": [
                _step(
                    "orient",
                    "E2",
                    _selector("selector", uid="purple_can"),
                )
            ]
        }

    interpret_and_ground_task_spec(
        "redacted_inventory",
        "扶正紫色易拉罐。",
        scene,
        robot_profile="ur10",
        caller=caller,
    )
    assert '"position"' not in captured["prompt"]
    assert '"label": "purple"' in captured["prompt"]


def test_default_llm_parser_requires_the_documented_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ACTION_ENGINE_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(interpretation_module, "_load_local_env", lambda: {})

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
        interpretation_module,
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
                    _selector("selector", category="can", color="purple"),
                )
            ]
        },
    )

    assert grounded.task_spec["metadata"]["instruction_model"] == "injected_caller"


def test_mimo_instruction_caller_uses_json_mode_and_disables_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MiMo-compatible endpoints must not use the lossy JSON-schema route."""
    import langchain_openai
    from embodichain.gen_sim.action_engine.planning import planner

    calls: list[dict] = []
    responses = [
        {
            "steps": [
                {
                    "id": "orient",
                    "task_type": "E2",
                    "object": _selector("selector", category="can", color="purple"),
                }
            ]
        },
        _handover_intent(),
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
        planner,
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
    assert len(calls) == 2
    for call in calls:
        assert call["structured_kwargs"] == {"method": "json_mode"}
        assert call["kwargs"]["max_completion_tokens"] == 4096
        assert call["kwargs"]["extra_body"] == {"thinking": {"type": "disabled"}}
    repair_messages = calls[1]["messages"]
    assert "previous JSON was invalid" in repair_messages[1].content


def test_instruction_prompt_contains_a_complete_shape_example() -> None:
    index = interpretation_module._SceneIndex(_scene(), robot_profile="ur10")
    prompt = interpretation_module._instruction_prompt("扶正紫色易拉罐。", index)
    selector_rules = interpretation_module._instruction_selector_rules()
    assert '"target_setting": 0' in prompt
    assert '"depends_on": []' in prompt
    assert "every step has all 14 step keys" in prompt
    assert "step_result" in prompt
    assert "Prefer an exact inventory UID" in prompt
    assert "conjunctive constraints" in prompt
    assert "step_result" in selector_rules
    assert "step_id" in selector_rules
    for field in ("uid", "category", "color", "side"):
        assert field in selector_rules


def test_scene_export_spatial_descriptions_do_not_create_false_supports() -> None:
    index = interpretation_module._SceneIndex(
        _scene_export_style_scene(), robot_profile="franka"
    )

    assert [entity.uid for entity in index.support] == ["table"]
    assert {entity.uid for entity in index.movable} == {
        "carrot_001",
        "cutting_board_001",
        "peeler_001",
    }


def test_scene_export_exact_uids_ground_pick_and_place() -> None:
    index = interpretation_module._SceneIndex(
        _scene_export_style_scene(), robot_profile="franka"
    )
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

    grounded = interpretation_module._ground_intent(
        "scene_export_pick_place",
        "先用左臂把胡萝卜放到砧板上",
        intent,
        index,
    )

    assert set(grounded.role_bindings.values()) == {
        "carrot_001",
        "cutting_board_001",
    }
    assert grounded.task_spec["task_instances"][0]["params"]["required_arm"] == (
        "left_arm"
    )


def test_deterministic_parser_handles_mixed_language_pronouns_and_handover() -> None:
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


def test_deterministic_parser_consumes_transfer_arm_retreat_as_handover_cleanup() -> (
    None
):
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


def test_deterministic_parser_keeps_target_side_distinct_from_relation_side() -> None:
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


def test_deterministic_parser_resolves_explicit_multi_object_count() -> None:
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


def test_deterministic_parser_does_not_treat_uid_digits_as_quantity() -> None:
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


def test_deterministic_parser_keeps_chinese_target_side_as_selector() -> None:
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
