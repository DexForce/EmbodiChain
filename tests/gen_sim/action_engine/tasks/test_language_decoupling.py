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

"""Acceptance tests for the structured-LLM language boundary."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import embodichain.gen_sim.action_engine.tasks as action_engine_tasks
from embodichain.gen_sim.action_engine.tasks import (
    instantiate_seed_graph,
    interpret_and_ground_task_spec,
    validate_instruction_intent,
)
from embodichain.gen_sim.action_engine.tasks.assembly import SceneInventory


def _selector(
    kind: str = "none",
    *,
    reference: str = "",
    step_id: str = "",
) -> dict:
    return {
        "kind": kind,
        "step_id": step_id,
        "reference": reference,
        "quantifier": "one",
        "count": 0,
    }


def _step(step_id: str, task_type: str, reference: str, **updates: object) -> dict:
    step = {
        "id": step_id,
        "task_type": task_type,
        "object": _selector("scene_ref", reference=reference),
        "target": _selector(),
        "relation": "none",
        "required_arm": "none",
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
    step.update(updates)
    return step


def _binding(reference_id: str, *uids: str) -> dict:
    return {
        "reference_id": reference_id,
        "status": "resolved",
        "uids": list(uids),
        "confidence": 1.0,
    }


def _grounding_caller(*bindings: dict):
    response = {"bindings": list(bindings)}
    return lambda **_kwargs: deepcopy(response)


def _open_scene() -> list[dict]:
    return [
        {
            "runtime_uid": "work_surface",
            "uid": "work_surface",
            "role": "support_surface",
            "category": "obsidian_dock",
            "name": "the landing ledge",
            "description": "A flat black ledge used as a work surface.",
            "init_pos": [0.0, 0.0, 0.0],
        },
        {
            "runtime_uid": "aerogel_fixture_7",
            "uid": "aerogel_fixture_7",
            "role": "rigid_object",
            "category": "aerogel_fixture",
            "name": "translucent fixture",
            "description": "A translucent rectangular fixture with a frosted edge.",
            "init_pos": [0.0, 0.1, 0.7],
        },
        {
            "runtime_uid": "plantain_marker",
            "uid": "plantain_marker",
            "role": "rigid_object",
            "category": "plantain_marker",
            "description": "A curved yellow marker behind the fixture.",
            "init_pos": [0.1, -0.2, 0.7],
        },
    ]


def test_scene_inventory_preserves_open_category_labels() -> None:
    scene = _open_scene()
    scene[1]["category"] = "Prototype.Fixture/V2"
    inventory = SceneInventory(scene, robot_profile="franka")

    assert inventory.by_uid["aerogel_fixture_7"].category == ("Prototype.Fixture/V2")


@pytest.mark.parametrize(
    ("step", "invalid_field"),
    [
        (
            _step(
                "place",
                "E1",
                "半透明的夹具",
                target=_selector("scene_ref", reference="黑色承台"),
                relation="左边",
            ),
            "relation",
        ),
        (
            _step("orient", "E2", "半透明的夹具", required_arm="左臂"),
            "required_arm",
        ),
        (
            _step(
                "orient",
                "E2",
                "半透明的夹具",
                orientation_goal="竖直",
            ),
            "orientation_goal",
        ),
    ],
)
def test_llm_intent_rejects_natural_language_aliases(
    step: dict,
    invalid_field: str,
) -> None:
    """Canonical protocol fields are not a second local language parser."""
    with pytest.raises(ValueError, match=invalid_field):
        validate_instruction_intent({"steps": [step]})


def test_noncanonical_llm_value_is_repaired_instead_of_locally_normalized() -> None:
    invalid = {
        "steps": [
            _step(
                "place",
                "E1",
                "半透明的夹具",
                target=_selector("scene_ref", reference="黑色承台"),
                relation="左边",
            )
        ]
    }
    valid = deepcopy(invalid)
    valid["steps"][0]["relation"] = "left_of"
    responses = [invalid, valid]
    prompts: list[str] = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return deepcopy(responses[len(prompts) - 1])

    grounded = interpret_and_ground_task_spec(
        "strict_canonical_repair",
        "把半透明的夹具搁到黑色承台左边。",
        _open_scene(),
        robot_profile="franka",
        model="test-model",
        caller=caller,
        grounding_caller=_grounding_caller(
            _binding("place.object", "aerogel_fixture_7"),
            _binding("place.target", "work_surface"),
        ),
    )

    assert len(prompts) == 2
    assert "previous JSON was invalid" in prompts[1]
    assert grounded.task_spec["metadata"]["instruction_call_count"] == 2
    assert grounded.task_spec["task_instances"][0]["params"]["relation"] == ("left_of")
    assert "instruction_intent_normalizations" not in grounded.task_spec["metadata"]


def test_two_noncanonical_llm_responses_fail_without_grounding_or_rule_fallback() -> (
    None
):
    invalid = {
        "steps": [
            _step(
                "place",
                "E1",
                "半透明的夹具",
                target=_selector("scene_ref", reference="黑色承台"),
                relation="左边",
            )
        ]
    }
    grounding_called = False

    def unexpected_grounding(**_kwargs):
        nonlocal grounding_called
        grounding_called = True
        raise AssertionError("invalid canonical intent must not reach grounding")

    with pytest.raises(ValueError, match="after one repair.*relation"):
        interpret_and_ground_task_spec(
            "strict_canonical_failure",
            "把半透明的夹具搁到黑色承台左边。",
            _open_scene(),
            robot_profile="franka",
            model="test-model",
            caller=lambda **_kwargs: deepcopy(invalid),
            grounding_caller=unexpected_grounding,
        )

    assert grounding_called is False


def test_legacy_instruction_parser_modules_and_api_are_absent() -> None:
    tasks_dir = Path(action_engine_tasks.__file__).resolve().parent

    assert not (tasks_dir / "deterministic.py").exists()
    assert not (tasks_dir / "planning.py").exists()
    assert not hasattr(action_engine_tasks, "plan_grounded_task_spec")


def test_production_sources_do_not_reference_legacy_instruction_parser() -> None:
    action_engine_dir = Path(action_engine_tasks.__file__).resolve().parent.parent
    forbidden = (
        "tasks.deterministic",
        "tasks.planning",
        "plan_grounded_task_spec",
        "instruction_parser",
        "deterministic_fallback",
    )
    offenders: dict[str, list[str]] = {}
    for path in action_engine_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        matches = [term for term in forbidden if term in source]
        if matches:
            offenders[str(path.relative_to(action_engine_dir))] = matches

    assert offenders == {}


def test_llm_caller_exception_propagates_without_scene_grounding() -> None:
    expected = RuntimeError("model unavailable")
    grounding_called = False

    def fail_model(**_kwargs):
        raise expected

    def unexpected_grounding(**_kwargs):
        nonlocal grounding_called
        grounding_called = True
        raise AssertionError("failed interpretation must not reach grounding")

    with pytest.raises(RuntimeError) as caught:
        interpret_and_ground_task_spec(
            "model_failure",
            "请把半透明构件安顿在落物台上。",
            _open_scene(),
            robot_profile="franka",
            model="test-model",
            caller=fail_model,
            grounding_caller=unexpected_grounding,
        )

    assert caught.value is expected
    assert grounding_called is False


def test_unfamiliar_wording_and_categories_flow_through_injected_llm_stages() -> None:
    intent = {
        "steps": [
            _step(
                "relocate_fixture",
                "E1",
                "那件带磨砂边的半透明构件",
                target=_selector("scene_ref", reference="黑色的落物台"),
                relation="on",
                required_arm="auto",
            )
        ]
    }

    grounded = interpret_and_ground_task_spec(
        "open_world_fixture",
        "请让那件带磨砂边的半透明构件安顿在黑色的落物台上。",
        _open_scene(),
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: deepcopy(intent),
        grounding_caller=_grounding_caller(
            _binding("relocate_fixture.object", "aerogel_fixture_7"),
            _binding("relocate_fixture.target", "work_surface"),
        ),
    )

    assert set(grounded.role_bindings.values()) == {
        "aerogel_fixture_7",
        "work_surface",
    }
    assert {item["category"] for item in grounded.scene_requirements["objects"]} == {
        "aerogel_fixture",
        "obsidian_dock",
    }
    assert grounded.task_spec["metadata"]["instruction_call_count"] == 1
    assert grounded.task_spec["metadata"]["scene_grounding_call_count"] == 1


@pytest.mark.parametrize(
    ("name", "instruction", "step", "bindings", "actions", "success"),
    [
        (
            "dual_lift",
            "用双臂把半透明构件端起来。",
            _step(
                "lift_fixture",
                "E5",
                "半透明构件",
                terminal_behavior="hold",
            ),
            [_binding("lift_fixture.object", "aerogel_fixture_7")],
            ["CoordinatedPickment"],
            "held_by_both_grippers",
        ),
        (
            "dual_move_place",
            "用双臂把半透明构件往左移动并放下。",
            _step(
                "move_fixture",
                "E5",
                "半透明构件",
                direction="left",
                terminal_behavior="place",
            ),
            [_binding("move_fixture.object", "aerogel_fixture_7")],
            ["CoordinatedPickment", "MoveJoints", "MoveJoints"],
            "semantic_goal",
        ),
        (
            "dual_relative",
            "用双臂把半透明构件移动到弯曲标记后面。",
            _step(
                "move_relative",
                "E5",
                "半透明构件",
                target=_selector("scene_ref", reference="弯曲的黄色标记"),
                relation="behind",
                terminal_behavior="hold",
            ),
            [
                _binding("move_relative.object", "aerogel_fixture_7"),
                _binding("move_relative.target", "plantain_marker"),
            ],
            ["CoordinatedPickment"],
            "held_by_both_grippers",
        ),
    ],
)
def test_e5_symbolic_intent_reaches_the_seed_graph(
    name: str,
    instruction: str,
    step: dict,
    bindings: list[dict],
    actions: list[str],
    success: str,
) -> None:
    grounded = interpret_and_ground_task_spec(
        name,
        instruction,
        _open_scene(),
        robot_profile="franka",
        model="test-model",
        caller=lambda **_kwargs: {"steps": [deepcopy(step)]},
        grounding_caller=_grounding_caller(*bindings),
    )
    instance = grounded.task_spec["task_instances"][0]
    graph = instantiate_seed_graph(grounded.task_spec, grounded.role_bindings)

    assert [node["atomic_action"] for node in graph["nodes"]] == actions
    assert grounded.task_spec["success"]["terms"] == [
        {"type": success, "task_instance_id": instance["id"]}
    ]
    assert instance["params"].get("direction") == (
        "up" if name == "dual_lift" else step["direction"]
    )
    if name == "dual_relative":
        assert graph["task_groups"][0]["goal"]["reference_object"] == (
            "plantain_marker"
        )
        assert graph["task_groups"][0]["goal"]["relation"] == "behind"
    if name == "dual_move_place":
        release_nodes = graph["nodes"][1:]
        assert {node["actor"]["arm"] for node in release_nodes} == {
            "left_arm",
            "right_arm",
        }
        assert len({node["sync_group"] for node in release_nodes}) == 1
