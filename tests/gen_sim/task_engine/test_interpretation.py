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
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from embodichain.gen_sim.task_engine import interpretation as interpretation_module


def _resource_step(step_id: str, task_type: str = "E4") -> dict[str, Any]:
    step = interpretation_module._instruction_shape_example()["steps"][0]
    step.update(
        id=step_id,
        task_type=task_type,
        orientation_goal="none",
        required_arm="none",
        transfer_arm="right_arm" if task_type == "E4" else "none",
        receive_arm="left_arm" if task_type == "E4" else "none",
        terminal_behavior="hold",
        direction="up" if task_type == "E5" else "none",
    )
    step["object"]["reference"] = step_id
    return step


def _result_selector(step_id: str) -> dict[str, Any]:
    return {
        "kind": "step_result",
        "step_id": step_id,
        "reference": "",
        "quantifier": "one",
        "count": 0,
    }


def _place_transfer(step: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(step)
    result.update(terminal_behavior="place", relation="inside")
    result["target"] = {
        "kind": "scene_ref",
        "step_id": "",
        "reference": "plate",
        "quantifier": "one",
        "count": 0,
    }
    return result


def test_intent_rejects_occupied_handover_receiver_before_grounding() -> None:
    first = _resource_step("first")
    second = _resource_step("second")
    second["depends_on"] = ["first"]

    with pytest.raises(ValueError, match="receiver.*occupied"):
        interpretation_module.validate_instruction_intent({"steps": [first, second]})


def test_resource_repair_uses_per_object_transfer_placement_without_rewriting() -> None:
    invalid = {"steps": [_resource_step("first"), _resource_step("second")]}
    repaired = {"steps": [_place_transfer(step) for step in invalid["steps"]]}
    repaired["steps"][1]["depends_on"] = ["first"]
    responses = iter((invalid, repaired))
    prompts = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return deepcopy(next(responses))

    result = interpretation_module.interpret_instruction_draft(
        "Transfer each fruit from right to left, then put each in the plate.",
        caller=caller,
    )

    assert result.attempts == 2
    assert result.intent == repaired
    assert invalid["steps"][0]["terminal_behavior"] == "hold"
    assert "per-object transfer and placement" in prompts[1]
    assert "Do not drop objects, arm requirements, targets, or ordering" in prompts[1]


def test_explicit_hold_conflict_stays_invalid_after_one_repair() -> None:
    invalid = {"steps": [_resource_step("first"), _resource_step("second")]}
    caller = Mock(return_value=invalid)

    with pytest.raises(ValueError, match="after one repair.*receiver.*occupied"):
        interpretation_module.interpret_instruction_draft(
            "Receive the first object in the left hand, keep holding it, then "
            "receive the second object in that same hand.",
            caller=caller,
        )

    assert caller.call_count == 2
    assert all(step["terminal_behavior"] == "hold" for step in invalid["steps"])


def test_held_object_can_be_transferred_back_through_explicit_result() -> None:
    first = _resource_step("first")
    second = _resource_step("second")
    second.update(
        object=_result_selector("first"),
        transfer_arm="left_arm",
        receive_arm="right_arm",
        depends_on=["first"],
    )

    intent = {"steps": [first, second]}
    assert interpretation_module.validate_instruction_intent(intent) == intent


def test_separate_explicit_placement_releases_receiver_before_next_transfer() -> None:
    first = _resource_step("first")
    place = _place_transfer(_resource_step("place"))
    place.update(
        task_type="E1",
        object=_result_selector("first"),
        required_arm="left_arm",
        transfer_arm="none",
        receive_arm="none",
        terminal_behavior="none",
        depends_on=["first"],
    )
    second = _resource_step("second")
    second["depends_on"] = ["place"]

    intent = {"steps": [first, place, second]}
    assert interpretation_module.validate_instruction_intent(intent) == intent


@pytest.mark.parametrize("transfer,receive", [("auto", "auto"), ("auto", "right_arm")])
def test_auto_handover_checks_possible_assignments_without_inventing_arms(
    transfer: str, receive: str
) -> None:
    first = _resource_step("first")
    second = _resource_step("second")
    second.update(
        object=_result_selector("first"),
        transfer_arm=transfer,
        receive_arm=receive,
        depends_on=["first"],
    )

    intent = {"steps": [first, second]}
    assert interpretation_module.validate_instruction_intent(intent) == intent


@pytest.mark.parametrize("first_type", ["E4", "E5"])
def test_new_dual_hold_cannot_use_occupied_hands(first_type: str) -> None:
    first = _resource_step("first", first_type)
    second = _resource_step("second", "E5")
    second["depends_on"] = ["first"]

    with pytest.raises(ValueError, match="both arms.*occupied"):
        interpretation_module.validate_instruction_intent({"steps": [first, second]})


def test_dual_hold_continuation_and_release_require_explicit_same_object() -> None:
    first = _resource_step("first", "E5")
    second = _resource_step("second", "E5")
    second.update(
        object=_result_selector("first"),
        direction="none",
        terminal_behavior="place",
        depends_on=["first"],
    )
    transfer = _resource_step("transfer")
    transfer["depends_on"] = ["second"]

    intent = {"steps": [first, second, transfer]}
    assert interpretation_module.validate_instruction_intent(intent) == intent


def test_prompt_same_part_continuation_does_not_merge_nouns() -> None:
    prompt = interpretation_module._instruction_prompt(
        "Open the sliding tray, unload it, then close the now-empty tray."
    )
    assert "same-entity continuation" in prompt
    assert "E6 opening and later closing the same part" in prompt
    assert "Do not merge independently selected native parts" in prompt
    assert "empty" in prompt


def test_prompt_preserves_containment_and_repeated_destination() -> None:
    prompt = interpretation_module._instruction_prompt(
        "Place the fruit inside a bowl, then put another fruit in the same bowl."
    )
    assert "relation=inside, not on" in prompt
    assert (
        "step_result identifies the manipulated object, not its destination" in prompt
    )
    assert "same bowl" in prompt
    assert "do not stack the second object on the first" in prompt


def _placement_step(step_id: str, obj: str, target: str, relation: str) -> dict:
    step = interpretation_module._instruction_shape_example()["steps"][0]
    step.update(id=step_id, task_type="E1", orientation_goal="none", relation=relation)
    step["object"]["reference"] = obj
    step["target"] = dict(step["object"], reference=target)
    return step


def test_literal_containment_conflict_requests_model_repair_without_rewriting() -> None:
    invalid = {"steps": [_placement_step("a", "fruit", "the bowl", "on")]}
    repaired = deepcopy(invalid)
    repaired["steps"][0]["relation"] = "inside"
    responses = iter((invalid, repaired))
    prompts = []

    def caller(**kwargs):
        prompts.append(kwargs["prompt"])
        return deepcopy(next(responses))

    result = interpretation_module.interpret_instruction_draft(
        "Put the fruit into the bowl.", caller=caller
    )
    assert result.attempts == 2
    assert result.intent == repaired
    assert invalid["steps"][0]["relation"] == "on"
    assert "Explicit containment" in prompts[1]


def test_persistent_literal_containment_conflict_fails_before_grounding() -> None:
    caller = Mock(return_value={"steps": [_placement_step("a", "fruit", "bowl", "on")]})
    with pytest.raises(ValueError, match="after one repair.*containment"):
        interpretation_module.interpret_instruction_draft(
            "Place fruit inside bowl.", caller=caller
        )
    assert caller.call_count == 2


@pytest.mark.parametrize(
    "instruction,obj",
    [
        ("Place fruit on the bowl.", "fruit"),
        ("Do not put fruit into the bowl; place it on the bowl.", "fruit"),
        ("Put fruit into a box, then place a cup on the bowl.", "cup"),
        ("Put fruit into bowl, then place a cup on bowl.", "cup"),
        ("Put fruit into bowl, then place fruit on bowl.", "fruit"),
    ],
)
def test_containment_guard_does_not_reinterpret_explicit_on(instruction, obj) -> None:
    intent = {"steps": [_placement_step("a", obj, "bowl", "on")]}
    result = interpretation_module.interpret_instruction_draft(
        instruction, caller=Mock(return_value=intent)
    )
    assert result.attempts == 1
    assert result.intent == intent


def test_chinese_literal_containment_and_same_target_request_repair() -> None:
    first = _placement_step("a", "苹果", "抽屉左侧的果盘", "on")
    second = _placement_step("b", "橙子", "同一果盘", "on")
    second["depends_on"] = ["a"]
    invalid = {"steps": [first, second]}
    repaired = deepcopy(invalid)
    for step in repaired["steps"]:
        step["relation"] = "inside"
    responses = iter((invalid, repaired))
    result = interpretation_module.interpret_instruction_draft(
        "把苹果放入抽屉左侧的果盘，再把橙子放入同一果盘。",
        caller=lambda **_: deepcopy(next(responses)),
    )
    assert result.attempts == 2
    assert result.intent == repaired


def test_same_destination_cannot_reference_prior_manipulated_object() -> None:
    first = _placement_step("a", "fruit", "large bowl", "inside")
    second = _placement_step("b", "cup", "same bowl", "inside")
    second.update(target=_result_selector("a"), depends_on=["a"])
    invalid = {"steps": [first, second]}
    repaired = deepcopy(invalid)
    repaired["steps"][1]["target"] = dict(first["target"], reference="same bowl")
    responses = iter((invalid, repaired))
    result = interpretation_module.interpret_instruction_draft(
        "Put fruit into large bowl, then put cup into the same bowl.",
        caller=lambda **_: deepcopy(next(responses)),
    )
    assert result.attempts == 2
    assert result.intent == repaired


def test_previous_manipulated_container_remains_valid_step_result_target() -> None:
    first = _placement_step("a", "bowl", "tray", "inside")
    second = _placement_step("b", "fruit", "same bowl", "inside")
    second.update(target=_result_selector("a"), depends_on=["a"])
    intent = {"steps": [first, second]}
    result = interpretation_module.interpret_instruction_draft(
        "Put bowl into tray, then put fruit into the same bowl.",
        caller=Mock(return_value=intent),
    )
    assert result.attempts == 1
    assert result.intent == intent


@pytest.mark.parametrize("result_object", [False, True])
def test_prior_containment_does_not_override_later_relative_placement(result_object):
    first = _placement_step("a", "fruit", "bowl", "inside")
    second = _placement_step("b", "fruit", "bowl", "left_of")
    second["depends_on"] = ["a"]
    if result_object:
        second["object"] = _result_selector("a")
    intent = {"steps": [first, second]}
    result = interpretation_module.interpret_instruction_draft(
        "Put fruit into bowl, then move fruit to the left of bowl.",
        caller=Mock(return_value=intent),
    )
    assert result.attempts == 1
    assert result.intent == intent


@pytest.mark.parametrize(
    "instruction",
    [
        "Move fruit previously put inside bowl to the left of bowl.",
        "I put fruit into bowl yesterday; now move it to the left of bowl.",
        "If I put fruit into bowl, move it to the left of bowl.",
    ],
)
def test_historical_inside_description_is_not_a_current_containment_command(
    instruction,
):
    intent = {"steps": [_placement_step("a", "fruit", "bowl", "left_of")]}
    result = interpretation_module.interpret_instruction_draft(
        instruction,
        caller=Mock(return_value=intent),
    )
    assert result.attempts == 1
    assert result.intent == intent


def test_same_container_location_description_does_not_replace_step_result_target():
    first = _placement_step("a", "tray", "bowl", "inside")
    second = _placement_step("b", "cup", "tray", "on")
    second.update(target=_result_selector("a"), depends_on=["a"])
    intent = {"steps": [first, second]}
    result = interpretation_module.interpret_instruction_draft(
        "Put tray into bowl, then put cup on the tray beside the same bowl.",
        caller=Mock(return_value=intent),
    )
    assert result.attempts == 1
    assert result.intent == intent


def test_e6_coreference_keeps_order_and_final_closed_state() -> None:
    from embodichain.gen_sim.task_engine.agent import TaskAgent
    from embodichain.gen_sim.task_engine.task_spec import task_spec_from_candidate
    from embodichain.gen_sim.task_spec import TaskSpec

    opened = _resource_step("opened")
    opened.update(
        task_type="E6",
        required_arm="right_arm",
        transfer_arm="none",
        receive_arm="none",
        terminal_behavior="none",
        target_state="open",
    )
    on_tray = _place_transfer(_resource_step("on_tray"))
    on_tray.update(
        task_type="E1",
        required_arm="right_arm",
        transfer_arm="none",
        receive_arm="none",
        terminal_behavior="none",
        target=_result_selector("opened"),
        relation="on",
        depends_on=["opened"],
    )
    on_table = deepcopy(on_tray)
    on_table.update(
        id="on_table",
        object=_result_selector("on_tray"),
        target={**on_tray["object"], "reference": "table"},
        depends_on=["on_tray"],
    )
    closed = deepcopy(opened)
    closed.update(
        id="closed",
        object=_result_selector("opened"),
        target_state="closed",
        depends_on=["on_table"],
    )
    candidates = TaskAgent(
        caller=lambda **_kwargs: {
            "steps": deepcopy([opened, on_tray, on_table, closed])
        }
    ).generate(
        "same_part", "Open, load, unload, and close the same tray.", candidate_count=1
    )
    candidate = candidates["candidates"][0]
    specification = task_spec_from_candidate(candidate)

    roles = specification.metadata["step_roles"]
    assert roles["step_01"]["object"] == roles["step_02"]["target"]
    assert roles["step_01"]["object"] == roles["step_04"]["object"]
    assert [
        predicate.arguments["target_state"]
        for predicate in specification.goal
        if predicate.name == "articulation_joint_near"
    ] == ["closed"]
    assert [
        predicate.arguments["target_state"]
        for milestone in specification.milestones
        for predicate in milestone.achieve
        if predicate.name == "articulation_joint_near"
    ] == ["open", "closed"]
    assert [milestone.after for milestone in specification.milestones] == [
        (),
        ("step_01",),
        ("step_02",),
        ("step_03",),
    ]
    assert (
        TaskSpec.from_dict(specification.to_dict()).to_dict() == specification.to_dict()
    )


def test_prompt_preserves_explicit_handover_orientation() -> None:
    prompt = interpretation_module._instruction_prompt(
        "Keep the received bottle upright."
    )
    assert "requires orientation_goal=upright" in prompt
    assert "terminal_behavior=hold" in prompt
    assert "If placement is a separate E1 step" in prompt
    assert "Never emit terminal_behavior=place with an empty E4 target" in prompt
    assert "For an E4 arm not named in the instruction" in prompt
    assert "Keep the received bottle upright." in prompt


def test_prompt_and_repair_guidance_use_mechanism_semantics() -> None:
    prompt = interpretation_module._instruction_prompt("Open and close the drawer.")
    assert "also E6 with target_state=closed" in prompt
    assert "revolute door is E7" in prompt
    assert "Closing or pushing in a drawer is E7" not in prompt
    catalog = interpretation_module._intent_capability_catalog()
    assert "prismatic" in catalog["E6"]["semantics"]
    assert "revolute" in catalog["E7"]["semantics"]
    guidance = interpretation_module._instruction_repair_guidance(
        interpretation_module._MissingRequiredObjectError("missing object")
    )
    assert "For E6 the drawer or sliding tray" in guidance
    assert "for E7 the hinged door" in guidance


def test_counted_set_repair_preserves_explicit_number() -> None:
    rules = interpretation_module._instruction_selector_rules()
    guidance = interpretation_module._instruction_repair_guidance(
        ValueError("InstructionIntent.steps[0].object quantifier=all requires count=0.")
    )

    assert "quantifier='count'" in rules
    assert "quantifier='all'" in rules
    assert "explicit number" in guidance
    assert "quantifier=count" in guidance
    assert "count=0" in guidance


def test_prompt_preserves_each_member_of_compound_noun() -> None:
    prompt = interpretation_module._instruction_prompt(
        "Put the knife and fork on the placemat."
    )
    assert "knife and fork" in prompt
    assert "quantifier=count" in prompt
    assert "two distinct objects" in prompt


def test_handover_arm_repair_uses_auto_without_guessing() -> None:
    guidance = interpretation_module._instruction_repair_guidance(
        ValueError("InstructionIntent.steps[0] E4 requires named arms or auto.")
    )
    assert "use auto" in guidance
    assert "preserve any explicitly named arm" in guidance


@pytest.mark.parametrize(
    "settings",
    [
        {"model": "mimo-v2-flash", "base_url": "https://api.xiaomimimo.com/v1"},
        {"model": "compatible-model", "base_url": "https://provider.example/v1"},
    ],
    ids=["mimo", "other-openai-compatible"],
)
def test_structured_output_uses_json_mode(settings: dict[str, str]) -> None:
    client = Mock()
    schema = {"type": "object", "properties": {"task": {"type": "string"}}}

    result = interpretation_module._structured_output_runnable(
        client, schema, settings=settings
    )

    client.with_structured_output.assert_called_once_with(schema, method="json_mode")
    assert result is client.with_structured_output.return_value


def _write_dotenv(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=dotenv-key",
                "OPENAI_BASE_URL=https://dotenv.example/v1",
                "OPENAI_MODEL=dotenv-model",
            )
        ),
        encoding="utf-8",
    )


def _clear_process_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "LLM_URL",
        "TASK_ENGINE_LLM_MODEL",
        "ACTION_ENGINE_LLM_MODEL",
        "OPENAI_MODEL",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_partial_process_transport_does_not_mix_with_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    _write_dotenv(env_path)
    monkeypatch.setattr(interpretation_module, "_GEN_SIM_ENV_PATH", env_path)
    monkeypatch.setattr(
        interpretation_module,
        "_GEN_CONFIG_PATH",
        tmp_path / "missing.json",
    )
    _clear_process_provider(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-process-key")

    settings = interpretation_module._load_llm_settings(model=None)

    assert settings["api_key"] == "dotenv-key"
    assert settings["base_url"] == "https://dotenv.example/v1"
    assert settings["model"] == "dotenv-model"


def test_complete_process_transport_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    _write_dotenv(env_path)
    monkeypatch.setattr(interpretation_module, "_GEN_SIM_ENV_PATH", env_path)
    monkeypatch.setattr(
        interpretation_module,
        "_GEN_CONFIG_PATH",
        tmp_path / "missing.json",
    )
    _clear_process_provider(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://process.example/v1/")
    monkeypatch.setenv("TASK_ENGINE_LLM_MODEL", "process-model")

    settings = interpretation_module._load_llm_settings(model=None)

    assert settings["api_key"] == "process-key"
    assert settings["base_url"] == "https://process.example/v1"
    assert settings["model"] == "process-model"


@pytest.mark.parametrize("source", ["json", "LLM_MODEL"])
def test_default_interpreter_uses_the_transport_model_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    import json

    for name in (
        "TASK_ENGINE_LLM_MODEL",
        "ACTION_ENGINE_LLM_MODEL",
        "OPENAI_MODEL",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "gen_config.json"
    config.write_text(
        json.dumps({"llm": {"openai_compatible": {"model": "configured-model"}}})
    )
    monkeypatch.setattr(interpretation_module, "_GEN_CONFIG_PATH", config)
    monkeypatch.setattr(interpretation_module, "_load_local_env", lambda: {})
    if source == "LLM_MODEL":
        monkeypatch.setenv("LLM_MODEL", "configured-model")
    selected = []

    def caller(**kwargs):
        selected.append(kwargs["model"])
        return interpretation_module._instruction_shape_example()

    monkeypatch.setattr(interpretation_module, "_default_instruction_caller", caller)
    result = interpretation_module.interpret_instruction_draft("Move the cup.")
    assert result.model == "configured-model"
    assert selected == ["configured-model"]
