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
from embodichain.gen_sim.task_engine.contracts import (
    TASK_DRAFT_SCHEMA,
    validate_task_candidate,
    validate_task_draft,
)
from embodichain.gen_sim.task_engine.agent import TaskAgent, TaskGenerationError
from embodichain.gen_sim.task_engine.interpretation import (
    InstructionDraftResult,
    interpret_instruction_draft,
)

_TEST_INSTRUCTION = "test-instruction"


def _selector(kind="none", *, step_id="", reference="", quantifier="one", count=0):
    return {
        "kind": kind,
        "step_id": step_id,
        "reference": reference,
        "quantifier": quantifier,
        "count": count,
    }


def _step(step_id="orient", reference="purple can"):
    return {
        "id": step_id,
        "task_type": "E2",
        "object": _selector("scene_ref", reference=reference),
        "target": _selector(),
        "relation": "none",
        "required_arm": "auto",
        "transfer_arm": "none",
        "receive_arm": "none",
        "orientation_goal": "upright",
        "target_state": "none",
        "target_setting": 0,
        "layout": "none",
        "axis": "none",
        "direction": "none",
        "terminal_behavior": "none",
        "depends_on": [],
    }


def _result(step):
    return InstructionDraftResult(
        intent={"steps": [deepcopy(step)]},
        model="injected_caller",
        attempts=1,
        latency_seconds=0.01,
        normalizations=(),
    )


def test_draft_rejects_grounded_fields_and_task_agent_fails_closed():
    draft = {
        "schema_version": TASK_DRAFT_SCHEMA,
        "task_id": "bad",
        "instruction": "bad",
        "steps": [_step()],
    }
    draft["steps"][0]["object"]["uid"] = "scene_uid"
    with pytest.raises(ValueError, match="forbidden|exactly fields"):
        validate_task_draft(draft)

    def invalid(_instruction, **_kwargs):
        raise ValueError("invalid draft after repair")

    with pytest.raises(TaskGenerationError, match="All Task Agent candidates"):
        TaskAgent(interpreter=invalid).generate("bad", "bad")


def test_task_candidate_rejects_scene_constraints_not_derived_from_draft():
    candidate = TaskAgent(
        interpreter=lambda *_args, **_kwargs: _result(_step())
    ).generate("upright", _TEST_INSTRUCTION, candidate_count=1)["candidates"][0]
    candidate["scene_request"]["references"][0]["affordances"] = []

    with pytest.raises(ValueError, match="derived exactly"):
        validate_task_candidate(candidate)


@pytest.mark.parametrize("content_format", ["json", "fenced", "partial"])
def test_default_instruction_transport_uses_supported_chat_parameters(
    monkeypatch, content_format
):
    import json
    from embodichain.gen_sim.services.adapters import chat
    from embodichain.gen_sim.services import legacy
    from embodichain.gen_sim.task_engine import interpretation

    expected = {"steps": [_step()]}
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            content = json.dumps(expected)
            if content_format == "fenced":
                content = "```json\n" + content + "\n```"
            elif content_format == "partial":
                content = content[:-1]
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def request(value, *, timeout):
        captured.update(json.loads(value.data))
        return Response()

    monkeypatch.setattr(
        legacy,
        "_load_llm_settings",
        lambda **_: {
            "api_key": "test-key",
            "model": "test-model",
            "base_url": "http://test/v1",
            "default_query": {},
        },
    )
    monkeypatch.setattr(chat, "urlopen", request)
    result = interpretation._default_instruction_caller(
        prompt="interpret",
        schema={"z": {"type": "number"}, "a": {"type": "string"}},
        model=None,
    )
    assert result == expected
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["stream"] is False
    assert captured["messages"] == [
        {
            "role": "system",
            "content": (
                "Return only the requested structured JSON response. Never "
                "return reasoning, coordinates, or AtomicSkill nodes."
            ),
        },
        {
            "role": "user",
            "content": (
                "interpret\n\nReturn one JSON object conforming exactly to this "
                "JSON Schema:\n"
                '{"a": {"type": "string"}, "z": {"type": "number"}}'
            ),
        },
    ]
    assert "http_socket_options" not in captured


def test_malformed_structured_response_receives_one_owner_repair(monkeypatch):
    import json
    from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg
    from embodichain.gen_sim.services.adapters import chat

    calls = []
    expected = {"steps": [_step()]}

    class Response:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": self.content}}]}
            ).encode()

    def request(value, *, timeout):
        calls.append(json.loads(value.data))
        return Response("invalid JSON" if len(calls) == 1 else json.dumps(expected))

    monkeypatch.setattr(chat, "urlopen", request)
    cfg = GenSimServicesCfg(
        profiles={
            "text": {
                "adapter": "openai_compatible",
                "model": "test-model",
                "api_key": "test-key",
                "base_url": "http://test/v1",
            }
        },
        roles={"task_interpretation": "text"},
    )
    with GenSimServices(cfg) as services:
        result = interpret_instruction_draft("扶正紫色罐头", services=services)
    assert result.intent == expected
    assert result.attempts == 2
    assert len(calls) == 3
    assert "REPAIR OVERRIDE" in calls[1]["messages"][1]["content"]
    assert "SOURCE AUDIT" in calls[2]["messages"][1]["content"]


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        ("把小球放进盒子。", "inside"),
        ("将小球装进盒子。", "inside"),
        ("把小球放在盒子顶上。", "on"),
    ],
)
def test_source_review_treats_placement_paraphrases_equivalently(instruction, expected):
    proposed = dict(
        _step(reference="小球"),
        task_type="E1",
        target=_selector("scene_ref", reference="盒子"),
        relation="on",
        orientation_goal="none",
    )
    reviewed = dict(proposed, id="review_step", relation=expected)
    passes = []

    def caller(**kwargs):
        if kwargs["schema"]["title"] == "ActionEngineInstructionSourceAudit":
            passes.append("source_review")
            return {"steps": [reviewed]}
        passes.append("interpretation")
        repair = "REPAIR OVERRIDE" in kwargs["prompt"]
        return {"steps": [dict(proposed, relation=expected if repair else "on")]}

    result = interpret_instruction_draft(instruction, caller=caller)
    assert result.intent["steps"][0]["relation"] == expected
    assert passes == ["interpretation", "source_review"] * (
        2 if expected == "inside" else 1
    )


def test_source_review_preserves_the_repeated_destination_not_the_previous_object():
    first = dict(
        _step("ball", "小球"),
        task_type="E1",
        relation="inside",
        target=_selector("scene_ref", reference="盒子"),
        orientation_goal="none",
    )
    second = dict(
        first,
        id="block",
        object=_selector("scene_ref", reference="木块"),
        depends_on=["ball"],
    )
    wrong = dict(second, target=_selector("step_result", step_id="ball"))

    def caller(*, schema, **_):
        return {
            "steps": [
                first,
                (
                    second
                    if schema["title"] == "ActionEngineInstructionSourceAudit"
                    else wrong
                ),
            ]
        }

    with pytest.raises(ValueError, match=r"after one repair: Source audit.*target"):
        interpret_instruction_draft(
            "先把小球放进盒子，再把木块装进那个盒子。", caller=caller
        )


@pytest.mark.parametrize(
    ("instruction", "wrong_height"),
    [
        ("Raise the cup by over 10 cm and keep holding it.", None),
        ("把杯子往上抬高至少十厘米，保持抓住。", 0.20),
    ],
)
def test_lift_height_is_explicit_typed_source_intent(instruction, wrong_height):
    step = dict(
        _step(reference="cup"),
        task_type="E1",
        orientation_goal="none",
        direction="up",
        terminal_behavior="hold",
    )
    proposed = step if wrong_height is None else dict(step, lift_height_m=wrong_height)

    def caller(*, prompt, schema, **_):
        if (
            schema["title"] == "ActionEngineInstructionSourceAudit"
            or "REPAIR OVERRIDE" in prompt
        ):
            return {"steps": [dict(step, lift_height_m=0.10)]}
        return {"steps": [proposed]}

    result = interpret_instruction_draft(instruction, caller=caller)
    assert result.attempts == 2
    assert result.intent["steps"][0]["lift_height_m"] == 0.10


@pytest.mark.parametrize(
    "destination", ["original position", "where it had been sitting"]
)
def test_source_review_distinguishes_own_start_pose_from_a_scene_role(destination):
    expected = dict(
        _step(reference="tray"),
        task_type="E5",
        orientation_goal="none",
        terminal_behavior="place",
    )
    invented = dict(
        expected,
        target=_selector("scene_ref", reference=destination),
        relation="on",
    )

    def caller(*, prompt, schema, **_):
        if (
            schema["title"] == "ActionEngineInstructionSourceAudit"
            or "REPAIR OVERRIDE" in prompt
        ):
            return {"steps": [expected]}
        return {"steps": [invented]}

    result = interpret_instruction_draft(
        (
            f"Lift the tray with both arms and set it back at its {destination}."
            if destination == "original position"
            else f"Lift the tray with both arms, then set it {destination}."
        ),
        caller=caller,
    )
    assert result.attempts == 2
    assert result.intent["steps"][0]["target"]["kind"] == "none"


def test_source_review_blocks_a_valid_draft_that_drops_an_unsupported_clause():
    quote = "再关上柜子铰链门"

    def caller(*, schema, **_):
        if schema["title"] == "ActionEngineInstructionSourceAudit":
            return {
                "issues": [
                    {"quote": quote, "reason": "Hinged closing is not represented."}
                ]
            }
        return {"steps": [_step(reference="罐子")]}

    with pytest.raises(ValueError, match=f"after one repair: Source audit.*{quote}"):
        interpret_instruction_draft(f"扶正罐子，{quote}。", caller=caller)


@pytest.mark.parametrize(
    ("reference", "draft_quote", "audit_quote", "diagnostic"),
    [
        ("cup", "the cup", "the cup", None),
        ("the cup", "the bottle", "the cup", "Source audit disagrees.*object"),
        ("bottle", "the bottle", "the cup", "Source audit disagrees.*object"),
        ("bottle", None, "the cup", "Source audit disagrees.*object"),
        ("cup", "the cup", "the glass", "quote must occur verbatim"),
    ],
)
def test_source_review_matches_referents_not_their_descriptions(
    reference, draft_quote, audit_quote, diagnostic
):
    def caller(*, schema, **_):
        is_review = schema["title"] == "ActionEngineInstructionSourceAudit"
        response = {"steps": [_step(reference="the cup" if is_review else reference)]}
        quote = audit_quote if is_review else draft_quote
        if quote is not None:
            response["scene_ref_quotes"] = [{"ref": "1.object", "quote": quote}]
        return response

    agent = TaskAgent(caller=caller)
    instruction = "Set the cup upright without moving the bottle."
    if diagnostic is not None:
        with pytest.raises(TaskGenerationError, match=diagnostic):
            agent.generate("upright", instruction, candidate_count=1)
        return
    (candidate,) = agent.generate("upright", instruction, candidate_count=1)[
        "candidates"
    ]
    assert candidate["draft"]["steps"][0]["object"]["reference"] == "the cup"
    assert candidate["attempts"] == 1


@pytest.mark.parametrize(
    ("draft_ref", "review_ref", "draft_quote", "review_quote", "expected"),
    [
        ("blue cup", "blue cup", "the blue cup", "blue cup", "blue cup"),
        (
            "blue cup",
            "red cup",
            "the blue cup next to the red cup",
            "the red cup",
            None,
        ),
        ("cup", "cup", "the blue cup next to the red cup", "the red cup", None),
    ],
)
def test_source_review_accepts_quote_boundary_variants_only_with_shared_anchor(
    draft_ref, review_ref, draft_quote, review_quote, expected
):
    def caller(*, schema, **_):
        reviewed = schema["title"] == "ActionEngineInstructionSourceAudit"
        return {
            "steps": [_step(reference=review_ref if reviewed else draft_ref)],
            "scene_ref_quotes": [
                {"ref": "1.object", "quote": review_quote if reviewed else draft_quote}
            ],
        }

    instruction = "Rotate the blue cup next to the red cup until it is upright."
    if expected is None:
        with pytest.raises(ValueError, match="Source audit disagrees.*object"):
            interpret_instruction_draft(instruction, caller=caller)
        return
    result = interpret_instruction_draft(instruction, caller=caller)
    assert result.intent["steps"][0]["object"]["reference"] == expected
    assert result.attempts == 1


@pytest.mark.parametrize(
    ("review_deps", "equivalent"),
    [(["a", "b"], True), (["a"], False)],
)
def test_source_review_compares_dependency_reachability(review_deps, equivalent):
    first = _step("a", "red cup")
    second = dict(_step("b", "blue cup"), depends_on=["a"])
    third = dict(_step("c", "green cup"), depends_on=["b"])

    def caller(*, schema, **_):
        last = (
            dict(third, depends_on=review_deps)
            if schema["title"] == "ActionEngineInstructionSourceAudit"
            else third
        )
        return {"steps": [first, second, last]}

    instruction = "Set the red, blue, then green cup upright, in that order."
    if not equivalent:
        with pytest.raises(ValueError, match="Source audit disagrees.*depends_on"):
            interpret_instruction_draft(instruction, caller=caller)
        return
    result = interpret_instruction_draft(instruction, caller=caller)
    assert result.attempts == 1
    assert result.intent["steps"][-1]["depends_on"] == ["b"]
