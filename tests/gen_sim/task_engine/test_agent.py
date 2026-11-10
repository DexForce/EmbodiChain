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
from embodichain.gen_sim.task_engine.interpretation import InstructionDraftResult

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
    from embodichain.gen_sim.task_engine.interpretation import (
        interpret_instruction_draft,
    )

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
    assert len(calls) == 2
    assert "REPAIR OVERRIDE" in calls[1]["messages"][1]["content"]
