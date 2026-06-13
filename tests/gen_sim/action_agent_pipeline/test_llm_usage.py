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

import json

import pytest

from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
    LLM_USAGE_PATH_ENV,
    UsageTrackedChatModel,
    build_usage_summary,
    configure_usage_tracking,
    disable_usage_tracking,
    normalize_usage,
    record_langchain_usage,
    scrub_usage_tracking_env,
)


class _FakeLangChainResponse:
    usage_metadata = {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "input_token_details": {"cache_read": 3},
        "output_token_details": {"reasoning": 2},
    }
    response_metadata = {
        "model_name": "gpt-test",
        "id": "chatcmpl-test",
        "finish_reason": "stop",
    }
    content = "{}"


class _FakeChatModel:
    model_name = "gpt-test"

    def __init__(self) -> None:
        self.inputs = []

    def invoke(self, value):
        self.inputs.append(value)
        return _FakeLangChainResponse()


@pytest.fixture(autouse=True)
def _clear_usage_env():
    disable_usage_tracking()
    yield
    disable_usage_tracking()


def test_normalize_usage_handles_openai_and_langchain_shapes():
    openai_usage = {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
        "prompt_tokens_details": {"cached_tokens": 7},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    assert normalize_usage(openai_usage) == {
        "input_tokens": 11,
        "output_tokens": 5,
        "total_tokens": 16,
        "cached_tokens": 7,
        "reasoning_tokens": 2,
    }

    langchain_usage = {
        "input_tokens": 10,
        "output_tokens": 4,
        "input_token_details": {"cache_read": 3},
        "output_token_details": {"reasoning": 2},
    }
    assert normalize_usage(langchain_usage) == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "cached_tokens": 3,
        "reasoning_tokens": 2,
    }


def test_record_langchain_usage_writes_jsonl_and_summary(tmp_path):
    usage_path = tmp_path / "llm_usage.jsonl"
    configure_usage_tracking(
        usage_path=usage_path,
        run_id="test-run",
        process_name="pytest",
        reset=True,
    )

    record_langchain_usage(
        _FakeLangChainResponse(),
        stage="Action Agent Task Graph",
        model="fallback-model",
    )

    records = [
        json.loads(line) for line in usage_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["stage"] == "action_agent_task_graph"
    assert records[0]["model"] == "gpt-test"
    assert records[0]["input_tokens"] == 10
    assert records[0]["output_tokens"] == 4

    summary = build_usage_summary(usage_path)
    assert summary["total"]["calls"] == 1
    assert summary["total"]["total_tokens"] == 14
    assert summary["by_stage"]["action_agent_task_graph"]["cached_tokens"] == 3


def test_usage_tracked_chat_model_records_invoke(tmp_path):
    usage_path = tmp_path / "llm_usage.jsonl"
    configure_usage_tracking(
        usage_path=usage_path,
        run_id="test-run",
        process_name="pytest",
        reset=True,
    )
    inner = _FakeChatModel()
    wrapped = UsageTrackedChatModel(inner, stage="action_agent.task_graph")

    response = wrapped.invoke("hello")

    assert response.content == "{}"
    assert inner.inputs == ["hello"]
    record = json.loads(usage_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["stage"] == "action_agent.task_graph"
    assert record["total_tokens"] == 14


def test_scrub_usage_tracking_env_removes_usage_keys(tmp_path):
    usage_path = tmp_path / "llm_usage.jsonl"
    configure_usage_tracking(
        usage_path=usage_path,
        run_id="test-run",
        process_name="pytest",
        reset=True,
    )

    cleaned = scrub_usage_tracking_env()

    assert LLM_USAGE_PATH_ENV not in cleaned
