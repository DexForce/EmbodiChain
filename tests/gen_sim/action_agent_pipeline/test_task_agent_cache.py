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

from types import SimpleNamespace

from embodichain.gen_sim.action_agent_pipeline.agents import (
    task_agent as task_agent_module,
)
from embodichain.gen_sim.action_agent_pipeline.agents.task_agent import TaskAgent


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return SimpleNamespace(content=f'{{"prompt": "{prompt}"}}')


class _FakePromptValue:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_string(self) -> str:
        return self.value

    def __str__(self) -> str:
        return f"unstable:{self.value}"


def test_task_agent_cache_uses_prompt_hash_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        task_agent_module.TaskPrompt,
        "unit_prompt",
        staticmethod(lambda **kwargs: f"task={kwargs['task']}"),
        raising=False,
    )
    llm = _FakeLLM()
    agent = TaskAgent(
        llm,
        prompt_name="unit_prompt",
        prompt_kwargs={},
        task_name="UnitTask",
    )

    first = agent.generate(log_dir=tmp_path, task="a")
    second = agent.generate(log_dir=tmp_path, task="a")
    third = agent.generate(log_dir=tmp_path, task="b")

    assert first == second
    assert first != third
    assert llm.calls == 2
    assert (tmp_path / "agent_task_graph.metadata.json").is_file()


def test_task_agent_cache_hashes_prompt_value_objects(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        task_agent_module.TaskPrompt,
        "unit_prompt_value",
        staticmethod(lambda **kwargs: _FakePromptValue(f"task={kwargs['task']}")),
        raising=False,
    )
    llm = _FakeLLM()
    agent = TaskAgent(
        llm,
        prompt_name="unit_prompt_value",
        prompt_kwargs={},
        task_name="UnitTask",
    )

    agent.generate(log_dir=tmp_path, task="a")
    agent.generate(log_dir=tmp_path, task="a")

    assert llm.calls == 1
