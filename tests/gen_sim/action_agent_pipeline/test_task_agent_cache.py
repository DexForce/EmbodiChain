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
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.agents import (
    task_agent as task_agent_module,
)
from embodichain.gen_sim.action_agent_pipeline.agents.agent_base import AgentBase
from embodichain.gen_sim.action_agent_pipeline.agents.compile_agent import CompileAgent
from embodichain.gen_sim.action_agent_pipeline.agents.task_agent import TaskAgent
from embodichain.gen_sim.action_agent_pipeline.utils.timing import (
    configure_timing_tracking,
    disable_timing_tracking,
)


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


def test_task_agent_uses_precomputed_task_graph_without_llm(tmp_path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "agent_config.json"
    task_graph_path = config_dir / "task_graph.json"
    task_graph = {
        "task": "unit",
        "start": "v0_start",
        "goal": "v1_done",
        "nodes": [{"id": "v0_start"}, {"id": "v1_done"}],
        "edges": [
            {
                "id": "e01_pick",
                "source": "v0_start",
                "target": "v1_done",
                "left_arm_action": {
                    "atomic_action_class": "PickUp",
                    "robot_name": "left_arm",
                    "control": "arm",
                    "target_object": {
                        "obj_name": "apple",
                        "affordance": "antipodal",
                    },
                    "cfg": {
                        "pre_grasp_distance": 0.08,
                        "lift_height": 0.3,
                        "sample_interval": 45,
                    },
                },
                "right_arm_action": None,
            }
        ],
    }
    task_graph_path.write_text(json.dumps(task_graph), encoding="utf-8")
    agent = TaskAgent(
        None,
        prompt_name="unused_prompt",
        precomputed_task_graph=task_graph_path.name,
        prompt_kwargs={},
        task_name="UnitTask",
        config_dir=str(config_path),
    )

    content = agent.generate(log_dir=tmp_path / "cache")

    assert json.loads(content) == task_graph
    assert (tmp_path / "cache/agent_task_graph.json").is_file()
    assert (tmp_path / "cache/agent_task_graph.metadata.json").is_file()


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


def test_task_agent_records_generation_timing(tmp_path, monkeypatch) -> None:
    disable_timing_tracking()
    timing_path = tmp_path / "timing.jsonl"
    configure_timing_tracking(
        timing_path=timing_path,
        run_id="test-run",
        process_name="pytest",
        reset=True,
    )
    monkeypatch.setattr(
        task_agent_module.TaskPrompt,
        "timed_prompt",
        staticmethod(lambda **kwargs: f"task={kwargs['task']}"),
        raising=False,
    )
    llm = _FakeLLM()
    agent = TaskAgent(
        llm,
        prompt_name="timed_prompt",
        prompt_kwargs={},
        task_name="TimedTask",
    )

    try:
        agent.generate(log_dir=tmp_path, task="a")
        agent.generate(log_dir=tmp_path, task="a")
    finally:
        disable_timing_tracking()

    stages = [
        json.loads(line)["stage"]
        for line in timing_path.read_text(encoding="utf-8").splitlines()
    ]
    assert stages.count("action_agent.task_graph.prompt_build") == 2
    assert stages.count("action_agent.task_graph.cache_lookup") == 2
    assert stages.count("action_agent.task_graph.llm_invoke") == 1
    assert stages.count("action_agent.task_graph.output_parse") == 1
    assert stages.count("action_agent.task_graph.cache_write") == 1
    assert stages.count("action_agent.task_graph.cache_read") == 1
    assert llm.calls == 1


def test_compile_agent_uses_base_prompt_loading(tmp_path) -> None:
    config_path = tmp_path / "agent_config.json"
    prompt_path = tmp_path / "compile_prompt.txt"
    prompt_path.write_text("compile instructions", encoding="utf-8")

    agent = CompileAgent(
        prompt_kwargs={
            "compile_prompt": {
                "type": "text",
                "name": prompt_path.name,
            }
        },
        task_name="UnitTask",
        config_dir=str(config_path),
    )

    composed = agent.get_composed_observations(task_graph="{}")

    assert composed["compile_prompt"] == "compile instructions"
    assert composed["task_graph"] == "{}"


def test_agent_base_requires_prompt_kwargs() -> None:
    class ConcreteAgent(AgentBase):
        def generate(self, *args, **kwargs):
            return None

        def act(self, *args, **kwargs):
            return None

    with pytest.raises(ValueError, match="prompt_kwargs"):
        ConcreteAgent(task_name="UnitTask")
