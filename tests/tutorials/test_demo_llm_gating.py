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

from embodichain.lab.gym.envs.tasks.tableware.base_agent_env import BaseAgentEnv
from scripts.tutorials.gym import rearrangement_atomic_graph as rearrangement_demo
from scripts.tutorials.gym import pour_water_recovery_compare as pour_water_demo


class _FakeAgent:
    task_name = "FakeTask"

    def __init__(self, output: str) -> None:
        self.output = output
        self.generate_calls = []

    def get_composed_observations(self, **kwargs):
        return dict(kwargs)

    def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        return self.output


class _FakeCompileAgent(_FakeAgent):
    task_name = "FakeTask"

    def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        return "agent_compiled_graph.json", kwargs, self.output


def test_base_agent_env_uses_phase_specific_regenerate_flags() -> None:
    env = object.__new__(BaseAgentEnv)
    env.task_agent = _FakeAgent("{}")
    env.recovery_agent = _FakeAgent("{}")
    env.compile_agent = _FakeCompileAgent("{}")
    env.get_obs_for_agent = lambda: {"rgb": "observation"}

    env.generate_graph_for_actions(
        regenerate=True,
        recovery=True,
        task_regenerate=False,
        recovery_regenerate=True,
        compile_regenerate=False,
    )

    assert env.task_agent.generate_calls[0]["regenerate"] is False
    assert env.recovery_agent.generate_calls[0]["regenerate"] is True
    assert env.compile_agent.generate_calls[0]["regenerate"] is False


def test_rearrangement_agent_regenerate_flags_are_phase_specific() -> None:
    assert rearrangement_demo._agent_regenerate_kwargs(
        SimpleNamespace(regenerate=False, runtime_llm_recovery=False)
    ) == {
        "task_regenerate": False,
        "recovery_regenerate": False,
        "compile_regenerate": False,
    }
    assert rearrangement_demo._agent_regenerate_kwargs(
        SimpleNamespace(regenerate=True, runtime_llm_recovery=False)
    ) == {
        "task_regenerate": True,
        "recovery_regenerate": False,
        "compile_regenerate": True,
    }
    assert rearrangement_demo._agent_regenerate_kwargs(
        SimpleNamespace(regenerate=False, runtime_llm_recovery=True)
    ) == {
        "task_regenerate": False,
        "recovery_regenerate": True,
        "compile_regenerate": True,
    }
    assert rearrangement_demo._agent_regenerate_kwargs(
        SimpleNamespace(regenerate=True, runtime_llm_recovery=True)
    ) == {
        "task_regenerate": True,
        "recovery_regenerate": True,
        "compile_regenerate": True,
    }


def test_pour_water_agent_regenerate_flags_are_phase_specific() -> None:
    assert pour_water_demo._agent_regenerate_kwargs(
        SimpleNamespace(regenerate=False, runtime_llm_recovery=False)
    ) == {
        "task_regenerate": False,
        "recovery_regenerate": False,
        "compile_regenerate": False,
    }
    assert pour_water_demo._agent_regenerate_kwargs(
        SimpleNamespace(regenerate=True, runtime_llm_recovery=False)
    ) == {
        "task_regenerate": True,
        "recovery_regenerate": False,
        "compile_regenerate": True,
    }
    assert pour_water_demo._agent_regenerate_kwargs(
        SimpleNamespace(regenerate=False, runtime_llm_recovery=True)
    ) == {
        "task_regenerate": False,
        "recovery_regenerate": True,
        "compile_regenerate": True,
    }
    assert pour_water_demo._agent_regenerate_kwargs(
        SimpleNamespace(regenerate=True, runtime_llm_recovery=True)
    ) == {
        "task_regenerate": True,
        "recovery_regenerate": True,
        "compile_regenerate": True,
    }
