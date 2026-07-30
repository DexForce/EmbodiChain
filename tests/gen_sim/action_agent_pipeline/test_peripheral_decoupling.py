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

import ast
from pathlib import Path
from types import ModuleType
import sys

import embodichain.gen_sim.action_agent_pipeline as action_agent_pipeline_package
from embodichain.gen_sim.action_agent_pipeline.generation.runtime_config import (
    make_runtime_agent_config,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.runner import (
    generate_action_agent_trajectory,
)
from embodichain.gen_sim.action_agent_pipeline.utils import mllm

_PACKAGE_ROOT = Path(next(iter(action_agent_pipeline_package.__path__))).resolve()


def test_runtime_agent_config_requires_only_the_seed_graph() -> None:
    assert make_runtime_agent_config() == {
        "TaskAgent": {"seed_task_graph": "seed_task_graph.json"},
        "CompileAgent": {},
        "Agent": {},
    }


def test_core_modules_do_not_import_peripheral_layers() -> None:
    forbidden_imports = {
        "generation/config_bundle_builders.py": {
            "generation.prompt_builders",
            "generation.seed_diagnostics",
        },
        "generation/config_io.py": {"graph_visualization"},
        "runtime/task_graph_artifact.py": {"graph_visualization"},
        "runtime/runner.py": {"cli", "utils.timing"},
        "env_adapters/tableware/agent_env.py": {"utils.timing"},
        "utils/mllm.py": {"utils.llm_usage"},
    }

    violations = []
    for relative_path, forbidden_suffixes in forbidden_imports.items():
        for imported in _imports(_PACKAGE_ROOT / relative_path):
            if any(suffix in imported for suffix in forbidden_suffixes):
                violations.append(f"{relative_path} imports {imported}")

    assert not violations, "\n".join(violations)


def test_legacy_peripheral_modules_remain_available() -> None:
    for relative_path in (
        "generation/prompt_builders.py",
        "generation/seed_diagnostics.py",
        "graph_visualization.py",
        "utils/llm_usage.py",
        "utils/timing.py",
        "cli/run_agent.py",
    ):
        assert (_PACKAGE_ROOT / relative_path).is_file()


def test_runtime_trajectory_executes_without_importing_the_cli() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.reset_count = 0
            self.actions = []

        def reset(self, *, seed=None):
            self.reset_count += 1
            self.seed = seed
            return None, {}

        def get_wrapper_attr(self, name):
            if name == "create_demo_action_list":
                return self.create_demo_action_list
            if name == "is_task_success":
                return lambda: True
            raise AttributeError(name)

        def step(self, action) -> None:
            self.actions.append(action)

        def create_demo_action_list(self, **kwargs):
            self.create_kwargs = kwargs
            return ["first", "second"]

    env = FakeEnv()

    assert generate_action_agent_trajectory(
        env=env,
        episode_index=0,
        runtime_run_id="test",
        seed=17,
        strict_serial=True,
    )
    assert env.reset_count == 1
    assert env.seed == 17
    assert env.create_kwargs["strict_serial"] is True
    assert env.actions == ["first", "second"]


def test_chat_client_is_not_wrapped_by_usage_statistics(monkeypatch) -> None:
    sentinel = object()
    fake_module = ModuleType("langchain_openai")
    fake_module.ChatOpenAI = lambda **_: sentinel
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(
        mllm,
        "_resolve_llm_config",
        lambda **_: {"api_key": "test", "model": "test-model"},
    )

    result = mllm.create_chat_openai(usage_stage="legacy-stage")

    assert result is sentinel


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports
