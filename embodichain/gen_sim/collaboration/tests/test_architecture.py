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

"""Guard ownership boundaries for the three-engine collaboration layout."""

from __future__ import annotations

import ast
from pathlib import Path

from embodichain import __main__ as root_cli
from embodichain.gen_sim.action_engine.agent import ActionAgent
from embodichain.gen_sim.action_engine.collaboration.action_agent import (
    ActionAgent as LegacyActionAgent,
)
from embodichain.gen_sim.action_engine.collaboration.task_agent import (
    TaskAgent as LegacyTaskAgent,
)
from embodichain.gen_sim.collaboration.scene_adapter import SceneAdapter
from embodichain.gen_sim.task_engine import TaskAgent

_GEN_SIM_ROOT = Path(__file__).resolve().parents[2]


def test_task_engine_has_no_action_scene_or_collaboration_imports() -> None:
    forbidden = {
        "embodichain.gen_sim.action_engine",
        "embodichain.gen_sim.scene_engine",
        "embodichain.gen_sim.collaboration",
    }
    offenders: list[str] = []
    for path in sorted((_GEN_SIM_ROOT / "task_engine").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(
                module == prefix or module.startswith(prefix + ".")
                for module in modules
                for prefix in forbidden
            ):
                offenders.append(path.name)
                break
    assert offenders == []


def test_public_agents_and_adapter_live_under_their_owning_packages() -> None:
    assert TaskAgent.__module__ == "embodichain.gen_sim.task_engine.agent"
    assert ActionAgent.__module__ == "embodichain.gen_sim.action_engine.agent"
    assert SceneAdapter.__module__ == "embodichain.gen_sim.collaboration.scene_adapter"


def test_legacy_collaboration_agent_imports_preserve_class_identity() -> None:
    assert LegacyTaskAgent is TaskAgent
    assert LegacyActionAgent is ActionAgent


def test_root_cli_dispatches_to_top_level_collaboration() -> None:
    command = next(item for item in root_cli.COMMANDS if item.name == "gen-sim-task")
    assert command.target == "embodichain.gen_sim.collaboration.cli:main"
