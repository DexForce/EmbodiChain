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

"""Guard the migration boundaries that make the rewrite meaningful."""

from __future__ import annotations

import ast
from pathlib import Path

import embodichain.gen_sim.action_engine as action_engine_package
from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
    build_default_registry,
)
from embodichain.gen_sim.action_engine.protocol import (
    ACTION_ENGINE_CONFIG_SCHEMA,
    ACTION_ENGINE_ENV_ID,
    EXECUTION_PROGRAM_FILENAME,
    SCENE_REQUIREMENTS_FILENAME,
    SCENE_REQUIREMENTS_SCHEMA,
    SEED_GRAPH_SCHEMA,
    TASK_SPEC_FILENAME,
    TASK_SPEC_SCHEMA,
)

_PACKAGE_ROOT = Path(action_engine_package.__file__).resolve().parent
_LEGACY_PACKAGE = "embodichain.gen_sim.action_agent_pipeline"


def _production_python_files() -> list[Path]:
    return sorted(
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(_PACKAGE_ROOT).parts
    )


def test_production_code_has_no_legacy_pipeline_imports() -> None:
    offenders: list[str] = []
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith(_LEGACY_PACKAGE) for name in names):
                offenders.append(path.relative_to(_PACKAGE_ROOT).as_posix())
                break
    assert offenders == []


def test_protocol_identifiers_are_new_and_stable() -> None:
    assert ACTION_ENGINE_ENV_ID == "ActionEngine-v1"
    assert ACTION_ENGINE_CONFIG_SCHEMA == "action_engine_config_v2"
    assert SEED_GRAPH_SCHEMA == "action_engine_seed_graph_v3"
    assert TASK_SPEC_SCHEMA == "action_engine_task_spec_v2"
    assert SCENE_REQUIREMENTS_SCHEMA == "action_engine_scene_requirements_v2"
    assert EXECUTION_PROGRAM_FILENAME == "seed_task_graph.json"
    assert TASK_SPEC_FILENAME == "task_spec.json"
    assert SCENE_REQUIREMENTS_FILENAME == "scene_requirements.json"


def test_planner_exposes_exactly_the_first_phase_skill_catalog() -> None:
    assert set(build_default_registry().operator_names()) == {
        "arrange_line",
        "build_stack",
        "coordinated_transport",
        "orient_object",
        "place_relative",
    }


def test_atomic_actions_have_one_runtime_capability_catalog() -> None:
    registry = build_atomic_capability_registry()
    assert set(registry.names()) == {
        "CoordinatedPickment",
        "CoordinatedPlacement",
        "HandOver",
        "MoveEndEffector",
        "MoveHeldObject",
        "MoveJoints",
        "PickUp",
        "Place",
        "Pour",
        "Press",
        "PullArticulatedPart",
        "PushArticulatedPart",
        "TurnKnob",
    }
    assert set(registry.executable_names()) == {
        "CoordinatedPickment",
        "CoordinatedPlacement",
        "HandOver",
        "MoveEndEffector",
        "MoveHeldObject",
        "MoveJoints",
        "PickUp",
        "Place",
        "Press",
    }


def test_action_class_dispatch_is_not_duplicated_across_runtime_layers() -> None:
    offenders = []
    for path in _production_python_files():
        if path.name == "atomic.py" and path.parent.name == "capabilities":
            continue
        source = path.read_text(encoding="utf-8")
        if "_ACTION_TYPES" in source:
            offenders.append(path.relative_to(_PACKAGE_ROOT).as_posix())
    assert offenders == []


def test_runtime_core_has_no_action_name_dispatch_branches() -> None:
    action_names = set(build_atomic_capability_registry().executable_names())
    offenders = {}
    for relative in (
        "runtime/actions.py",
        "runtime/executor.py",
        "runtime/grounding.py",
    ):
        path = _PACKAGE_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        duplicated = sorted(literals & action_names)
        if duplicated:
            offenders[relative] = duplicated
    assert offenders == {}
