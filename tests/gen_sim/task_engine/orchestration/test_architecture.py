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
from dataclasses import fields
from inspect import signature
from pathlib import Path

import embodichain.gen_sim as gen_sim_package
from embodichain.gen_sim.task_engine import TaskAgent
from embodichain.gen_sim.task_engine import __main__ as task_engine_main
from embodichain.gen_sim.task_engine import cli as task_engine_cli
from embodichain.gen_sim.task_engine.orchestration.coordinator import (
    TaskEngineCoordinator,
)
from embodichain.gen_sim.task_engine.orchestration.artifacts import (
    TaskEngineArtifactPaths,
    write_task_engine_artifacts,
)
from embodichain.gen_sim.task_engine.orchestration.scene_adapter import SceneAdapter

_GEN_SIM_ROOT = Path(gen_sim_package.__file__).resolve().parent
_PURE_TASK_MODULES = (
    "agent.py",
    "config.py",
    "contracts.py",
    "interpretation.py",
    "ontology.py",
    "state_machine.py",
    "workflow_contracts.py",
)


def test_task_semantic_core_does_not_import_scene_action_or_orchestration() -> None:
    forbidden = {
        "embodichain.gen_sim.action_engine",
        "embodichain.gen_sim.scene_engine",
        "embodichain.gen_sim.task_engine.orchestration",
        "embodichain.gen_sim.task_engine.scene",
    }
    offenders: list[str] = []
    for filename in _PURE_TASK_MODULES:
        path = _GEN_SIM_ROOT / "task_engine" / filename
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
                offenders.append(filename)
                break
    assert offenders == []


def test_task_engine_does_not_import_atomic_execution_layer() -> None:
    forbidden = "embodichain.lab.sim.atomic_actions"
    offenders: list[str] = []
    task_root = _GEN_SIM_ROOT / "task_engine"
    for path in task_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(
                module == forbidden or module.startswith(forbidden + ".")
                for module in modules
            ):
                offenders.append(path.relative_to(task_root).as_posix())
                break
    assert offenders == []


def test_default_task_runtime_has_no_legacy_physical_executor() -> None:
    task_root = _GEN_SIM_ROOT / "task_engine"
    default_runtime = (
        task_root / "workflow.py",
        task_root / "_bundle_runner.py",
        task_root / "semantic_planner.py",
        task_root / "task_program_bundle.py",
        task_root / "orchestration/coordinator.py",
    )
    forbidden_names = {
        "ActionAgent",
        "ActionGrounder",
        "AtomicActionAdapter",
        "ProgramExecutor",
    }
    offenders = {
        path.relative_to(task_root).as_posix(): sorted(
            forbidden_names.intersection(
                node.id
                for node in ast.walk(
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                )
                if isinstance(node, ast.Name)
            )
        )
        for path in default_runtime
    }
    assert {path: names for path, names in offenders.items() if names} == {}


def test_cross_engine_owners_are_explicit() -> None:
    assert TaskAgent.__module__ == "embodichain.gen_sim.task_engine.agent"
    assert SceneAdapter.__module__.startswith(
        "embodichain.gen_sim.task_engine.orchestration"
    )
    assert TaskEngineCoordinator.__module__.startswith(
        "embodichain.gen_sim.task_engine.orchestration"
    )


def test_task_engine_owns_its_module_entry_point() -> None:
    assert task_engine_main.main is task_engine_cli.main


def test_task_engine_artifacts_have_no_legacy_grounded_plan_boundary() -> None:
    assert "grounded_task_plan" not in {
        field.name for field in fields(TaskEngineArtifactPaths)
    }
    assert "grounded_task_plan" not in signature(write_task_engine_artifacts).parameters


def test_legacy_cross_engine_packages_are_deleted() -> None:
    assert not any((_GEN_SIM_ROOT / "action_engine").rglob("*.py"))
    assert not (_GEN_SIM_ROOT / "scene_bridge").exists()
    assert not (_GEN_SIM_ROOT / "collaboration").exists()
