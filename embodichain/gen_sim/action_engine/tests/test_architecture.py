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
import json
from pathlib import Path

from embodichain.gen_sim.action_engine.capabilities import build_default_registry
from embodichain.gen_sim.action_engine.protocol import (
    ACTION_ENGINE_CONFIG_SCHEMA,
    ACTION_ENGINE_ENV_ID,
    EXECUTION_PROGRAM_FILENAME,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_PACKAGE = "embodichain.gen_sim.action_agent_pipeline"


def _production_python_files() -> list[Path]:
    return sorted(
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(_PACKAGE_ROOT).parts
    )


def test_legacy_imports_are_isolated_to_runtime_adapters() -> None:
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
    assert offenders == [
        "runtime/motion_policy.py",
        "runtime/pipeline_backend.py",
    ]


def test_protocol_identifiers_are_new_and_stable() -> None:
    assert ACTION_ENGINE_ENV_ID == "ActionEngine-v1"
    assert ACTION_ENGINE_CONFIG_SCHEMA == "action_engine_config_v1"
    assert EXECUTION_PROGRAM_FILENAME == "seed_task_graph.json"


def test_planner_exposes_exactly_the_first_phase_skill_catalog() -> None:
    assert set(build_default_registry().operator_names()) == {
        "arrange_line",
        "build_stack",
        "coordinated_transport",
        "orient_object",
        "place_relative",
    }


def test_acceptance_manifest_covers_twenty_supported_tasks() -> None:
    manifest_path = (
        _PACKAGE_ROOT.parents[2] / "texts" / "action_engine" / "acceptance_tasks.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    names = [task["task_name"] for task in tasks]
    visible = set(build_default_registry().operator_names())

    assert len(tasks) == 20
    assert len(names) == len(set(names))
    assert all(set(task["expected_skills"]) <= visible for task in tasks)


def test_recovery_branch_stays_inside_temporary_physical_line_ceiling() -> None:
    # P7 keeps 10,000 as the optimization target. During parity recovery, the
    # explicit production-backend adapter temporarily raises the physical
    # ceiling. This still prevents unbounded growth while the independent
    # runtime is characterized and removed.
    line_count = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in _production_python_files()
    )
    # The main-sync adapter resolves semantic arms into the new per-control-part
    # WorldState contract without weakening the independent runtime boundary.
    assert line_count <= 12_650, f"Action Engine production LOC grew to {line_count}"
