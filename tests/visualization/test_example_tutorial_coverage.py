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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_TUTORIAL_ROOTS = (
    PROJECT_ROOT / "examples",
    PROJECT_ROOT / "scripts" / "tutorials",
)
MANUAL_VISER_TUTORIAL = (
    PROJECT_ROOT / "scripts" / "tutorials" / "visualization" / "viser_scene.py"
)


def _simulation_manager_cfg_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr if isinstance(node.func, ast.Attribute) else None
        )
        if function_name == "SimulationManagerCfg":
            calls.append(node)
    return calls


def test_simulation_examples_and_tutorials_are_viser_aware() -> None:
    missing_visualization_cfg: list[str] = []

    for root in EXAMPLE_TUTORIAL_ROOTS:
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            calls = _simulation_manager_cfg_calls(tree)
            if not calls:
                continue
            if path == MANUAL_VISER_TUTORIAL:
                assert "VisualizationRuntime(" in source
                continue
            for call in calls:
                if not any(keyword.arg == "visualization" for keyword in call.keywords):
                    relative_path = path.relative_to(PROJECT_ROOT)
                    missing_visualization_cfg.append(f"{relative_path}:{call.lineno}")

    assert not missing_visualization_cfg, (
        "Simulation examples/tutorials must pass a visualization configuration: "
        + ", ".join(missing_visualization_cfg)
    )
