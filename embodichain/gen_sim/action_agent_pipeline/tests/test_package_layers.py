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

"""Protect package-level ownership, dependency direction, and compatibility."""

from __future__ import annotations

import ast
import os
from pathlib import Path

from embodichain.gen_sim.action_agent_pipeline import defaults as legacy_defaults
from embodichain.gen_sim.action_agent_pipeline.cli import pipeline_defaults
from embodichain.gen_sim.action_agent_pipeline.config import defaults
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware import (
    success as legacy_success,
)
from embodichain.gen_sim.action_agent_pipeline.runtime import success_evaluator

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PREFIX = "embodichain.gen_sim.action_agent_pipeline"
_COMPATIBILITY_FACADES = {
    f"{_PACKAGE_PREFIX}.contracts": {
        f"{_PACKAGE_PREFIX}.config.defaults",
        f"{_PACKAGE_PREFIX}.protocol.actions",
        f"{_PACKAGE_PREFIX}.protocol.artifacts",
        f"{_PACKAGE_PREFIX}.protocol.success",
        f"{_PACKAGE_PREFIX}.protocol.tasks",
    },
    f"{_PACKAGE_PREFIX}.defaults": {
        f"{_PACKAGE_PREFIX}.config.defaults",
    },
    f"{_PACKAGE_PREFIX}.semantics": {
        f"{_PACKAGE_PREFIX}.domain.object_semantics",
        f"{_PACKAGE_PREFIX}.generation.relation_language",
    },
    f"{_PACKAGE_PREFIX}.env_adapters.tableware.success": {
        f"{_PACKAGE_PREFIX}.runtime.success_evaluator",
    },
}
_LAYER_NAMES = {
    "agents",
    "cli",
    "compatibility",
    "config",
    "domain",
    "env_adapters",
    "generation",
    "prompts",
    "protocol",
    "runtime",
    "tests",
    "utils",
}

# Lower layers may not reach into orchestration or concrete integration layers.
# Scanning every AST node also catches imports intentionally delayed in a
# function body, which are still architectural dependencies.
_FORBIDDEN_TARGETS = {
    "config": _LAYER_NAMES - {"config"},
    "domain": _LAYER_NAMES - {"domain"},
    "protocol": _LAYER_NAMES - {"protocol"},
    "core": {
        "agents",
        "cli",
        "compatibility",
        "env_adapters",
        "generation",
        "prompts",
        "runtime",
        "utils",
    },
    "compatibility": set(),
    "prompts": {"agents", "cli", "env_adapters", "generation", "runtime"},
    "utils": {"agents", "cli", "env_adapters", "generation", "runtime"},
    "generation": {"agents", "cli", "compatibility", "env_adapters", "runtime"},
    "runtime": {"agents", "cli", "compatibility", "env_adapters", "generation"},
    "agents": {"cli", "compatibility", "env_adapters", "generation"},
    "env_adapters": {"cli", "compatibility", "generation"},
    "cli": set(),
}


def test_shared_defaults_have_package_ownership_and_legacy_identity() -> None:
    assert defaults._DEFAULTS_PATH == _PACKAGE_ROOT / "config" / "defaults.yaml"
    assert not (
        _PACKAGE_ROOT / "generation" / "action_agent_config_defaults.yaml"
    ).exists()
    assert (
        legacy_defaults.ACTION_AGENT_CONFIG_DEFAULTS is defaults.ACTION_AGENT_DEFAULTS
    )
    assert legacy_defaults.defaults_section is defaults.defaults_section
    assert legacy_defaults.generation_defaults_section is defaults.defaults_section


def test_distinct_task_name_policies_keep_their_compatibility_aliases() -> None:
    expected_pipeline_name = os.getenv(
        "ACTION_AGENT_DEFAULT_TASK_NAME", "ActionAgentTask"
    )
    assert defaults.DEFAULT_GENERATED_CONFIG_TASK_NAME == "gen_sim"
    assert pipeline_defaults.DEFAULT_PIPELINE_TASK_NAME == expected_pipeline_name
    assert defaults.DEFAULT_TASK_NAME == defaults.DEFAULT_GENERATED_CONFIG_TASK_NAME
    assert (
        pipeline_defaults.DEFAULT_TASK_NAME
        == pipeline_defaults.DEFAULT_PIPELINE_TASK_NAME
    )


def test_success_evaluator_legacy_path_is_an_identity_facade() -> None:
    assert (
        legacy_success.evaluate_configured_success
        is success_evaluator.evaluate_configured_success
    )
    assert legacy_success._FALLBACKS is success_evaluator._FALLBACKS


def test_package_layers_follow_the_declared_dependency_direction() -> None:
    violations: list[str] = []
    for path in _production_module_paths():
        source_layer = _path_layer(path)
        for target_module, line_number in _pipeline_imports(path):
            target_layer = _module_layer(target_module)
            if target_layer in _FORBIDDEN_TARGETS[source_layer]:
                violations.append(
                    f"{path.relative_to(_PACKAGE_ROOT)}:{line_number} "
                    f"{source_layer} imports forbidden {target_layer} module "
                    f"{target_module}"
                )
    assert not violations, "\n".join(violations)


def test_production_modules_do_not_import_compatibility_facades() -> None:
    violations: list[str] = []
    for path in _production_module_paths():
        source_module = _path_module(path)
        if source_module in _COMPATIBILITY_FACADES:
            continue
        for target_module, line_number in _pipeline_imports(path):
            if target_module in _COMPATIBILITY_FACADES:
                violations.append(
                    f"{path.relative_to(_PACKAGE_ROOT)}:{line_number} imports "
                    f"compatibility facade {target_module}"
                )
    assert not violations, "\n".join(violations)


def test_compatibility_facades_only_import_canonical_owners() -> None:
    module_paths = {_path_module(path): path for path in _production_module_paths()}
    violations: list[str] = []
    for facade_module, allowed_targets in _COMPATIBILITY_FACADES.items():
        path = module_paths[facade_module]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef))
            for node in tree.body
        ):
            violations.append(
                f"{path.relative_to(_PACKAGE_ROOT)} defines behavior instead of "
                "only re-exporting canonical owners"
            )
        for target_module, line_number in _pipeline_imports(path):
            if target_module not in allowed_targets:
                violations.append(
                    f"{path.relative_to(_PACKAGE_ROOT)}:{line_number} imports "
                    f"non-canonical facade target {target_module}"
                )
    assert not violations, "\n".join(violations)


def test_package_internal_import_graph_is_acyclic() -> None:
    module_paths = {_path_module(path): path for path in _production_module_paths()}
    edges = {
        module_name: {
            imported
            for imported, _ in _pipeline_imports(path)
            if imported in module_paths
        }
        for module_name, path in module_paths.items()
    }
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module_name: str) -> None:
        if module_name in visiting:
            cycle_start = visiting.index(module_name)
            cycle = " -> ".join([*visiting[cycle_start:], module_name])
            raise AssertionError(f"Package import cycle detected: {cycle}")
        if module_name in visited:
            return
        visiting.append(module_name)
        for dependency in sorted(edges[module_name]):
            visit(dependency)
        visiting.pop()
        visited.add(module_name)

    for module_name in sorted(edges):
        visit(module_name)


def _production_module_paths() -> list[Path]:
    return [
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(_PACKAGE_ROOT).parts
    ]


def _pipeline_imports(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name.startswith(_PACKAGE_PREFIX):
                imports.append((module_name, node.lineno))
        elif isinstance(node, ast.Import):
            imports.extend(
                (alias.name, node.lineno)
                for alias in node.names
                if alias.name.startswith(_PACKAGE_PREFIX)
            )
    return imports


def _path_module(path: Path) -> str:
    relative = path.relative_to(_PACKAGE_ROOT).with_suffix("")
    parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
    suffix = ".".join(parts)
    return f"{_PACKAGE_PREFIX}.{suffix}" if suffix else _PACKAGE_PREFIX


def _path_layer(path: Path) -> str:
    if _path_module(path) in _COMPATIBILITY_FACADES:
        return "compatibility"
    relative = path.relative_to(_PACKAGE_ROOT)
    return relative.parts[0] if len(relative.parts) > 1 else "core"


def _module_layer(module_name: str) -> str:
    if module_name in _COMPATIBILITY_FACADES:
        return "compatibility"
    relative = module_name.removeprefix(f"{_PACKAGE_PREFIX}.")
    first_component = relative.split(".", maxsplit=1)[0]
    return first_component if first_component in _LAYER_NAMES else "core"
