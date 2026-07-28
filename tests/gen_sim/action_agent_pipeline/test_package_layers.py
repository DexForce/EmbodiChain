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
from collections.abc import Mapping
import importlib.util
import os
from pathlib import Path

import pytest

import embodichain.gen_sim.action_agent_pipeline as action_agent_pipeline_package
from embodichain.gen_sim.action_agent_pipeline import defaults as legacy_defaults
from embodichain.gen_sim.action_agent_pipeline.cli import pipeline_defaults
from embodichain.gen_sim.action_agent_pipeline.config import defaults
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware import (
    success as legacy_success,
)
from embodichain.gen_sim.action_agent_pipeline.runtime import success_evaluator

# Resolve production ownership from the imported package, not the test location.
_PACKAGE_ROOT = Path(next(iter(action_agent_pipeline_package.__path__))).resolve()
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
    _assert_acyclic_import_graph(edges)


def test_production_dynamic_import_targets_are_statically_resolvable() -> None:
    violations: list[str] = []
    for path in _production_module_paths():
        for line_number, call_name in _unresolved_dynamic_imports(path):
            violations.append(
                f"{path.relative_to(_PACKAGE_ROOT)}:{line_number} uses "
                f"non-literal dynamic import {call_name}"
            )
    assert not violations, "\n".join(violations)


def test_current_relative_and_literal_dynamic_imports_are_graph_edges() -> None:
    module_paths = _production_module_index()
    known_modules = set(module_paths)
    prompt_imports = {
        module_name
        for module_name, _ in _pipeline_imports(
            module_paths[f"{_PACKAGE_PREFIX}.prompts"],
            known_modules=known_modules,
        )
    }
    runtime_imports = {
        module_name
        for module_name, _ in _pipeline_imports(
            module_paths[f"{_PACKAGE_PREFIX}.runtime.graph_compiler"],
            known_modules=known_modules,
        )
    }

    assert f"{_PACKAGE_PREFIX}.prompts.template_loader" in prompt_imports
    assert f"{_PACKAGE_PREFIX}.runtime.task_graph" in runtime_imports
    assert f"{_PACKAGE_PREFIX}.runtime.atom_actions" in runtime_imports


def test_import_collector_resolves_relative_submodule_and_dynamic_imports() -> None:
    source_module = f"{_PACKAGE_PREFIX}.generation.synthetic_module"
    known_modules = {
        _PACKAGE_PREFIX,
        f"{_PACKAGE_PREFIX}.contracts",
        f"{_PACKAGE_PREFIX}.domain",
        f"{_PACKAGE_PREFIX}.domain.object_semantics",
        f"{_PACKAGE_PREFIX}.generation",
        f"{_PACKAGE_PREFIX}.generation.relative_intent",
        f"{_PACKAGE_PREFIX}.generation.relative_spec",
        f"{_PACKAGE_PREFIX}.runtime.atom_actions",
        f"{_PACKAGE_PREFIX}.runtime.success_evaluator",
        f"{_PACKAGE_PREFIX}.runtime.task_graph",
    }
    source = f"""\
from .relative_intent import _normalize_relative_relation
from . import relative_spec
from ..domain import object_semantics
from {_PACKAGE_PREFIX} import contracts
import importlib as module_loader
from importlib import import_module as load_module
module_loader.import_module("{_PACKAGE_PREFIX}.runtime.task_graph")
load_module("{_PACKAGE_PREFIX}.runtime.atom_actions")
__import__("{_PACKAGE_PREFIX}.domain.object_semantics")
module_loader.import_module(".success_evaluator", "{_PACKAGE_PREFIX}.runtime")
"""

    imports, unresolved = _pipeline_imports_from_source(
        source,
        source_module=source_module,
        source_is_package=False,
        known_modules=known_modules,
    )

    assert unresolved == []
    assert {
        (f"{_PACKAGE_PREFIX}.generation.relative_intent", 1),
        (f"{_PACKAGE_PREFIX}.generation.relative_spec", 2),
        (f"{_PACKAGE_PREFIX}.domain.object_semantics", 3),
        (f"{_PACKAGE_PREFIX}.contracts", 4),
        (f"{_PACKAGE_PREFIX}.runtime.task_graph", 7),
        (f"{_PACKAGE_PREFIX}.runtime.atom_actions", 8),
        (f"{_PACKAGE_PREFIX}.domain.object_semantics", 9),
        (f"{_PACKAGE_PREFIX}.runtime.success_evaluator", 10),
    } <= set(imports)


def test_import_collector_rejects_non_literal_dynamic_targets() -> None:
    source = """\
import importlib
module_name = "some.plugin"
importlib.import_module(module_name)
importlib.import_module(f"{module_name}.child")
"""

    imports, unresolved = _pipeline_imports_from_source(
        source,
        source_module=f"{_PACKAGE_PREFIX}.runtime.synthetic_module",
        source_is_package=False,
        known_modules=set(),
    )

    assert imports == []
    assert unresolved == [
        (3, "importlib.import_module"),
        (4, "importlib.import_module"),
    ]


def test_cycle_report_includes_the_complete_dependency_path() -> None:
    module_sources = {
        f"{_PACKAGE_PREFIX}.synthetic.alpha": "from . import beta\n",
        f"{_PACKAGE_PREFIX}.synthetic.beta": "from . import gamma\n",
        f"{_PACKAGE_PREFIX}.synthetic.gamma": "from . import alpha\n",
    }
    known_modules = set(module_sources)
    edges = {}
    for module_name, source in module_sources.items():
        imports, unresolved = _pipeline_imports_from_source(
            source,
            source_module=module_name,
            source_is_package=False,
            known_modules=known_modules,
        )
        assert unresolved == []
        edges[module_name] = {
            imported for imported, _ in imports if imported in known_modules
        }

    with pytest.raises(AssertionError) as exc_info:
        _assert_acyclic_import_graph(edges)
    assert str(exc_info.value) == (
        "Package import cycle detected: "
        f"{_PACKAGE_PREFIX}.synthetic.alpha -> "
        f"{_PACKAGE_PREFIX}.synthetic.beta -> "
        f"{_PACKAGE_PREFIX}.synthetic.gamma -> "
        f"{_PACKAGE_PREFIX}.synthetic.alpha"
    )


def _production_module_paths() -> list[Path]:
    return [
        path
        for path in _PACKAGE_ROOT.rglob("*.py")
        if "tests" not in path.relative_to(_PACKAGE_ROOT).parts
    ]


def _production_module_index() -> dict[str, Path]:
    return {_path_module(path): path for path in _production_module_paths()}


def _pipeline_imports(
    path: Path,
    *,
    known_modules: set[str] | None = None,
) -> list[tuple[str, int]]:
    imports, _ = _pipeline_imports_from_source(
        path.read_text(encoding="utf-8"),
        source_module=_path_module(path),
        source_is_package=path.name == "__init__.py",
        known_modules=(
            known_modules
            if known_modules is not None
            else set(_production_module_index())
        ),
    )
    return imports


def _unresolved_dynamic_imports(path: Path) -> list[tuple[int, str]]:
    _, unresolved = _pipeline_imports_from_source(
        path.read_text(encoding="utf-8"),
        source_module=_path_module(path),
        source_is_package=path.name == "__init__.py",
        known_modules=set(_production_module_index()),
    )
    return unresolved


def _pipeline_imports_from_source(
    source: str,
    *,
    source_module: str,
    source_is_package: bool,
    known_modules: set[str],
) -> tuple[list[tuple[str, int]], list[tuple[int, str]]]:
    tree = ast.parse(source)
    source_package = (
        source_module if source_is_package else source_module.rpartition(".")[0]
    )
    imports: list[tuple[str, int]] = []
    unresolved: list[tuple[int, str]] = []
    seen_imports: set[tuple[str, int]] = set()
    importlib_aliases = {"importlib"}
    import_module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "importlib"
        ):
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_aliases.add(alias.asname or alias.name)

    def add_import(module_name: str, line_number: int) -> None:
        dependency = (module_name, line_number)
        if (
            module_name.startswith(_PACKAGE_PREFIX)
            and module_name != source_module
            and dependency not in seen_imports
        ):
            seen_imports.add(dependency)
            imports.append(dependency)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = _resolve_import_from_module(
                node,
                source_package=source_package,
            )
            add_import(module_name, node.lineno)
            # ``from package import name`` may import a child module or merely
            # an attribute. The package index is the reliable discriminator.
            for alias in node.names:
                child_module = f"{module_name}.{alias.name}"
                if alias.name != "*" and child_module in known_modules:
                    add_import(child_module, node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                add_import(alias.name, node.lineno)
        elif isinstance(node, ast.Call):
            call_name = _dynamic_import_call_name(
                node,
                importlib_aliases=importlib_aliases,
                import_module_aliases=import_module_aliases,
            )
            if call_name is None:
                continue
            module_name = _literal_dynamic_import_module(
                node,
                call_name=call_name,
            )
            if module_name is None:
                unresolved.append((node.lineno, call_name))
                continue
            add_import(module_name, node.lineno)
    return imports, unresolved


def _resolve_import_from_module(
    node: ast.ImportFrom,
    *,
    source_package: str,
) -> str:
    if node.level == 0:
        return node.module or ""
    relative_name = f"{'.' * node.level}{node.module or ''}"
    try:
        # Python resolves relative imports from the containing package, not
        # from the importing module's full dotted name.
        return importlib.util.resolve_name(relative_name, source_package)
    except ImportError as exc:
        raise AssertionError(
            f"Cannot resolve relative import {relative_name!r} from "
            f"{source_package!r} at line {node.lineno}."
        ) from exc


def _dynamic_import_call_name(
    node: ast.Call,
    *,
    importlib_aliases: set[str],
    import_module_aliases: set[str],
) -> str | None:
    if isinstance(node.func, ast.Name):
        if node.func.id == "__import__" or node.func.id in import_module_aliases:
            return node.func.id
        return None
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_aliases
    ):
        return f"{node.func.value.id}.import_module"
    return None


def _literal_dynamic_import_module(
    node: ast.Call,
    *,
    call_name: str,
) -> str | None:
    name_arg = _call_argument(node, position=0, keyword="name")
    if not isinstance(name_arg, ast.Constant) or not isinstance(name_arg.value, str):
        return None
    module_name = name_arg.value
    if not module_name.startswith("."):
        return module_name
    if call_name == "__import__":
        return None

    package_arg = _call_argument(node, position=1, keyword="package")
    if not isinstance(package_arg, ast.Constant) or not isinstance(
        package_arg.value, str
    ):
        return None
    try:
        return importlib.util.resolve_name(module_name, package_arg.value)
    except ImportError:
        return None


def _call_argument(
    node: ast.Call,
    *,
    position: int,
    keyword: str,
) -> ast.expr | None:
    if len(node.args) > position:
        return node.args[position]
    return next(
        (item.value for item in node.keywords if item.arg == keyword),
        None,
    )


def _assert_acyclic_import_graph(
    edges: Mapping[str, set[str]],
) -> None:
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
        for dependency in sorted(edges.get(module_name, set())):
            visit(dependency)
        visiting.pop()
        visited.add(module_name)

    for module_name in sorted(edges):
        visit(module_name)


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
