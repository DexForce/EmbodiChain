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

"""Build a conservative, explainable pytest plan from a repository diff.

The selector deliberately works without importing EmbodiChain.  CI can run it
on a small Python installation before starting the CUDA container.  Static
rules and the AST graph provide the first line of impact analysis; callers may
add coverage-derived selectors to the same plan in a later stage.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Python 3.11 (the CI planner runtime) has this in the standard library.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10.
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]

__all__ = [
    "ChangedPath",
    "ImportGraph",
    "TestPlan",
    "build_plan",
    "collect_changed_paths",
    "load_manifest",
    "main",
]

PLAN_VERSION = 1
DEFAULT_MANIFEST = ".ci/test-impact.toml"
DEFAULT_MAP = "agent_context/MAP.yaml"
TEST_ROOT = "tests"
_GLOB_CHARS = "*?["
_SOURCE_ROOTS = ("embodichain", "embodichain_tasks", "scripts", "examples", "tests")
_DOC_PATTERNS = (
    "docs/**",
    "docs/scripts/**",
    "tests/docs/**",
    "design/**",
    "agent_context/topics/**",
    "agent_context/conventions/**",
    "README.md",
    "AGENTS.md",
    "CONTRIBUTORS.md",
    "CONTRIBUTING.md",
)
_HIGH_RISK_TOKENS = (
    "SimulationManager",
    "register_env",
    "entry_points",
    "__all__",
    "configclass",
    "PhysicsBackend",
)
_LANES = ("docs", "fast", "sim", "distributed", "gpu")
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _is_test_file(path: Path) -> bool:
    return path.name.startswith("test_") or path.name.endswith("_test.py")


@dataclass(frozen=True)
class ChangedPath:
    """One path changed between the PR base and head revisions."""

    path: str
    status: str = "M"
    old_path: str | None = None

    @property
    def paths(self) -> tuple[str, ...]:
        """Return both old and new names, preserving rename impact."""
        if self.old_path and self.old_path != self.path:
            return (self.old_path, self.path)
        return (self.path,)


@dataclass
class TestPlan:
    """Serializable test selection and its audit information."""

    mode: str
    risk: str
    base_sha: str | None
    head_sha: str | None
    changed: list[dict[str, Any]] = field(default_factory=list)
    selectors: list[str] = field(default_factory=list)
    lanes: dict[str, list[str]] = field(default_factory=dict)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    resource_hints: list[str] = field(default_factory=list)
    install: dict[str, bool] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    graph_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON representation consumed by CI runners."""
        return {
            "version": PLAN_VERSION,
            "mode": self.mode,
            "risk": self.risk,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "changed": self.changed,
            "selectors": self.selectors,
            "lanes": self.lanes,
            "reasons": self.reasons,
            "resource_hints": self.resource_hints,
            "install": self.install,
            "topics": self.topics,
            "fallback_reason": self.fallback_reason,
            "graph_warnings": self.graph_warnings,
        }


def _normalize_path(value: str) -> str:
    """Normalize a repository path to a safe POSIX spelling."""
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"unsafe repository path: {value!r}")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if not normalized:
        raise ValueError(f"unsafe repository path: {value!r}")
    return normalized


def path_matches(path: str, pattern: str) -> bool:
    """Match a repository path against a file, directory, or glob pattern."""
    path = _normalize_path(path)
    pattern = _normalize_path(pattern)
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(f"{prefix}/")
    if pattern.endswith("/"):
        prefix = pattern.rstrip("/")
        return path == prefix or path.startswith(f"{prefix}/")
    return fnmatch.fnmatchcase(path, pattern)


def _patterns_match(path: str, patterns: Iterable[str]) -> bool:
    return any(path_matches(path, pattern) for pattern in patterns)


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def collect_changed_paths(
    root: str | Path,
    base: str,
    head: str = "HEAD",
) -> list[ChangedPath]:
    """Collect changed paths using git's rename-aware NUL format.

    Args:
        root: Repository root.
        base: Base revision or merge-base revision.
        head: Head revision, defaulting to ``HEAD``.

    Returns:
        Changed paths in git's reported order.

    Raises:
        RuntimeError: If git cannot resolve either revision.
    """
    try:
        output = _git_output(
            Path(root),
            ["diff", "--name-status", "--find-renames", "-z", base, head, "--"],
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else "unknown git error"
        raise RuntimeError(f"cannot compute diff {base}..{head}: {detail}") from error

    tokens = [token for token in output.split("\0") if token]
    changed: list[ChangedPath] = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if status[:1] in {"R", "C"}:
            if index + 1 >= len(tokens):
                raise RuntimeError("malformed rename entry in git diff")
            old_path, new_path = tokens[index], tokens[index + 1]
            index += 2
            changed.append(
                ChangedPath(
                    path=_normalize_path(new_path),
                    old_path=_normalize_path(old_path),
                    status=status[:1],
                )
            )
            continue
        if index >= len(tokens):
            raise RuntimeError("malformed entry in git diff")
        changed.append(
            ChangedPath(path=_normalize_path(tokens[index]), status=status[:1] or "M")
        )
        index += 1
    return changed


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the CI impact manifest."""
    if tomllib is None:  # pragma: no cover - Python 3.10 without tomli.
        raise RuntimeError(
            "Python 3.11 or the tomli package is required to read the impact manifest"
        )
    manifest_path = Path(path)
    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"cannot read impact manifest {manifest_path}: {error}"
        ) from error
    if data.get("version") != PLAN_VERSION:
        raise RuntimeError(
            f"unsupported impact manifest version: {data.get('version')!r}"
        )
    if not isinstance(data.get("rules", []), list):
        raise RuntimeError("impact manifest 'rules' must be an array of tables")
    if not isinstance(data.get("topics", {}), dict):
        raise RuntimeError("impact manifest 'topics' must be a table")
    for field_name in ("always", "full_pr_if"):
        value = data.get(field_name, [])
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise RuntimeError(
                f"impact manifest '{field_name}' must be an array of strings"
            )
    depth = data.get("reverse_import_depth", 4)
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise RuntimeError(
            "impact manifest 'reverse_import_depth' must be a positive integer"
        )
    for index, rule in enumerate(data.get("rules", [])):
        if not isinstance(rule, dict):
            raise RuntimeError(f"impact rule {index} must be a table")
        for field_name in ("paths", "tests", "contracts", "resource_hints"):
            value = rule.get(field_name, [])
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise RuntimeError(
                    f"impact rule {index} '{field_name}' must be an array of strings"
                )
        if not rule.get("paths"):
            raise RuntimeError(f"impact rule {index} is missing path patterns")
        mode = rule.get("mode", "partial")
        if not isinstance(mode, str) or mode not in {"partial", "full-pr", "full"}:
            raise RuntimeError(f"impact rule {index} has an unsupported mode: {mode!r}")
        if mode == "partial" and not (rule.get("tests") or rule.get("contracts")):
            raise RuntimeError(f"partial impact rule {index} is missing test selectors")
        rule_risk = rule.get("risk", "medium")
        if not isinstance(rule_risk, str) or rule_risk not in _RISK_ORDER:
            raise RuntimeError(f"impact rule {index} has an unsupported risk")
    for topic_id, topic in data.get("topics", {}).items():
        if not isinstance(topic, dict):
            raise RuntimeError(f"impact topic {topic_id!r} must be a table")
        for field_name in ("tests", "contracts", "resource_hints"):
            value = topic.get(field_name, [])
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise RuntimeError(
                    f"impact topic {topic_id!r} '{field_name}' must be an array of strings"
                )
    return data


def _load_map(path: Path) -> dict[str, Any]:
    """Load MAP.yaml when PyYAML is available; absence is a warning upstream."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _map_topics(
    map_data: Mapping[str, Any], changed: Sequence[ChangedPath]
) -> list[str]:
    topics = map_data.get("topics", [])
    if not isinstance(topics, list):
        return []
    result: list[str] = []
    for topic in topics:
        if not isinstance(topic, Mapping) or not isinstance(topic.get("id"), str):
            continue
        mapped: list[str] = []
        for field in ("source_of_truth", "watch_paths"):
            values = topic.get(field, [])
            if isinstance(values, list):
                mapped.extend(value for value in values if isinstance(value, str))
        if any(
            _patterns_match(path, mapped) for change in changed for path in change.paths
        ):
            result.append(topic["id"])
    return result


def _topic_related(map_data: Mapping[str, Any], topic_ids: Sequence[str]) -> list[str]:
    by_id = {
        topic.get("id"): topic
        for topic in map_data.get("topics", [])
        if isinstance(topic, Mapping) and isinstance(topic.get("id"), str)
    }
    related: list[str] = []
    for topic_id in topic_ids:
        topic = by_id.get(topic_id, {})
        values = topic.get("related_topics", []) if isinstance(topic, Mapping) else []
        if isinstance(values, list):
            related.extend(value for value in values if isinstance(value, str))
    return list(dict.fromkeys(related))


def _module_name(path: Path, root: Path) -> tuple[str, bool] | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _discover_modules(root: Path) -> tuple[dict[str, Path], dict[str, bool]]:
    modules: dict[str, Path] = {}
    package_modules: dict[str, bool] = {}
    for source_root in _SOURCE_ROOTS:
        directory = root / source_root
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            if any(part in {"__pycache__", ".venv"} for part in path.parts):
                continue
            result = _module_name(path, root)
            if result is None:
                continue
            name, is_package = result
            if name:
                modules[name] = path
                package_modules[name] = is_package
    return modules, package_modules


def _resolve_import(
    imported: str,
    modules: Mapping[str, Path],
) -> set[str]:
    """Resolve an import to the longest known module and package prefixes."""
    candidates: set[str] = set()
    pieces = imported.split(".") if imported else []
    for end in range(len(pieces), 0, -1):
        candidate = ".".join(pieces[:end])
        if candidate in modules:
            candidates.add(candidate)
            break
    return candidates


def _relative_module(
    current: str,
    level: int,
    module: str | None,
    *,
    current_is_package: bool,
) -> str:
    current_parts = current.split(".") if current else []
    package_parts = current_parts if current_is_package else current_parts[:-1]
    if level <= 1:
        base = package_parts
    else:
        base = package_parts[: max(0, len(package_parts) - level + 1)]
    suffix = module.split(".") if module else []
    return ".".join((*base, *suffix))


def _literal_imports(
    tree: ast.AST,
    current: str,
    *,
    current_is_package: bool,
) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = (
                _relative_module(
                    current,
                    node.level,
                    node.module,
                    current_is_package=current_is_package,
                )
                if node.level
                else (node.module or "")
            )
            if module:
                imports.add(module)
                imports.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "import_module" or not node.args:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                imports.add(argument.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in {"__import__", "import_module"} or not node.args:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                imports.add(argument.value)
    return imports


class ImportGraph:
    """A lightweight repository-local reverse import graph."""

    def __init__(
        self,
        root: str | Path,
        modules: Mapping[str, Path],
        reverse: Mapping[str, set[str]],
        warnings: Sequence[str] = (),
    ) -> None:
        self.root = Path(root)
        self.modules = dict(modules)
        self.reverse = {key: set(value) for key, value in reverse.items()}
        self.warnings = list(warnings)

    @classmethod
    def build(cls, root: str | Path) -> "ImportGraph":
        """Build a graph without importing project modules."""
        repository_root = Path(root)
        modules, package_modules = _discover_modules(repository_root)
        reverse: dict[str, set[str]] = defaultdict(set)
        warnings: list[str] = []
        for name, path in modules.items():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as error:
                warnings.append(f"cannot parse {path}: {error}")
                continue
            for imported in _literal_imports(
                tree,
                name,
                current_is_package=package_modules[name],
            ):
                for target in _resolve_import(imported, modules):
                    reverse[target].add(name)
        return cls(repository_root, modules, reverse, warnings)

    def impacted_test_paths(
        self,
        changed: Sequence[ChangedPath],
        *,
        max_depth: int = 4,
    ) -> set[str]:
        """Return nearby test modules that transitively import changed modules.

        Import graphs in framework packages can fan out through a package
        initializer and make every test appear related to a leaf helper.  A
        bounded walk keeps the static hint useful; explicit manifest rules and
        topic contracts remain authoritative for deeper integration coverage.
        """
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        targets: set[str] = set()
        for change in changed:
            for path_value in change.paths:
                path = self.root / path_value
                result = _module_name(path, self.root)
                if result is not None:
                    targets.add(result[0])
        impacted_modules: set[str] = set(targets)
        queue: deque[tuple[str, int]] = deque((target, 0) for target in targets)
        while queue:
            target, depth = queue.popleft()
            for importer in self.reverse.get(target, ()):
                if importer not in impacted_modules:
                    impacted_modules.add(importer)
                    if depth + 1 < max_depth:
                        queue.append((importer, depth + 1))
        result: set[str] = set()
        for module in impacted_modules:
            path = self.modules.get(module)
            if path is None or not module.startswith("tests"):
                continue
            if path.name != "conftest.py" and not _is_test_file(path):
                continue
            result.add(path.relative_to(self.root).as_posix())
        return result


def _as_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _rule_matches(rule: Mapping[str, Any], changed: Sequence[ChangedPath]) -> bool:
    patterns = _as_strings(rule.get("paths"))
    return any(
        _patterns_match(path, patterns) for change in changed for path in change.paths
    )


def _expand_selector(root: Path, selector: str) -> list[str]:
    """Validate a manifest selector and return paths pytest can consume.

    ``pytest`` does not consistently expand shell globs when it receives an
    argv list.  Expand file and recursive directory globs here so every
    selector in the serialized plan is an actual path.
    """
    normalized = _normalize_path(selector)
    if normalized == TEST_ROOT:
        return [normalized]
    candidate = root / normalized
    if candidate.exists():
        return [normalized]
    if not any(char in normalized for char in _GLOB_CHARS):
        return []

    if normalized.endswith("/**"):
        prefix = normalized[:-3].rstrip("/")
        directory = root / prefix
        if directory.is_dir():
            matches = [
                path.relative_to(root).as_posix()
                for path in directory.rglob("*.py")
                if _is_test_file(path)
            ]
            if matches:
                return sorted(dict.fromkeys(matches))
            return []

    matches: list[str] = []
    for path in (root / TEST_ROOT).rglob("*.py"):
        if not _is_test_file(path):
            continue
        relative = path.relative_to(root).as_posix()
        if fnmatch.fnmatchcase(relative, normalized) or path_matches(
            relative, normalized
        ):
            matches.append(relative)
    return sorted(dict.fromkeys(matches))


def _add_selector(
    selectors: set[str],
    reasons: dict[str, set[str]],
    root: Path,
    selector: str,
    reason: str,
    *,
    selector_hints: dict[str, set[str]] | None = None,
    resource_hints: Iterable[str] = (),
) -> bool:
    expanded = _expand_selector(root, selector)
    if not expanded:
        return False
    for value in expanded:
        selectors.add(value)
        reasons.setdefault(value, set()).add(reason)
        if selector_hints is not None:
            selector_hints.setdefault(value, set()).update(resource_hints)
    return True


def _is_doc_path(path: str) -> bool:
    return _patterns_match(path, _DOC_PATTERNS)


def _is_test_path(path: str) -> bool:
    return path == TEST_ROOT or path.startswith(f"{TEST_ROOT}/")


def _is_global_path(path: str, manifest: Mapping[str, Any]) -> bool:
    return _patterns_match(path, _as_strings(manifest.get("full_pr_if")))


def _content_risk(changed: Sequence[ChangedPath], diff_text: str | None = None) -> str:
    """Raise risk for changed code with global registration/lifecycle tokens."""
    risk = "low"
    if diff_text:
        added_lines = "\n".join(
            line[1:]
            for line in diff_text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        if any(token in added_lines for token in _HIGH_RISK_TOKENS):
            risk = "high"
    # A package entry point or a deleted implementation cannot be classified by
    # the added-line scan alone.
    for change in changed:
        for path_value in change.paths:
            path = _normalize_path(path_value)
            if path.endswith("/__init__.py") or path in {
                "embodichain/__init__.py",
                "embodichain_tasks/embodichain_tasks/__init__.py",
            }:
                risk = "high"
    return risk


def _max_risk(*values: str) -> str:
    normalized = [value if value in _RISK_ORDER else "high" for value in values]
    return max(normalized, key=lambda value: _RISK_ORDER[value])


def _test_hints(root: Path, selector: str) -> set[str]:
    """Infer resource lanes conservatively from a test file or directory."""
    hints: set[str] = set()
    normalized = _normalize_path(selector)
    paths: list[Path] = []
    candidate = root / normalized
    if candidate.is_file():
        paths = [candidate]
    elif candidate.is_dir():
        paths = [path for path in candidate.rglob("*.py") if _is_test_file(path)]
    else:
        for path in (root / TEST_ROOT).rglob("*.py"):
            if not _is_test_file(path):
                continue
            relative = path.relative_to(root).as_posix()
            if fnmatch.fnmatchcase(relative, normalized) or path_matches(
                relative, normalized
            ):
                paths.append(path)
    base_hints: set[str] = set()
    if (
        normalized in {"tests/sim", "tests/gym/envs", "tests/lab/task_program"}
        or normalized.startswith(
            ("tests/sim/", "tests/gym/envs/", "tests/lab/task_program/")
        )
        or "subprocess_sim" in normalized
    ):
        base_hints.add("sim")
    if "test_rl_distributed.py" in normalized:
        base_hints.add("distributed")
    if any(token in normalized.lower() for token in ("cuda", "gpu")):
        base_hints.add("gpu")
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        lowered = text.lower()
        file_hints = set(base_hints)
        explicitly_no_sim = "no_sim" in lowered
        if "pytest.mark.gpu" in lowered or "pytestmark = pytest.mark.gpu" in lowered:
            file_hints.add("gpu")
        if (
            "requires_sim" in lowered
            or "simulationmanager" in lowered
            or "subprocess_sim" in lowered
        ):
            file_hints.add("sim")
        if "cuda" in path.name.lower() or "gpu" in path.name.lower():
            file_hints.add("gpu")
        if "test_rl_distributed.py" == path.name:
            file_hints.add("distributed")
        if explicitly_no_sim:
            file_hints.discard("sim")
        hints.update(file_hints)
    return hints


def _is_slow_marker(node: ast.AST) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    if not isinstance(node, ast.Attribute) or node.attr != "slow":
        return False
    return (isinstance(node.value, ast.Attribute) and node.value.attr == "mark") or (
        isinstance(node.value, ast.Name) and node.value.id == "mark"
    )


def _selector_has_slow_marker(root: Path, selector: str) -> bool:
    """Detect actual slow decorators/pytestmark assignments, ignoring strings."""
    normalized = _normalize_path(selector)
    candidate = root / normalized
    if candidate.is_file():
        paths = [candidate]
    elif candidate.is_dir():
        paths = [path for path in candidate.rglob("*.py") if _is_test_file(path)]
    else:
        paths = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            return True
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if any(_is_slow_marker(decorator) for decorator in node.decorator_list):
                    return True
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if any(
                    isinstance(target, ast.Name) and target.id == "pytestmark"
                    for target in targets
                ):
                    if node.value is not None and any(
                        _is_slow_marker(value) for value in ast.walk(node.value)
                    ):
                        return True
    return False


def _lane_selectors(
    root: Path,
    selectors: Sequence[str],
    selector_hints: Mapping[str, Iterable[str]] | None,
    mode: str,
) -> dict[str, list[str]]:
    if mode in {"full", "full-pr"}:
        return {
            "docs": ["tests/docs"],
            "fast": ["tests"],
            "sim": ["tests"],
            "distributed": ["tests/learning/test_rl_distributed.py"],
            "gpu": ["tests"],
        }
    lanes: dict[str, set[str]] = {lane: set() for lane in _LANES}
    for selector in selectors:
        if _patterns_match(selector, ("tests/docs/**", "tests/docs")):
            lanes["docs"].add(selector)
            continue
        lanes["fast"].add(selector)
        hints = _test_hints(root, selector)
        if selector_hints is not None:
            hints.update(selector_hints.get(selector, ()))
        for lane in ("sim", "distributed", "gpu"):
            if lane in hints:
                lanes[lane].add(selector)
    return {lane: sorted(values) for lane, values in lanes.items() if values}


def _install_requirements(
    selectors: Sequence[str],
    changed: Sequence[ChangedPath],
    mode: str,
    resource_hints: set[str],
) -> dict[str, bool]:
    all_paths = [path for change in changed for path in change.paths]
    all_paths.extend(selectors)
    if mode in {"full", "full-pr"}:
        return {"gensim": True, "curobo": True}
    gensim = any(
        path.startswith("embodichain/gen_sim/") or path.startswith("tests/gen_sim/")
        for path in all_paths
    )
    curobo = "curobo" in " ".join(all_paths).lower()
    if "gensim" in resource_hints:
        gensim = True
    if "curobo" in resource_hints:
        curobo = True
    return {"gensim": gensim, "curobo": curobo}


def _full_plan(
    root: Path,
    *,
    mode: str,
    base_sha: str | None,
    head_sha: str | None,
    changed: Sequence[ChangedPath],
    reason: str,
    risk: str = "high",
    topics: Sequence[str] = (),
    graph_warnings: Sequence[str] = (),
) -> TestPlan:
    return TestPlan(
        mode=mode,
        risk=risk,
        base_sha=base_sha,
        head_sha=head_sha,
        changed=[change.__dict__ for change in changed],
        selectors=["tests"],
        lanes=_lane_selectors(root, ["tests"], None, mode),
        reasons={"*": [reason]},
        resource_hints=["sim", "gpu"],
        install={"gensim": True, "curobo": True},
        topics=list(topics),
        fallback_reason=reason,
        graph_warnings=list(graph_warnings),
    )


def build_plan(
    root: str | Path,
    changed: Sequence[ChangedPath],
    manifest: Mapping[str, Any],
    *,
    base_sha: str | None = None,
    head_sha: str | None = None,
    map_data: Mapping[str, Any] | None = None,
    force_full: bool = False,
    diff_text: str | None = None,
) -> TestPlan:
    """Build a test plan from changed paths and repository metadata.

    Args:
        root: Repository root.
        changed: Rename-aware changed paths.
        manifest: Parsed impact manifest.
        base_sha: Optional base revision for audit output.
        head_sha: Optional head revision for audit output.
        map_data: Optional parsed ``agent_context/MAP.yaml`` data.
        force_full: Select all tests, including slow tests.

    Returns:
        A serializable :class:`TestPlan`.
    """
    repository_root = Path(root)
    changed = [
        ChangedPath(
            _normalize_path(change.path),
            status=change.status,
            old_path=_normalize_path(change.old_path) if change.old_path else None,
        )
        for change in changed
    ]
    if force_full:
        return _full_plan(
            repository_root,
            mode="full",
            base_sha=base_sha,
            head_sha=head_sha,
            changed=changed,
            reason="forced full run",
        )

    if not changed:
        return _full_plan(
            repository_root,
            mode="full-pr",
            base_sha=base_sha,
            head_sha=head_sha,
            changed=changed,
            reason="empty diff cannot establish impact safely",
        )

    map_data = map_data or _load_map(repository_root / DEFAULT_MAP)
    direct_topic_ids = _map_topics(map_data, changed)
    topic_ids = list(direct_topic_ids)
    slow_test_changed = any(
        _selector_has_slow_marker(repository_root, path)
        for change in changed
        for path in change.paths
        if _is_test_path(path)
    )

    for change in changed:
        if any(_is_global_path(path, manifest) for path in change.paths):
            return _full_plan(
                repository_root,
                mode="full" if slow_test_changed else "full-pr",
                base_sha=base_sha,
                head_sha=head_sha,
                changed=changed,
                reason=f"global-risk path: {change.path}",
                topics=topic_ids,
            )
        if change.status in {"D"} and not _is_test_path(change.path):
            return _full_plan(
                repository_root,
                mode="full" if slow_test_changed else "full-pr",
                base_sha=base_sha,
                head_sha=head_sha,
                changed=changed,
                reason=f"deleted source path: {change.path}",
                topics=topic_ids,
            )
        if change.status in {"D"} and _is_test_path(change.path):
            return _full_plan(
                repository_root,
                mode="full" if slow_test_changed else "full-pr",
                base_sha=base_sha,
                head_sha=head_sha,
                changed=changed,
                reason=f"deleted test path: {change.path}",
                topics=topic_ids,
            )

    docs_changed = any(
        _is_doc_path(path) for change in changed for path in change.paths
    )
    source_paths = {
        _normalize_path(path)
        for change in changed
        for path in change.paths
        if not _is_doc_path(path) and not _is_test_path(path)
    }
    source_changed = bool(source_paths)
    test_changed = any(
        _is_test_path(path) and not _is_doc_path(path)
        for change in changed
        for path in change.paths
    )
    if slow_test_changed:
        return _full_plan(
            repository_root,
            mode="full",
            base_sha=base_sha,
            head_sha=head_sha,
            changed=changed,
            reason="changed slow test requires a full run",
            risk="high",
            topics=topic_ids,
        )
    if docs_changed and not source_changed and not test_changed:
        docs_selectors = ["tests/docs"]
        return TestPlan(
            mode="docs-only",
            risk="low",
            base_sha=base_sha,
            head_sha=head_sha,
            changed=[change.__dict__ for change in changed],
            selectors=docs_selectors,
            lanes={"docs": docs_selectors},
            reasons={"tests/docs": ["documentation-only change"]},
            resource_hints=[],
            install={"gensim": False, "curobo": False},
            topics=topic_ids,
        )

    selectors: set[str] = set()
    reasons: dict[str, set[str]] = {}
    selector_hints: dict[str, set[str]] = {}
    resource_hints: set[str] = set()
    risk = _content_risk(changed, diff_text)
    rule_covered_paths: set[str] = set()
    reverse_paths: set[str] = set()

    for selector in _as_strings(manifest.get("always")):
        _add_selector(
            selectors,
            reasons,
            repository_root,
            selector,
            "always",
            selector_hints=selector_hints,
        )

    for change in changed:
        for path in change.paths:
            if _is_test_path(path):
                _add_selector(
                    selectors,
                    reasons,
                    repository_root,
                    path,
                    "changed-test",
                    selector_hints=selector_hints,
                )

    for index, rule in enumerate(manifest.get("rules", [])):
        if not isinstance(rule, Mapping) or not _rule_matches(rule, changed):
            continue
        matching_source_paths = {
            path
            for path in source_paths
            if _patterns_match(path, _as_strings(rule.get("paths")))
        }
        rule_covered_paths.update(matching_source_paths)
        rule_id = str(rule.get("id", f"rule-{index}"))
        risk = _max_risk(risk, str(rule.get("risk", "medium")))
        rule_hints = _as_strings(rule.get("resource_hints"))
        resource_hints.update(rule_hints)
        if rule.get("include_reverse", False):
            reverse_paths.update(matching_source_paths)
        if str(rule.get("mode", "partial")) in {"full", "full-pr"}:
            return _full_plan(
                repository_root,
                mode=str(rule.get("mode")),
                base_sha=base_sha,
                head_sha=head_sha,
                changed=changed,
                reason=f"rule:{rule_id}",
                topics=topic_ids,
            )
        rule_has_tests = False
        for selector in (
            *_as_strings(rule.get("tests")),
            *_as_strings(rule.get("contracts")),
        ):
            rule_has_tests |= _add_selector(
                selectors,
                reasons,
                repository_root,
                selector,
                f"rule:{rule_id}",
                selector_hints=selector_hints,
                resource_hints=rule_hints,
            )
        if matching_source_paths and not rule_has_tests:
            return _full_plan(
                repository_root,
                mode="full-pr",
                base_sha=base_sha,
                head_sha=head_sha,
                changed=changed,
                reason=f"rule:{rule_id} has no runnable test selector",
                topics=topic_ids,
            )

    fallback_paths = source_paths - rule_covered_paths
    fallback_topic_ids: list[str] = []
    for path in sorted(fallback_paths):
        path_topics = _map_topics(map_data, [ChangedPath(path)])
        if not path_topics:
            return _full_plan(
                repository_root,
                mode="full-pr",
                base_sha=base_sha,
                head_sha=head_sha,
                changed=changed,
                reason=f"source path has no impact rule or topic: {path}",
                topics=topic_ids,
            )
        path_has_mapping = False
        for topic_id in path_topics:
            topic_cfg = manifest.get("topics", {}).get(topic_id)
            if not isinstance(topic_cfg, Mapping):
                continue
            configured = (
                *_as_strings(topic_cfg.get("tests")),
                *_as_strings(topic_cfg.get("contracts")),
            )
            if any(
                _expand_selector(repository_root, selector) for selector in configured
            ):
                path_has_mapping = True
                break
        if not path_has_mapping:
            return _full_plan(
                repository_root,
                mode="full-pr",
                base_sha=base_sha,
                head_sha=head_sha,
                changed=changed,
                reason=f"source topic has no runnable test mapping: {path}",
                topics=topic_ids,
            )
        fallback_topic_ids.extend(path_topics)
    fallback_topic_ids = list(dict.fromkeys(fallback_topic_ids))
    reverse_paths.update(fallback_paths)
    related_seeds = fallback_topic_ids if source_changed else direct_topic_ids
    related_ids = _topic_related(map_data, related_seeds)
    topic_ids = list(dict.fromkeys((*direct_topic_ids, *related_ids)))

    for topic_id in topic_ids:
        topic_cfg = manifest.get("topics", {}).get(topic_id, {})
        if not isinstance(topic_cfg, Mapping):
            continue
        topic_contracts = _as_strings(topic_cfg.get("contracts"))
        topic_hints = _as_strings(topic_cfg.get("resource_hints"))
        resource_hints.update(topic_hints)
        for selector in topic_contracts:
            _add_selector(
                selectors,
                reasons,
                repository_root,
                selector,
                f"topic:{topic_id}:contract",
                selector_hints=selector_hints,
                resource_hints=topic_hints,
            )
        if topic_id in fallback_topic_ids:
            for selector in _as_strings(topic_cfg.get("tests")):
                _add_selector(
                    selectors,
                    reasons,
                    repository_root,
                    selector,
                    f"topic:{topic_id}:fallback",
                    selector_hints=selector_hints,
                    resource_hints=topic_hints,
                )

    graph_warnings: list[str] = []
    try:
        graph_depth = int(manifest.get("reverse_import_depth", 4))
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "impact manifest 'reverse_import_depth' must be an integer"
        ) from error
    # Explicit rules are curated boundaries.  Their tests can opt into the
    # bounded graph when a leaf has important consumers; otherwise skipping a
    # high-fanout transitive closure keeps partial plans genuinely partial.
    if reverse_paths:
        graph = ImportGraph.build(repository_root)
        graph_warnings = graph.warnings
        impacted_paths = graph.impacted_test_paths(
            [ChangedPath(path) for path in sorted(reverse_paths)],
            max_depth=graph_depth,
        )
        if any(
            path == "conftest.py" or path.endswith("/conftest.py")
            for path in impacted_paths
        ):
            return _full_plan(
                repository_root,
                mode="full-pr",
                base_sha=base_sha,
                head_sha=head_sha,
                changed=changed,
                reason="reverse-import reaches shared test fixture",
                topics=topic_ids,
                graph_warnings=graph_warnings,
            )
        for selector in sorted(impacted_paths):
            _add_selector(
                selectors, reasons, repository_root, selector, "reverse-import"
            )

    if not selectors:
        return _full_plan(
            repository_root,
            mode="full-pr",
            base_sha=base_sha,
            head_sha=head_sha,
            changed=changed,
            reason="impact analysis produced no test selector",
            topics=topic_ids,
            graph_warnings=graph_warnings,
        )

    if graph_warnings and source_changed:
        # A parse failure makes the dependency evidence incomplete.  Keep the
        # explicit selectors in the audit output, but require the conservative
        # PR-wide non-slow run instead of treating the partial graph as proof.
        risk = _max_risk(risk, "high")

    mode = "partial"
    if risk == "high":
        mode = "full-pr"
    lanes = _lane_selectors(repository_root, sorted(selectors), selector_hints, mode)
    if docs_changed:
        _add_selector(
            selectors,
            reasons,
            repository_root,
            "tests/docs",
            "documentation-change",
            selector_hints=selector_hints,
        )
        lanes = _lane_selectors(
            repository_root, sorted(selectors), selector_hints, mode
        )
    install = _install_requirements(
        sorted(selectors),
        changed,
        mode,
        resource_hints,
    )
    return TestPlan(
        mode=mode,
        risk=risk,
        base_sha=base_sha,
        head_sha=head_sha,
        changed=[change.__dict__ for change in changed],
        selectors=sorted(selectors),
        lanes=lanes,
        reasons={key: sorted(value) for key, value in sorted(reasons.items())},
        resource_hints=sorted(resource_hints),
        install=install,
        topics=topic_ids,
        graph_warnings=graph_warnings,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base", help="base revision for a PR diff")
    parser.add_argument("--head", default="HEAD", help="head revision")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--map", dest="map_path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("test-plan.json"))
    parser.add_argument("--full", action="store_true", help="force a full plan")
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="explicit changed path (repeatable; useful outside git)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by the CI planning job."""
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    manifest_path = args.manifest or root / DEFAULT_MANIFEST
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    try:
        manifest = load_manifest(manifest_path)
        if args.changed_file:
            changed = [
                ChangedPath(_normalize_path(path), status="M")
                for path in args.changed_file
            ]
        elif args.base:
            changed = collect_changed_paths(root, args.base, args.head)
        else:
            changed = []
        map_path = args.map_path or root / DEFAULT_MAP
        if not map_path.is_absolute():
            map_path = root / map_path
        map_data = _load_map(map_path.resolve())
        diff_text = None
        if args.base:
            try:
                diff_text = _git_output(
                    root,
                    ["diff", "--unified=0", args.base, args.head, "--"],
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    "cannot inspect diff content for risk classification"
                ) from error
        plan = build_plan(
            root,
            changed,
            manifest,
            base_sha=args.base,
            head_sha=args.head,
            map_data=map_data,
            force_full=args.full,
            diff_text=diff_text,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"impact analysis failed: {error}", file=sys.stderr)
        return 2
    output = args.output
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "mode": plan.mode,
                "risk": plan.risk,
                "selectors": len(plan.selectors),
                "lanes": {lane: len(values) for lane, values in plan.lanes.items()},
                "topics": plan.topics,
                "fallback_reason": plan.fallback_reason,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
