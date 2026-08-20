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

"""Check Sphinx API-reference coverage without modifying repository files.

Every non-private module's static ``__all__`` declaration is treated as its
public API contract. The checker compares those exports with explicit Sphinx
autodoc and autosummary directives and reports any missing import paths.

Usage:
    python docs/scripts/check_api_docs.py
    python docs/scripts/check_api_docs.py --format json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = [
    "ApiDocsError",
    "CheckResult",
    "MissingExport",
    "PackageRoot",
    "PublicModule",
    "check_api_docs",
    "collect_documented_exports",
    "discover_public_modules",
    "find_missing_exports",
    "format_json_report",
    "format_text_report",
]


REPO_ROOT = Path(__file__).resolve().parents[2]
API_REFERENCE_ROOT = REPO_ROOT / "docs" / "source" / "api_reference"

_DIRECTIVE_RE = re.compile(
    r"^(?P<indent>[ \t]*)\.\.\s+(?P<name>[\w-]+)::\s*(?P<argument>.*?)\s*$"
)
_OPTION_RE = re.compile(r"^:(?P<name>[\w-]+):\s*(?P<value>.*)$")
_OBJECT_DIRECTIVES = frozenset(
    {
        "autoattribute",
        "autoclass",
        "autodata",
        "autoexception",
        "autofunction",
        "automethod",
    }
)


class ApiDocsError(ValueError):
    """Raised when the API contract cannot be checked statically."""


@dataclass(frozen=True)
class PackageRoot:
    """Map a Python import package to its source directory."""

    module: str
    path: Path


@dataclass(frozen=True)
class PublicModule:
    """Public exports declared by one Python module."""

    name: str
    exports: tuple[str, ...]
    source: Path


@dataclass(frozen=True)
class MissingExport:
    """One public import path missing from the API reference."""

    module: str
    name: str
    source: Path

    @property
    def qualified_name(self) -> str:
        """Return the complete public import path."""
        return f"{self.module}.{self.name}"


@dataclass(frozen=True)
class CheckResult:
    """Coverage result returned by the read-only checker."""

    total_exports: int
    missing: tuple[MissingExport, ...]

    @property
    def documented_exports(self) -> int:
        """Return the number of public exports covered by API docs."""
        return self.total_exports - len(self.missing)

    @property
    def is_aligned(self) -> bool:
        """Return whether every declared export is documented."""
        return not self.missing


DEFAULT_PACKAGE_ROOTS = (
    PackageRoot("embodichain", REPO_ROOT / "embodichain"),
    PackageRoot(
        "embodichain_tasks",
        REPO_ROOT / "embodichain_tasks" / "embodichain_tasks",
    ),
    PackageRoot(
        "embodichain_tasks.configs",
        REPO_ROOT / "embodichain_tasks" / "configs",
    ),
)


def _static_all(tree: ast.Module, source: Path) -> tuple[str, ...] | None:
    value: ast.expr | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value = node.value

    if value is None:
        return None

    try:
        exports = ast.literal_eval(value)
    except (ValueError, TypeError) as exc:
        raise ApiDocsError(
            f"{source}: __all__ must be a static list of strings"
        ) from exc

    if not isinstance(exports, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in exports
    ):
        raise ApiDocsError(
            f"{source}: __all__ must contain only valid Python identifiers"
        )
    if len(exports) != len(set(exports)):
        raise ApiDocsError(f"{source}: __all__ contains duplicate names")
    return tuple(exports)


def discover_public_modules(
    package_roots: Sequence[PackageRoot] = DEFAULT_PACKAGE_ROOTS,
) -> tuple[PublicModule, ...]:
    """Discover module APIs declared through static ``__all__`` values.

    Args:
        package_roots: Import-package to source-directory mappings to inspect.

    Returns:
        Public modules sorted by import path.

    Raises:
        ApiDocsError: If a package root is missing or an ``__all__`` declaration
            cannot be evaluated statically.
    """
    modules: list[PublicModule] = []
    seen_modules: set[str] = set()

    for package_root in package_roots:
        if not package_root.path.is_dir():
            raise ApiDocsError(f"Package root does not exist: {package_root.path}")

        for source in sorted(package_root.path.rglob("*.py")):
            relative_parts = list(
                source.relative_to(package_root.path).with_suffix("").parts
            )
            if relative_parts[-1] == "__init__":
                relative_parts.pop()
            if any(part.startswith("_") for part in relative_parts):
                continue

            tree = ast.parse(
                source.read_text(encoding="utf-8-sig"), filename=str(source)
            )
            exports = _static_all(tree, source)
            if not exports:
                continue

            module = ".".join((package_root.module, *relative_parts))
            if module in seen_modules:
                raise ApiDocsError(f"Duplicate package mapping for {module}")
            seen_modules.add(module)
            modules.append(PublicModule(module, exports, source))

    return tuple(sorted(modules, key=lambda item: item.name))


def _directive_end(lines: list[str], start: int, indent: int) -> int:
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        index += 1
    return index


def _directive_options(lines: list[str], start: int, end: int) -> dict[str, str]:
    options: dict[str, str] = {}
    active_option: str | None = None
    active_indent = 0

    for line in lines[start + 1 : end]:
        stripped = line.strip()
        option_match = _OPTION_RE.match(stripped)
        if option_match:
            active_option = option_match.group("name")
            active_indent = len(line) - len(line.lstrip())
            options[active_option] = option_match.group("value")
        elif (
            active_option
            and stripped
            and len(line) - len(line.lstrip()) > active_indent
        ):
            options[active_option] = f"{options[active_option]} {stripped}".strip()
        elif stripped:
            active_option = None
    return options


def _normalize_target(target: str) -> str:
    target = target.strip().lstrip("~")
    link_match = re.fullmatch(r".*<([^>]+)>", target)
    if link_match:
        target = link_match.group(1).strip()
    return re.sub(r"\(.*\)$", "", target).strip()


def _qualify_target(
    target: str, context: str | None, public_exports: set[str]
) -> str | None:
    normalized = _normalize_target(target)
    if normalized in public_exports:
        return normalized
    if context:
        qualified = f"{context}.{normalized}"
        if qualified in public_exports:
            return qualified
    return None


def _autosummary_entries(lines: list[str], start: int, end: int) -> tuple[str, ...]:
    entries: list[str] = []
    for line in lines[start + 1 : end]:
        stripped = line.strip()
        if not stripped or stripped.startswith((":", "..")):
            continue
        entries.append(stripped)
    return tuple(entries)


def collect_documented_exports(
    api_reference_root: Path,
    public_modules: Sequence[PublicModule],
) -> set[str]:
    """Collect public exports covered by Sphinx API-reference directives.

    Args:
        api_reference_root: Root containing API-reference RST files.
        public_modules: Public modules to check.

    Returns:
        Fully qualified public export paths covered by API documentation.
    """
    exports_by_module = {
        module.name: {f"{module.name}.{name}" for name in module.exports}
        for module in public_modules
    }
    public_exports = set().union(*exports_by_module.values())
    documented: set[str] = set()

    for rst_path in sorted(api_reference_root.rglob("*.rst")):
        relative_path = rst_path.relative_to(api_reference_root)
        if "_autosummary" in relative_path.parts:
            continue
        lines = rst_path.read_text(encoding="utf-8-sig").splitlines()
        directives: list[tuple[int, int, str, str, int]] = []
        for index, line in enumerate(lines):
            match = _DIRECTIVE_RE.match(line)
            if not match:
                continue
            indent = len(match.group("indent"))
            directives.append(
                (
                    index,
                    indent,
                    match.group("name"),
                    match.group("argument"),
                    _directive_end(lines, index, indent),
                )
            )

        primary_module = next(
            (
                argument
                for _, _, name, argument, _ in directives
                if name == "automodule" and argument
            ),
            None,
        )
        current_module: str | None = None

        for index, indent, name, argument, end in directives:
            if name == "currentmodule":
                current_module = argument
                continue

            enclosing_module = next(
                (
                    parent_argument
                    for parent_index, parent_indent, parent_name, parent_argument, parent_end in reversed(
                        directives
                    )
                    if parent_name == "automodule"
                    and parent_index < index < parent_end
                    and parent_indent < indent
                ),
                None,
            )
            context = enclosing_module or current_module or primary_module

            if name == "automodule" and argument in exports_by_module:
                options = _directive_options(lines, index, end)
                if "members" in options:
                    member_option = options["members"]
                    if member_option:
                        member_names = {
                            item.strip()
                            for item in member_option.split(",")
                            if item.strip()
                        }
                        covered_members = {
                            f"{argument}.{member}"
                            for member in member_names
                            if f"{argument}.{member}" in public_exports
                        }
                    else:
                        covered_members = exports_by_module[argument]

                    excluded = {
                        f"{argument}.{item.strip()}"
                        for item in options.get("exclude-members", "").split(",")
                        if item.strip()
                    }
                    documented.update(covered_members - excluded)

            if name in _OBJECT_DIRECTIVES:
                qualified = _qualify_target(argument, context, public_exports)
                if qualified:
                    documented.add(qualified)
            elif name == "autosummary":
                for entry in _autosummary_entries(lines, index, end):
                    qualified = _qualify_target(entry, context, public_exports)
                    if qualified:
                        documented.add(qualified)

    return documented


def find_missing_exports(
    public_modules: Sequence[PublicModule], documented: set[str]
) -> tuple[MissingExport, ...]:
    """Return declared public exports absent from API documentation."""
    return tuple(
        MissingExport(module.name, name, module.source)
        for module in public_modules
        for name in module.exports
        if f"{module.name}.{name}" not in documented
    )


def check_api_docs(
    *,
    package_roots: Sequence[PackageRoot] = DEFAULT_PACKAGE_ROOTS,
    api_reference_root: Path = API_REFERENCE_ROOT,
) -> CheckResult:
    """Check API-reference coverage without writing any files.

    Args:
        package_roots: Import-package to source-directory mappings to inspect.
        api_reference_root: Root containing API-reference RST files.

    Returns:
        Coverage counts and missing public import paths.
    """
    public_modules = discover_public_modules(package_roots)
    documented = collect_documented_exports(api_reference_root, public_modules)
    return CheckResult(
        total_exports=sum(len(module.exports) for module in public_modules),
        missing=find_missing_exports(public_modules, documented),
    )


def _source_label(source: Path) -> str:
    try:
        return source.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(source)


def format_text_report(result: CheckResult) -> str:
    """Format a human-readable coverage report."""
    if result.is_aligned:
        return (
            "API docs are aligned: "
            f"{result.documented_exports}/{result.total_exports} exports documented."
        )

    lines = [
        "API docs are missing "
        f"{len(result.missing)} of {result.total_exports} public exports:"
    ]
    lines.extend(
        f"- {item.qualified_name} ({_source_label(item.source)})"
        for item in result.missing
    )
    lines.extend(
        [
            "",
            "Use $update-api-docs to generate or update the corresponding API pages.",
        ]
    )
    return "\n".join(lines)


def format_json_report(result: CheckResult) -> str:
    """Format a machine-readable coverage report for agent workflows."""
    payload = {
        "documented_exports": result.documented_exports,
        "missing": [
            {
                "module": item.module,
                "name": item.name,
                "qualified_name": item.qualified_name,
                "source": _source_label(item.source),
            }
            for item in result.missing
        ],
        "missing_count": len(result.missing),
        "total_exports": result.total_exports,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Sphinx coverage of module-level __all__ exports."
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Report format (default: text).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only API documentation checker."""
    args = _parse_args(argv)
    try:
        result = check_api_docs()
    except (ApiDocsError, OSError, SyntaxError) as exc:
        print(f"API docs check failed: {exc}", file=sys.stderr)
        return 1

    report = (
        format_json_report(result)
        if args.format == "json"
        else format_text_report(result)
    )
    output = sys.stdout if args.format == "json" or result.is_aligned else sys.stderr
    print(report, file=output)
    return 0 if result.is_aligned else 1


if __name__ == "__main__":
    raise SystemExit(main())
