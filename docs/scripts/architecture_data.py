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

"""Validate architecture graphs and evidence against an immutable Git revision."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import yaml
from jsonschema import Draft202012Validator

__all__ = [
    "ArchitectureDataError",
    "SourceRepository",
    "load_snapshot",
    "validate_snapshot",
]

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "docs/architecture/architecture.schema.json"


class ArchitectureDataError(ValueError):
    """An architecture graph or its pinned evidence cannot be verified."""


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load a JSON object with actionable file errors.

    Args:
        path: Snapshot or curated seed path.

    Returns:
        Parsed JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ArchitectureDataError(f"{path}: {error}") from error
    if not isinstance(data, dict):
        raise ArchitectureDataError(f"{path}: expected a JSON object")
    return data


class SourceRepository:
    """Cached Git source access; never imports or executes repository modules."""

    def __init__(self, root: Path, revision: str) -> None:
        self.root = root
        try:
            self.revision = self.git(
                "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
            ).strip()
        except ArchitectureDataError as error:
            raise ArchitectureDataError(
                f"Revision {revision!r} unavailable; fetch that commit before "
                f"generating or validating. Underlying error: {error}"
            ) from error
        self.files = set(
            self.git("ls-tree", "-rz", "--name-only", self.revision).split("\0")
        ) - {""}
        self._texts: dict[str, str] = {}
        self._trees: dict[str, ast.Module] = {}
        self._symbols: dict[str, dict[str, list[tuple[int, int]]]] = {}

    def git(self, *arguments: str) -> str:
        """Run a Git read command in this repository.

        Args:
            *arguments: Git command arguments.

        Returns:
            Standard output decoded as UTF-8.
        """
        try:
            return subprocess.run(
                ["git", "-C", str(self.root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            raise ArchitectureDataError(
                f"Git source read failed: {detail.strip()}"
            ) from error

    def read(self, path: str) -> str:
        """Read a tracked file at the pinned commit.

        Args:
            path: Repository-relative tracked file path.

        Returns:
            Committed file text, independent of working-tree changes.
        """
        if path not in self.files:
            raise ArchitectureDataError(
                f"Missing source path {path!r} at {self.revision}"
            )
        if path not in self._texts:
            self._texts[path] = self.git("show", f"{self.revision}:{path}")
        return self._texts[path]

    def tree(self, path: str) -> ast.Module:
        """Parse Python source statically.

        Args:
            path: Python source path.

        Returns:
            Cached syntax tree.
        """
        if path not in self._trees:
            try:
                self._trees[path] = ast.parse(self.read(path), filename=path)
            except SyntaxError as error:
                raise ArchitectureDataError(
                    f"{path}: cannot parse Python: {error}"
                ) from error
        return self._trees[path]

    def symbols(self, path: str) -> dict[str, list[tuple[int, int]]]:
        """Index qualified class/function names without collapsing duplicate definitions.

        Args:
            path: Python source path.

        Returns:
            Qualified names mapped to lexical line ranges.
        """
        if path not in self._symbols:
            result: dict[str, list[tuple[int, int]]] = {}

            def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
                if isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    parents = (*parents, node.name)
                    start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                    result.setdefault(".".join(parents), []).append(
                        (start, node.end_lineno or node.lineno)
                    )
                for child in ast.iter_child_nodes(node):
                    visit(child, parents)

            visit(self.tree(path))
            self._symbols[path] = result
        return self._symbols[path]

    def scope(self, proof: dict[str, Any]) -> tuple[int, int]:
        """Resolve an evidence symbol's unique lexical scope.

        Args:
            proof: Source evidence including path and symbol.

        Returns:
            Inclusive line range of its lexical scope.
        """
        path, symbol = proof["path"], proof["symbol"]
        size = len(self.read(path).splitlines())
        if symbol == "<document>" and not path.endswith(".py"):
            return 1, size
        if not path.endswith(".py"):
            raise ArchitectureDataError(
                f"{path}: non-Python evidence requires symbol <document>"
            )
        if symbol == "<module>":
            self.tree(path)
            return 1, size
        matches = self.symbols(path).get(symbol, [])
        if len(matches) != 1:
            raise ArchitectureDataError(
                f"{path}: symbol {symbol!r} is missing or ambiguous"
            )
        return matches[0]

    def in_scope(self, proof: dict[str, Any], start: int, end: int) -> bool:
        """Check lexical containment, excluding nested definitions for module evidence.

        Args:
            proof: Evidence descriptor.
            start: Candidate first line.
            end: Candidate last line.

        Returns:
            Whether the entire evidence range belongs to the declared scope.
        """
        first, last = self.scope(proof)
        if not first <= start <= end <= last:
            return False
        if proof["symbol"] == "<module>":
            return not any(
                start <= b and a <= end
                for spans in self.symbols(proof["path"]).values()
                for a, b in spans
            )
        return True


def _index(items: list[dict[str, Any]], context: str) -> dict[str, dict[str, Any]]:
    result = {}
    for item in items:
        if item["id"] in result:
            raise ArchitectureDataError(f"{context}: duplicate ID {item['id']}")
        result[item["id"]] = item
    return result


def validate_snapshot(
    data: dict[str, Any], *, repo_root: Path, schema_path: Path = SCHEMA
) -> None:
    """Validate structure, references, lexical evidence, and pinned documentation.

    Args:
        data: Static architecture snapshot.
        repo_root: Git repository containing its revision.
        schema_path: JSON Schema Draft 2020-12 contract.
    """
    schema = load_snapshot(schema_path)
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(data), key=lambda e: str(list(e.absolute_path))
    )
    if errors:
        error = errors[0]
        field = ".".join(map(str, error.absolute_path))
        raise ArchitectureDataError(f"{field}: {error.message}")
    source = SourceRepository(repo_root, data["revision"])
    if source.read("VERSION").strip() != data["package_version"]:
        raise ArchitectureDataError("package_version does not match pinned VERSION")
    try:
        topic_map = yaml.safe_load(source.read(data["topic_index"]))
        topics = {item["id"] for item in topic_map["topics"]}
    except (yaml.YAMLError, KeyError, TypeError) as error:
        raise ArchitectureDataError(f"Invalid topic index: {error}") from error
    nodes, edges, views = (
        _index(data[key], key) for key in ("nodes", "edges", "views")
    )
    for item in [*nodes.values(), *edges.values()]:
        for proof in item["evidence"]:
            try:
                lines = source.read(proof["path"]).splitlines()
                start, end = proof["start_line"], proof["end_line"]
                if not 1 <= start <= end <= len(lines):
                    raise ArchitectureDataError("invalid evidence line range")
                if "\n".join(lines[start - 1 : end]) != proof["excerpt"]:
                    raise ArchitectureDataError("stale evidence excerpt")
                if not source.in_scope(proof, start, end):
                    raise ArchitectureDataError(
                        "evidence outside declared symbol scope"
                    )
            except ArchitectureDataError as error:
                raise ArchitectureDataError(
                    f"{item['id']} ({proof['path']}:{proof['start_line']}): {error}"
                ) from error
    for node in nodes.values():
        missing = set(node["topic_ids"]) - topics
        if missing:
            raise ArchitectureDataError(
                f"{node['id']}: unknown topic {sorted(missing)}"
            )
        for doc in node["documentation"]:
            matches = [
                ext
                for ext in (".md", ".rst")
                if f"docs/source/{doc['docname']}{ext}" in source.files
            ]
            if len(matches) != 1:
                raise ArchitectureDataError(
                    f"{node['id']}: docname {doc['docname']} is missing or ambiguous"
                )
            if doc.get("source_extension", matches[0]) != matches[0]:
                raise ArchitectureDataError(
                    f"{node['id']}: incorrect documentation source extension"
                )
    for edge in edges.values():
        for endpoint in ("source", "target"):
            if edge[endpoint] not in nodes:
                raise ArchitectureDataError(
                    f"{edge['id']}: unknown {endpoint} {edge[endpoint]}"
                )
    for view in views.values():
        selected = set(view["node_ids"])
        if not selected <= nodes.keys() or not set(view["edge_ids"]) <= edges.keys():
            raise ArchitectureDataError(f"{view['id']}: unknown view member")
        groups = _index(view["groups"], f"{view['id']} groups")
        members = [id for group in groups.values() for id in group["node_ids"]]
        if len(members) != len(set(members)) or set(members) != selected:
            raise ArchitectureDataError(
                f"{view['id']}: group members must partition the view exactly"
            )
        for id in view["edge_ids"]:
            if not {edges[id]["source"], edges[id]["target"]} <= selected:
                raise ArchitectureDataError(
                    f"{view['id']}: edge {id} has excluded endpoint"
                )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        data = load_snapshot(args.snapshot)
        validate_snapshot(data, repo_root=args.repo_root)
    except ArchitectureDataError as error:
        parser.exit(1, f"Architecture validation failed: {error}\n")
    print(
        f"Valid: {len(data['nodes'])} nodes, {len(data['edges'])} edges at {data['revision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
