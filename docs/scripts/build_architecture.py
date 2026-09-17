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

"""Generate deterministic, source-validated architecture data without importing packages."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml
from jsonschema import Draft202012Validator

from architecture_data import (
    ArchitectureDataError,
    SCHEMA,
    SourceRepository,
    load_snapshot,
    validate_snapshot,
)

__all__ = ["build_snapshot", "render_summary"]
ROOT = Path(__file__).resolve().parents[2]
CURATED = ROOT / "docs/architecture/curated.json"


def _relocate(source: SourceRepository, proof: dict[str, Any], owner: str) -> None:
    try:
        first, last = source.scope(proof)
        lines = source.read(proof["path"]).splitlines()
        excerpt = proof["excerpt"].splitlines()
        matches = [
            start
            for start in range(first, last - len(excerpt) + 2)
            if lines[start - 1 : start - 1 + len(excerpt)] == excerpt
            and source.in_scope(proof, start, start + len(excerpt) - 1)
        ]
        if len(matches) != 1:
            raise ArchitectureDataError(
                "excerpt missing or ambiguous inside declared symbol; review and update the curated evidence"
            )
        proof["start_line"] = matches[0]
        proof["end_line"] = matches[0] + len(excerpt) - 1
    except ArchitectureDataError as error:
        raise ArchitectureDataError(
            f"{owner} ({proof['path']}::{proof['symbol']}): {error}"
        ) from error


def _module_name(path: str) -> str:
    if path.startswith("embodichain_tasks/embodichain_tasks/"):
        path = path.removeprefix("embodichain_tasks/")
    return path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def _module_statements(
    statements: list[ast.stmt], scope: tuple[str, ...] = ()
) -> Iterator[tuple[ast.stmt, tuple[str, ...]]]:
    for statement in statements:
        yield statement, scope
        if isinstance(statement, ast.If):
            condition = ast.unparse(statement.test)
            yield from _module_statements(statement.body, (*scope, f"if {condition}"))
            yield from _module_statements(
                statement.orelse, (*scope, f"else of {condition}")
            )
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            yield from _module_statements(statement.body, (*scope, "try"))
            for handler in statement.handlers:
                yield from _module_statements(handler.body, (*scope, "except"))
            yield from _module_statements(statement.orelse, (*scope, "try else"))
            yield from _module_statements(statement.finalbody, (*scope, "finally"))
        elif isinstance(
            statement, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)
        ):
            yield from _module_statements(statement.body, (*scope, "conditional block"))
            yield from _module_statements(
                getattr(statement, "orelse", []), (*scope, "conditional else")
            )
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                yield from _module_statements(case.body, (*scope, "match case"))


def _binding_writes(node: ast.AST) -> Iterator[str]:
    """Conservatively find module binding changes without entering definition bodies."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        yield node.name
        expressions = list(node.decorator_list)
        if isinstance(node, ast.ClassDef):
            expressions.extend(node.bases)
            expressions.extend(keyword.value for keyword in node.keywords)
        else:
            expressions.extend(node.args.defaults)
            expressions.extend(default for default in node.args.kw_defaults if default)
        for expression in expressions:
            yield from _binding_writes(expression)
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        yield node.id
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
        # Mutating an imported module's attributes also invalidates base resolution.
        value = node.value
        while isinstance(value, ast.Attribute):
            value = value.value
        if isinstance(value, ast.Name):
            yield value.id
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
        yield node.name
    if isinstance(node, ast.MatchMapping) and node.rest:
        yield node.rest
    for child in ast.iter_child_nodes(node):
        yield from _binding_writes(child)


def _identity(edge: dict[str, Any]) -> tuple[Any, ...]:
    return (
        edge["source"],
        edge["target"],
        edge["relation"],
        tuple(
            (p["path"], p["start_line"], p["end_line"], p["excerpt"])
            for p in edge["evidence"]
        ),
    )


def _extract(source: SourceRepository, data: dict[str, Any]) -> list[dict[str, Any]]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    classes: dict[str, list[str]] = {}
    safe_bases: set[str] = set()
    for node in data["nodes"]:
        for proof in node["evidence"][:1]:
            if not proof["path"].endswith(".py"):
                continue
            by_file.setdefault(proof["path"], []).append(node)
            if node["kind"] == "class":
                qualified = f"{_module_name(proof['path'])}.{proof['symbol']}"
                classes.setdefault(qualified, []).append(node["id"])
                tree = source.tree(proof["path"])
                writes = list(_binding_writes(tree))
                imported_names = {
                    alias.asname
                    or (
                        alias.name.split(".")[0]
                        if isinstance(statement, ast.Import)
                        else alias.name
                    )
                    for statement, _ in _module_statements(tree.body)
                    if isinstance(statement, (ast.Import, ast.ImportFrom))
                    for alias in statement.names
                }
                for definition in tree.body:
                    if (
                        isinstance(definition, ast.ClassDef)
                        and definition.name == proof["symbol"]
                        and not definition.decorator_list
                        and not definition.keywords
                        and writes.count(definition.name) == 1
                        and definition.name not in imported_names
                        and "*" not in imported_names
                    ):
                        safe_bases.add(qualified)
    owners: dict[str, str] = {}
    for path, nodes in by_file.items():
        module_nodes = [n for n in nodes if n["kind"] in ("module", "package")]
        candidates = module_nodes or nodes
        if len(candidates) == 1:
            owners[_module_name(path)] = candidates[0]["id"]

    def target(name: str) -> str | None:
        candidates = classes.get(name, [])
        if len(candidates) == 1:
            return candidates[0]
        return owners.get(name) if not candidates else None

    extracted: list[dict[str, Any]] = []

    def append(
        src: str,
        dst: str | None,
        relation: str,
        path: str,
        statement: ast.stmt,
        symbol: str,
        scope: str,
    ) -> None:
        if dst is None or src == dst:
            return
        lines = source.read(path).splitlines()
        start, end = statement.lineno, statement.end_lineno or statement.lineno
        if relation == "inherits":
            # Base expressions may span multiple lines; retain the complete class header.
            assert isinstance(statement, ast.ClassDef)
            end = max(
                [start, *[base.end_lineno or base.lineno for base in statement.bases]]
            )
        proof = dict(
            path=path,
            symbol=symbol,
            start_line=start,
            end_line=end,
            excerpt="\n".join(lines[start - 1 : end]),
        )
        digest = hashlib.sha256(
            json.dumps(
                [src, dst, relation, path, start, end], separators=(",", ":")
            ).encode()
        ).hexdigest()[:12]
        extracted.append(
            dict(
                id=f"static-{relation}-{digest}",
                source=src,
                target=dst,
                relation=relation,
                description=(
                    "The selected source module declares an absolute import associated with this target."
                    if relation == "imports"
                    else "The class declares this statically resolved base class."
                ),
                provenance="static-extracted",
                scope=scope,
                evidence=[proof],
            )
        )

    for path in sorted(by_file):
        statements = list(_module_statements(source.tree(path).body))
        owner = owners.get(_module_name(path))
        bindings: dict[str, list[str | None]] = {}
        import_lines: dict[str, int] = {}
        for statement, condition in statements:
            scope = (
                "Module-level declaration"
                + (" under " + "; ".join(condition) if condition else "")
                + ". Not a runtime call or proof that every class in the file uses the imported symbol."
            )
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                for alias in statement.names:
                    if isinstance(statement, ast.Import):
                        name = alias.name
                        local = alias.asname or name.split(".")[0]
                        bound = name if alias.asname else name.split(".")[0]
                        dst = target(name)
                    else:
                        if alias.name == "*":
                            # Wildcards can change any binding; inheritance resolution is unsafe.
                            bindings.setdefault("*", []).append(None)
                            continue
                        local = alias.asname or alias.name
                        name = f"{statement.module}.{alias.name}"
                        bound = name if statement.level == 0 else None
                        dst = (
                            target(name) or owners.get(statement.module or "")
                            if statement.level == 0
                            else None
                        )
                    bindings.setdefault(local, []).append(
                        bound if not condition else None
                    )
                    import_lines[local] = statement.lineno
                    if owner:
                        append(
                            owner, dst, "imports", path, statement, "<module>", scope
                        )
        for name in _binding_writes(source.tree(path)):
            bindings.setdefault(name, []).append(None)
        local_classes = {
            statement.name: statement.lineno
            for statement, condition in statements
            if isinstance(statement, ast.ClassDef) and not condition
        }
        for statement, condition in statements:
            if not isinstance(statement, ast.ClassDef) or condition or "*" in bindings:
                continue
            srcs = classes.get(f"{_module_name(path)}.{statement.name}", [])
            if len(srcs) != 1:
                continue
            for base in statement.bases:
                name = ast.unparse(base)
                first, _, rest = name.partition(".")
                values = bindings.get(first, [])
                resolved = None
                if (
                    len(values) == 1
                    and values[0] is not None
                    and import_lines[first] < statement.lineno
                ):
                    resolved = values[0] + ("." + rest if rest else "")
                elif (
                    len(values) == 1
                    and not rest
                    and local_classes.get(first, statement.lineno) < statement.lineno
                    and f"{_module_name(path)}.{first}" in classes
                ):
                    resolved = f"{_module_name(path)}.{first}"
                if resolved in safe_bases and len(classes.get(resolved, [])) == 1:
                    append(
                        srcs[0],
                        classes[resolved][0],
                        "inherits",
                        path,
                        statement,
                        statement.name,
                        "Unconditional class header with a unique static base binding; no dynamic or protocol-to-concrete resolution.",
                    )
    return extracted


def build_snapshot(
    repo_root: Path, curated_path: Path, *, revision: str = "HEAD"
) -> dict[str, Any]:
    """Relocate reviewed evidence and extract conservative relationships at a commit.

    Args:
        repo_root: Git repository to read without importing its code.
        curated_path: Editable v1 semantic seed.
        revision: Git commit/ref to pin, defaulting to the checked-out commit.

    Returns:
        Validated, deterministically ordered snapshot.
    """
    seed = load_snapshot(curated_path)
    errors = list(Draft202012Validator(load_snapshot(SCHEMA)).iter_errors(seed))
    if errors:
        raise ArchitectureDataError(
            f"curated seed {list(errors[0].absolute_path)}: {errors[0].message}"
        )
    data = copy.deepcopy(seed)
    source = SourceRepository(repo_root, revision)
    data["revision"] = source.revision
    data["source_ref"] = revision
    data["package_version"] = source.read("VERSION").strip()
    extracted_ids = {
        e["id"] for e in data["edges"] if e["provenance"] == "static-extracted"
    }
    data["edges"] = [e for e in data["edges"] if e["provenance"] == "source-reviewed"]
    for view in data["views"]:
        view["edge_ids"] = [id for id in view["edge_ids"] if id not in extracted_ids]
    paths = {"VERSION", data["topic_index"]}
    for item in [*data["nodes"], *data["edges"]]:
        for proof in item["evidence"]:
            paths.add(proof["path"])
    for node in data["nodes"]:
        for doc in node["documentation"]:
            paths.update(
                f"docs/source/{doc['docname']}{ext}" for ext in (".md", ".rst")
            )
    if source.revision == source.git("rev-parse", "HEAD").strip():
        dirty = source.git(
            "status", "--porcelain=v1", "--untracked-files=all", "--", *sorted(paths)
        )
        if dirty:
            raise ArchitectureDataError(
                f"Relevant source has uncommitted changes; commit or restore it before generating:\n{dirty}"
            )
    for item in [*data["nodes"], *data["edges"]]:
        for proof in item["evidence"]:
            _relocate(source, proof, item["id"])
    identities = {_identity(e) for e in data["edges"]}
    for edge in _extract(source, data):
        if _identity(edge) in identities:
            continue
        views = [
            v
            for v in data["views"]
            if {edge["source"], edge["target"]} <= set(v["node_ids"])
        ]
        if not views:
            continue
        data["edges"].append(edge)
        identities.add(_identity(edge))
        for view in views:
            view["edge_ids"].append(edge["id"])
    data["nodes"].sort(key=lambda n: n["id"])
    data["edges"].sort(key=lambda e: e["id"])
    validate_snapshot(data, repo_root=repo_root)
    return data


def render_summary(data: dict[str, Any], repo_root: Path) -> str:
    """Render a Markdown overview with source and documentation links at one revision.

    Args:
        data: Validated architecture snapshot.
        repo_root: Git repository containing the pinned documentation.

    Returns:
        Deterministic Markdown text, suitable for a documentation fallback page.
    """
    source = SourceRepository(repo_root, data["revision"])
    view = next((v for v in data["views"] if v["id"] == "overview"), data["views"][0])
    nodes = {n["id"]: n for n in data["nodes"]}
    base = f"https://github.com/{data['repository']}/blob/{data['revision']}"

    def cell(text: str) -> str:
        return (
            text.replace("|", "\\|")
            .replace("\n", " ")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    lines = [
        "# Architecture overview",
        "",
        f"Source revision: `{data['revision']}` · package {data['package_version']}",
        "",
        "Curated static relationships; not a complete dependency graph or runtime trace.",
        "",
        "| Module | Responsibility | Evidence and documentation |",
        "| --- | --- | --- |",
    ]
    for id in view["node_ids"]:
        node = nodes[id]
        proof = node["evidence"][0]
        links = [
            f"[Source]({base}/{proof['path']}#L{proof['start_line']}-L{proof['end_line']})"
        ]
        for doc in node["documentation"]:
            ext = next(
                ext
                for ext in (".md", ".rst")
                if f"docs/source/{doc['docname']}{ext}" in source.files
            )
            links.append(
                f"[{cell(doc['label'])}]({base}/docs/source/{doc['docname']}{ext})"
            )
        lines.append(
            f"| {cell(node['label'])} | {cell(node['summary'])} | {' · '.join(links)} |"
        )
    topics = yaml.safe_load(source.read(data["topic_index"]))["topics"]
    represented = {topic for node in data["nodes"] for topic in node["topic_ids"]}
    missing_topics = [
        topic.get("title", topic["id"])
        for topic in topics
        if topic["id"] not in represented
    ]
    view_edges = [edge for edge in data["edges"] if edge["id"] in view["edge_ids"]]
    connected = {
        endpoint for edge in view_edges for endpoint in (edge["source"], edge["target"])
    }
    unmapped = [nodes[id]["label"] for id in view["node_ids"] if id not in connected]
    reviewed = sum(edge["provenance"] == "source-reviewed" for edge in data["edges"])
    extracted = sum(edge["provenance"] == "static-extracted" for edge in data["edges"])
    lines.extend(
        [
            "",
            "## Coverage and interpretation",
            "",
            f"Snapshot: {len(data['nodes'])} nodes, {reviewed} source-reviewed relationships, and {extracted} statically extracted relationships across all views.",
            "",
            "Import declarations do not establish runtime calls or execution order. Missing edges do not imply architectural independence.",
            "",
            "Topics without represented nodes: "
            + (", ".join(map(cell, missing_topics)) or "none")
            + ". Topic representation does not imply complete coverage.",
            "",
            "Overview nodes without mapped relationships: "
            + (", ".join(map(cell, unmapped)) or "none")
            + ".",
            "",
            *["- " + cell(limit) for limit in data["limitations"]],
        ]
    )
    return "\n".join(lines) + "\n"


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--curated", type=Path, default=CURATED)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = build_snapshot(args.repo_root, args.curated, revision=args.revision)
        contents = {
            "architecture.json": json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            "summary.md": render_summary(data, args.repo_root),
        }
        # All validation and rendering finish before any output is replaced.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, content in contents.items():
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=args.output_dir, delete=False
                ) as stream:
                    temporary = stream.name
                    stream.write(content)
                os.replace(temporary, args.output_dir / name)
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
    except (ArchitectureDataError, OSError) as error:
        parser.exit(1, f"Architecture generation failed: {error}\n")
    print(
        f"Generated {len(data['nodes'])} nodes and {len(data['edges'])} edges at {data['revision']} in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
