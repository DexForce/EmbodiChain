# Architecture Explorer

A runnable [frontend preview](web/README.md) is now available. It provides the
28-module overview and 16-object Task Program view, with search, relationship
filters, direct-neighbour exploration, source evidence, and shareable navigation. Run it with the commands in
the frontend README. It uses [a pinned preview snapshot](preview.snapshot.json);
automatic generation and Sphinx integration are still pending.

The data contract and original sample below remain the reference for the preview
and future generator. Neither snapshot represents a runtime trace or a complete graph.

- [Design and scope](../superpowers/specs/2026-09-17-architecture-explorer-design.md)
- [Implementation plan](../superpowers/plans/2026-09-17-architecture-explorer.md)
- [JSON Schema](architecture.schema.json)
- [Task Program sample](task-program.sample.json): 16 nodes, 25 scoped relations.

The sample is pinned to `3224ac1ee28b6730b245d5cb69dc25a8d2d8dd94`.
`agent_context/MAP.yaml` remains the sole topic inventory. A view group is a
presentation choice, not a new topic or proof of a dependency. Every relationship
has a source excerpt and an explicit scope. Follow `evidence` to inspect a claim;
do not infer runtime order from adjacency or interpret `holds` as exclusive ownership.

`architecture.schema.json` uses JSON Schema Draft 2020-12. Source and documentation
paths are repository-relative. Documentation uses Sphinx docnames without an
extension; no symbol anchor is claimed. All evidence in one snapshot refers to
its root `revision`. The schema validates structure; the graph and source checks
below validate references and exact excerpts. They do not prove that prose
correctly interprets a source fragment; that remains a review responsibility.

## Reproduce the sample checks

Run from the repository root with Python, `jsonschema`, and `PyYAML` available.
These are validation-tool requirements, not new simulator runtime dependencies.
The pinned commit must exist locally. The historical sample is checked against
that commit even after later source edits; a future release generator must instead
produce a fresh snapshot from the checkout being documented.

```bash
python - <<'PY'
import ast
import json
import subprocess
from functools import cache
from pathlib import Path

import jsonschema
import yaml

root = Path.cwd()
folder = root / "docs/architecture"
schema = json.loads((folder / "architecture.schema.json").read_text())
data = json.loads((folder / "task-program.sample.json").read_text())
jsonschema.Draft202012Validator.check_schema(schema)
jsonschema.Draft202012Validator(schema).validate(data)

@cache
def source(path):
    return subprocess.check_output(
        ["git", "show", f"{data['revision']}:{path}"], text=True
    )

def symbols(text):
    result = {}
    def visit(node, parents=()):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parents = (*parents, node.name)
            result[".".join(parents)] = (node.lineno, node.end_lineno)
        for child in ast.iter_child_nodes(node):
            visit(child, parents)
    visit(ast.parse(text))
    return result

def index(items):
    result = {item["id"]: item for item in items}
    assert len(result) == len(items), "Duplicate ID"
    return result

nodes, edges, views = (index(data[key]) for key in ("nodes", "edges", "views"))
topics = {topic["id"] for topic in yaml.safe_load(source(data["topic_index"]))["topics"]}
tracked = set(subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", data["revision"]], text=True
).splitlines())
assert source("VERSION").strip() == data["package_version"]
proof_count = 0
for item in [*nodes.values(), *edges.values()]:
    for proof in item["evidence"]:
        text = source(proof["path"])
        lines = text.splitlines()
        start, end = proof["start_line"], proof["end_line"]
        assert 1 <= start <= end <= len(lines), item["id"]
        assert "\n".join(lines[start - 1:end]) == proof["excerpt"], item["id"]
        first, last = symbols(text)[proof["symbol"]]
        assert first <= start <= end <= last, item["id"]
        proof_count += 1
for node in nodes.values():
    assert set(node["topic_ids"]) <= topics, node["id"]
    for doc in node["documentation"]:
        matches = [suffix for suffix in (".md", ".rst")
                   if f"docs/source/{doc['docname']}{suffix}" in tracked]
        assert len(matches) == 1, doc["docname"]
for edge in edges.values():
    assert edge["source"] in nodes and edge["target"] in nodes, edge["id"]
for view in views.values():
    selected = set(view["node_ids"])
    assert selected <= nodes.keys(), view["id"]
    assert set(view["edge_ids"]) <= edges.keys(), view["id"]
    groups = index(view["groups"])
    members = [node for group in groups.values() for node in group["node_ids"]]
    assert len(members) == len(set(members)) and set(members) == selected
    for edge_id in view["edge_ids"]:
        edge = edges[edge_id]
        assert {edge["source"], edge["target"]} <= selected, edge_id
print(f"PASS: {len(nodes)} nodes, {len(edges)} edges, {proof_count} source excerpts")
PY
```

The future production validator and its regression tests are specified in the
implementation plan. This inline check keeps the current design/data-only change
independently inspectable without introducing a new production script.
