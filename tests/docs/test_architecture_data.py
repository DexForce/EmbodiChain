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

"""Static architecture contracts verified in isolated Git repositories."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "docs/scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from architecture_data import ArchitectureDataError, load_snapshot, validate_snapshot


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def commit(root: Path) -> str:
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "fixture",
    )
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def valid_snapshot(tmp_path: Path):
    git(tmp_path, "init", "-q")
    write(tmp_path, "VERSION", "1.0.0\n")
    write(tmp_path, "agent_context/MAP.yaml", "topics:\n- id: test-topic\n")
    write(tmp_path, "pkg/a.py", "class Alpha:\n    def run(self):\n        return 1\n")
    write(tmp_path, "pkg/b.py", "class Beta:\n    pass\n")
    write(tmp_path, "docs/source/alpha.md", "Alpha\n")
    rev = commit(tmp_path)

    def node(id, label, path):
        return dict(
            id=id,
            label=label,
            kind="class",
            topic_ids=["test-topic"],
            summary=label + " role",
            boundaries=["Example only"],
            evidence=[
                dict(
                    path=path,
                    symbol=label,
                    start_line=1,
                    end_line=1,
                    excerpt=f"class {label}:",
                )
            ],
            documentation=[dict(docname="alpha", label="Alpha docs")],
        )

    data = dict(
        schema_version=1,
        kind="static-architecture",
        coverage="curated-sample",
        repository="DexForce/EmbodiChain",
        revision=rev,
        source_ref="HEAD",
        package_version="1.0.0",
        topic_index="agent_context/MAP.yaml",
        limitations=["Example"],
        nodes=[node("alpha", "Alpha", "pkg/a.py"), node("beta", "Beta", "pkg/b.py")],
        edges=[
            dict(
                id="alpha-calls-beta",
                source="alpha",
                target="beta",
                relation="calls",
                description="Example scoped relationship",
                provenance="source-reviewed",
                scope="Test fixture only",
                evidence=[
                    dict(
                        path="pkg/a.py",
                        symbol="Alpha.run",
                        start_line=3,
                        end_line=3,
                        excerpt="        return 1",
                    )
                ],
            )
        ],
        views=[
            dict(
                id="overview",
                label="Overview",
                description="Example",
                node_ids=["alpha", "beta"],
                edge_ids=["alpha-calls-beta"],
                groups=[dict(id="all", label="All", node_ids=["alpha", "beta"])],
            )
        ],
    )
    return data, tmp_path, ROOT / "docs/architecture/architecture.schema.json"


def test_historical_validation_ignores_worktree_source(valid_snapshot):
    data, root, schema = valid_snapshot
    write(root, "pkg/a.py", "raise RuntimeError('must not import')\n")
    validate_snapshot(data, repo_root=root, schema_path=schema)


@pytest.mark.parametrize(
    "case,match",
    [
        ("target", "missing"),
        ("duplicate", "duplicate"),
        ("relation", "relation"),
        ("path", "path"),
        ("range", "range"),
        ("excerpt", "excerpt"),
        ("symbol", "symbol"),
        ("topic", "topic"),
        ("doc", "doc"),
        ("endpoint", "endpoint"),
        ("group", "group"),
        ("group-omitted", "group"),
        ("version", "VERSION"),
        ("revision", "fetch"),
    ],
)
def test_rejects_invalid_contract(valid_snapshot, case, match):
    data, root, schema = valid_snapshot
    if case == "target":
        data["edges"][0]["target"] = "missing"
    if case == "duplicate":
        data["nodes"].append(copy.deepcopy(data["nodes"][0]))
    if case == "relation":
        data["edges"][0]["relation"] = "teleports"
    if case == "path":
        data["nodes"][0]["evidence"][0]["path"] = "../a.py"
    if case == "range":
        data["edges"][0]["evidence"][0]["end_line"] = 1
    if case == "excerpt":
        data["nodes"][0]["evidence"][0]["excerpt"] = "class Renamed:"
    if case == "symbol":
        data["edges"][0]["evidence"][0]["symbol"] = "Beta.run"
    if case == "topic":
        data["nodes"][0]["topic_ids"] = ["unknown"]
    if case == "doc":
        data["nodes"][0]["documentation"][0]["docname"] = "missing"
    if case == "endpoint":
        data["views"][0]["node_ids"] = ["alpha"]
        data["views"][0]["groups"][0]["node_ids"] = ["alpha"]
    if case == "group":
        data["views"][0]["groups"][0]["node_ids"] = ["alpha", "alpha"]
    if case == "group-omitted":
        data["views"][0]["groups"][0]["node_ids"] = ["alpha"]
    if case == "version":
        data["package_version"] = "9.9.9"
    if case == "revision":
        data["revision"] = "0" * 40
    with pytest.raises(ArchitectureDataError, match=match):
        validate_snapshot(data, repo_root=root, schema_path=schema)


def test_rejects_ambiguous_docname(valid_snapshot):
    data, root, schema = valid_snapshot
    write(root, "docs/source/alpha.rst", "Alpha\n")
    data["revision"] = commit(root)
    with pytest.raises(ArchitectureDataError, match="doc"):
        validate_snapshot(data, repo_root=root, schema_path=schema)


def test_module_and_document_evidence_and_wrong_lexical_scope(valid_snapshot):
    data, root, schema = valid_snapshot
    write(
        root,
        "pkg/a.py",
        "import os\nclass Alpha:\n    def run(self):\n        return 1\n",
    )
    write(root, "config.txt", "setting = 1\n")
    data["revision"] = commit(root)
    data["nodes"][0]["evidence"] = [
        dict(
            path="pkg/a.py",
            symbol="<module>",
            start_line=1,
            end_line=1,
            excerpt="import os",
        ),
        dict(
            path="config.txt",
            symbol="<document>",
            start_line=1,
            end_line=1,
            excerpt="setting = 1",
        ),
    ]
    data["edges"][0]["evidence"][0].update(start_line=4, end_line=4)
    validate_snapshot(data, repo_root=root, schema_path=schema)
    data["edges"][0]["evidence"][0]["symbol"] = "<module>"
    with pytest.raises(ArchitectureDataError, match="scope"):
        validate_snapshot(data, repo_root=root, schema_path=schema)


def test_load_snapshot_reports_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(ArchitectureDataError, match="bad.json"):
        load_snapshot(path)


def test_module_excerpt_cannot_overlap_a_definition(valid_snapshot):
    data, root, schema = valid_snapshot
    write(
        root,
        "pkg/a.py",
        "class Alpha:\n    def run(self):\n        return 1\nimport os\n",
    )
    data["revision"] = commit(root)
    data["edges"][0]["evidence"][0].update(
        symbol="<module>", end_line=4, excerpt="        return 1\nimport os"
    )
    with pytest.raises(ArchitectureDataError, match="scope"):
        validate_snapshot(data, repo_root=root, schema_path=schema)
