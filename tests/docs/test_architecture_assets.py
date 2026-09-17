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

"""Architecture assets publish only complete builds and preserve source versions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/scripts"))
from build_architecture_assets import build_assets


@pytest.mark.parametrize("fail", ["generator", "frontend"])
def test_failed_build_preserves_previous_bundle(tmp_path, monkeypatch, fail):
    root = tmp_path / "repo"
    (root / "docs/architecture/web/node_modules").mkdir(parents=True)
    output = tmp_path / "published"
    output.mkdir()
    (output / "index.html").write_text("previous complete app")

    def run(command, **kwargs):
        phase = "generator" if command[0] == sys.executable else "frontend"
        if phase == fail:
            raise subprocess.CalledProcessError(1, command)
        target = Path(command[command.index("--output-dir") + 1])
        target.mkdir(exist_ok=True)
        (target / "architecture.json").write_text("{}")
        (target / "summary.md").write_text("summary")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        build_assets(root, output)
    assert (output / "index.html").read_text() == "previous complete app"
    assert list(output.iterdir()) == [output / "index.html"]


def test_success_publishes_matching_data_and_frontend(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "docs/architecture/web/node_modules").mkdir(parents=True)
    output = tmp_path / "published"
    output.mkdir()
    (output / "stale.js").write_text("old")

    def run(command, **kwargs):
        if command[0] == sys.executable:
            target = Path(command[command.index("--output-dir") + 1])
            target.mkdir(exist_ok=True)
            (target / "architecture.json").write_text('{"revision":"fixture"}')
            (target / "summary.md").write_text("summary")
        else:
            target = Path(command[command.index("--outDir") + 1])
            target.mkdir(exist_ok=True)
            snapshot = Path(kwargs["env"]["ARCHITECTURE_DATA_PATH"]).read_text()
            (target / "architecture.json").write_text(snapshot)
            (target / "index.html").write_text("app")

    monkeypatch.setattr(subprocess, "run", run)
    build_assets(root, output)
    assert sorted(p.name for p in output.iterdir()) == [
        "architecture.json",
        "index.html",
        "summary.md",
    ]
    assert (
        json.loads((output / "architecture.json").read_text())["revision"] == "fixture"
    )


def test_text_build_needs_no_node_toolchain(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert command[0] == sys.executable
        target = Path(command[command.index("--output-dir") + 1])
        (target / "architecture.json").write_text("{}")
        (target / "summary.md").write_text("searchable summary")

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "published"
    build_assets(tmp_path, output, html=False)
    assert (output / "summary.md").read_text() == "searchable summary"
    assert not (output / "index.html").exists()


def test_sphinx_entry_uses_searchable_summary_and_relative_links(tmp_path):
    from types import SimpleNamespace
    from architecture_sphinx import render_entry

    bundle = tmp_path / "_static/architecture"
    bundle.mkdir(parents=True)
    data = {
        "revision": "abc",
        "repository": "DexForce/EmbodiChain",
        "nodes": [
            {
                "documentation": [
                    {"docname": "api_reference/engine", "label": "Engine docs"}
                ]
            }
        ],
    }
    (bundle / "architecture.json").write_text(json.dumps(data))
    (bundle / "summary.md").write_text(
        "# Architecture overview\n\nEngine responsibility [Engine docs](https://github.com/DexForce/EmbodiChain/blob/abc/docs/source/api_reference/engine.rst)\n"
    )
    dependencies = []
    app = SimpleNamespace(
        srcdir=tmp_path,
        env=SimpleNamespace(note_dependency=dependencies.append),
        builder=SimpleNamespace(
            format="html", get_target_uri=lambda doc: doc + ".html"
        ),
    )
    result = render_entry(app, "overview/architecture/index")
    assert "Engine responsibility" in result
    assert "{doc}`Engine docs </api_reference/engine>`" in result
    assert "../../_static/architecture/index.html?docsRoot=../../" in result
    assert 'title="EmbodiChain architecture explorer"' in result
    assert "theme=light" in result
    assert dependencies
    app.builder.format = "text"
    text = render_entry(app, "overview/architecture/index")
    assert "<iframe" not in text and "Engine responsibility" in text


def test_two_source_revisions_keep_independent_snapshots_and_summaries(tmp_path):
    import shutil

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_architecture_data import commit, git, write

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    write(root, "VERSION", "1.0.0\n")
    write(root, "agent_context/MAP.yaml", "topics:\n- id: fixture\n")
    write(root, "pkg/engine.py", "class Engine:\n    pass\n")
    write(root, "docs/source/engine.md", "# Engine\n")
    revision = commit(root)
    seed = {
        "schema_version": 1,
        "kind": "static-architecture",
        "coverage": "curated-sample",
        "repository": "DexForce/EmbodiChain",
        "revision": revision,
        "source_ref": "HEAD",
        "package_version": "1.0.0",
        "topic_index": "agent_context/MAP.yaml",
        "limitations": ["Fixture"],
        "nodes": [
            {
                "id": "engine",
                "label": "First engine",
                "kind": "class",
                "summary": "First responsibility",
                "boundaries": ["Fixture"],
                "topic_ids": ["fixture"],
                "evidence": [
                    {
                        "path": "pkg/engine.py",
                        "symbol": "Engine",
                        "start_line": 1,
                        "end_line": 1,
                        "excerpt": "class Engine:",
                    }
                ],
                "documentation": [{"docname": "engine", "label": "Engine docs"}],
            }
        ],
        "edges": [],
        "views": [
            {
                "id": "overview",
                "label": "Overview",
                "description": "Fixture",
                "node_ids": ["engine"],
                "edge_ids": [],
                "groups": [{"id": "all", "label": "All", "node_ids": ["engine"]}],
            }
        ],
    }
    write(root, "docs/architecture/curated.json", json.dumps(seed))
    repo = Path(__file__).resolve().parents[2]
    for relative in (
        "docs/scripts/build_architecture.py",
        "docs/scripts/architecture_data.py",
        "docs/architecture/architecture.schema.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    first_revision = commit(root)
    first = tmp_path / "site/v1/_static/architecture"
    build_assets(root, first, html=False)
    write(root, "pkg/engine.py", "# moved\nclass Engine:\n    pass\n")
    seed["nodes"][0].update(label="Second engine", summary="Second responsibility")
    write(root, "docs/architecture/curated.json", json.dumps(seed))
    second_revision = commit(root)
    second = tmp_path / "site/v2/_static/architecture"
    build_assets(root, second, html=False)
    for output, revision, line, label in (
        (first, first_revision, 1, "First engine"),
        (second, second_revision, 2, "Second engine"),
    ):
        data = json.loads((output / "architecture.json").read_text())
        assert data["revision"] == revision
        assert data["nodes"][0]["evidence"][0]["start_line"] == line
        summary = (output / "summary.md").read_text()
        assert label in summary and f"/blob/{revision}/docs/source/engine.md" in summary


def test_project_myst_configuration_remains_valid(monkeypatch):
    import runpy
    from myst_parser.config.main import MdParserConfig

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fixture.invalid/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fixture")
    config = runpy.run_path(str(repo / "docs/source/conf.py"))
    parsed = MdParserConfig(enable_extensions=set(config["myst_enable_extensions"]))
    assert "colon_fence" in parsed.enable_extensions
    assert "architecture_sphinx" in config["extensions"]
