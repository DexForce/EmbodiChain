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

"""Generation uses committed text and conservative static relationships."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_architecture_data import ROOT, commit, git, valid_snapshot, write
from architecture_data import ArchitectureDataError, validate_snapshot
from build_architecture import build_snapshot, render_summary


@pytest.fixture
def seed(valid_snapshot):
    data, root, schema = valid_snapshot
    path = root / "seed.json"
    path.write_text(json.dumps(data))
    return data, root, path, schema


def test_generation_resolves_line_moves_and_is_deterministic(seed):
    data, root, path, schema = seed
    write(root, "pkg/a.py", "# shifted\n\n" + (root / "pkg/a.py").read_text())
    revision = commit(root)
    first = build_snapshot(root, path)
    assert first == build_snapshot(root, path)
    assert first["revision"] == revision
    assert first["nodes"][0]["evidence"][0]["start_line"] == 3
    assert first["edges"][0]["evidence"][0]["start_line"] == 5
    assert "generated_at" not in first
    validate_snapshot(first, repo_root=root, schema_path=schema)
    summary = render_summary(first, root)
    assert "Alpha role" in summary and "Beta role" in summary
    assert f"/blob/{revision}/docs/source/alpha.md" in summary


@pytest.mark.parametrize(
    "text,reason",
    [
        ("class Renamed:\n    def run(self):\n        return 1\n", "symbol"),
        ("class Alpha:\n    def run(self):\n        return 2\n", "excerpt"),
        (
            "class Alpha:\n    def run(self):\n        return 1\n        return 1\n",
            "ambiguous",
        ),
        (
            "class Alpha:\n    def run(self):\n        return 2\n    def other(self):\n        return 1\n",
            "excerpt",
        ),
    ],
)
def test_generation_fails_closed_for_changed_or_ambiguous_evidence(seed, text, reason):
    _, root, path, _ = seed
    write(root, "pkg/a.py", text)
    commit(root)
    with pytest.raises(ArchitectureDataError, match=reason):
        build_snapshot(root, path)


@pytest.mark.parametrize(
    "file", ["pkg/a.py", "agent_context/MAP.yaml", "docs/source/alpha.md", "VERSION"]
)
def test_rejects_relevant_dirty_inputs(seed, file):
    _, root, path, _ = seed
    with (root / file).open("a") as stream:
        stream.write("\n# dirty\n")
    with pytest.raises(ArchitectureDataError, match="uncommitted"):
        build_snapshot(root, path)


def test_dirty_check_checks_index_and_worktree_independently(seed):
    _, root, path, _ = seed
    original = (root / "pkg/a.py").read_text()
    write(root, "pkg/a.py", original + "# staged\n")
    git(root, "add", "pkg/a.py")
    write(root, "pkg/a.py", original)
    with pytest.raises(ArchitectureDataError, match="uncommitted"):
        build_snapshot(root, path)


def test_extracts_only_unambiguous_absolute_module_imports_and_inheritance(seed):
    data, root, path, schema = seed
    write(
        root,
        "pkg/a.py",
        "from pkg.b import Beta as B\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pkg.b\nclass Alpha(B):\n    def run(self):\n        return 1\n    def hidden(self):\n        import pkg.b\nraise RuntimeError('must never execute')\n",
    )
    data["nodes"][0]["evidence"][0]["excerpt"] = "class Alpha(B):"
    path.write_text(json.dumps(data))
    commit(root)
    result = build_snapshot(root, path)
    extracted = [e for e in result["edges"] if e["provenance"] == "static-extracted"]
    assert {e["relation"] for e in extracted} == {"imports", "inherits"}
    assert all(e["source"] == "alpha" and e["target"] == "beta" for e in extracted)
    assert not any(p["start_line"] == 9 for e in extracted for p in e["evidence"])
    assert any("TYPE_CHECKING" in e["scope"] for e in extracted)
    assert all(e["id"] in result["views"][0]["edge_ids"] for e in extracted)
    assert "pkg.a" not in sys.modules
    validate_snapshot(result, repo_root=root, schema_path=schema)


def test_ambiguous_file_ownership_and_rebound_alias_are_not_guessed(seed):
    data, root, path, _ = seed
    write(
        root,
        "pkg/a.py",
        "from pkg.b import Beta as B\nB = object\nclass Alpha(B):\n    def run(self):\n        return 1\nclass Extra:\n    pass\n",
    )
    data["nodes"][0]["evidence"][0]["excerpt"] = "class Alpha(B):"
    path.write_text(json.dumps(data))
    commit(root)
    result = build_snapshot(root, path)
    assert not any(e["relation"] == "inherits" for e in result["edges"])
    extra = json.loads(json.dumps(data["nodes"][0]))
    extra.update(id="extra", label="Extra")
    extra["evidence"][0].update(symbol="Extra", excerpt="class Extra:")
    data["nodes"].append(extra)
    data["views"][0]["node_ids"].append("extra")
    data["views"][0]["groups"][0]["node_ids"].append("extra")
    path.write_text(json.dumps(data))
    result = build_snapshot(root, path)
    assert not any(
        e["provenance"] == "static-extracted" and e["relation"] == "imports"
        for e in result["edges"]
    )


def test_failed_cli_preserves_existing_outputs(seed):
    _, root, path, _ = seed
    out = root / "output"
    out.mkdir()
    (out / "architecture.json").write_text("previous snapshot")
    (out / "summary.md").write_text("previous summary")
    write(root, "pkg/a.py", "class Renamed:\n    pass\n")
    commit(root)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "docs/scripts/build_architecture.py"),
            "--repo-root",
            str(root),
            "--curated",
            str(path),
            "--output-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "symbol" in result.stderr
    assert (out / "architecture.json").read_text() == "previous snapshot"
    assert (out / "summary.md").read_text() == "previous summary"


@pytest.mark.parametrize(
    "rebinding",
    [
        "with manager() as B:\n    pass\n",
        "try:\n    pass\nexcept Exception as B:\n    pass\n",
        "if (B := object):\n    pass\n",
        "del B\n",
        "match value:\n    case {'base': B}:\n        pass\n",
    ],
)
def test_other_module_binding_forms_do_not_claim_inheritance(seed, rebinding):
    data, root, path, _ = seed
    write(
        root,
        "pkg/a.py",
        "from pkg.b import Beta as B\n"
        + rebinding
        + "class Alpha(B):\n    def run(self):\n        return 1\n",
    )
    data["nodes"][0]["evidence"][0]["excerpt"] = "class Alpha(B):"
    path.write_text(json.dumps(data))
    commit(root)
    assert not any(
        e["relation"] == "inherits" for e in build_snapshot(root, path)["edges"]
    )


def test_unknown_curated_view_edge_is_not_silently_removed(seed):
    data, root, path, _ = seed
    data["views"][0]["edge_ids"].append("typo-edge")
    path.write_text(json.dumps(data))
    with pytest.raises(ArchitectureDataError, match="unknown view member"):
        build_snapshot(root, path)


@pytest.mark.parametrize("local", [True, False])
def test_decorated_base_identity_is_not_assumed(seed, local):
    data, root, path, _ = seed
    base = "def replace(cls):\n    return object\n@replace\nclass Beta:\n    pass\n"
    derived = "class Alpha(Beta):\n    def run(self):\n        return 1\n"
    write(
        root,
        "pkg/a.py",
        base + derived if local else "from pkg.b import Beta\n" + derived,
    )
    write(root, "pkg/b.py", base)
    data["nodes"][0]["evidence"][0]["excerpt"] = "class Alpha(Beta):"
    if local:
        data["nodes"][1]["evidence"][0]["path"] = "pkg/a.py"
    path.write_text(json.dumps(data))
    commit(root)
    assert not any(
        e["relation"] == "inherits" for e in build_snapshot(root, path)["edges"]
    )


def test_import_after_class_is_not_used_to_resolve_its_base(seed):
    data, root, path, _ = seed
    write(
        root,
        "pkg/a.py",
        "class Alpha(Beta):\n    def run(self):\n        return 1\nfrom pkg.b import Beta\n",
    )
    data["nodes"][0]["evidence"][0]["excerpt"] = "class Alpha(Beta):"
    path.write_text(json.dumps(data))
    commit(root)
    assert not any(
        e["relation"] == "inherits" for e in build_snapshot(root, path)["edges"]
    )


def test_summary_reports_unmapped_topics_and_edges_from_pinned_inputs(seed):
    data, root, path, _ = seed
    with (root / "agent_context/MAP.yaml").open("a") as stream:
        stream.write("- id: future\n  title: Future subsystem\n")
    commit(root)
    data["edges"] = []
    for view in data["views"]:
        view["edge_ids"] = []
    path.write_text(json.dumps(data))
    snapshot = build_snapshot(root, path)
    # A dirty topic index must not change a summary for the already pinned snapshot.
    with (root / "agent_context/MAP.yaml").open("a") as stream:
        stream.write("- id: dirty-only\n  title: Uncommitted topic\n")
    summary = render_summary(snapshot, root)
    assert "Topics without represented nodes: Future subsystem" in summary
    assert "Overview nodes without mapped relationships: Alpha, Beta" in summary
    assert "Uncommitted topic" not in summary
    assert "Missing edges do not imply architectural independence." in summary


def test_generation_records_actual_document_source_extensions(seed):
    data, root, path, _ = seed
    write(root, "docs/source/guide.rst", "Guide\n=====\n")
    commit(root)
    data["nodes"][0]["documentation"].append({"docname": "guide", "label": "Guide"})
    path.write_text(json.dumps(data))
    snapshot = build_snapshot(root, path)
    docs = snapshot["nodes"][0]["documentation"]
    assert docs[0]["source_extension"] == ".md"
    assert docs[-1]["source_extension"] == ".rst"
