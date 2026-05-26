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

"""Tests for multi-version docs merge (CI GitHub Pages)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from .conftest import load_versions_manifest, merge_published_site


def _write_published_site(root: Path, versions: list[str], latest: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "latest": latest,
        "versions": [
            {
                "name": v,
                "url": f"./{v}/index.html",
                "type": "tag" if v.startswith("v") else "branch",
            }
            for v in versions
        ],
    }
    (root / "versions.json").write_text(json.dumps(manifest), encoding="utf-8")
    for v in versions:
        d = root / v
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(f"<html>{v} published</html>", encoding="utf-8")


@pytest.fixture
def published_site(tmp_path: Path) -> Path:
    published = tmp_path / "published"
    _write_published_site(published, ["v0.1.0", "v0.2.0", "main"], latest="v0.2.0")
    return published


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    build = tmp_path / "build" / "html"
    build.mkdir(parents=True)
    (build / "main").mkdir()
    (build / "main" / "index.html").write_text(
        "<html>stale main from cache</html>", encoding="utf-8"
    )
    return build


def test_load_manifest_from_local(published_site: Path) -> None:
    manifest = load_versions_manifest(published_root=published_site)
    assert manifest is not None
    assert manifest["latest"] == "v0.2.0"
    assert len(manifest["versions"]) == 3


def test_merge_fills_missing_tags_from_published(
    build_dir: Path, published_site: Path
) -> None:
    """Simulates main push: cache only has main/, live site has release tags."""
    merged = merge_published_site(
        build_dir,
        published_root=published_site,
        skip_versions=frozenset({"main"}),
    )
    assert merged == ["v0.1.0", "v0.2.0"]
    assert (
        (build_dir / "v0.1.0" / "index.html")
        .read_text(encoding="utf-8")
        .startswith("<html>v0.1.0")
    )
    assert (build_dir / "v0.2.0").is_dir()
    assert (build_dir / "main" / "index.html").read_text(encoding="utf-8") == (
        "<html>stale main from cache</html>"
    )


def test_merge_does_not_overwrite_existing_version(
    build_dir: Path, published_site: Path
) -> None:
    (build_dir / "v0.2.0").mkdir()
    (build_dir / "v0.2.0" / "index.html").write_text(
        "<html>v0.2.0 local cache</html>", encoding="utf-8"
    )
    merged = merge_published_site(
        build_dir,
        published_root=published_site,
        skip_versions=frozenset({"main"}),
    )
    assert merged == ["v0.1.0"]
    assert "local cache" in (build_dir / "v0.2.0" / "index.html").read_text(
        encoding="utf-8"
    )


def test_merge_skip_version_for_fresh_tag_build(
    build_dir: Path, published_site: Path
) -> None:
    """Simulates tag push: do not pull the tag being built from published."""
    merged = merge_published_site(
        build_dir,
        published_root=published_site,
        skip_versions=frozenset({"v0.3.0"}),
    )
    assert "v0.3.0" not in merged
    assert (build_dir / "v0.1.0").is_dir()


def test_main_push_after_tag_preserves_releases(
    build_dir: Path, published_site: Path, tmp_path: Path
) -> None:
    """End-to-end: stale cache + published site (post-tag) + rebuild main/."""
    _write_published_site(
        published_site,
        ["v0.1.0", "v0.2.0", "v0.3.0", "main"],
        latest="v0.3.0",
    )
    (published_site / "v0.3.0" / "index.html").write_text(
        "<html>v0.3.0 published</html>", encoding="utf-8"
    )

    merge_published_site(
        build_dir,
        published_root=published_site,
        skip_versions=frozenset({"main"}),
    )

    shutil.rmtree(build_dir / "main")
    (build_dir / "main").mkdir()
    (build_dir / "main" / "index.html").write_text(
        "<html>main rebuilt</html>", encoding="utf-8"
    )

    for name in ("v0.1.0", "v0.2.0", "v0.3.0"):
        assert (build_dir / name).is_dir(), f"missing {name} after main push simulation"
    assert "rebuilt" in (build_dir / "main" / "index.html").read_text(encoding="utf-8")
