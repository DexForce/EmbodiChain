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

"""Tests for pre-deployment docs site validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import validate_docs_site


def _write_site(root: Path, versions: list[str], latest: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "latest": latest or versions[0],
        "versions": [
            {
                "name": version,
                "url": f"./{version}/index.html",
                "type": "tag" if version.startswith("v") else "branch",
            }
            for version in versions
        ],
    }
    (root / "versions.json").write_text(json.dumps(manifest), encoding="utf-8")
    for version in versions:
        version_dir = root / version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "index.html").write_text(
            f"<html>{version}</html>", encoding="utf-8"
        )


def test_validate_requires_release_version(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    _write_site(build_dir, ["v0.2.2", "main"], latest="v0.2.2")

    validate_docs_site(build_dir, required_versions={"v0.2.2"})


def test_validate_fails_when_required_release_missing(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    _write_site(build_dir, ["main"], latest="main")

    with pytest.raises(RuntimeError, match="v0.2.2"):
        validate_docs_site(build_dir, required_versions={"v0.2.2"})


def test_validate_fails_when_manifest_points_to_missing_dir(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    _write_site(build_dir, ["v0.2.2", "main"], latest="v0.2.2")
    (build_dir / "v0.2.2" / "index.html").unlink()

    with pytest.raises(FileNotFoundError, match="v0.2.2"):
        validate_docs_site(build_dir)


def test_validate_preserves_live_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir = tmp_path / "build"
    _write_site(build_dir, ["v0.2.2", "main"], latest="v0.2.2")

    live_manifest = {
        "latest": "v0.2.2",
        "versions": [
            {"name": "v0.2.2", "url": "./v0.2.2/index.html", "type": "tag"},
            {"name": "main", "url": "./main/index.html", "type": "branch"},
        ],
    }

    monkeypatch.setattr(
        "validate_docs_site.load_live_manifest",
        lambda site_base_url: live_manifest,
    )

    validate_docs_site(
        build_dir,
        required_versions={"main"},
        preserve_live_tags=True,
        site_base_url="https://example.invalid/EmbodiChain",
    )


def test_validate_rejects_main_only_site_when_live_has_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir = tmp_path / "build"
    _write_site(build_dir, ["main"], latest="main")

    live_manifest = {
        "latest": "v0.2.2",
        "versions": [
            {"name": "v0.2.2", "url": "./v0.2.2/index.html", "type": "tag"},
            {"name": "main", "url": "./main/index.html", "type": "branch"},
        ],
    }

    monkeypatch.setattr(
        "validate_docs_site.load_live_manifest",
        lambda site_base_url: live_manifest,
    )

    with pytest.raises(RuntimeError, match="v0.2.2"):
        validate_docs_site(
            build_dir,
            required_versions={"main"},
            preserve_live_tags=True,
            site_base_url="https://example.invalid/EmbodiChain",
        )
