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

from __future__ import annotations

import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path
from urllib.error import HTTPError

import pytest

from .test_merge_published_site import _write_published_site

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "docs/scripts/prepare_docs_deployment.py"
)
_spec = importlib.util.spec_from_file_location("prepare_docs_deployment", _SCRIPT)
assert _spec is not None and _spec.loader is not None
_deploy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_deploy)


@pytest.mark.parametrize("version", ["main", "v0.3.1"])
def test_queued_artifact_preserves_versions_published_after_build(
    tmp_path: Path, version: str
) -> None:
    candidate, published = tmp_path / "candidate", tmp_path / "published"
    _write_published_site(candidate, ["v0.3.0", version], latest="v0.3.0")
    _write_published_site(published, ["v0.3.0", "v0.3.1", "main"], latest="v0.3.1")
    (candidate / version / "index.html").write_text("freshly built")
    (candidate / "v0.3.0" / "index.html").write_text("stale cached release")
    if version != "main":
        (candidate / "main").mkdir()
        (candidate / "main" / "index.html").write_text("stale cached main")

    _deploy.prepare_deployment(candidate, version, published_root=published)

    manifest = json.loads((candidate / "versions.json").read_text())
    assert manifest["latest"] == "v0.3.1"
    assert {entry["name"] for entry in manifest["versions"]} == {
        "v0.3.0",
        "v0.3.1",
        "main",
    }
    assert (candidate / version / "index.html").read_text() == "freshly built"
    for name in {"v0.3.0", "v0.3.1", "main"} - {version}:
        assert (candidate / name / "index.html").read_bytes() == (
            published / name / "index.html"
        ).read_bytes()
    assert "./v0.3.1/index.html" in (candidate / "index.html").read_text()


@pytest.mark.parametrize("missing", ["manifest", "homepage", "target"])
def test_incomplete_sites_stop_preparation(tmp_path: Path, missing: str) -> None:
    candidate, published = tmp_path / "candidate", tmp_path / "published"
    _write_published_site(candidate, ["main"], latest="main")
    _write_published_site(published, ["main", "v0.3.1"], latest="v0.3.1")
    if missing == "manifest":
        (published / "versions.json").unlink()
    elif missing == "homepage":
        (published / "v0.3.1" / "index.html").unlink()
    else:
        (candidate / "main" / "index.html").unlink()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        _deploy.prepare_deployment(candidate, "main", published_root=published)


def test_retention_applies_after_merging_live_releases(tmp_path: Path) -> None:
    candidate, published = tmp_path / "candidate", tmp_path / "published"
    tags = [f"v0.3.{i}" for i in range(6)]
    _write_published_site(candidate, ["main"], latest="main")
    _write_published_site(published, [*tags, "main"], latest=tags[-1])
    _deploy.prepare_deployment(candidate, "main", published_root=published)
    manifest = json.loads((candidate / "versions.json").read_text())
    assert [entry["name"] for entry in manifest["versions"]] == [
        *reversed(tags[1:]),
        "main",
    ]
    assert not (candidate / tags[0]).exists()


def test_old_release_outside_retention_is_rejected(tmp_path: Path) -> None:
    candidate, published = tmp_path / "candidate", tmp_path / "published"
    _write_published_site(candidate, ["v0.1.0"], latest="v0.1.0")
    _write_published_site(published, ["v0.3.0", "main"], latest="v0.3.0")
    with pytest.raises(ValueError, match="outside"):
        _deploy.prepare_deployment(
            candidate, "v0.1.0", published_root=published, max_versions=1
        )


@pytest.mark.parametrize("failure", [None, "marker", "manifest", "homepage"])
def test_verification_checks_versions_and_homepages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    _write_published_site(tmp_path, ["main", "v0.3.1"], latest="v0.3.1")
    requests = []

    def read_url(site: str, path: str) -> bytes:
        requests.append(path)
        if path == "pages-deployment-id.txt":
            return b"old" if failure == "marker" else b"123-1"
        if path == "versions.json" and failure == "manifest":
            return b'{"latest": "main", "versions": []}'
        if path == "v0.3.1/index.html" and failure == "homepage":
            raise HTTPError(site + path, 404, "Not found", None, None)
        return (tmp_path / path).read_bytes()

    monkeypatch.setattr(_deploy, "_read_url", read_url)
    if failure:
        with pytest.raises((RuntimeError, HTTPError)):
            _deploy.verify_deployment(tmp_path, "https://example.org", "123-1")
    else:
        _deploy.verify_deployment(tmp_path, "https://example.org", "123-1")
        assert "main/index.html" in requests
        assert "v0.3.1/index.html" in requests


def test_deployed_snapshot_uses_exact_pages_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site, downloaded = tmp_path / "site", tmp_path / "downloaded"
    _write_published_site(site, ["main", "v0.3.1"], latest="v0.3.1")
    (site / "pages-deployment-id.txt").write_text("123-2")
    downloaded.mkdir()
    with tarfile.open(downloaded / "artifact.tar", "w") as archive:
        for path in site.rglob("*"):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(site))
    monkeypatch.setattr(_deploy, "_read_url", lambda *args: b"123-2")

    def download(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["gh", "run", "download", "123"]
        assert "github-pages-2" in command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", download)
    snapshot = _deploy._published_snapshot(
        "https://example.org", "owner/repo", downloaded
    )
    assert snapshot is not None
    assert (snapshot / "v0.3.1" / "index.html").read_bytes() == (
        site / "v0.3.1" / "index.html"
    ).read_bytes()


def test_expired_artifact_falls_back_to_live_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_deploy, "_read_url", lambda *args: b"123-2")
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1)
    )
    assert (
        _deploy._published_snapshot("https://example.org", "owner/repo", tmp_path)
        is None
    )


@pytest.mark.parametrize("version", ["v0.3.1.post1", "v0.4.0rc1"])
def test_pep440_release_tags_are_supported(tmp_path: Path, version: str) -> None:
    candidate, published = tmp_path / "candidate", tmp_path / "published"
    _write_published_site(candidate, [version], latest=version)
    _write_published_site(published, ["main"], latest="main")
    _deploy.prepare_deployment(candidate, version, published_root=published)
    assert json.loads((candidate / "versions.json").read_text())["latest"] == version


@pytest.mark.parametrize("version", ["../main", "v../main", "feature/docs", "version"])
def test_invalid_target_cannot_select_other_paths(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError):
        _deploy.prepare_deployment(tmp_path, version, published_root=tmp_path)


def test_required_manifest_read_failure_is_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import merge_published_site

    _write_published_site(tmp_path, ["main"], latest="main")

    def unavailable(*args: object, **kwargs: object) -> None:
        raise HTTPError("https://example.org", 503, "Unavailable", None, None)

    monkeypatch.setattr(merge_published_site, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="Cannot read published manifest"):
        _deploy.prepare_deployment(tmp_path, "main", site_url="https://example.org")
