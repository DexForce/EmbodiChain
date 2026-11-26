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

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_SCRIPT = _DOCS_DIR / "scripts/generate_versions_json.py"


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["v0.2.4", "v0.2.4.post1"], ["v0.2.4.post1", "v0.2.4"]),
        (
            ["v0.2.4.post2", "v0.2.5", "v0.2.4.post10"],
            ["v0.2.5", "v0.2.4.post10", "v0.2.4.post2"],
        ),
        (["v0.2.3", "v0.2.4"], ["v0.2.4", "v0.2.3"]),
        ([], []),
    ],
)
def test_manifest_and_redirect_use_newest_version(
    tmp_path: Path, tags: list[str], expected: list[str]
) -> None:
    for tag in [*tags, "main"]:
        (tmp_path / tag).mkdir()
    subprocess.run(
        [sys.executable, str(_SCRIPT), "--build-dir", str(tmp_path)], check=True
    )

    manifest = json.loads((tmp_path / "versions.json").read_text())
    latest = expected[0] if expected else "main"
    assert manifest["latest"] == latest
    assert [entry["name"] for entry in manifest["versions"]] == [*expected, "main"]
    assert f"url=./{latest}/index.html" in (tmp_path / "index.html").read_text()


@pytest.mark.skipif(shutil.which("make") is None, reason="requires GNU Make")
def test_current_docs_preserves_preview_homepage(tmp_path: Path) -> None:
    pytest.importorskip("sphinx")
    shutil.copyfile(_DOCS_DIR / "Makefile", tmp_path / "Makefile")
    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text("project = 'Preview'\n", encoding="utf-8")
    homepage_content = "This is the locally rendered documentation."
    (source / "index.rst").write_text(
        f"Preview\n=======\n\n{homepage_content}\n", encoding="utf-8"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(_SCRIPT, scripts / _SCRIPT.name)

    subprocess.run(
        [
            "make",
            "-o",
            "sync-readme",
            "current-docs",
            "BUILDDIR=preview",
            f"SPHINXBUILD={shlex.quote(sys.executable)} -m sphinx",
        ],
        cwd=tmp_path,
        check=True,
    )

    homepage = tmp_path / "preview/html/index.html"
    assert homepage_content in homepage.read_text(encoding="utf-8")
