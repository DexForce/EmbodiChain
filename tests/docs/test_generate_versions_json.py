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
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "docs/scripts/generate_versions_json.py"


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
