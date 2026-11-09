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

"""Regression coverage for isolated USD acceptance result handling."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validation import accept_usd_import as acceptance


@pytest.mark.parametrize("writes_result", [False, True])
def test_rerun_requires_a_result_from_the_current_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writes_result: bool
) -> None:
    output = tmp_path / "output"
    case_dir = output / "example"
    case_dir.mkdir(parents=True)
    result_path = case_dir / "result.json"
    result_path.write_text(
        json.dumps({"id": "example", "status": "passed", "previous_run": True})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "example", "file_path": "source.usda"}]))
    monkeypatch.setattr(
        acceptance.sys,
        "argv",
        ["accept_usd_import", "--manifest", str(manifest), "--output", str(output)],
    )

    def run_child(*args, **kwargs):
        if writes_result:
            result_path.write_text(json.dumps({"id": "example", "status": "passed"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(acceptance.subprocess, "run", run_child)
    if writes_result:
        acceptance.main()
    else:
        with pytest.raises(SystemExit, match="1"):
            acceptance.main()
    result = json.loads((output / "results.json").read_text())[0]
    assert "previous_run" not in result
    if writes_result:
        assert result["status"] == "passed"
    else:
        assert result["status"] == "failed"
        assert "without a result" in result["error"]
