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


@pytest.mark.parametrize(
    "case_id", ["../outside", "..\\outside", "/outside", "C:outside", "..", "", None]
)
def test_manifest_rejects_unsafe_case_ids_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case_id: object
) -> None:
    from unittest.mock import Mock

    output = tmp_path / "output"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "result.json"
    sentinel.write_text("keep")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": case_id, "file_path": "source.usda"}]))
    monkeypatch.setattr(
        acceptance.sys,
        "argv",
        ["accept_usd_import", "--manifest", str(manifest), "--output", str(output)],
    )
    child = Mock()
    monkeypatch.setattr(acceptance.subprocess, "run", child)
    with pytest.raises(ValueError, match="single path component"):
        acceptance.main()
    child.assert_not_called()
    assert sentinel.read_text() == "keep"
    assert not (outside / "case.json").exists()


@pytest.mark.parametrize(
    "target",
    [
        "directory",
        "case.json",
        "runtime.log",
        "result.json",
        "overview.png",
        "results.json",
    ],
)
def test_acceptance_rejects_output_symlinks_without_touching_external_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    from unittest.mock import Mock

    output = tmp_path / "output"
    case_dir = output / "example"
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "result.json"
    external.write_text("keep")
    if target == "directory":
        output.mkdir()
        case_dir.symlink_to(outside, target_is_directory=True)
    else:
        case_dir.mkdir(parents=True)
        destination = output / target if target == "results.json" else case_dir / target
        destination.symlink_to(external)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"id": "example", "file_path": "source.usda"}]))
    monkeypatch.setattr(
        acceptance.sys,
        "argv",
        ["accept_usd_import", "--manifest", str(manifest), "--output", str(output)],
    )
    child = Mock()
    monkeypatch.setattr(acceptance.subprocess, "run", child)
    with pytest.raises(ValueError, match="symlink|escapes"):
        acceptance.main()
    child.assert_not_called()
    assert external.read_text() == "keep"
    assert not (outside / "case.json").exists()
