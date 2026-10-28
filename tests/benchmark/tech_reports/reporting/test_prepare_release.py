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

"""Protect raw evidence and distinguish input candidates from public artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark.tech_reports.reporting.prepare_release import prepare


def test_failed_run_is_retained_but_runtime_and_generic_assets_are_excluded(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    raw = workspace / "outputs/benchmarks/P-03/runs/failed"
    raw.mkdir(parents=True)
    (raw / "error.log").write_bytes(b"abc")
    runtime = workspace / "outputs/benchmarks/P-03/runtime"
    runtime.mkdir()
    (runtime / "engine.bin").write_bytes(b"private runtime")
    (raw / "robot.usd").write_bytes(b"generic upstream asset")
    output = tmp_path / "stage"
    summary = prepare(workspace, output)
    report = output / "reports" / summary["report_id"]
    inventory = json.loads((report / "inventory.json").read_text())
    assert summary["publication_status"] == "not_uploaded"
    assert summary["dataset_revision"] is None
    assert len(inventory["artifacts"]) == 1
    record = inventory["artifacts"][0]
    assert record["source"].endswith("failed/error.log")
    assert (
        record["sha256"]
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert record["size_bytes"] == 3
    assert record["license"] == "NOASSERTION"
    assert len(inventory["excluded"]) == 2
    index = json.loads((output / "index.fragment.json").read_text())
    assert index["bundles"]
    for bundle in index["bundles"].values():
        for item in bundle["files"]:
            assert "/inputs/" in item["path"]
            assert (output / item["path"]).stat().st_size == item["size_bytes"]
    assert (raw / "error.log").read_bytes() == b"abc"
    with pytest.raises(FileExistsError):
        prepare(workspace, output)


def test_staging_cannot_write_inside_evidence_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        prepare(tmp_path, tmp_path / "staging")
