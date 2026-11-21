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

"""Run publication preserves the filesystem identity of sealed declarations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.orchestration.workflow_artifacts import (
    WorkflowRunArtifacts,
    validate_run_output_directory,
    write_run_json,
)


def _seal_marker(bundle: Path) -> None:
    bundle.mkdir(parents=True)
    (bundle / "integration_fingerprint.json").write_bytes(b"independent sealed bytes")


def test_completion_keeps_the_original_bundle_path_and_bytes(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    bundle = destination / "attempts" / "scene_0001" / "bundle"
    with WorkflowRunArtifacts(destination) as artifacts:
        assert artifacts.staging_dir == destination
        _seal_marker(bundle)
        inode = bundle.stat().st_ino
        write_run_json(destination / "run_manifest.json", {"status": "prepared"})
        assert artifacts.commit() == destination
    assert bundle.stat().st_ino == inode
    assert (
        bundle / "integration_fingerprint.json"
    ).read_bytes() == b"independent sealed bytes"
    assert json.loads((destination / "run_manifest.json").read_bytes()) == {
        "status": "prepared"
    }


def test_cancellation_preserves_sealed_inputs_at_their_original_path(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run"
    bundle = destination / "attempts" / "scene_0001" / "bundle"
    with pytest.raises(KeyboardInterrupt):
        with WorkflowRunArtifacts(destination):
            _seal_marker(bundle)
            raise KeyboardInterrupt("cancelled after preparation")
    assert (
        bundle / "integration_fingerprint.json"
    ).read_bytes() == b"independent sealed bytes"
    interrupted = json.loads((destination / "workflow_interruption.json").read_bytes())
    assert interrupted["error_type"] == "KeyboardInterrupt"
    assert interrupted["status"] == "interrupted"
    with pytest.raises(ValueError, match="sealed bundles"):
        with WorkflowRunArtifacts(destination, overwrite=True):
            pytest.fail("sealed history was opened for replacement")


def test_unsealed_failed_replacement_restores_previous_output(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    destination.mkdir()
    (destination / "old-audit.txt").write_text("previous audit")
    with pytest.raises(RuntimeError, match="before bundle preparation"):
        with WorkflowRunArtifacts(destination, overwrite=True):
            write_run_json(destination / "partial.json", {"status": "running"})
            raise RuntimeError("before bundle preparation")
    assert (destination / "old-audit.txt").read_text() == "previous audit"
    assert not (destination / "partial.json").exists()


def test_audit_refresh_cannot_reach_a_sealed_bundle_through_a_link(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _seal_marker(bundle)
    alias = tmp_path / "bundle-alias"
    alias.symlink_to(bundle, target_is_directory=True)
    for path in (bundle / "declaration.json", alias / "declaration.json"):
        with pytest.raises(ValueError, match="sealed bundle"):
            write_run_json(path, {"changed": True})
    assert not (bundle / "declaration.json").exists()
    for path in (bundle, alias, alias / "run"):
        with pytest.raises(ValueError, match="sealed bundle"):
            validate_run_output_directory(path)
    with pytest.raises(ValueError, match="sealed bundle"):
        with WorkflowRunArtifacts(alias / "run"):
            pytest.fail("a sealed bundle admitted run outputs")


def test_invalid_audit_does_not_replace_the_previous_complete_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run" / "attempt.json"
    write_run_json(path, {"status": "prepared"})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="JSON compliant"):
        write_run_json(path, {"measurement": float("nan")})
    assert path.read_bytes() == original
    assert list(path.parent.glob("*.tmp")) == []


def test_run_output_validation_is_read_only(tmp_path: Path) -> None:
    destination = tmp_path / "new-run"
    assert validate_run_output_directory(destination) == destination
    assert not destination.exists()
