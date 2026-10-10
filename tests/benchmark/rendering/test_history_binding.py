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

"""Portable history imports preserve every original timing byte."""

from __future__ import annotations

from pathlib import Path
import runpy
import shutil
from typing import Any

import pytest

from scripts.benchmark.rendering.frozen_source import load_original_run, sha256
from scripts.benchmark.rendering.history_binding import import_history_proof
from scripts.benchmark.rendering.history_contract import validate_history_proof


def original_experiment(fixture: dict[str, Any], target: Path) -> dict[str, Any]:
    """Materialize a fake original formal experiment using the proof's exact bytes."""
    run_id = fixture["expected_original"]["run_id"]
    mapping = {
        "result": f"{run_id}/result.json",
        "worker_config": f"{run_id}/worker_config.json",
        "capture": f"{run_id}/capture.json",
        "freshness": f"{run_id}/freshness_probe.json",
        "raw": "raw.jsonl",
        "definition": "definition.json",
        "scene": "original/scene.json",
        "replay": "original/replay.json",
    }
    for evidence in fixture["manifest"]["original"]["artifacts"]:
        if evidence["role"] in mapping:
            path = target / mapping[evidence["role"]]
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture["root"] / evidence["path"], path)
    return load_original_run(target, run_id)


def test_import_copies_verified_closure_without_touching_originals(
    tmp_path: Path,
) -> None:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(tmp_path / "proof")
    target = tmp_path / "target"
    original = original_experiment(fixture, target)
    receipt = import_history_proof(fixture["path"], target)
    assert receipt["quality_status"] == "not_evaluated"
    assert receipt["performance_claim"] is False
    proof_path = target / receipt["history_evidence"]["path"]
    checked = validate_history_proof(target, proof_path, original["row"])
    assert checked["valid"], checked["reasons"]
    assert len(checked["frames"]) == 7
    assert {name: sha256(path) for name, path in original["paths"].items()} == original[
        "input_hashes"
    ]
    assert import_history_proof(fixture["path"], target) == receipt
    assert not (target / "quality.json").exists()
    assert not (target / "runs.json").exists()


@pytest.mark.parametrize("changed", ["input", "target", "symlink"])
def test_import_rejects_tampering_or_unsafe_destination(
    tmp_path: Path, changed: str
) -> None:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(tmp_path / "proof")
    target = tmp_path / "target"
    original = original_experiment(fixture, target)
    if changed == "input":
        next((fixture["root"] / "capture/frames").glob("*.npz")).write_bytes(b"changed")
    elif changed == "target":
        original["paths"]["capture"].write_text('{"changed":true}\n')
    else:
        (target / "history_imports").symlink_to(
            tmp_path / "elsewhere", target_is_directory=True
        )
    with pytest.raises(ValueError):
        import_history_proof(fixture["path"], target)
    assert not (tmp_path / "elsewhere").exists()
