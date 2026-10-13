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

from scripts.benchmark.rendering.frozen_source import (
    load_original_run,
    read_json,
    sha256,
    write_json,
)
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
    for relative in (
        "artifact_index.json",
        "capture/resources.json",
        "capture/renderer_validation.json",
        "capture/worker.log",
        "capture/freshness_probe.npz",
    ):
        assert sha256(proof_path.parent / relative) == sha256(
            fixture["root"] / relative
        )
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


def test_legacy_import_repair_is_append_only_and_idempotent(tmp_path: Path) -> None:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(tmp_path / "proof")
    target = tmp_path / "target"
    original_experiment(fixture, target)
    complete = import_history_proof(fixture["path"], target)
    destination = (target / complete["history_evidence"]["path"]).parent
    omitted = {
        "artifact_index.json",
        "capture/resources.json",
        "capture/renderer_validation.json",
        "capture/worker.log",
        "capture/freshness_probe.npz",
    }
    # Recreate the exact old importer behavior: its receipt and copied files
    # contain the direct descriptor graph, without the auxiliary closure.
    legacy = {
        **complete,
        "artifacts": [
            item
            for item in complete["artifacts"]
            if (target / item["path"]).relative_to(destination).as_posix()
            not in omitted
        ],
    }
    for name in omitted:
        (destination / name).unlink()
    write_json(destination / "import_receipt.json", legacy)
    preserved = {
        path: sha256(path) for path in destination.rglob("*") if path.is_file()
    }
    source = {
        path: sha256(path) for path in fixture["root"].rglob("*") if path.is_file()
    }
    repaired = import_history_proof(fixture["path"], target)
    assert (
        repaired["previous_import_receipt"]["sha256"]
        == preserved[destination / "import_receipt.json"]
    )
    assert {path: sha256(path) for path in preserved} == preserved
    assert {path: sha256(path) for path in source} == source
    assert all((destination / name).is_file() for name in omitted)
    assert len(list((destination / "closure_receipts").glob("*.json"))) == 1
    assert import_history_proof(fixture["path"], target) == repaired
    assert len(list((destination / "closure_receipts").glob("*.json"))) == 1


@pytest.mark.parametrize(
    "filename",
    ["resources.json", "renderer_validation.json", "worker.log", "freshness_probe.npz"],
)
@pytest.mark.parametrize("damage", ["missing", "changed", "unindexed"])
def test_auxiliary_missing_or_damaged_bytes_block_import(
    tmp_path: Path, filename: str, damage: str
) -> None:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(tmp_path / "proof")
    target = tmp_path / "target"
    original_experiment(fixture, target)
    path = fixture["root"] / "capture" / filename
    if damage == "missing":
        path.unlink()
    elif damage == "changed":
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        index = read_json(fixture["root"] / "artifact_index.json")
        index["artifacts"] = [
            entry
            for entry in index["artifacts"]
            if entry["path"] != "capture/" + filename
        ]
        write_json(fixture["root"] / "artifact_index.json", index)
    with pytest.raises(ValueError, match="Cannot import invalid history"):
        import_history_proof(fixture["path"], target)
    assert not (target / "history_imports").exists()


@pytest.mark.parametrize(
    "changed", ["gpu", "pid", "renderer", "log_prefix", "freshness"]
)
def test_resealed_index_does_not_replace_semantic_evidence_bindings(
    tmp_path: Path, changed: str
) -> None:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(tmp_path)
    filename = {
        "gpu": "resources.json",
        "pid": "resources.json",
        "renderer": "renderer_validation.json",
        "log_prefix": "worker.log",
        "freshness": "freshness_probe.npz",
    }[changed]
    path = tmp_path / "capture" / filename
    if changed in {"gpu", "pid", "renderer"}:
        data = read_json(path)
        if changed == "gpu":
            data["gpu"]["uuid"] = "GPU-other"
        elif changed == "pid":
            data["pid"] += 1
        else:
            data["checkpoint"] = "after_native_close"
        write_json(path, data)
    else:
        path.write_bytes(b"modified evidence\n")
    index = read_json(tmp_path / "artifact_index.json")
    for entry in index["artifacts"]:
        if entry["path"] == path.relative_to(tmp_path).as_posix():
            entry.update(sha256=sha256(path), bytes=path.stat().st_size)
    write_json(tmp_path / "artifact_index.json", index)
    checked = validate_history_proof(
        tmp_path, fixture["path"], fixture["expected_original"]
    )
    assert not checked["valid"]
    assert any(
        word in checked["reasons"][0] for word in ("resource", "renderer", "Freshness")
    )
