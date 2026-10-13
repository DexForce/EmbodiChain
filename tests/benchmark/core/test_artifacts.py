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

"""Batch evidence publication without partial indexes or repeated index writes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts.benchmark.core import artifacts
from scripts.benchmark.core.artifacts import ArtifactStore, read_json_object, write_json


def evidence(root: Path, name: str, *, kind: str = "fixture") -> dict[str, Any]:
    path = root / name
    path.write_text(name)
    return {
        "path": name,
        "kind": kind,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_batch_reads_and_writes_index_once_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "artifact_index.json"
    write_json(
        index_path, {"schema_version": 1, "retained": "metadata", "artifacts": []}
    )
    entries = [evidence(tmp_path, f"file_{index:04d}.txt") for index in range(500)]
    reads, writes = [], []
    read, write = artifacts.read_json_object, artifacts.write_json

    def observed_read(path: Path) -> dict[str, Any]:
        reads.append(path)
        return read(path)

    def observed_write(path: Path, value: object) -> Path:
        writes.append(path)
        return write(path, value)

    monkeypatch.setattr(artifacts, "read_json_object", observed_read)
    monkeypatch.setattr(artifacts, "write_json", observed_write)
    store = ArtifactStore(tmp_path)
    result = store.register_artifacts([*entries, entries[0]])
    assert len(result) == 500
    assert reads == writes == [index_path]
    assert read(index_path) == {
        "schema_version": 1,
        "retained": "metadata",
        "artifacts": result,
    }
    before = index_path.read_bytes()
    reads.clear()
    writes.clear()
    assert store.register_artifacts(entries) == result
    assert reads == [index_path]
    assert writes == []
    assert index_path.read_bytes() == before


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "checksum",
        "size",
        "duplicate_kind",
        "registered_kind",
        "changed_registered_bytes",
        "invalid_entry",
    ],
)
def test_invalid_batch_never_partially_updates_index(
    tmp_path: Path, failure: str
) -> None:
    store = ArtifactStore(tmp_path)
    registered = evidence(tmp_path, "registered.txt")
    store.register_artifact(tmp_path / registered["path"], kind=registered["kind"])
    index_path = tmp_path / "artifact_index.json"
    before = index_path.read_bytes()
    first = evidence(tmp_path, "valid_new.txt")
    bad = evidence(tmp_path, "invalid_new.txt")
    entries = [first, bad]
    if failure == "missing":
        (tmp_path / bad["path"]).unlink()
    elif failure == "checksum":
        bad["sha256"] = "0" * 64
    elif failure == "size":
        bad["bytes"] = 999
    elif failure == "duplicate_kind":
        entries.append({**first, "kind": "different"})
    elif failure == "registered_kind":
        entries.append({**registered, "kind": "different"})
    elif failure == "changed_registered_bytes":
        (tmp_path / registered["path"]).write_text("changed")
        entries.append({"path": registered["path"], "kind": registered["kind"]})
    else:
        entries.append({"path": "invalid-kind.txt"})
    with pytest.raises((ValueError, FileNotFoundError)):
        store.register_artifacts(entries)
    assert index_path.read_bytes() == before
    assert len(read_json_object(index_path)["artifacts"]) == 1


def test_failed_new_store_does_not_publish_an_index(tmp_path: Path) -> None:
    entries = [
        evidence(tmp_path, "valid.txt"),
        {"path": "missing.txt", "kind": "fixture"},
    ]
    with pytest.raises(FileNotFoundError):
        ArtifactStore(tmp_path).register_artifacts(entries)
    assert not (tmp_path / "artifact_index.json").exists()


@pytest.mark.parametrize("conflict", [False, True])
def test_registered_duplicates_are_idempotent_or_rejected(
    tmp_path: Path, conflict: bool
) -> None:
    store = ArtifactStore(tmp_path)
    entry = evidence(tmp_path, "existing.txt")
    record = store.register_artifact(tmp_path / entry["path"], kind=entry["kind"])
    index_path = tmp_path / "artifact_index.json"
    write_json(
        index_path,
        {
            "artifacts": [
                record,
                {**record, **({"kind": "conflict"} if conflict else {})},
            ]
        },
    )
    before = index_path.read_bytes()
    if conflict:
        with pytest.raises(ValueError, match="Conflicting registered"):
            store.register_artifacts([evidence(tmp_path, "new.txt")])
        assert index_path.read_bytes() == before
    else:
        assert store.register_artifacts([entry, entry]) == [record]
        assert index_path.read_bytes() == before


def test_kind_preservation_is_explicit_and_still_requires_identical_bytes(
    tmp_path: Path,
) -> None:
    entry = evidence(tmp_path, "existing.txt", kind="original")
    store = ArtifactStore(tmp_path)
    registered = store.register_artifact(tmp_path / entry["path"], kind="original")
    changed_kind = {**entry, "kind": "later_classification"}
    with pytest.raises(ValueError, match="different content"):
        store.register_artifacts([changed_kind])
    with pytest.raises(ValueError, match="different content"):
        store.register_artifact(tmp_path / entry["path"], kind="later_classification")
    assert store.register_artifacts([changed_kind], preserve_existing_kind=True) == [
        registered
    ]
    (tmp_path / entry["path"]).write_text("changed")
    with pytest.raises(ValueError):
        store.register_artifacts([changed_kind], preserve_existing_kind=True)


def test_batch_rejects_paths_outside_root_and_invalid_index(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    with pytest.raises(ValueError, match="inside"):
        ArtifactStore(root).register_artifacts([{"path": outside, "kind": "fixture"}])
    write_json(root / "artifact_index.json", {"artifacts": {}})
    with pytest.raises(ValueError, match="must be a list"):
        ArtifactStore(root).register_artifacts([])
