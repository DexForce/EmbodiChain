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

"""Exact-identity shared reference closure and corrupt-input tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.benchmark.core.artifacts import stable_hash, write_json, read_json_object
from scripts.benchmark.rendering.reference_cache import transfer_reference


def _source(root: Path) -> tuple[str, dict]:
    identity = {
        "scene_sha256": "scene",
        "camera": {"intrinsics": [1, 2, 3]},
        "width": 128,
        "height": 128,
        "settings": {"spp": 4096, "seed": 0, "threads": 8},
        "reference_implementation_sha256": "generator",
    }
    key = stable_hash(identity)
    entry = root / "references" / key
    entry.mkdir(parents=True)
    records = []
    for name in ("reference.json", "frame_0000.json", "frame_0000.npz"):
        path = entry / name
        path.write_bytes(name.encode())
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kind": "independent_reference",
            }
        )
    write_json(
        entry / "cache_binding.json",
        {"identity": identity, "frame": records[1]["path"], "artifacts": records},
    )
    return key, identity


def test_transfer_preserves_portable_bindings_and_reuses_identical_entry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    key, identity = _source(source)
    assert transfer_reference(source, target, key, identity)
    for path in (source / "references" / key).iterdir():
        assert (
            path.read_bytes() == (target / "references" / key / path.name).read_bytes()
        )
    assert transfer_reference(source, target, key, identity)
    changed = {**identity, "width": 256}
    with pytest.raises(ValueError, match="key"):
        transfer_reference(source, tmp_path / "other", key, changed)


def test_corrupt_source_never_publishes_a_partial_entry(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    key, identity = _source(source)
    (source / "references" / key / "frame_0000.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        transfer_reference(source, target, key, identity)
    assert not target.exists()


def test_reference_cannot_escape_its_immutable_entry(tmp_path: Path) -> None:
    source = tmp_path / "source"
    key, identity = _source(source)
    path = source / "references" / key / "cache_binding.json"
    binding = read_json_object(path)
    binding["artifacts"][0]["path"] = "../outside.json"
    write_json(path, binding)
    with pytest.raises(ValueError, match="escapes"):
        transfer_reference(source, tmp_path / "target", key, identity)


def test_conflicting_target_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    key, identity = _source(source)
    destination = target / "references" / key
    destination.mkdir(parents=True)
    (destination / "frame_0000.npz").write_bytes(b"original target")
    with pytest.raises(ValueError, match="conflicts"):
        transfer_reference(source, target, key, identity)
    assert (destination / "frame_0000.npz").read_bytes() == b"original target"
