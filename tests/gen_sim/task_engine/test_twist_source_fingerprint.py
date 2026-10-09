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
"""Scene provenance covers optional asset-local twist semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.orchestration.scene_source import (
    _asset_dependency_files,
    fingerprint_scene_source,
    verify_scene_source_fingerprint,
)


def _project(tmp_path: Path):
    source = tmp_path / "device.usda"
    source.write_text("Opaque source fixture; fingerprinting does not parse USD.")
    config = tmp_path / "scene_config.json"
    config.write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "articulation": [{"uid": "unseen", "fpath": source.name}],
            }
        )
    )
    return source, config


def test_source_semantics_are_hashed_as_local_asset_dependencies(tmp_path):
    source, config = _project(tmp_path)
    sidecar = source.with_suffix(".twist.json")
    sidecar.write_text('{"annotation": "original"}')
    assert _asset_dependency_files(source) == tuple(sorted((source, sidecar)))
    fingerprint = fingerprint_scene_source(config)
    assert (
        fingerprint.asset_sha256[sidecar.as_posix()]
        == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    )
    verify_scene_source_fingerprint(fingerprint.to_dict())


@pytest.mark.parametrize("change", ("add", "edit", "remove"))
def test_annotation_changes_invalidate_a_prepared_scene(tmp_path, change):
    source, config = _project(tmp_path)
    sidecar = source.with_suffix(".twist.json")
    if change != "add":
        sidecar.write_text('{"annotation": "original"}')
    old_source = source.read_bytes()
    fingerprint = fingerprint_scene_source(config)
    if change == "remove":
        sidecar.unlink()
    else:
        sidecar.write_text('{"annotation": "changed"}')
    assert source.read_bytes() == old_source
    with pytest.raises(RuntimeError, match="changed after"):
        verify_scene_source_fingerprint(fingerprint.to_dict())


def test_unannotated_assets_keep_the_original_dependency_identity(tmp_path):
    source, config = _project(tmp_path)
    assert _asset_dependency_files(source) == (source,)
    assert set(fingerprint_scene_source(config).asset_sha256) == {source.as_posix()}


def test_non_usd_assets_do_not_acquire_a_twist_sidecar_dependency(tmp_path):
    source = tmp_path / "ordinary.glb"
    source.write_bytes(b"opaque geometry")
    source.with_suffix(".twist.json").write_text("unrelated companion")
    assert _asset_dependency_files(source) == (source,)
