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

"""Scene-export file checks, not semantic or physical qualification.

Set GEN_SIM_SCENE_EXPORT_MANIFEST to a JSON file containing an explicit list
of export directories to audit external data. Relative entries are resolved
against the manifest directory. Missing entries fail rather than being dropped.
Ordinary tests build their own files and do not require a local scene corpus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _manifest_exports(manifest: Path) -> list[Path]:
    manifest = manifest.expanduser().resolve()
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(entries, list)
        or not entries
        or any(not isinstance(entry, str) or not entry.strip() for entry in entries)
    ):
        raise ValueError("Scene manifest must be a non-empty list of directory paths.")
    exports = [(manifest.parent / entry).resolve() for entry in entries]
    if len(set(exports)) != len(exports):
        raise ValueError("Scene manifest contains duplicate export directories.")
    return exports


def _assert_export_files(export: Path) -> None:
    documents = {
        name: json.loads((export / name).read_text(encoding="utf-8"))
        for name in ("scene_config.json", "scene.json", "scene_graph.json")
    }
    assert all(isinstance(document, dict) for document in documents.values())
    assert documents["scene_config.json"]["format"] == "embodichain.scene-export/v1"

    asset_paths: list[str] = []

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"fpath", "urdf_path", "usd_path"} and isinstance(child, str):
                    asset_paths.append(child)
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(documents["scene_config.json"])
    assert asset_paths
    missing = [path for path in asset_paths if not (export / path).is_file()]
    assert not missing, f"Missing scene assets in {export}: {missing}"


@pytest.fixture
def scene_export(tmp_path: Path) -> Path:
    export = tmp_path / "sample_scene"
    export.mkdir()
    (export / "asset.obj").write_text("v 0 0 0\n", encoding="utf-8")
    documents = {
        "scene_config.json": {
            "format": "embodichain.scene-export/v1",
            "rigid_object": [{"uid": "sample_object", "shape": {"fpath": "asset.obj"}}],
        },
        "scene.json": {"objects": [{"id": "sample_object"}]},
        "scene_graph.json": {"nodes": [{"id": "sample_object"}]},
    }
    for name, document in documents.items():
        (export / name).write_text(json.dumps(document), encoding="utf-8")
    return export


def test_export_file_check_is_independent_of_working_directory(
    scene_export: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _assert_export_files(scene_export)


@pytest.mark.parametrize(
    "missing", ["scene.json", "scene_graph.json", "scene_config.json", "asset.obj"]
)
def test_export_file_check_rejects_missing_inputs(
    scene_export: Path, missing: str
) -> None:
    (scene_export / missing).unlink()
    with pytest.raises((FileNotFoundError, AssertionError)):
        _assert_export_files(scene_export)


@pytest.mark.parametrize("entries", [[], {}, [""], [1], ["scene", "./scene"]])
def test_manifest_rejects_invalid_or_duplicate_entries(
    tmp_path: Path, entries: object
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries), encoding="utf-8")
    with pytest.raises(ValueError):
        _manifest_exports(manifest)


def test_manifest_can_explicitly_select_a_valid_export(
    scene_export: Path, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([scene_export.name]), encoding="utf-8")
    exports = _manifest_exports(manifest)
    assert exports == [scene_export]
    _assert_export_files(exports[0])


def test_manifest_resolves_relative_paths_and_keeps_missing_entries(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(["missing_scene"]), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    assert _manifest_exports(manifest) == [tmp_path / "missing_scene"]
    with pytest.raises(FileNotFoundError):
        _assert_export_files(_manifest_exports(manifest)[0])


_manifest = os.environ.get("GEN_SIM_SCENE_EXPORT_MANIFEST")


@pytest.mark.parametrize(
    "export", _manifest_exports(Path(_manifest)) if _manifest else [None]
)
def test_explicit_external_scene_export_files(export: Path | None) -> None:
    if export is None:
        pytest.skip(
            "Set GEN_SIM_SCENE_EXPORT_MANIFEST to audit an explicit external corpus."
        )
    _assert_export_files(export)
