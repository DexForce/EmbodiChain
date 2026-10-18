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

"""Local provider preparation, dependencies and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodichain.data.asset_catalog import (
    AssetQuery,
    Bounds3D,
    DirectoryAssetProvider,
    ExplicitAssetProvider,
    PreparationContext,
)

# A tetrahedron with unit X/Y/Z bounds is sufficient to verify mesh extent handling.
_MESH = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"


def _package(root: Path, *, category: str = "cup", asset_id: str = "cup-1") -> Path:
    root.mkdir(parents=True)
    (root / "mesh.obj").write_text(_MESH)
    manifest = {
        "descriptor": {
            "asset_id": asset_id,
            "category": category,
            "bounds_m": Bounds3D((0, 0, 0), (1, 1, 1)).to_dict(),
            "normalized": True,
            "capabilities": ["antipodal_grasp"],
        },
        "visual_path": "mesh.obj",
    }
    path = root / "asset.catalog.json"
    path.write_text(json.dumps(manifest))
    return path


def _prepare(provider, tmp_path):
    candidate = provider.candidates(AssetQuery(), limit=10)[0]
    return provider.materialize(
        candidate, context=PreparationContext(str(tmp_path / "cache"))
    )


def test_directory_and_explicit_providers_prepare_identical_cached_package(tmp_path):
    manifest = _package(tmp_path / "assets" / "cup")
    directory = DirectoryAssetProvider(tmp_path / "assets")
    explicit = ExplicitAssetProvider([manifest.parent / "mesh.obj"])
    first = _prepare(directory, tmp_path)
    second = _prepare(explicit, tmp_path)
    assert first == second
    assert first.visual_path == first.collision_path
    assert Path(first.visual_path).read_text() == _MESH
    assert set(first.provenance["files"]) == {"asset.catalog.json", "mesh.obj"}
    assert Path(first.provenance["package_root"]).name == first.content_sha256


def test_selection_does_not_discard_alias_candidates(tmp_path):
    mug = _package(tmp_path / "mug", category="mug", asset_id="mug-1")
    cup = _package(tmp_path / "cup", category="cup", asset_id="cup-1")
    provider = ExplicitAssetProvider([mug, cup])
    candidates = provider.candidates(AssetQuery("cup"), limit=2)
    assert [c.descriptor.category for c in candidates] == ["cup", "mug"]
    with pytest.raises(ValueError, match="positive integer"):
        provider.candidates(AssetQuery(), limit=True)


def test_source_changes_create_new_content_identity(tmp_path):
    manifest = _package(tmp_path / "cup")
    provider = ExplicitAssetProvider([manifest])
    first = _prepare(provider, tmp_path)
    (manifest.parent / "mesh.obj").write_text(_MESH + "# a revised package\n")
    second = _prepare(provider, tmp_path)
    assert first.content_sha256 != second.content_sha256
    assert Path(first.visual_path).read_text() == _MESH


def test_changed_cache_is_rejected(tmp_path):
    provider = ExplicitAssetProvider([_package(tmp_path / "cup")])
    prepared = _prepare(provider, tmp_path)
    Path(prepared.visual_path).write_text("corrupt")
    with pytest.raises(ValueError, match="cache is corrupt"):
        _prepare(provider, tmp_path)


def test_obj_texture_dependencies_are_copied_and_hashed(tmp_path):
    manifest = _package(tmp_path / "cup")
    root = manifest.parent
    (root / "mesh.obj").write_text("mtllib material.mtl\n" + _MESH)
    (root / "material.mtl").write_text("newmtl surface\nmap_Kd color.png\n")
    # Trimesh accepts a real tiny PNG; use Pillow to keep the fixture portable.
    from PIL import Image

    Image.new("RGB", (1, 1), "red").save(root / "color.png")
    provider = ExplicitAssetProvider([manifest])
    prepared = _prepare(provider, tmp_path)
    assert "color.png" in prepared.provenance["files"]
    assert (Path(prepared.provenance["package_root"]) / "color.png").is_file()
    (root / "color.png").unlink()
    with pytest.raises(ValueError, match="Missing package dependency"):
        _prepare(provider, tmp_path)


def test_external_texture_and_package_symlinks_rejected(tmp_path):
    manifest = _package(tmp_path / "cup")
    root = manifest.parent
    (tmp_path / "outside.png").write_bytes(b"bytes")
    (root / "mesh.obj").write_text("mtllib material.mtl\n" + _MESH)
    (root / "material.mtl").write_text("newmtl surface\nmap_Kd ../outside.png\n")
    provider = ExplicitAssetProvider([manifest])
    with pytest.raises(ValueError, match="escapes"):
        _prepare(provider, tmp_path)
    (root / "material.mtl").write_text("newmtl surface\n")
    (root / "link.png").symlink_to(tmp_path / "outside.png")
    with pytest.raises(ValueError, match="symlinks"):
        _prepare(provider, tmp_path)


def test_missing_annotations_and_false_geometry_are_rejected(tmp_path):
    raw = tmp_path / "raw.obj"
    raw.write_text(_MESH)
    with pytest.raises(ValueError, match="annotate normalized geometry"):
        ExplicitAssetProvider([raw])
    manifest = _package(tmp_path / "cup")
    data = json.loads(manifest.read_text())
    data["descriptor"]["bounds_m"]["upper"] = [2, 2, 2]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        _prepare(ExplicitAssetProvider([manifest]), tmp_path)


def test_simready_category_is_imported_but_readiness_is_explicit(tmp_path):
    manifest = _package(tmp_path / "simready")
    (manifest.parent / "asset.json").write_text(
        json.dumps(
            {
                "asset_id": "external-1",
                "identity": {"category": "cup"},
                "simulation": {"sim_ready": True},
                "provenance": {"source": "manual"},
            }
        )
    )
    data = json.loads(manifest.read_text())
    del data["descriptor"]["category"]
    data["simready_metadata"] = "asset.json"
    manifest.write_text(json.dumps(data))
    prepared = _prepare(ExplicitAssetProvider([manifest]), tmp_path)
    assert prepared.descriptor.category == "cup"
    assert prepared.descriptor.provenance["simready"]["asset_id"] == "external-1"
    data["descriptor"]["normalized"] = False
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="explicit normalized"):
        _prepare(ExplicitAssetProvider([manifest]), tmp_path)


def test_gltf_external_dependencies_checked_before_geometry_loading(tmp_path):
    manifest = _package(tmp_path / "gltf")
    data = json.loads(manifest.read_text())
    data["visual_path"] = "mesh.gltf"
    manifest.write_text(json.dumps(data))
    (manifest.parent / "mesh.gltf").write_text(
        json.dumps({"asset": {"version": "2.0"}, "buffers": [{"uri": "missing.bin"}]})
    )
    with pytest.raises(ValueError, match="Missing package dependency"):
        _prepare(ExplicitAssetProvider([manifest]), tmp_path)


def test_cache_must_be_outside_source_package(tmp_path):
    manifest = _package(tmp_path / "cup")
    provider = ExplicitAssetProvider([manifest])
    candidate = provider.candidates(AssetQuery(), limit=1)[0]
    with pytest.raises(ValueError, match="cannot be inside"):
        provider.materialize(
            candidate, context=PreparationContext(str(manifest.parent / "cache"))
        )


def test_prepared_snapshot_reverification_detects_mutation(tmp_path):
    from embodichain.data.asset_catalog import verify_prepared_asset

    provider = ExplicitAssetProvider([_package(tmp_path / "cup")])
    prepared = _prepare(provider, tmp_path)
    verify_prepared_asset(prepared)
    Path(prepared.visual_path).write_text("mutated after preparation")
    with pytest.raises(ValueError, match="files changed"):
        verify_prepared_asset(prepared)
