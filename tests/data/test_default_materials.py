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

"""Offline discovery, CLI registration and immutable resource distribution."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from embodichain.data import list_default_materials, get_default_material_info
from embodichain.data.default_materials import _catalog
from embodichain.data.assets.materials import DefaultMaterials
from embodichain.data.download import get_registry

pytestmark = pytest.mark.no_sim


def test_offline_catalog_is_independent_and_includes_geometry_guidance(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Offline metadata must not construct a dataset")

    monkeypatch.setattr(DefaultMaterials, "__init__", fail)
    names = list_default_materials()
    assert len(names) == len(set(names)) == 24
    assert "SoapBubbleInterior" not in names
    assert (
        sum(get_default_material_info(name)["renderers"] == ["rt"] for name in names)
        == 10
    )
    glass = get_default_material_info("GlassClear")
    assert glass["renderers"] == ["rt"]
    assert glass["nominal_size_m"] == [0.5, 0.5]
    glass["renderers"].clear()
    assert get_default_material_info("GlassClear")["renderers"] == ["rt"]
    assert get_default_material_info("SoapBubble")["geometry_notes"]
    assert get_default_material_info("SoapBubbleInterior")["auxiliary"]
    assert (
        sum(not get_default_material_info(name)["requires_download"] for name in names)
        == 6
    )
    builtin = get_default_material_info("PlasticMatteBlue")
    assert builtin["source"] == "builtin"
    assert builtin["texture_resolution"] is None
    assert builtin["nominal_size_m"] is None
    assert builtin["revision"] == "builtin-v1"
    assert builtin["parameters"]["base_color"] == [0.05, 0.22, 0.6, 1]
    builtin["parameters"]["base_color"][0] = 1.0
    assert (
        get_default_material_info("PlasticMatteBlue")["parameters"]["base_color"][0]
        == 0.05
    )
    assert ("DefaultMaterials", DefaultMaterials) in get_registry()["materials"]
    with pytest.raises(ValueError, match="Unknown default material"):
        get_default_material_info("glassclear")


def test_verified_archive_repairs_missing_dependency_in_custom_cache(
    tmp_path, monkeypatch
):
    # Exercise the production constructor, lock and Open3D cache path without
    # depending on a network server in unit tests.
    payload = {
        "Material/SoapBubble/Color.png": b"original pixels",
        "Material/SoapBubbleInterior/material.json": b"{}",
    }
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for directory in (
            "Material/",
            "Material/SoapBubble/",
            "Material/SoapBubbleInterior/",
        ):
            bundle.writestr(directory, b"")
        for name, content in payload.items():
            bundle.writestr(name, content)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    metadata = dict(
        _catalog(),
        bundle={
            **_catalog()["bundle"],
            "revision": "test",
            "archive_name": "fixture.zip",
            "md5": hashlib.md5(archive.read_bytes()).hexdigest(),
            "sha256": digest,
            "upstream_url": str(archive),
            "files": {
                name: {
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for name, content in payload.items()
            },
        },
    )
    monkeypatch.setattr("embodichain.data.assets.materials._catalog", lambda: metadata)
    monkeypatch.setattr(DefaultMaterials, "cache_prefix", "DefaultMaterials_test")
    monkeypatch.delenv("EMBODICHAIN_DOWNLOAD_PREFIX", raising=False)
    # Preseed a valid downloaded archive, as if downloaded on another run.
    root = tmp_path / "custom-cache"
    download = root / "download/DefaultMaterials_test"
    download.mkdir(parents=True)
    (download / "fixture.zip").write_bytes(archive.read_bytes())
    extract = root / "extract/DefaultMaterials_test"
    extract.mkdir(parents=True)
    (extract / "unrelated").write_bytes(b"partial extraction")
    first = DefaultMaterials(str(root))
    texture = Path(first.get_material_path("SoapBubble")) / "Color.png"
    assert texture.read_bytes() == b"original pixels"
    texture.write_bytes(b"corrupted pixels")
    second = DefaultMaterials(str(root))
    assert Path(second.extract_dir) == extract
    assert texture.read_bytes() == b"original pixels"
    assert (extract / "Material/SoapBubbleInterior/material.json").read_bytes() == b"{}"
    with pytest.raises(ValueError, match="built-in parameter preset"):
        second.get_material_path("PlasticMatteBlue")


def test_data_cli_reports_the_versioned_cache_as_ready(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from embodichain.data import download

    monkeypatch.setattr(download, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(tmp_path))
    extracted = tmp_path / "extract" / DefaultMaterials.cache_prefix
    extracted.mkdir(parents=True)
    (extracted / "Material").mkdir()
    download.cmd_list(Namespace(category="materials"))
    assert "[✓] DefaultMaterials" in capsys.readouterr().out
    observed = []
    monkeypatch.setattr(
        download, "_ensure_extract", lambda obj, prefix: observed.append(prefix)
    )

    class CachedMaterials:
        cache_prefix = DefaultMaterials.cache_prefix

    assert download.download_asset("DefaultMaterials", CachedMaterials)
    assert observed == [DefaultMaterials.cache_prefix]
