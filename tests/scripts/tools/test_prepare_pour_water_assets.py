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

"""Network-free coverage of pour-water mesh adaptation and selective download."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import trimesh

from scripts.tools import prepare_pour_water_assets as preparation


def test_zip_reader_fetches_only_requested_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise zipfile's end-relative seeks and member reads through HTTP ranges."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("visual/model.obj", b"mesh data" * 100)
        archive.writestr("unrelated.bin", bytes(range(256)) * 1000)
    raw = buffer.getvalue()
    requested = []

    def open_range(request: object, timeout: int) -> io.BytesIO:
        start, end = map(
            int, request.get_header("Range").removeprefix("bytes=").split("-")
        )
        requested.append((start, end))
        response = io.BytesIO(raw[start : end + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(raw)}"}
        return response

    monkeypatch.setattr(preparation.urllib.request, "urlopen", open_range)
    with zipfile.ZipFile(
        preparation._RangeFile("https://example.org/assets.zip", len(raw))
    ) as archive:
        assert archive.read("visual/model.obj") == b"mesh data" * 100
    assert requested
    assert all(end - start + 1 < len(raw) for start, end in requested)


def test_zip_reader_refuses_ignored_range(monkeypatch: pytest.MonkeyPatch) -> None:
    response = io.BytesIO(b"whole archive")
    response.status = 200
    response.headers = {}
    monkeypatch.setattr(
        preparation.urllib.request, "urlopen", lambda *args, **kwargs: response
    )
    with pytest.raises(RuntimeError, match="refusing a full archive"):
        preparation._RangeFile("https://example.org/assets.zip", 100).read(10)


def test_upload_archive_creates_parent_directories_before_nested_files(
    tmp_path: Path,
) -> None:
    """Open3D's extractor needs directory records, unlike Python extractall."""
    output = tmp_path / "assets"
    source = output / "source" / "cup" / "visual"
    source.mkdir(parents=True)
    (source / "model.obj").write_bytes(b"source mesh")
    (output / "cup.glb").write_bytes(b"adapted mesh")
    path = tmp_path / "PourWaterAssets.zip"
    preparation._write_archive(output, path)
    with zipfile.ZipFile(path) as archive:
        entries = archive.namelist()
        for directory in ("source/", "source/cup/", "source/cup/visual/"):
            assert archive.getinfo(directory).is_dir()
            assert entries.index(directory) < entries.index(
                "source/cup/visual/model.obj"
            )
        assert archive.read("source/cup/visual/model.obj") == b"source mesh"
        assert archive.read("cup.glb") == b"adapted mesh"
        assert archive.testzip() is None


def test_upload_archive_cannot_include_itself(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the asset directory"):
        preparation._write_archive(tmp_path, tmp_path / "bundle.zip")


@pytest.mark.parametrize("kind", ["bottle", "cup"])
def test_object_conversion_preserves_task_origin_and_embeds_materials(
    tmp_path: Path, kind: str
) -> None:
    """Check actual exported glTF bounds and UV-backed material persistence."""
    relative, height, bottom, diameter = preparation._OBJECTS[kind]
    source = tmp_path / "source"
    directory = source / "objaverse" / relative
    visual = directory / "visual"
    visual.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 2.0])
    mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.vertices[:, :2] + 0.5)
    mesh.export(visual / "mesh.obj")
    Image.new("RGB", (8, 8), (128, 32, 16)).save(visual / "color.png")
    (directory / "model.xml").write_text(
        '<mujoco><asset><mesh name="mesh_vis" file="visual/mesh.obj" scale="0.1 0.1 0.1"/>'
        '<texture name="texture" file="visual/color.png"/>'
        '<material name="paint" texture="texture"/></asset><worldbody><body>'
        '<geom group="1" mesh="mesh_vis" material="paint"/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    metadata = preparation._build_object(source, tmp_path, kind)
    scene = trimesh.load(tmp_path / f"{kind}.glb", force="scene")
    # Apply the same glTF Y-up -> simulation Z-up conversion as DexSim.
    scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    np.testing.assert_allclose(scene.bounds[:, 2], [bottom, bottom + height], atol=1e-7)
    np.testing.assert_allclose(scene.extents[:2], [diameter, diameter], atol=1e-7)
    converted = next(iter(scene.geometry.values()))
    assert converted.visual.uv is not None
    assert converted.visual.material.baseColorTexture.size == (8, 8)
    assert converted.visual.material.metallicFactor == 0.0
    assert metadata["triangles"] == 12


def test_table_preserves_support_height_and_packs_roughness(tmp_path: Path) -> None:
    Image.new("RGB", (8, 8), (70, 35, 20)).save(tmp_path / "wood_table_001_diff_1k.jpg")
    Image.new("L", (8, 8), 160).save(tmp_path / "wood_table_001_rough_1k.jpg")
    preparation._build_table(tmp_path, tmp_path)
    scene = trimesh.load(tmp_path / "table.glb", force="scene")
    scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    np.testing.assert_allclose(
        scene.bounds, [[-0.5, -0.5, -0.05], [0.5, 0.5, 0.05]], atol=1e-7
    )
    top = scene.geometry["wood_top"]
    assert np.allclose(top.vertices[:, 2], 0.05)
    packed = np.asarray(top.visual.material.metallicRoughnessTexture)
    np.testing.assert_array_equal(packed[0, 0], [255, 160, 0])
