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

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

trimesh = pytest.importorskip("trimesh")

BOX_VERTEX_COUNT = 8
CONCATENATED_BOX_VERTEX_COUNT = BOX_VERTEX_COUNT * 2
DEFAULT_VISUAL_RESULT: dict[str, Any] = {
    "visual_category": "None",
    "material_kind": None,
    "material": {"textures": {}},
    "uv_present": False,
    "texture_count_total": 0,
}


def _import_ingest_utils():
    return importlib.import_module(
        "embodichain.gen_sim.simready_pipeline.utils.ingest_utils"
    )


def _write_box_obj(path: Path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.export(path)


def test_load_one_trimesh_uses_first_scene_geometry(monkeypatch) -> None:
    ingest_utils = _import_ingest_utils()
    first_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    scene = trimesh.Scene({"first": first_box, "second": second_box})
    monkeypatch.setattr(ingest_utils.trimesh, "load_mesh", lambda _: scene)

    mesh = ingest_utils.load_one_trimesh("unused.obj", scene_mesh_strategy="first")

    assert len(mesh.vertices) == BOX_VERTEX_COUNT


def test_load_one_trimesh_concatenates_scene_geometry(monkeypatch) -> None:
    ingest_utils = _import_ingest_utils()
    first_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    second_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    scene = trimesh.Scene({"first": first_box, "second": second_box})
    monkeypatch.setattr(ingest_utils.trimesh, "load_mesh", lambda _: scene)

    mesh = ingest_utils.load_one_trimesh(
        "unused.obj", scene_mesh_strategy="concatenate"
    )

    assert len(mesh.vertices) == CONCATENATED_BOX_VERTEX_COUNT


def test_trimesh_parse_ingest_writes_canonical_obj(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ingest_utils = _import_ingest_utils()
    source_file = tmp_path / "source.obj"
    asset_source = tmp_path / "asset_source"
    _write_box_obj(source_file)
    monkeypatch.setattr(
        ingest_utils,
        "classify_visual",
        lambda _: DEFAULT_VISUAL_RESULT,
    )

    result = ingest_utils.trimesh_parse_ingest(
        source_file=source_file,
        asset_source=asset_source,
        obj_name="asset.obj",
        config={
            "visual": {"default_face_color": [128, 128, 128, 255]},
            "export": {
                "include_normals": True,
                "include_color": True,
                "include_texture": True,
                "write_texture": False,
            },
        },
    )

    assert (asset_source / "asset.obj").is_file()
    assert result["visual_ingest"] == "no visual"
    assert result["visual_source"]["visual_category"] == "None"
    assert result["visual_source"]["uv_present"] is False
    assert result["visual_source"]["textures"] == {}


def test_trimesh_parse_ingest_passes_export_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ingest_utils = _import_ingest_utils()
    source_file = tmp_path / "source.obj"
    asset_source = tmp_path / "asset_source"
    captured_export_kwargs: dict[str, Any] = {}
    _write_box_obj(source_file)
    monkeypatch.setattr(
        ingest_utils,
        "classify_visual",
        lambda _: DEFAULT_VISUAL_RESULT,
    )

    def fake_export_obj(mesh, **kwargs):
        captured_export_kwargs.update(kwargs)
        return "o asset\n", {}

    monkeypatch.setattr(
        ingest_utils.trimesh.exchange.obj, "export_obj", fake_export_obj
    )

    ingest_utils.trimesh_parse_ingest(
        source_file=source_file,
        asset_source=asset_source,
        obj_name="asset.obj",
        config={
            "mtl_name": "custom_asset.mtl",
            "export": {
                "include_normals": False,
                "include_color": False,
                "include_texture": False,
                "write_texture": True,
            },
        },
    )

    assert captured_export_kwargs["mtl_name"] == "custom_asset.mtl"
    assert captured_export_kwargs["include_normals"] is False
    assert captured_export_kwargs["include_color"] is False
    assert captured_export_kwargs["include_texture"] is False
    assert captured_export_kwargs["write_texture"] is True
