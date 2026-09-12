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

import json
from pathlib import Path

import pytest

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
    import_scene_from_output_root,
)


def _write_scene_export(
    output_root: Path,
    *,
    include_table: bool = True,
    include_asset_mesh: bool = True,
) -> None:
    scene_export_root = output_root / "scene_export"
    table_mesh_path = scene_export_root / "mesh_assets" / "table" / "table.glb"
    asset_mesh_path = scene_export_root / "mesh_assets" / "cup" / "cup.glb"
    table_mesh_path.parent.mkdir(parents=True)
    asset_mesh_path.parent.mkdir(parents=True)
    table_mesh_path.write_bytes(b"glTF-table")
    if include_asset_mesh:
        asset_mesh_path.write_bytes(b"glTF-cup")

    background = []
    if include_table:
        background.append(
            {
                "uid": "table",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": "mesh_assets/table/table.glb",
                },
                "attrs": {"mass_props": {"mass": 1.0}},
                "body_type": "kinematic",
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
                "is_articulated": False,
                "max_convex_hull_num": 16,
            }
        )
    scene_config = {
        "background": background,
        "rigid_object": [
            {
                "uid": "cup",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": "mesh_assets/cup/cup.glb",
                },
                "attrs": {"mass_props": {"mass": 1.0}},
                "body_type": "dynamic",
                "init_pos": [1.0, -3.0, 2.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 2.0, 3.0],
                "is_articulated": False,
                "center_xy": [1.0, -3.0],
                "max_convex_hull_num": 32,
            }
        ],
    }
    (scene_export_root / "scene_config.json").write_text(
        json.dumps(scene_config),
        encoding="utf-8",
    )


def test_import_scene_from_output_root_writes_y_up_scene_json(tmp_path: Path) -> None:
    _write_scene_export(tmp_path)
    (tmp_path / "scene_export" / "scene.json").write_text(
        '{"old": true}',
        encoding="utf-8",
    )

    scene = import_scene_from_output_root(tmp_path)
    scene_json = json.loads(
        (tmp_path / "scene_export" / "scene.json").read_text(encoding="utf-8")
    )

    assert scene.table is not None
    assert scene.table.id == "table"
    assert scene.assets[0].id == "cup"
    assert scene.assets[0].pos == [1.0, 2.0, 3.0]
    assert scene.assets[0].scale == [1.0, 2.0, 3.0]
    assert scene.assets[0].center_xy == [1.0, -3.0]
    assert scene.assets[0].simready_glb_path == str(
        (tmp_path / "scene_export" / "mesh_assets" / "cup" / "cup.glb").resolve()
    )
    assert scene_json["objects"][1]["id"] == "cup"
    assert scene_json["objects"][1]["pos"] == [1.0, 2.0, 3.0]


def test_check_scene_export_for_edit_requires_export_directories(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Output root"):
        import_scene_from_output_root(tmp_path)

    (tmp_path / "scene_export").mkdir()
    with pytest.raises(FileNotFoundError, match="mesh assets"):
        import_scene_from_output_root(tmp_path)


def test_check_scene_export_for_edit_requires_table(tmp_path: Path) -> None:
    _write_scene_export(tmp_path, include_table=False)

    with pytest.raises(ValueError, match="table"):
        import_scene_from_output_root(tmp_path)


def test_check_scene_export_for_edit_requires_rigid_object_glb(
    tmp_path: Path,
) -> None:
    _write_scene_export(tmp_path, include_asset_mesh=False)

    with pytest.raises(FileNotFoundError, match="cup"):
        import_scene_from_output_root(tmp_path)
