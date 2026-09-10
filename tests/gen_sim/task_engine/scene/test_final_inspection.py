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

import numpy as np
import trimesh

from embodichain.gen_sim.task_engine.orchestration.scene_source import (
    scene_revision_id,
)
from embodichain.gen_sim.task_engine.scene.final_inspection import (
    inspect_final_scene,
)


def _scene_export(root: Path, *, scene_id: str, can_rotation: list[float]) -> Path:
    export = root / "scene_export"
    assets = export / "mesh_assets"
    assets.mkdir(parents=True)
    trimesh.creation.box(extents=[1.0, 0.1, 1.0]).export(
        assets / "table.glb", file_type="glb"
    )
    can = trimesh.creation.cylinder(radius=0.04, height=0.2)
    can.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    )
    can.export(assets / "can.glb", file_type="glb")
    config = {
        "format": "embodichain.scene-export/v1",
        "scene_id": scene_id,
        "background": [
            {
                "uid": "table",
                "name": "table",
                "description": "A support table.",
                "category": "table",
                "shape": {"shape_type": "Mesh", "fpath": "mesh_assets/table.glb"},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "rigid_object": [
            {
                "uid": "can",
                "name": "red can",
                "description": "A red can.",
                "category": "can",
                "shape": {"shape_type": "Mesh", "fpath": "mesh_assets/can.glb"},
                "init_pos": [0.0, 0.0, 0.15],
                "init_rot": can_rotation,
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
    }
    path = export / "scene_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_scene_revision_id_ignores_exporter_timestamp_and_location(
    tmp_path: Path,
) -> None:
    first = _scene_export(
        tmp_path / "first", scene_id="scene-100", can_rotation=[0, 0, 0]
    )
    second = _scene_export(
        tmp_path / "second", scene_id="scene-200", can_rotation=[0, 0, 0]
    )

    assert scene_revision_id(first) == scene_revision_id(second)

    value = json.loads(second.read_text(encoding="utf-8"))
    value["rigid_object"][0]["init_pos"][0] = 0.25
    second.write_text(json.dumps(value), encoding="utf-8")
    assert scene_revision_id(first) != scene_revision_id(second)


def test_final_inspection_recomputes_support_and_orientation(tmp_path: Path) -> None:
    source = _scene_export(
        tmp_path / "standing", scene_id="scene", can_rotation=[0.0, 0.0, 0.0]
    )

    inspection = inspect_final_scene(source, revision_id=scene_revision_id(source))

    can = next(item for item in inspection["objects"] if item["uid"] == "can")
    assert can["orientation"] == "standing"
    assert can["support"]["parent_uid"] == "table"
    assert can["support"]["relation"] == "on"
    assert can["support"]["xy_overlap_ratio"] > 0.9


def test_final_inspection_detects_lying_rotation(tmp_path: Path) -> None:
    source = _scene_export(
        tmp_path / "lying", scene_id="scene", can_rotation=[90.0, 0.0, 0.0]
    )

    inspection = inspect_final_scene(source, revision_id=scene_revision_id(source))

    can = next(item for item in inspection["objects"] if item["uid"] == "can")
    assert can["orientation"] == "lying"
