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
from scipy.spatial.transform import Rotation

from embodichain.gen_sim.scene_engine.core.asset import Asset
from embodichain.gen_sim.scene_engine.core.scene import Scene
from embodichain.gen_sim.scene_engine.core.table import Table
from embodichain.gen_sim.scene_engine.pipeline import scene_export


def test_export_scene_copies_meshes_and_converts_y_up_layout(tmp_path: Path) -> None:
    table_glb = tmp_path / "source_table.glb"
    asset_glb = tmp_path / "source_cup.glb"
    table_glb.write_bytes(b"glTFtable")
    asset_glb.write_bytes(b"glTFasset")
    table = Table(
        id="table",
        category="table",
        name="table",
        description="A table.",
        simready_glb_path=str(table_glb),
        rot=[0.0, 0.0, 0.0],
        pos=[0.0, 0.0, 0.0],
        scale=[1.0, 1.0, 1.0],
    )
    asset = Asset(
        id="cup",
        category="cup",
        name="cup",
        description="A cup.",
        simready_glb_path=str(asset_glb),
        rot=[20.0, -35.0, 40.0],
        pos=[1.0, 2.0, 3.0],
        scale=[1.0, 2.0, 3.0],
    )

    config_path = scene_export.export_scene(
        scene=Scene(table=table, assets=[asset]),
        output_root=tmp_path / "output",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    exported_asset = config["rigid_object"][0]

    assert config["format"] == "embodichain.scene-export/v1"
    assert "robot" not in config
    assert "env" not in config
    assert exported_asset["init_pos"] == [1.0, -3.0, 2.0]
    assert exported_asset["body_scale"] == [1.0, 2.0, 3.0]
    assert (
        config_path.parent / "mesh_assets" / "table" / "table.glb"
    ).read_bytes() == b"glTFtable"
    assert (
        config_path.parent / "mesh_assets" / "cup" / "cup.glb"
    ).read_bytes() == b"glTFasset"

    expected_rotation = (
        scene_export._Y_UP_TO_Z_UP_ROTATION
        @ Rotation.from_euler("xyz", asset.rot, degrees=True).as_matrix()
        @ scene_export._Y_UP_TO_Z_UP_ROTATION.T
    )
    actual_rotation = Rotation.from_euler(
        "XYZ", exported_asset["init_rot"], degrees=True
    ).as_matrix()
    np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-8)
