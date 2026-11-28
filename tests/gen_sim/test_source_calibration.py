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
from PIL import Image, ImageDraw
import pytest
import trimesh
import yaml

from embodichain.gen_sim._source_calibration import (
    _Camera,
    calibrate_scene,
    load_source_calibration,
)

_CALIBRATION = {
    "schema_version": "embodichain.gensim-source-calibration/v1",
    "image_size": [320, 240],
    "camera": {
        "position": [-0.032, -0.45, 1.35],
        "forward": [0.0, 0.6, -0.8],
        "left": [-1.0, 0.0, 0.0],
        "fovy_deg": 37.0,
    },
    "table": {
        "center_xy": [0.0, 0.0],
        "size_xy": [1.2, 0.7],
        "surface_z": 0.74,
        "thickness": 0.05,
    },
    "scene_yaw_deg": 180.0,
}


def test_projection_matches_a_known_robotwin_object() -> None:
    # RoboTwin beat_block_hammer places the hammer at (0, -0.06, 0.783); its
    # head-camera frame shows the hammer centred near pixel (176, 132).
    pixels, depth = _Camera(_CALIBRATION).project(np.array([[0.0, -0.06, 0.783]]))
    assert depth[0] > 0
    assert pixels[0] == pytest.approx([176.7, 134.6], abs=1.0)


def test_generated_scene_is_rescaled_onto_the_calibrated_tabletop(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(_CALIBRATION))
    calibration = load_source_calibration(calibration_path)
    export, masks = tmp_path / "export", tmp_path / "masks"
    export.mkdir()
    masks.mkdir()
    # The generated scene is three times too large, as a close-up photograph yields.
    trimesh.creation.box(extents=[0.15, 0.15, 0.15]).export(export / "cube.glb")
    trimesh.creation.box(extents=[2.0, 0.1, 1.6]).export(export / "table.glb")
    # Ground truth: a 0.05 m cube resting on the tabletop at (0.1, -0.05).
    truth = trimesh.creation.box(extents=[0.05, 0.05, 0.05]).vertices
    truth = truth + np.array([0.1, -0.05, 0.74 + 0.025 + 0.001])
    pixels, _ = _Camera(calibration).project(truth)
    mask = Image.new("L", (320, 240), 0)
    ImageDraw.Draw(mask).rectangle(
        [*np.floor(pixels.min(axis=0)), *np.ceil(pixels.max(axis=0)) - 1], fill=255
    )
    mask.save(masks / "cube_001_mask.png")
    data = {
        "format": "embodichain.scene-export/v1",
        "background": [
            {
                "uid": "table",
                "shape": {"fpath": "table.glb"},
                "init_rot": [0, 0, 180],
                "max_convex_hull_num": 16,
            }
        ],
        "rigid_object": [
            {
                "uid": "cube_001",
                "shape": {"fpath": "cube.glb"},
                "init_pos": [0.5, 0.5, 0.5],
                "init_rot": [0, 0, 180],
            }
        ],
    }

    result, report = calibrate_scene(data, export, masks, calibration)

    (fit,) = report["objects"]
    assert fit["scale"] == pytest.approx(1 / 3, rel=0.03)
    assert fit["center_error_px"] < 1.0
    cube = result["rigid_object"][0]
    assert cube["init_pos"] == pytest.approx([0.1, -0.05, 0.766], abs=0.01)
    assert cube["init_rot"] == pytest.approx([0, 0, 0], abs=1e-6)
    table = result["background"][0]
    # Z-up (x, y, z) extents (2.0, 1.6, 0.1) become (1.2, 0.7, 0.05); body_scale
    # is authored in the Y-up axes (x, height, depth).
    assert table["body_scale"] == pytest.approx([0.6, 0.5, 0.4375])
    assert table["init_pos"] == pytest.approx([0.0, 0.0, 0.715])
    # V-HACD keeps the collision surface on the calibrated tabletop height and
    # leaves the table its own hull budget.
    collision = table["shape"]["collision"]
    assert collision["approximation"] == "convex_decomposition"
    assert (collision["max_hulls"], collision["acd_method"]) == (16, "vhacd")
    assert data["rigid_object"][0]["init_pos"] == [0.5, 0.5, 0.5]
    # Deployments are written as YAML; numpy scalars must not leak into them.
    yaml.safe_dump(result)


def test_calibration_requires_a_mask_for_every_object(tmp_path: Path) -> None:
    trimesh.creation.box(extents=[1, 0.1, 1]).export(tmp_path / "table.glb")
    trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(tmp_path / "cube.glb")
    data = {
        "background": [{"uid": "table", "shape": {"fpath": "table.glb"}}],
        "rigid_object": [{"uid": "cube_001", "shape": {"fpath": "cube.glb"}}],
    }
    with pytest.raises(ValueError, match="No image mask"):
        calibrate_scene(data, tmp_path, tmp_path, _CALIBRATION)


def test_calibration_rejects_floor_objects_before_reading_assets(
    tmp_path: Path,
) -> None:
    data = {
        "background": [{"uid": "table", "shape": {"fpath": "table.glb"}}],
        "rigid_object": [
            {"uid": "cup", "shape": {"fpath": "cup.glb"}},
            {
                "uid": "dustbin",
                "shape": {"fpath": "dustbin.glb"},
                "placement_support": "floor",
            },
        ],
    }
    with pytest.raises(ValueError, match=r"tabletop.*dustbin"):
        calibrate_scene(data, tmp_path, tmp_path, _CALIBRATION)


def test_calibration_rejects_off_axis_tables_and_articulations(
    tmp_path: Path,
) -> None:
    trimesh.creation.box(extents=[1, 0.1, 1]).export(tmp_path / "table.glb")
    data = {"background": [{"uid": "table", "shape": {"fpath": "table.glb"}}]}
    # A box table yawed by 45 degrees cannot match a world-axis tabletop size.
    with pytest.raises(ValueError, match="aligned with the world axes"):
        calibrate_scene(
            data, tmp_path, tmp_path, {**_CALIBRATION, "scene_yaw_deg": 45.0}
        )
    # Exported articulations carry a native USD root and a proxy mesh.
    drawer = {"uid": "drawer_001", "fpath": "drawer.usdc", "proxy_glb_fpath": "d.glb"}
    with pytest.raises(ValueError, match="rigid objects only"):
        calibrate_scene(
            {**data, "articulation": [drawer]}, tmp_path, tmp_path, _CALIBRATION
        )


def test_incomplete_calibration_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({**_CALIBRATION, "table": {}}))
    with pytest.raises(ValueError, match="missing camera, table"):
        load_source_calibration(path)
