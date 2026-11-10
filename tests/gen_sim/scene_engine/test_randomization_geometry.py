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

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from embodichain.gen_sim.scene_engine.randomization._geometry import (
    measure_scene_geometry,
)
from embodichain.lab.sim.scene._support_cpu import SupportSurfaceDetector


def _box(
    extents: tuple[float, float, float], center: tuple[float, float, float]
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def _entry(
    export_root: Path, uid: str, mesh: trimesh.Trimesh | trimesh.Scene
) -> dict[str, Any]:
    path = Path("mesh_assets") / uid / f"{uid}.glb"
    (export_root / path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(export_root / path)
    return {
        "uid": uid,
        "shape": {"fpath": path.as_posix()},
        "init_pos": [0.0, 0.0, 0.0],
        "init_rot": [0.0, 0.0, 0.0],
        "body_scale": [1.0, 1.0, 1.0],
    }


def _table(export_root: Path) -> dict[str, Any]:
    # GLB is Y-up: the 10 cm slab has top height 80 cm and footprint 2 m x 1 m.
    entry = _entry(export_root, "table", _box((2.0, 0.1, 1.0), (0.0, 0.75, 0.0)))
    entry["support_surface_z"] = 0.8
    entry["support_optimization_rect_xy"] = [
        [-0.9, -0.4],
        [0.9, -0.4],
        [0.9, 0.4],
        [-0.9, 0.4],
    ]
    return entry


def test_final_aabb_applies_node_transform_y_up_scale_and_z_up_rotation_once(
    tmp_path: Path,
) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(0.2, 0.3, 0.4)),
        transform=trimesh.transformations.translation_matrix([0.4, 0.15, -0.2]),
    )
    asset = _entry(tmp_path, "asset", scene)
    asset.update(
        body_scale=[2.0, 3.0, 4.0],
        init_rot=[90.0, 0.0, 90.0],
        init_pos=[0.7, -0.2, 0.9],
    )

    measured = measure_scene_geometry([_table(tmp_path), asset], tmp_path)

    # The two intrinsic rotations map (x, y, z) in Y-up GLB to (4z, -3y, 2x).
    expected_bounds = [[-0.9, -1.1, 1.5], [0.7, -0.2, 1.9]]
    np.testing.assert_allclose(
        measured.aabbs_by_uid["asset"], expected_bounds, atol=1e-7
    )
    np.testing.assert_allclose(
        measured.poses_by_uid["asset"][:3, :3],
        Rotation.from_euler("XYZ", asset["init_rot"], degrees=True).as_matrix(),
    )


def test_entity_origin_is_kept_separate_from_aabb_center(tmp_path: Path) -> None:
    asset = _entry(tmp_path, "asset", _box((0.2, 0.3, 0.4), (0.4, 0.15, -0.2)))
    asset["init_pos"] = [0.7, -0.2, 0.9]

    measured = measure_scene_geometry([_table(tmp_path), asset], tmp_path)

    np.testing.assert_allclose(measured.poses_by_uid["asset"][:3, 3], asset["init_pos"])
    np.testing.assert_allclose(
        measured.aabbs_by_uid["asset"].mean(axis=0), [1.1, 0.0, 1.05], atol=1e-7
    )


def test_missing_table_metadata_uses_shared_detector(tmp_path: Path) -> None:
    table = _table(tmp_path)
    del table["support_surface_z"]
    del table["support_optimization_rect_xy"]

    measured = measure_scene_geometry([table], tmp_path)

    assert measured.table["entity_id"] == "table"
    np.testing.assert_allclose(measured.table["center_xy"], [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(measured.table["size_xy"], [2.0, 1.0], atol=1e-7)
    assert measured.table["surface_z"] == pytest.approx(0.8)


def test_saved_work_area_survives_float32_glb_round_trip(tmp_path: Path) -> None:
    # SimReady detects support on its double-precision mesh after exporting it.
    # GLB float32 positions round both 1.3 m and 0.7 m edges slightly inward.
    mesh = _box((2.6, 0.1, 1.4), (0.0, 0.05, 0.0))
    table = _entry(tmp_path, "table", mesh)
    y_up_to_z_up = np.eye(4)
    y_up_to_z_up[:3, :3] = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
    mesh.apply_transform(y_up_to_z_up)
    detected = SupportSurfaceDetector(mesh.vertices, mesh.faces).detect()
    table["support_surface_z"] = detected.top_z
    table["support_optimization_rect_xy"] = np.asarray(
        detected.optimization_rectangle.exterior.coords[:-1]
    ).tolist()

    measured = measure_scene_geometry([table], tmp_path)

    # Accept only the numerical boundary error and retain the saved work area.
    np.testing.assert_allclose(
        measured.table["size_xy"], [2.6, 1.4], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        measured.table["center_xy"], [0.0, 0.0], atol=1e-12, rtol=0
    )
    assert measured.table["surface_z"] == table["support_surface_z"]
    assert measured.aabbs_by_uid["table"][1, 0] < 1.3


@pytest.mark.parametrize(
    ("translation_x", "mesh_offset_x"), [(0.0, 0.0), (1e6, 0.0), (0.0, 1e6)]
)
def test_work_area_overhang_above_local_precision_is_rejected(
    tmp_path: Path, translation_x: float, mesh_offset_x: float
) -> None:
    table = _table(tmp_path)
    if mesh_offset_x:
        _entry(tmp_path, "table", _box((2.0, 0.1, 1.0), (mesh_offset_x, 0.75, 0.0)))
    table["init_pos"][0] = translation_x
    # A ten-micrometre overhang exceeds GLB precision for this 2 m table,
    # including large arena coordinates or a large local mesh offset, where the
    # accepted numerical error remains capped at one micrometre.
    center_x = translation_x + mesh_offset_x
    table["support_optimization_rect_xy"] = [
        [center_x - 1.0, -0.4],
        [center_x + 1.0 + 1e-5, -0.4],
        [center_x + 1.0 + 1e-5, 0.4],
        [center_x - 1.0, 0.4],
    ]

    with pytest.raises(ValueError, match="connected support"):
        measure_scene_geometry([table], tmp_path)


@pytest.mark.parametrize(
    "corners",
    [
        [[-1.1, -0.4], [0.9, -0.4], [0.9, 0.4], [-1.1, 0.4]],
        [[-0.9, -0.4], [0.8, -0.3], [0.9, 0.4], [-0.9, 0.4]],
    ],
)
def test_unsafe_or_nonrectangular_saved_work_area_is_rejected(
    tmp_path: Path, corners: list[list[float]]
) -> None:
    table = _table(tmp_path)
    table["support_optimization_rect_xy"] = corners

    with pytest.raises(ValueError, match="rectangle"):
        measure_scene_geometry([table], tmp_path)


def test_table_height_must_match_final_geometry(tmp_path: Path) -> None:
    table = _table(tmp_path)
    table["support_surface_z"] = 0.82

    with pytest.raises(ValueError, match="height"):
        measure_scene_geometry([table], tmp_path)


@pytest.mark.parametrize("kind", ["hole", "gap", "different_height"])
def test_work_area_cannot_cover_holes_gaps_or_different_levels(
    tmp_path: Path, kind: str
) -> None:
    if kind == "hole":
        parts = [
            _box((0.5, 0.1, 2.0), (-0.75, 0.75, 0.0)),
            _box((0.5, 0.1, 2.0), (0.75, 0.75, 0.0)),
            _box((1.0, 0.1, 0.5), (0.0, 0.75, -0.75)),
            _box((1.0, 0.1, 0.5), (0.0, 0.75, 0.75)),
        ]
    else:
        parts = [
            _box((0.9, 0.1, 1.0), (-0.55, 0.75, 0.0)),
            _box(
                (0.9, 0.1, 1.0),
                (0.55, 0.78 if kind == "different_height" else 0.75, 0.0),
            ),
        ]
    table = _table(tmp_path)
    _entry(tmp_path, "table", trimesh.util.concatenate(parts))

    with pytest.raises(ValueError, match="height|connected support"):
        measure_scene_geometry([table], tmp_path)


def test_detached_geometry_outside_the_work_area_is_allowed(tmp_path: Path) -> None:
    table = _table(tmp_path)
    parts = [
        _box((2.0, 0.1, 1.0), (0.0, 0.75, 0.0)),
        _box((0.1, 0.1, 0.1), (1.5, 0.75, 0.0)),
    ]
    _entry(tmp_path, "table", trimesh.util.concatenate(parts))

    measured = measure_scene_geometry([table], tmp_path)

    np.testing.assert_allclose(measured.table["size_xy"], [1.8, 0.8])


def test_nonhorizontal_table_is_rejected(tmp_path: Path) -> None:
    table = _table(tmp_path)
    table["init_rot"] = [0.0, 4.0, 0.0]

    with pytest.raises(ValueError, match="horizontal|height"):
        measure_scene_geometry([table], tmp_path)
