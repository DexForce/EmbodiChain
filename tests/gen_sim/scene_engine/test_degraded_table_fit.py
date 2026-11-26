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

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, box

from embodichain.gen_sim.scene_engine.pipeline.utils.degraded_table_fit import (
    fit_degraded_table,
    tabletop_level_rotation,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.table_support_surface import (
    TableSupportRegion,
)


def _replacement_table() -> trimesh.Trimesh:
    top = trimesh.creation.box(
        extents=[1.2, 0.8, 0.04],
        transform=trimesh.transformations.translation_matrix([0.2, 0.1, 0.83]),
    )
    legs = [
        trimesh.creation.box(
            extents=[0.05, 0.05, 0.76],
            transform=trimesh.transformations.translation_matrix([x, y, 0.43]),
        )
        for x in (-0.3, 0.7)
        for y in (-0.25, 0.45)
    ]
    return trimesh.util.concatenate([top, *legs])


def test_reference_plane_levels_with_either_triangle_winding() -> None:
    # An open, tilted mesh provides an unsigned geometric normal only.
    tilted = Rotation.from_euler("x", 21, degrees=True)
    mesh = trimesh.Trimesh(
        vertices=tilted.apply([[0, 0, 0], [1.4, 0, 0], [1.4, 0.8, 0], [0, 0.8, 0]]),
        faces=[[0, 2, 1], [0, 3, 2]],
        process=False,
    )
    correction = tabletop_level_rotation(mesh, complete=False)

    np.testing.assert_allclose(
        mesh.vertices @ correction.T,
        [[0, 0, 0], [1.4, 0, 0], [1.4, 0.8, 0], [0, 0.8, 0]],
        atol=1e-8,
    )
    mesh.invert()
    np.testing.assert_allclose(
        tabletop_level_rotation(mesh, complete=False), correction, atol=1e-8
    )


@pytest.mark.parametrize("roll_degrees", [0, 180])
def test_complete_table_levels_with_legs_below_top(roll_degrees: int) -> None:
    upright = _replacement_table()
    mesh = upright.copy()
    orientation = Rotation.from_euler(
        "xyz", [roll_degrees, 17, 0], degrees=True
    ).as_matrix()
    mesh.vertices = upright.vertices @ orientation.T
    correction = tabletop_level_rotation(mesh, complete=True)

    level_vertices = mesh.vertices @ correction.T
    # The full tabletop top was 0.8 m higher than the four feet before rotation.
    assert np.ptp(level_vertices[:8, 2]) == pytest.approx(0.04)
    assert level_vertices[:8, 2].max() - level_vertices[8:, 2].min() == pytest.approx(
        0.8
    )
    assert level_vertices[:8, 2].min() >= level_vertices[8:, 2].max() - 1e-8
    assert np.linalg.det(correction) == pytest.approx(1.0)


def test_fit_covers_reference_and_shares_ground_lift_with_objects() -> None:
    mesh = _replacement_table()
    table_faces = mesh.faces[
        (mesh.face_normals[:, 2] > 0.9) & (mesh.triangles_center[:, 2] > 0.84)
    ]
    rectangle = box(-0.4, -0.3, 0.8, 0.5)
    region = TableSupportRegion(
        top_z=0.85,
        vertices=mesh.vertices.copy(),
        faces=table_faces,
        support_polygon=rectangle,
        optimization_rectangle=rectangle,
    )
    reference_bounds = np.array([[0.0, -0.8], [0.8, 0.8]])

    fit = fit_degraded_table(
        replacement_mesh_z_up=mesh,
        replacement_support=region,
        reference_bounds_xy=reference_bounds,
        reference_top_z=0.08,
        grounded_tabletop_height_m=0.75,
    )

    # Width shrinks, depth doubles, and the requested height is independent.
    assert fit.scale_xyz == pytest.approx((2.0 / 3.0, 2.0, 0.9375))
    assert fit.aligned_bottom_z == pytest.approx(-0.67)
    assert fit.ground_shift_z == pytest.approx(0.67)
    assert fit.mesh_z_up.bounds[0, 2] == pytest.approx(0.0)
    assert fit.mesh_z_up.bounds[1, 2] == pytest.approx(0.75)
    assert fit.support_region.top_z == pytest.approx(0.75)
    assert fit.support_region.support_polygon.covers(box(*reference_bounds.ravel()))
    assert fit.support_region.optimization_rectangle.centroid.coords[
        0
    ] == pytest.approx((0.4, 0.0))
    np.testing.assert_allclose(fit.support_region.vertices, fit.mesh_z_up.vertices)
    np.testing.assert_array_equal(fit.support_region.faces, table_faces)
    object_bottom_z = 0.11 + fit.ground_shift_z
    assert object_bottom_z - fit.support_region.top_z == pytest.approx(0.03)
    assert mesh.bounds[0, 2] == pytest.approx(0.05)


def test_fit_rejects_a_nominal_rectangle_covering_a_support_hole() -> None:
    mesh = _replacement_table()
    rectangle = box(-0.4, -0.3, 0.8, 0.5)
    cutout = box(0.15, 0.05, 0.25, 0.15)
    region = TableSupportRegion(
        top_z=0.85,
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[:2],
        support_polygon=Polygon(rectangle.exterior, [cutout.exterior]),
        optimization_rectangle=rectangle,
    )
    with pytest.raises(ValueError, match="cannot cover"):
        fit_degraded_table(
            replacement_mesh_z_up=mesh,
            replacement_support=region,
            reference_bounds_xy=np.array([[0.0, -0.8], [0.8, 0.8]]),
            reference_top_z=0.08,
            grounded_tabletop_height_m=0.75,
        )
