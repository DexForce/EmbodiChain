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

import numpy as np
import pytest

from embodichain.gen_sim.prompt2scene.workflows.asset_orientation_normalization import (
    asset_orientation_is_upper_larger,
    export_z_axis_normalized_asset,
    match_asset_orientation_keyword,
)
from embodichain.gen_sim.prompt2scene.workflows.gym_export import (
    _glb_to_sim_rotation,
    _load_mesh_as_trimesh,
)


@pytest.mark.parametrize(
    (
        "object_id",
        "name",
        "description",
        "expected_keyword",
        "expected_is_upper_larger",
    ),
    [
        (
            "interact_bottle_1",
            "bottle",
            "clear plastic bottle",
            "bottle",
            False,
        ),
        ("soda_can_1", "soda can", "red soda can", "can", False),
        (
            "interact_canned_food_0",
            "canned food",
            "cylindrical canned food",
            "canned food",
            False,
        ),
        (
            "interact_paper_cup_0",
            "paper cup",
            "tapered disposable paper cup",
            "paper cup",
            True,
        ),
        ("interact_cup_0", "cup", "plain cup", "cup", True),
        ("interact_object_0", "纸杯", "一次性纸杯", "纸杯", True),
        ("apple_1", "apple", "red apple", None, False),
    ],
)
def test_z_axis_normalization_keyword_resolves_upright_asset_policy(
    object_id: str,
    name: str,
    description: str,
    expected_keyword: str | None,
    expected_is_upper_larger: bool,
) -> None:
    keyword = match_asset_orientation_keyword(
        object_id=object_id,
        name=name,
        description=description,
    )

    assert keyword == expected_keyword
    assert asset_orientation_is_upper_larger(keyword) is expected_is_upper_larger


@pytest.mark.parametrize(
    (
        "name",
        "is_upper_larger",
        "lower_radius",
        "upper_radius",
        "upper_should_be_larger",
    ),
    [
        ("bottle", False, 0.06, 0.02, False),
        ("cup", True, 0.02, 0.06, True),
        ("can", False, 0.06, 0.02, False),
    ],
)
def test_export_z_axis_normalized_asset_keeps_bottom_at_origin(
    tmp_path: Path,
    name: str,
    is_upper_larger: bool,
    lower_radius: float,
    upper_radius: float,
    upper_should_be_larger: bool,
) -> None:
    source_path = tmp_path / f"{name}_lying.glb"
    output_path = tmp_path / f"{name}_standard.glb"
    _write_frustum_glb(
        source_path,
        lower_radius=lower_radius,
        upper_radius=upper_radius,
    )

    result = export_z_axis_normalized_asset(
        source_path,
        output_path,
        glb_to_sim_rotation=_glb_to_sim_rotation(),
        is_upper_larger=is_upper_larger,
    )

    vertices = _load_sim_vertices(output_path)
    extents = np.ptp(vertices, axis=0)
    lower_extent = _slice_xy_extent(vertices, upper=False)
    upper_extent = _slice_xy_extent(vertices, upper=True)

    assert len(result.init_pos) == 3
    assert len(result.init_rot) == 3
    assert vertices[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert extents[2] > extents[:2].max() * 4.0
    if upper_should_be_larger:
        assert upper_extent > lower_extent * 1.5
    else:
        assert upper_extent * 1.5 < lower_extent


def _write_frustum_glb(
    path: Path,
    *,
    lower_radius: float,
    upper_radius: float,
    height: float = 0.6,
    segments: int = 32,
) -> None:
    import trimesh

    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    vertices_sim = []
    for x_value, radius in (
        (-height / 2.0, lower_radius),
        (height / 2.0, upper_radius),
    ):
        for angle in angles:
            vertices_sim.append(
                [x_value, radius * np.cos(angle), radius * np.sin(angle)]
            )
    lower_center_index = len(vertices_sim)
    vertices_sim.append([-height / 2.0, 0.0, 0.0])
    upper_center_index = len(vertices_sim)
    vertices_sim.append([height / 2.0, 0.0, 0.0])

    faces = []
    for index in range(segments):
        next_index = (index + 1) % segments
        lower_a = index
        lower_b = next_index
        upper_a = index + segments
        upper_b = next_index + segments
        faces.append([lower_a, lower_b, upper_a])
        faces.append([lower_b, upper_b, upper_a])
        faces.append([lower_center_index, lower_a, lower_b])
        faces.append([upper_center_index, upper_b, upper_a])

    basis = _glb_to_sim_rotation()
    vertices_glb = np.asarray(vertices_sim, dtype=np.float64) @ basis
    mesh = trimesh.Trimesh(vertices=vertices_glb, faces=faces, process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def _load_sim_vertices(path: Path) -> np.ndarray:
    mesh = _load_mesh_as_trimesh(path)
    basis = _glb_to_sim_rotation()
    return (basis @ np.asarray(mesh.vertices, dtype=np.float64).T).T


def _slice_xy_extent(vertices: np.ndarray, *, upper: bool) -> float:
    z_values = vertices[:, 2]
    z_min = float(z_values.min())
    z_max = float(z_values.max())
    threshold = z_min + (z_max - z_min) * (0.8 if upper else 0.2)
    selected = (
        vertices[z_values > threshold] if upper else vertices[z_values < threshold]
    )
    extents = np.ptp(selected[:, :2], axis=0)
    return float(extents.max())
