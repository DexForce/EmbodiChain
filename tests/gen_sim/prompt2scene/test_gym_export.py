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

from embodichain.gen_sim.prompt2scene.workflows.gym_export import (
    _BOTTLE_FRAME_STANDARDIZATION,
    _CAN_FRAME_STANDARDIZATION,
    _CUP_FRAME_STANDARDIZATION,
    _bake_glb_bottom_center_to_origin,
    _glb_to_sim_rotation,
    _load_mesh_as_trimesh,
    _upright_frame_standardization_for_object,
)


def test_upright_frame_standardization_detects_cup_bottle_and_can() -> None:
    cup = _upright_frame_standardization_for_object(
        object_id="paper_cup_1",
        description="white paper cup",
        mesh_path=Path("mesh_assets/paper_cup/paper_cup_1.glb"),
    )
    bottle = _upright_frame_standardization_for_object(
        object_id="interact_bottle_1",
        description="clear plastic bottle",
        mesh_path=Path("mesh_assets/bottle/bottle_1.glb"),
    )
    can = _upright_frame_standardization_for_object(
        object_id="soda_can_1",
        description="red soda can",
        mesh_path=Path("mesh_assets/soda_can/soda_can_1.glb"),
    )
    apple = _upright_frame_standardization_for_object(
        object_id="apple_1",
        description="red apple",
        mesh_path=Path("mesh_assets/apple/apple_1.glb"),
    )

    assert cup == _CUP_FRAME_STANDARDIZATION
    assert cup.is_upper_larger is True
    assert bottle == _BOTTLE_FRAME_STANDARDIZATION
    assert bottle.is_upper_larger is False
    assert can == _CAN_FRAME_STANDARDIZATION
    assert can.is_upper_larger is False
    assert apple is None


@pytest.mark.parametrize(
    ("profile", "lower_radius", "upper_radius", "upper_should_be_larger"),
    [
        (_BOTTLE_FRAME_STANDARDIZATION, 0.09, 0.03, False),
        (_CUP_FRAME_STANDARDIZATION, 0.03, 0.09, True),
        (_CAN_FRAME_STANDARDIZATION, 0.09, 0.03, False),
    ],
)
def test_bake_glb_standardizes_upright_axis_and_keeps_bottom_at_origin(
    tmp_path: Path,
    profile,
    lower_radius: float,
    upper_radius: float,
    upper_should_be_larger: bool,
) -> None:
    source_path = tmp_path / f"{profile.name}_lying.glb"
    output_path = tmp_path / f"{profile.name}_standard.glb"
    _write_frustum_glb(
        source_path,
        lower_radius=lower_radius,
        upper_radius=upper_radius,
    )

    report = _bake_glb_bottom_center_to_origin(
        source_path,
        output_path,
        upright_frame_standardization=profile,
    )

    vertices = _load_sim_vertices(output_path)
    extents = np.ptp(vertices, axis=0)
    lower_extent = _slice_xy_extent(vertices, upper=False)
    upper_extent = _slice_xy_extent(vertices, upper=True)

    assert report is not None
    assert report["profile"] == profile.name
    assert report["status"] == "applied"
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
