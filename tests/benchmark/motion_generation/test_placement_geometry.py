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

"""Shared placement geometry retains grasp and support-clearance semantics."""

from __future__ import annotations

import pytest
import torch

from embodichain_benchmarks.motion_generation.scenarios._placement_geometry import (
    build_flat_object_grasp_pose,
    build_pan_handle_grasp_pose,
    build_placing_object_target_pose,
    build_support_object_target_pose,
    get_pan_basin_vertices,
)
from embodichain_benchmarks.motion_generation.scenarios._placement_qualification import (
    _placement_input_geometry,
)

pytestmark = pytest.mark.no_sim

_CPU = torch.device("cpu")


def test_flat_grasp_uses_transformed_center_and_world_top() -> None:
    vertices = torch.tensor([[-0.1, -0.05, 0.0], [0.1, 0.05, 0.02]])
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    grasp = build_flat_object_grasp_pose(
        pose,
        vertices,
        vertices.min(0).values,
        vertices.max(0).values,
        _CPU,
        z_clearance=0.018,
    )
    torch.testing.assert_close(grasp[:3, 3], torch.tensor([1.0, 2.0, 3.038]))
    rotation = grasp[:3, :3]
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(3), atol=1e-6, rtol=0)
    assert torch.linalg.det(rotation).item() == pytest.approx(1.0, abs=1e-6)
    # Preserve the measured downward tilt rather than replacing it with identity.
    torch.testing.assert_close(
        rotation[:, 2], torch.tensor([-0.0022, -0.0401, -0.9992]), atol=5e-5, rtol=0
    )
    torch.testing.assert_close(pose[:3, 3], torch.tensor([1.0, 2.0, 3.0]))


def test_handle_grasp_points_down_and_moves_toward_handle_root() -> None:
    pan = torch.tensor(
        [[-0.1, -0.1, 0.0], [0.1, 0.1, 0.02], [0.0, 0.3, 0.06], [0.0, 0.5, 0.08]]
    )
    grasp = build_pan_handle_grasp_pose(torch.eye(4), pan, _CPU)
    torch.testing.assert_close(grasp[:3, 3], torch.tensor([0.0, 0.365, 0.086]))
    torch.testing.assert_close(grasp[:3, 2], torch.tensor([0.0, 0.0, -1.0]))
    torch.testing.assert_close(grasp[:3, :3].T @ grasp[:3, :3], torch.eye(3))
    assert torch.linalg.det(grasp[:3, :3]).item() == pytest.approx(1.0)


def test_placing_target_clears_basin_and_ignores_tall_handle() -> None:
    bread = torch.tensor([[-0.1, -0.05, 0.0], [0.1, 0.05, 0.02]])
    pan = torch.tensor([[-0.1, -0.1, 0.0], [0.1, 0.1, 0.02], [0.0, 0.5, 0.08]])
    initial = torch.eye(4)
    support = build_support_object_target_pose(initial, _CPU)
    placing = build_placing_object_target_pose(initial, bread, pan, support, _CPU)
    # The basin top is 0.731 m; the bread bottom retains 6 mm clearance.
    torch.testing.assert_close(support[:3, 3], torch.tensor([-0.06, 0.0, 0.711]))
    torch.testing.assert_close(placing[:3, 3], torch.tensor([-0.06, 0.0, 0.737]))
    torch.testing.assert_close(initial, torch.eye(4))


def test_empty_basin_selection_preserves_full_mesh() -> None:
    pan = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.2, 0.2]])
    torch.testing.assert_close(get_pan_basin_vertices(pan), pan)


def test_placing_target_centers_offset_mesh_footprints_on_the_basin() -> None:
    # The bread's asset origin is displaced 0.3m along X from its mesh center.
    bread = torch.tensor([[0.2, -0.04, 0.0], [0.4, 0.04, 0.02]])
    pan = torch.tensor([[-0.12, -0.12, 0.0], [0.12, 0.12, 0.02]])
    initial = torch.eye(4)
    initial[:3, 3] = torch.tensor([4.0, 3.0, 2.0])
    support = torch.eye(4)
    support[:3, 3] = torch.tensor([1.0, 2.0, 0.5])
    placing = build_placing_object_target_pose(initial, bread, pan, support, _CPU)
    # Center the actual bread footprint at (1,2); bottom sits 6mm above pan top.
    torch.testing.assert_close(placing[:3, 3], torch.tensor([0.7, 2.0, 0.526]))
    torch.testing.assert_close(placing[:3, :3], torch.eye(3))


def test_placing_target_uses_inner_floor_instead_of_rim_height() -> None:
    bread = torch.tensor([[-0.02, -0.02, -0.03], [0.02, 0.02, 0.0]])
    # Two floor layers plus a raised rim vertex: the ray must choose the upper
    # floor surface, not the lower exterior or the basin's maximum height.
    floor = torch.tensor(
        [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]]
    )
    pan = torch.cat(
        (
            floor,
            floor + torch.tensor([0.0, 0.0, 0.005]),
            torch.tensor([[0.1, 0.1, 0.04]]),
        )
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    support = torch.eye(4)
    support[2, 3] = 0.7
    target = build_placing_object_target_pose(
        torch.eye(4), bread, pan, support, _CPU, pan_triangles=faces
    )
    assert target[2, 3].item() == pytest.approx(0.741, abs=1e-6)


def test_placing_target_rejects_missing_center_support_surface() -> None:
    bread = torch.tensor([[-0.02, -0.02, 0.0], [0.02, 0.02, 0.02]])
    pan = torch.tensor(
        [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, -0.09, 0.0], [-0.1, 0.1, 0.0]]
    )
    with pytest.raises(ValueError, match="no supporting mesh surface"):
        build_placing_object_target_pose(
            torch.eye(4),
            bread,
            pan,
            torch.eye(4),
            _CPU,
            pan_triangles=torch.tensor([[0, 1, 2]]),
        )


@pytest.mark.parametrize(
    "condition",
    [
        "supported",
        "overhang",
        "near_overhang",
        "penetrating",
        "hole",
        "thin_hole",
        "handle",
    ],
)
def test_frozen_placement_mesh_audit_checks_full_footprint_and_sampled_gaps(condition):
    """Origins, bounds or a support somewhere under the object are insufficient."""
    bread = torch.tensor(
        [[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0], [0.02, 0.02, 0.0], [-0.02, 0.02, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    pan = bread * 5
    pan_faces = faces.clone()
    placing, support = torch.eye(4)[None], torch.eye(4)[None]
    placing[:, 2, 3] = 0.006
    if condition in {"overhang", "near_overhang"}:
        # Even a real 10 nm overhang must fail the picometre overlay grid.
        placing[:, 0, 3] = 0.095 if condition == "overhang" else 0.08000001
    elif condition == "penetrating":
        placing[:, 2, 3] = -0.001
    elif condition in {"hole", "thin_hole"}:
        inner = bread / 2
        if condition == "thin_hole":
            inner[:, 0] *= 1e-5
        pan = torch.cat((pan, inner))
        pan_faces = torch.tensor(
            [
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ]
        )
    elif condition == "handle":
        pan = torch.cat((pan, bread + torch.tensor([0.2, 0.0, 0.1])))
        pan_faces = torch.cat((faces, faces + 4))
        placing[:, 0, 3] = 0.2
        placing[:, 2, 3] = 0.106
    original = placing.clone()
    result = _placement_input_geometry(bread, faces, pan, pan_faces, placing, support)
    row = result["rows"][0]
    assert row["qualified"] is (condition == "supported")
    if condition in {"overhang", "near_overhang", "hole", "thin_hole", "handle"}:
        assert not row["footprint_covered"]
        assert row["unsupported_area_m2"] > 0
    else:
        assert row["footprint_covered"]
        assert row["unsupported_area_m2"] == 0
        assert row["minimum_sampled_vertical_gap_m"] == pytest.approx(
            0.006 if condition == "supported" else -0.001
        )
    if condition == "handle":
        assert result["basin_triangle_count"] == 2
    assert row["sample_count"] > len(bread)
    assert len(result["meshes"]["placing"]["scaled_vertices_float64_le_sha256"]) == 64
    torch.testing.assert_close(placing, original, rtol=0, atol=0)


def test_placement_mesh_audit_retains_each_transformed_target_row():
    bread = torch.tensor(
        [[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0], [0.02, 0.02, 0.0], [-0.02, 0.02, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    support = torch.eye(4).repeat(2, 1, 1)
    support[1, :3, 3] = torch.tensor([1.0, -2.0, 0.7])
    support[1, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    placing = support.clone()
    placing[:, 2, 3] += 0.006
    result = _placement_input_geometry(bread, faces, bread * 5, faces, placing, support)
    assert [row["env_index"] for row in result["rows"]] == [0, 1]
    assert all(row["qualified"] for row in result["rows"])
    assert [
        row["minimum_sampled_vertical_gap_m"] for row in result["rows"]
    ] == pytest.approx([0.006, 0.006], abs=1e-7)


def test_placement_overlay_covers_an_explicitly_included_support_triangle() -> None:
    # The first support face is exactly the complete placing footprint. Other
    # overlapping faces exposed a GEOS overlay boundary/false-hole regression.
    support = torch.tensor(
        [
            [0.011445822376227668, -0.0213415672648513, 0.0],
            [-0.02159610273972397, -0.038208493535725, 0.0],
            [-0.042596265478847185, 0.07063838905816863, 0.0],
            [0.03952113001942484, -0.058749279891731246, 0.0],
            [0.04599055897515414, -0.08793302096735432, 0.0],
            [0.0758258280248719, -0.014193465619503895, 0.0],
            [-0.09968107017292843, 0.08224718721033761, 0.0],
            [-0.0005873569081919633, -0.06513274093354135, 0.0],
            [0.046434362177007193, 0.07395736233171199, 0.0],
        ],
        dtype=torch.float64,
    )
    placing_pose = torch.eye(4, dtype=torch.float64)[None]
    placing_pose[:, 2, 3] = 0.006
    result = _placement_input_geometry(
        support[:3],
        torch.tensor([[0, 1, 2]]),
        support,
        torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        placing_pose,
        torch.eye(4, dtype=torch.float64)[None],
    )
    row = result["rows"][0]
    assert row["qualified"]
    assert row["footprint_covered"]
    assert row["unsupported_area_m2"] == 0
    assert row["minimum_sampled_vertical_gap_m"] == pytest.approx(0.006)
