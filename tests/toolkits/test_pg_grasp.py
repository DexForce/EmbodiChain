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

"""Regression tests for mesh-center and grasp-region selection."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import open3d as o3d
import pytest
import torch

from embodichain.toolkits.graspkit.pg_grasp._antipodal_backend import (
    _AntipodalMeshBackend,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import AntipodalSampler


def _box_geometry(*, subdivide_end: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep a 20 x 4 x 4 cm closed box while refining only its positive X face."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.04, depth=0.04)
    vertices = (np.asarray(mesh.vertices) - np.array([0.1, 0.02, 0.02])).tolist()
    triangles = np.asarray(mesh.triangles).tolist()
    if subdivide_end:
        # Interior barycenters preserve face boundaries and the closed surface.
        for _ in range(4):
            refined = []
            for triangle in triangles:
                corners = np.asarray([vertices[index] for index in triangle])
                if np.allclose(corners[:, 0], 0.1):
                    center_index = len(vertices)
                    vertices.append(corners.mean(axis=0).tolist())
                    a, b, c = triangle
                    refined.extend(
                        [
                            [a, b, center_index],
                            [b, c, center_index],
                            [c, a, center_index],
                        ]
                    )
                else:
                    refined.append(triangle)
            triangles = refined
    return torch.tensor(vertices, dtype=torch.float32), torch.tensor(triangles)


def _prepared_backend(
    vertices: torch.Tensor, pairs: torch.Tensor
) -> _AntipodalMeshBackend:
    """Bypass sampling and collision construction to test prepared-pair logic."""
    backend = _AntipodalMeshBackend.__new__(_AntipodalMeshBackend)
    backend.device = vertices.device
    backend.vertices = vertices
    backend._hit_point_pairs = pairs
    return backend


def _object_pose() -> torch.Tensor:
    """Rotate local X onto world Y and translate away from the origin."""
    return torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.3],
            [1.0, 0.0, 0.0, 7.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_default_grasp_mode_ranks_by_transformed_bounding_box_center() -> None:
    vertices, _ = _box_geometry(subdivide_end=True)
    # One pair is at the geometric center; another is near the densely meshed end.
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
    contact_offset = torch.tensor([0.0, 0.02, 0.0])
    pairs = torch.stack([centers - contact_offset, centers + contact_offset], dim=1)
    backend = _prepared_backend(vertices, pairs)
    backend._max_deviation_angle = 0.1
    backend._approach_direction_samples = 1
    backend._max_candidates = 10
    backend._filter_ground_collision = False
    backend._collision_checker = Mock()
    backend._collision_checker.query.return_value = (
        torch.zeros(len(pairs), dtype=torch.bool),
        torch.zeros(len(pairs)),
    )
    object_pose = _object_pose()

    success, poses, widths, costs = backend.get_valid_grasp_poses(
        object_pose=object_pose,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )

    assert success
    assert torch.isfinite(costs).all()
    torch.testing.assert_close(poses[costs.argmin(), :3, 3], object_pose[:3, 3])
    torch.testing.assert_close(widths, torch.full((2,), 0.04))
    assert costs.min() == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.parametrize(
    "middle_empty_ratio, side_count", [(0.0, 7), (0.4, 4), (0.8, 1)]
)
def test_dual_arm_gap_excludes_middle_and_boundaries_by_pair_center(
    monkeypatch: pytest.MonkeyPatch, middle_empty_ratio: float, side_count: int
) -> None:
    vertices, _ = _box_geometry()
    vertices = vertices * 50.0  # Local X bounds become [-5, 5].
    # Includes the center and both exclusion boundaries for all three ratios.
    local_x = torch.tensor(
        [
            -4.5,
            -4.0,
            -3.5,
            -2.5,
            -2.0,
            -1.5,
            -0.5,
            0.0,
            0.5,
            1.5,
            2.0,
            2.5,
            3.5,
            4.0,
            4.5,
        ]
    )
    centers = torch.zeros(len(local_x), 3)
    centers[:, 0] = local_x
    # Endpoints straddle some boundaries, so selection must use each pair's center.
    contact_offset = torch.tensor([0.4, 0.02, 0.0])
    pairs = torch.stack([centers - contact_offset, centers + contact_offset], dim=1)
    backend = _prepared_backend(vertices, pairs)
    filter_poses = Mock(
        return_value=(True, torch.eye(4)[None], torch.ones(1), torch.zeros(1))
    )
    monkeypatch.setattr(backend, "_filter_valid_grasp_poses", filter_poses)
    object_pose = _object_pose()

    result = backend.get_dual_arm_valid_grasp_poses(
        object_pose=object_pose,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        left_to_right_arm_direction=torch.tensor([0.0, 1.0, 0.0]),
        middle_empty_ratio=middle_empty_ratio,
    )

    assert result is not None
    assert filter_poses.call_count == 2
    transformed_pairs = pairs @ object_pose[:3, :3].T + object_pose[:3, 3]
    for call, expected_pairs, side in zip(
        filter_poses.call_args_list,
        [transformed_pairs[:side_count], transformed_pairs[-side_count:]],
        [-1.0, 1.0],
    ):
        torch.testing.assert_close(call.kwargs["origin_points_"], expected_pairs[:, 0])
        torch.testing.assert_close(call.kwargs["hit_points_"], expected_pairs[:, 1])
        expected_center = object_pose[:3, 3] + torch.tensor([0.0, side * 2.5, 0.0])
        torch.testing.assert_close(call.kwargs["mesh_center"], expected_center)


def _raycast_box(*, subdivide_end: bool) -> tuple[torch.Tensor, torch.Tensor]:
    vertices, triangles = _box_geometry(subdivide_end=subdivide_end)
    sampler = AntipodalSampler()
    sampler.mesh = o3d.t.geometry.TriangleMesh()
    sampler.mesh.vertex.positions = o3d.core.Tensor(vertices.numpy())
    sampler.mesh.triangle.indices = o3d.core.Tensor(triangles.numpy().astype(np.int32))
    # A few thousand deterministic rays cover both halves without a simulation.
    return sampler._sample_surface_by_fibonacci_raycast(vertices, n_sample=4096)


def test_fibonacci_raycast_is_invariant_to_uneven_face_triangulation() -> None:
    points, normals = _raycast_box(subdivide_end=False)
    refined_points, refined_normals = _raycast_box(subdivide_end=True)

    torch.testing.assert_close(refined_points, points, atol=2.0e-6, rtol=0.0)
    torch.testing.assert_close(refined_normals, normals, atol=1.0e-6, rtol=0.0)
    # An off-center sphere focus previously left the sparse half almost unsampled.
    left_fraction = (refined_points[:, 0] < 0.0).float().mean().item()
    assert 0.45 < left_fraction < 0.55


def test_fibonacci_raycast_zero_samples_preserves_shape_and_dtype() -> None:
    vertices, _ = _box_geometry()
    vertices = vertices.to(dtype=torch.float64)

    points, normals = AntipodalSampler()._sample_surface_by_fibonacci_raycast(
        vertices, n_sample=0
    )

    assert points.shape == normals.shape == (0, 3)
    assert points.dtype == normals.dtype == vertices.dtype
    assert points.device == normals.device == vertices.device
