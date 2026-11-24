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
import math

import numpy as np
import open3d as o3d
import pytest
import torch

from embodichain.toolkits.graspkit.pg_grasp._antipodal_backend import (
    _AntipodalMeshBackend,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler import (
    AntipodalSampler,
    AntipodalSamplerCfg,
)
from embodichain.toolkits.graspkit.pg_grasp._ranking import _normalized_ranking_weights


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
    vertices: torch.Tensor, pairs: torch.Tensor, *, center_mode: str = "bounds"
) -> _AntipodalMeshBackend:
    """Bypass sampling and collision construction to test prepared-pair logic."""
    backend = _AntipodalMeshBackend.__new__(_AntipodalMeshBackend)
    backend.device = vertices.device
    backend.vertices = vertices
    backend._hit_point_pairs = pairs
    backend._center_mode = center_mode
    backend._approach_alignment_angle = None
    backend._closing_geometry = None
    backend._ranking_weights = (0.5, 0.5, 0.0)
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


def test_closing_floor_rejection_precedes_top_candidate_ranking() -> None:
    from embodichain.toolkits.graspkit.pg_grasp.contact_fit import SampledHandGeometry

    vertices, _ = _box_geometry()
    centers = torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.06]])
    offset = torch.tensor([0.02, 0.0, 0.0])
    backend = _prepared_backend(
        vertices, torch.stack((centers - offset, centers + offset), 1)
    )
    backend._max_deviation_angle = 0.1
    backend._approach_direction_samples = 1
    backend._max_candidates = 1
    backend._filter_ground_collision = False
    backend._collision_checker = Mock()
    backend._collision_checker.query.side_effect = lambda obj, poses, **kwargs: (
        torch.zeros(len(poses), dtype=torch.bool),
        torch.zeros(len(poses)),
    )
    query = dict(
        object_pose=torch.eye(4), approach_direction=torch.tensor([0.0, 0.0, -1.0])
    )
    success, poses, _, _ = backend.get_valid_grasp_poses(**query)
    assert success and float(poses[0, 2, 3]) == pytest.approx(0.02)
    backend._closing_geometry = SampledHandGeometry(
        torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.04]], [[0.0, 0.0, 0.08]]]),
        torch.zeros(3, 2, 1, 3),
        torch.tensor([0.1, 0.04, 0.0]),
    )
    backend._closing_ground_height_m = 0.0
    backend._closing_ground_clearance_m = 0.002
    backend._contact_half_turn = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    success, poses, widths, costs = backend.get_valid_grasp_poses(**query)
    assert success and len(poses) == 1 and torch.isfinite(costs).all()
    assert float(poses[0, 2, 3]) == pytest.approx(0.06)
    assert float(widths[0]) == pytest.approx(0.04)


@pytest.mark.parametrize(("second_center_x", "narrow_cost"), [(0.08, 0.5), (0.0, 0.0)])
def test_default_grasp_mode_ranks_by_transformed_bounding_box_center(
    second_center_x: float, narrow_cost: float
) -> None:
    vertices, _ = _box_geometry(subdivide_end=True)
    # Grasp across the long and short sides. Both closing axes are perpendicular
    # to the approach, so the costs should tie when both grasps are centered.
    centers = torch.tensor([[0.0, 0.0, 0.0], [second_center_x, 0.0, 0.0]])
    offsets = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.02, 0.0]])
    pairs = torch.stack([centers - offsets, centers + offsets], dim=1)
    backend = _prepared_backend(vertices, pairs)
    backend._max_deviation_angle = 0.1
    backend._approach_direction_samples = 1
    backend._max_candidates = 10
    backend._filter_ground_collision = False
    backend._collision_checker = Mock()
    backend._collision_checker.query.side_effect = lambda obj, poses, **kwargs: (
        torch.zeros(len(poses), dtype=torch.bool),
        torch.zeros(len(poses)),
    )
    object_pose = _object_pose()

    success, poses, widths, costs = backend.get_valid_grasp_poses(
        object_pose=object_pose,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )

    assert success
    torch.testing.assert_close(poses[costs.argmin(), :3, 3], object_pose[:3, 3])
    by_width = widths.argsort()
    torch.testing.assert_close(widths[by_width], torch.tensor([0.04, 0.2]))
    torch.testing.assert_close(costs[by_width], torch.tensor([narrow_cost, 0.0]))


@pytest.mark.parametrize(
    ("weights", "expected_costs", "best_center_x"),
    [
        ((0.5, 0.5, 0.0), [1 / 12, 0.05, 0.5], 0.01),
        ((1e308, 1e308, 0.0), [1 / 12, 0.05, 0.5], 0.01),
        ((0.25, 0.5, 0.25), [1 / 24, 0.175, 0.5], 0.0),
        ((0.0, 1.0, 0.0), [0.0, 0.1, 1.0], 0.0),
    ],
)
def test_ranking_weights_control_costs_and_top_candidate(
    weights: tuple[float, float, float],
    expected_costs: list[float],
    best_center_x: float,
) -> None:
    """A centered 15-degree grasp competes with a narrower aligned grasp."""
    centers = torch.tensor([[0.0, 0, 0], [0.01, 0, 0], [0.1, 0, 0]])
    axes = torch.tensor(
        [[math.cos(math.pi / 12), 0, math.sin(math.pi / 12)], [0, 1, 0], [1, 0, 0]]
    )
    offsets = axes * torch.tensor([0.08, 0.04, 0.08])[:, None] / 2
    backend = _prepared_backend(
        torch.tensor([[-0.2, -0.1, -0.1], [0.2, 0.1, 0.1]]),
        torch.stack([centers - offsets, centers + offsets], dim=1),
    )
    backend._ranking_weights = _normalized_ranking_weights(*weights)
    backend._max_deviation_angle = math.pi / 6
    backend._approach_direction_samples = 1
    backend._max_candidates = 10
    backend._collision_checker = None
    query = dict(
        object_pose=torch.eye(4), approach_direction=torch.tensor([0.0, 0, -1])
    )
    success, poses, _, costs = backend.get_valid_grasp_poses(**query)
    assert success
    by_center = poses[:, 0, 3].argsort()
    torch.testing.assert_close(costs[by_center], torch.tensor(expected_costs))

    backend._max_candidates = 1
    success, poses, _, costs = backend.get_valid_grasp_poses(**query)
    assert success and len(poses) == 1
    assert float(poses[0, 0, 3]) == pytest.approx(best_center_x)
    assert float(costs[0]) == pytest.approx(min(expected_costs), abs=1e-7)


@pytest.mark.parametrize("hard_angle", [None, math.radians(10)])
def test_hard_approach_filter_precedes_nms_and_top_candidate(
    monkeypatch, hard_angle
) -> None:
    """An 11degree representative cannot suppress a nearby legal9degree grasp."""
    from embodichain.toolkits.graspkit.pg_grasp import _antipodal_backend as module
    from embodichain.utils.math import axis_angle_to_rotation_matrix

    centers = torch.tensor([[0.01, 0, 0], [0.011, 0, 0], [0.1, 0, 0]])
    widths = torch.tensor([0.04, 0.02, 0.03])
    delta = torch.zeros_like(centers)
    delta[:, 0] = widths / 2
    pairs = torch.stack((centers - delta, centers + delta), dim=1)
    backend = _prepared_backend(torch.tensor([[-0.1, 0, 0], [0.1, 0, 0]]), pairs)
    backend._max_deviation_angle = math.pi / 6
    backend._approach_alignment_angle = hard_angle
    backend._approach_direction_samples = 1
    backend._max_candidates = 1
    backend._filter_ground_collision = False
    backend._collision_checker = Mock()
    backend._collision_checker.query.side_effect = lambda obj, poses, **kwargs: (
        torch.zeros(len(poses), dtype=torch.bool),
        torch.zeros(len(poses)),
    )
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, :3, 3] = centers
    rotation = axis_angle_to_rotation_matrix(
        torch.tensor([[math.radians(d), 0, 0] for d in [11, 9, 20]])
    )
    poses[:, :3, :3] = rotation @ torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    monkeypatch.setattr(
        _AntipodalMeshBackend,
        "_grasp_pose_from_approach_direction",
        staticmethod(lambda *args: poses),
    )
    nms = Mock(wraps=module.pose_nms)
    monkeypatch.setattr(module, "pose_nms", nms)
    success, selected, selected_width, cost = backend.get_valid_grasp_poses(
        object_pose=torch.eye(4), approach_direction=torch.tensor([0.0, 0.0, -1.0])
    )
    index = 0 if hard_angle is None else 1
    assert success and len(selected) == 1 and torch.isfinite(cost).all()
    torch.testing.assert_close(selected[0], poses[index])
    torch.testing.assert_close(selected_width, widths[index : index + 1])
    assert len(nms.call_args.args[0]) == (3 if hard_angle is None else 1)
    backend._approach_alignment_angle = math.radians(5)
    nms.reset_mock()
    backend._collision_checker.query.reset_mock()
    success, *_ = backend.get_valid_grasp_poses(
        object_pose=torch.eye(4), approach_direction=torch.tensor([0.0, 0.0, -1.0])
    )
    assert not success
    nms.assert_not_called()
    backend._collision_checker.query.assert_not_called()


def test_centroid_mode_preserves_vertex_weighted_ranking() -> None:
    vertices, _ = _box_geometry(subdivide_end=True)
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
    contact_offset = torch.tensor([0.0, 0.02, 0.0])
    pairs = torch.stack([centers - contact_offset, centers + contact_offset], dim=1)
    backend = _prepared_backend(vertices, pairs, center_mode="centroid")
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

    success, poses, _, costs = backend.get_valid_grasp_poses(
        object_pose=object_pose,
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )

    assert success
    expected = centers[1] @ object_pose[:3, :3].T + object_pose[:3, 3]
    torch.testing.assert_close(poses[costs.argmin(), :3, 3], expected)


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


def test_centroid_dual_arm_partition_does_not_duplicate_crossing_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices, _ = _box_geometry()
    vertices[:, 0] *= 10.0
    backend = _prepared_backend(
        vertices,
        # Both contacts lie beyond the gap, but their center lies inside it.
        torch.tensor([[[-0.9, 0.0, 0.0], [0.9, 0.0, 0.0]]]),
        center_mode="centroid",
    )
    filter_poses = Mock(
        return_value=(True, torch.eye(4)[None], torch.ones(1), torch.zeros(1))
    )
    monkeypatch.setattr(backend, "_filter_valid_grasp_poses", filter_poses)

    result = backend.get_dual_arm_valid_grasp_poses(
        object_pose=torch.eye(4),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        left_to_right_arm_direction=torch.tensor([1.0, 0.0, 0.0]),
        middle_empty_ratio=0.8,
    )

    assert result is not None
    assert filter_poses.call_count == 2
    assert filter_poses.call_args_list[0].kwargs["origin_points_"].numel() == 0
    assert filter_poses.call_args_list[1].kwargs["origin_points_"].numel() == 0


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


def test_antipodal_ray_deviation_adds_bounded_reproducible_contact_directions() -> None:
    """Honor configured coverage while retaining the original normal-ray pairs."""
    vertices, triangles = _box_geometry()
    cfg = AntipodalSamplerCfg(n_sample=512, max_angle=0.0, max_length=0.3)
    normal = AntipodalSampler(cfg).sample(vertices, triangles)
    assert len(normal) > 0
    cfg = cfg.replace(max_angle=float(np.pi / 12))
    torch.manual_seed(43001)
    state = torch.random.get_rng_state()
    varied = AntipodalSampler(cfg).sample(vertices, triangles)
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.manual_seed(43002)
    repeated = AntipodalSampler(cfg).sample(vertices, triangles)
    torch.testing.assert_close(varied, repeated, rtol=0.0, atol=0.0)
    torch.testing.assert_close(varied[: len(normal)], normal)
    assert len(varied) > len(normal)
    axes = torch.nn.functional.normalize(varied[:, 1] - varied[:, 0], dim=1)
    assert ((axes.abs() > 1.0e-4).sum(dim=1) > 1).any()
    deviation = torch.acos(axes.abs().amax(dim=1).clamp(0.0, 1.0))
    assert (deviation <= cfg.max_angle + 1.0e-5).all()
    half_extents = torch.tensor([0.1, 0.02, 0.02])
    assert (varied.abs() <= half_extents + 2.0e-6).all()
    surface_distance = (varied.abs() - half_extents).abs().amin(dim=2)
    assert (surface_distance < 2.0e-6).all()
