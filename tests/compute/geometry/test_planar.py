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

"""Arena planar group transform oracles and device/dtype contracts."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from embodichain.compute.geometry.planar import compose_planar_offsets


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_planar_yaw_preserves_tilt_mounting_height_and_inputs(dtype):
    reference = torch.tensor(
        [[[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]]], dtype=dtype
    )
    before = reference.clone()
    offsets = torch.tensor([[[0.5, -0.5, math.pi / 2]], [[0, 0, 0]]], dtype=dtype)
    result = compose_planar_offsets(reference, offsets)
    expected = torch.tensor(
        [[0, 0, 1, 1.5], [1, 0, 0, 1.5], [0, 1, 0, 3], [0, 0, 0, 1]], dtype=dtype
    )
    assert torch.allclose(result[0, 0], expected, atol=1e-6, rtol=0)
    assert torch.equal(result[1, 0], before[0]) and torch.equal(reference, before)
    result.zero_()
    assert torch.equal(reference, before)


@pytest.mark.gpu
def test_planar_cuda_matches_cpu_with_noncontiguous_offsets():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    reference = torch.eye(4).repeat(2, 1, 1)
    offsets = torch.tensor([[[0.1, 0.2, 0.3], [0, 0, 0], [0.4, 0.5, 0.6], [0, 0, 0]]])[
        :, ::2
    ]
    assert not offsets.is_contiguous()
    actual = compose_planar_offsets(reference.cuda(), offsets.cuda()).cpu()
    assert torch.allclose(
        actual, compose_planar_offsets(reference, offsets), atol=1e-6, rtol=0
    )


def test_closed_mesh_membership_preserves_disconnected_components_and_chunking():
    import trimesh
    from embodichain.compute.geometry.solid import points_in_closed_mesh

    left = trimesh.creation.box()
    left.apply_translation((-2, 0, 0))
    right = trimesh.creation.box()
    right.apply_translation((2, 0, 0))
    mesh = trimesh.util.concatenate((left, right))
    vertices = torch.tensor(np.array(mesh.vertices, copy=True), dtype=torch.float64)
    faces = torch.tensor(np.array(mesh.faces, copy=True), dtype=torch.int64)
    points = torch.tensor(
        [[-2.0, 0, 0], [2.0, 0, 0], [0.0, 0, 0], [4.0, 0, 0]], dtype=torch.float64
    )
    expected = torch.tensor([True, True, False, False])
    assert torch.equal(
        points_in_closed_mesh(
            points, vertices, faces, point_chunk_size=1, triangle_chunk_size=3
        ),
        expected,
    )
    assert torch.equal(points_in_closed_mesh(points, vertices, faces), expected)


@pytest.mark.gpu
def test_closed_mesh_cuda_membership_matches_cpu_without_autograd_history():
    import trimesh
    from embodichain.compute.geometry.solid import points_in_closed_mesh

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    vertices = torch.tensor(np.array(mesh.vertices), dtype=torch.float64)
    triangles = torch.tensor(np.array(mesh.faces), dtype=torch.int64)
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, -0.9, 0.9], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actual = points_in_closed_mesh(
        points.cuda(), vertices.cuda(), triangles.cuda(), triangle_chunk_size=3
    )
    assert actual.is_cuda and not actual.requires_grad
    assert actual.cpu().tolist() == [True, True, False]
    assert torch.equal(actual.cpu(), points_in_closed_mesh(points, vertices, triangles))
