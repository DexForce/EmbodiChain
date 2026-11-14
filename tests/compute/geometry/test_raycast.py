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

"""Analytic closest-hit ray query checks."""

from __future__ import annotations

import pytest
import torch

from embodichain.compute.geometry.raycast import TriangleRayMesh, cast_rays


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_batched_mesh_rays_have_independent_closest_hits(device: str) -> None:
    vertices = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], device=device
    )
    mesh = TriangleRayMesh(vertices, torch.tensor([[0, 1, 2]], device=device))
    poses = torch.eye(4, device=device).repeat(2, 1, 1)
    poses[1, 2, 3] = 1.0
    nearer = poses.clone()
    nearer[:, 2, 3] += 0.5
    origins = torch.tensor(
        [[[0.0, 0.0, 3.0], [2.0, 0.0, 3.0], [0.0, 0.0, 3.0]]], device=device
    ).expand(2, -1, -1)
    directions = torch.tensor(
        [[[0.0, 0.0, -2.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]], device=device
    ).expand_as(origins)
    distances, indices = cast_rays(
        origins, directions, [(mesh, poses), (mesh, nearer)], 10.0
    )
    torch.testing.assert_close(distances[:, 0], torch.tensor([2.5, 1.5], device=device))
    assert torch.isinf(distances[:, 1:]).all()
    assert indices.tolist() == [[1, -1, -1], [1, -1, -1]]
