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

"""Cooperative CUDA inference agrees with the serial differentiable solver."""

from __future__ import annotations

import pytest
import torch

from embodichain.compute.trajectory._toppra_warp import _retime_toppra_warp

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dofs", [1, 14])
@pytest.mark.parametrize("batch", [4, 129])
@pytest.mark.parametrize("sampling", [{"sample_count": 17}, {"sample_dt": 0.2}])
def test_cuda_cooperative_inference_preserves_serial_solution(
    dofs: int, batch: int, sampling: dict[str, int | float]
) -> None:
    """Cover multi-warp planes, both projection routes, and isolated failure rows."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(804)
    points = 0.03 * torch.randn(
        batch, 6, dofs, generator=generator, dtype=torch.float64
    ).cumsum(dim=1)
    points[1] = points[1, :1].clone()
    points[2, 0, 0] = torch.nan
    points[3, 0, 0] = torch.inf
    points = points.cuda()
    lengths = (4 + torch.arange(batch) % 3).to(device="cuda", dtype=torch.int32)
    velocity = torch.stack(
        (-torch.linspace(0.7, 1.1, dofs), torch.linspace(0.8, 1.2, dofs)), dim=-1
    ).to(device="cuda", dtype=torch.float64)
    acceleration = velocity * 2.0
    # Requires-grad uses the independently retained serial projection/solve
    # kernels; inference uses cooperative plane reductions and, for large
    # batches, the fast projection.
    reference = _retime_toppra_warp(
        points.clone().requires_grad_(),
        velocity,
        acceleration,
        path_lengths=lengths,
        **sampling,
    )
    with torch.no_grad():
        actual = _retime_toppra_warp(
            points, velocity, acceleration, path_lengths=lengths, **sampling
        )
    assert reference["success"][:2].all()
    assert not reference["success"][2:4].any()
    assert reference["success"][4:].all()
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_cuda_cooperative_solver_rejects_finite_infeasible_rows(scale: float) -> None:
    """One-sided motion fails inside solve while frozen motion fails in prepare."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # A large mixed batch also exercises fast projection. The decreasing
    # path contains finite inputs but cannot move under a nonnegative speed.
    points = torch.tensor(
        [
            [[0.0, 2.0], [1.0, 2.0]],
            [[0.0, 2.0], [-1.0, 2.0]],
            [[0.0, 2.0], [1.0, 3.0]],
            [[0.5, 2.0], [0.5, 2.0]],
        ],
        dtype=torch.float64,
        device="cuda",
    ).repeat(32, 1, 1)
    velocity = [[0.0, scale], [0.0, 0.0]]
    acceleration = [[-2.0 * scale**2, scale**2], [0.0, 0.0]]
    reference = _retime_toppra_warp(
        points.clone().requires_grad_(), velocity, acceleration, sample_count=33
    )
    with torch.no_grad():
        actual = _retime_toppra_warp(points, velocity, acceleration, sample_count=33)
    assert actual["success"].tolist() == [True, False, False, True] * 32
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], rtol=1e-9, atol=1e-9)
