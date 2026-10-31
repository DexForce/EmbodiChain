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

"""Plain tensor retiming and stateless Torch composition contracts."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from embodichain.compute.trajectory import retime_joint_paths
from embodichain.learning.layers import DifferentiableRetimingLayer


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_mixed_lengths_time_quantity_and_padding(device, dtype):
    points = torch.tensor(
        [
            [[0.0], [0.05], [0.1], [99.0]],
            [[0.2], [0.2], [0.2], [99.0]],
            [[0.0], [float("nan")], [0.1], [99.0]],
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    path_lengths = torch.tensor([3, 3, 3], device=device)
    velocity, acceleration = points.new_tensor(1.0), points.new_tensor(2.0)
    layer = DifferentiableRetimingLayer(sample_count=17).eval()
    outputs = layer(points, velocity, acceleration, path_lengths=path_lengths)
    assert type(outputs) is tuple and len(outputs) == 6
    assert all(isinstance(x, torch.Tensor) for x in outputs)
    q, qd, qdd, dt, success, lengths = outputs
    assert q.shape == (3, 17, 1) and qd.shape == qdd.shape == q.shape
    assert success.tolist() == [True, True, False] and lengths.tolist() == [17, 2, 0]
    torch.testing.assert_close(q[1], points[1, :1].expand(17, 1))
    assert dt[1:].count_nonzero() == 0 and q[2].count_nonzero() == 0
    trimmed = retime_joint_paths(
        points[:1, :3], velocity, acceleration, sample_count=17
    )
    torch.testing.assert_close(q[:1], trimmed[0])
    torch.testing.assert_close(dt[:1].sum(-1), trimmed[3].sum(-1))
    _, _, _, time_dt, _, time_lengths = retime_joint_paths(
        points[:2],
        velocity,
        acceleration,
        path_lengths=path_lengths[:2],
        sample_dt=0.07,
    )
    assert time_lengths.tolist() == [int(torch.ceil(time_dt[0].sum() / 0.07)) + 1, 2]
    torch.testing.assert_close(time_dt.sum(-1), dt[:2].sum(-1))
    assert time_dt.max() <= 0.07 + 1e-6
    grad = torch.autograd.grad(dt.sum(), points)[0]
    assert torch.isfinite(grad).all() and grad[0, :3].norm() > 0
    assert grad[:, 3].count_nonzero() == grad[1:].count_nonzero() == 0
    for row, valid_count in ((1, 2), (2, 0)):
        static_or_failed = retime_joint_paths(
            points[row : row + 1, :3], velocity, acceleration, sample_count=17
        )
        assert static_or_failed[0].shape == (1, 17, 1)
        assert static_or_failed[5].item() == valid_count


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([2**32 + 3]),
        torch.tensor([-(2**32) + 3]),
        torch.tensor([1]),
        torch.tensor([4]),
        torch.tensor([3.0]),
    ],
)
def test_lengths_rejected_before_integer_narrowing(bad):
    with pytest.raises(ValueError, match="path_lengths"):
        retime_joint_paths(torch.zeros(1, 3, 1), 1.0, 2.0, path_lengths=bad)


def test_stateless_layer_composes_with_model_buffers_and_standard_state_dict():
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Linear(2, 3, dtype=torch.float64)
            self.retimer = DifferentiableRetimingLayer(sample_count=19, grid_size=35)
            self.register_buffer("velocity", torch.tensor(1.0, dtype=torch.float64))
            self.register_buffer("acceleration", torch.tensor(2.0, dtype=torch.float64))

        def forward(self, features):
            q = self.network(features).reshape(-1, 3, 1)
            return self.retimer(q, self.velocity, self.acceleration)

    torch.manual_seed(37)
    model = Predictor().eval()
    features = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    first = model(features)
    second = model(features + 0.001)
    snapshot = first[0].detach().clone()
    duration = first[3].sum()
    grad = torch.autograd.grad(duration, model.network.weight, retain_graph=True)[0]
    repeated = torch.autograd.grad(duration, model.network.weight, retain_graph=True)[0]
    torch.testing.assert_close(grad, repeated)
    assert torch.isfinite(grad).all() and grad.norm() > 0
    (duration + second[3].sum()).backward()
    torch.testing.assert_close(first[0], snapshot)
    assert not list(model.retimer.parameters()) and not list(model.retimer.buffers())
    assert model.retimer.state_dict() == {}
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = Predictor()
    restored.load_state_dict(torch.load(stream, weights_only=True), assign=True)
    with torch.no_grad():
        actual = restored(features)
    for a, b in zip(actual, first):
        torch.testing.assert_close(a, b)
    assert not actual[3].requires_grad
    # Numerical configuration is construction state, not implicit weight state.
    other = DifferentiableRetimingLayer(sample_count=7, grid_size=9)
    other.load_state_dict(model.retimer.state_dict())
    assert other.sample_count == 7 and other.grid_size == 9
    assert "sample_count=7" in repr(other)
    # Requests provide their own budgets without mutating module state.
    q = model.network(features).reshape(1, 3, 1)
    with torch.no_grad():
        slow = model.retimer(q, q.new_tensor(0.5), q.new_tensor(0.5))
    assert (slow[3].sum(-1) >= first[3].sum(-1)).all()


def test_fixed_endpoint_first_order_update_reduces_duration():
    points = torch.tensor(
        [[[0.0, 0.0], [0.25, 0.4], [0.7, -0.15], [1.0, 0.2]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    def solve(q):
        return retime_joint_paths(q, [1.0, 0.6], [2.0, 1.0], sample_count=65)

    result = solve(points)
    gradient = torch.autograd.grad(result[3].sum(), points)[0]
    direction = gradient.clone()
    direction[:, [0, -1]] = 0
    direction /= direction.norm()
    eps = 1e-6
    plus, minus = solve(points.detach() + eps * direction), solve(
        points.detach() - eps * direction
    )
    finite = (plus[3].sum(-1) - minus[3].sum(-1)) / (2 * eps)
    torch.testing.assert_close(
        finite, (gradient * direction).sum().reshape(1), rtol=2e-3, atol=2e-4
    )
    improved = False
    for step in [0.01, 0.001, 0.0001]:
        candidate = points.detach() - step * direction
        next_result = solve(candidate)
        if next_result[4].all() and next_result[3].sum() < result[3].sum():
            torch.testing.assert_close(candidate[:, [0, -1]], points[:, [0, -1]])
            assert next_result[0].abs().max() < 2
            improved = True
            break
    assert improved


def test_uint8_lengths_on_paths_longer_than_255_nodes():
    result = retime_joint_paths(
        torch.zeros(1, 300, 1),
        1.0,
        2.0,
        path_lengths=torch.tensor([250], dtype=torch.uint8),
        sample_count=2,
    )
    assert result[4].item() and result[5].item() == 2


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_public_output_precision_keeps_gradient_to_float32_input(device):
    q = torch.tensor([[[0.0], [0.2], [0.5]]], requires_grad=True, device=device)
    positions, _, _, dt, _, _ = retime_joint_paths(
        q, 1.0, 2.0, output_dtype=torch.float64
    )
    assert positions.dtype == dt.dtype == torch.float64
    dt.sum().backward()
    assert q.grad is not None and q.grad.dtype == torch.float32
    assert torch.isfinite(q.grad).all() and q.grad.norm() > 0
