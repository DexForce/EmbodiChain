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

"""Public retiming and Torch-layer contracts beyond the shared solver tests."""

from __future__ import annotations

import io

import pytest
import torch

from embodichain.compute.trajectory import retime_joint_paths
from embodichain.learning.layers import DifferentiableRetimingLayer


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_mixed_lengths_time_quantity_and_layer_padding(device, dtype):
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
    lengths = torch.tensor([3, 3, 3], device=device)
    layer = (
        DifferentiableRetimingLayer(1.0, 2.0, sample_count=17)
        .to(device=device, dtype=dtype)
        .eval()
    )
    result = layer(points, path_lengths=lengths)
    assert result.positions.shape == (3, 17, 1)
    assert result.success.tolist() == [True, True, False]
    assert result.sample_lengths.tolist() == [17, 2, 0]
    torch.testing.assert_close(result.positions[1], points[1, :1].expand(17, 1))
    assert result.dt[1:].count_nonzero() == 0
    assert result.positions[2].count_nonzero() == 0
    trimmed = layer(points[:1, :3])
    torch.testing.assert_close(result.positions[:1], trimmed.positions)
    torch.testing.assert_close(result.duration[:1], trimmed.duration)
    time = retime_joint_paths(
        points[:2],
        constraints={"velocity": 1.0, "acceleration": 2.0},
        path_lengths=lengths[:2],
        sample_dt=0.07,
    )
    assert time.sample_lengths.tolist() == [
        int(torch.ceil(time.duration[0] / 0.07)) + 1,
        2,
    ]
    torch.testing.assert_close(time.duration, result.duration[:2])
    assert time.dt.max() <= 0.07 + 1e-6
    gradients = torch.autograd.grad(result.duration.sum(), points)[0]
    assert torch.isfinite(gradients).all()
    assert gradients[0, :3].norm() > 0
    assert gradients[:, 3].count_nonzero() == 0
    assert gradients[1:].count_nonzero() == 0
    # The all-stationary batch is still fixed-width through the module.
    stationary = layer(points[1:2, :2])
    assert stationary.positions.shape == (1, 17, 1)
    assert stationary.sample_lengths.item() == 2
    failed = layer(points[2:3, :3])
    assert failed.positions.shape == (1, 17, 1)
    assert failed.sample_lengths.item() == 0


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
        retime_joint_paths(
            torch.zeros(1, 3, 1),
            constraints={"velocity": 1.0, "acceleration": 2.0},
            path_lengths=bad,
        )


def test_time_gradient_reaches_predictor_and_checkpoint_roundtrip():
    torch.manual_seed(37)
    predictor = torch.nn.Linear(2, 3, dtype=torch.float64)
    layer = (
        DifferentiableRetimingLayer(1.0, 2.0, sample_count=19, grid_size=35)
        .double()
        .eval()
    )
    features = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
    points = predictor(features).reshape(1, 3, 1)
    result = layer(points)
    other = layer(points + 0.001)
    snapshot = result.positions.detach().clone()
    grad = torch.autograd.grad(
        result.duration.sum(), predictor.weight, retain_graph=True
    )[0]
    again = torch.autograd.grad(
        result.duration.sum(), predictor.weight, retain_graph=True
    )[0]
    torch.testing.assert_close(grad, again)
    assert grad.norm() > 0 and torch.isfinite(grad).all()
    (result.duration.sum() + other.duration.sum()).backward()
    torch.testing.assert_close(result.positions, snapshot)
    assert not list(layer.parameters())
    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)
    restored = DifferentiableRetimingLayer(9.0, 9.0).double()
    restored.load_state_dict(torch.load(buffer, weights_only=True), assign=True)
    assert restored.get_extra_state() == layer.get_extra_state()
    with torch.no_grad():
        actual = restored(points)
        override = restored(points, velocity_limits=0.5, acceleration_limits=0.5)
    torch.testing.assert_close(actual.duration, result.duration)
    assert (override.duration >= actual.duration).all()
    assert not actual.duration.requires_grad
    with pytest.raises(ValueError, match="dtype"):
        restored(points.float())
    with pytest.raises(ValueError, match="float32 and float64"):
        restored.half()


def test_fixed_endpoint_first_order_update_reduces_duration():
    points = torch.tensor(
        [[[0.0, 0.0], [0.25, 0.4], [0.7, -0.15], [1.0, 0.2]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    constraints = {"velocity": [1.0, 0.6], "acceleration": [2.0, 1.0]}
    result = retime_joint_paths(points, constraints=constraints, sample_count=65)
    gradient = torch.autograd.grad(result.duration.sum(), points)[0]
    # Directional derivative, with endpoints held fixed, on this smooth branch.
    direction = gradient.clone()
    direction[:, [0, -1]] = 0
    direction /= direction.norm()
    eps = 1e-6
    plus = retime_joint_paths(
        points.detach() + eps * direction, constraints=constraints, sample_count=65
    )
    minus = retime_joint_paths(
        points.detach() - eps * direction, constraints=constraints, sample_count=65
    )
    finite = (plus.duration - minus.duration) / (2 * eps)
    torch.testing.assert_close(
        finite, (gradient * direction).sum().reshape(1), rtol=2e-3, atol=2e-4
    )
    improved = False
    for step in [0.01, 0.001, 0.0001]:
        candidate = points.detach() - step * direction
        next_result = retime_joint_paths(
            candidate, constraints=constraints, sample_count=65
        )
        if (
            next_result.success.all()
            and next_result.duration.item() < result.duration.item()
        ):
            torch.testing.assert_close(candidate[:, [0, -1]], points[:, [0, -1]])
            assert next_result.positions.abs().max() < 2
            improved = True
            break
    assert improved


def test_uint8_lengths_on_paths_longer_than_255_nodes():
    result = retime_joint_paths(
        torch.zeros(1, 300, 1),
        constraints={"velocity": 1.0, "acceleration": 2.0},
        path_lengths=torch.tensor([250], dtype=torch.uint8),
        sample_count=2,
    )
    assert result.success.item() and result.sample_lengths.item() == 2
